"""Multi-agent PR reviewer.

fetch -> supervisor -> Send(review) per (file, specialist) -> report (dedupe + rank)
      -> [approve: human-in-the-loop interrupt] -> post (GitHub review with inline comments)
"""

import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import httpx
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from reviewer import github
from reviewer.diff import FileDiff, parse_diff
from reviewer.github import parse_pr_url  # re-exported for callers

__all__ = ["build_graph", "make_llm", "parse_pr_url", "PROVIDERS", "Finding", "Plan", "Review"]

Focus = Literal["security", "performance", "style"]

FOCUS_PROMPTS: dict[str, str] = {
    "security": "injection, authn/authz gaps, secrets in code, unsafe deserialization, path traversal, SSRF, XSS, missing input validation",
    "performance": "N+1 queries, needless allocations or copies, blocking I/O in hot or async paths, quadratic loops, missing caching or batching",
    "style": "naming, dead code, duplication, unclear control flow, swallowed or missing error handling, missing tests for new logic",
}
SEVERITY_GUIDE = ("Severity: high = exploitable vulnerability, data loss, or a crash/outage in normal use; "
                  "medium = a likely bug or a significant, measurable slowdown; low = maintainability or readability. "
                  "Missing tests, docs, or type hints are never above low.")
MAX_TOOL_STEPS = 4  # read_file rounds per reviewer before it must answer
# Waits 2s, 4s, 8s, 16s (+jitter, capped at 30s): long enough to outlast per-minute rate limits (429) and
# 5xx "high demand" blips, which short retries didn't on Groq's free tier.
RETRY = {"stop_after_attempt": 5, "wait_exponential_jitter": True, "exponential_jitter_params": {"initial": 2, "max": 30}}
MAX_FILE_CHARS = 60_000
FULL_DIFF_FOR_SUPERVISOR = 100_000  # above this the supervisor sees only the file list
LOCKFILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock", "Pipfile.lock",
             "Cargo.lock", "go.sum", "Gemfile.lock", "composer.lock"}  # generated; reviewing them burns tokens


class Finding(BaseModel):
    file: str
    line: int | None = Field(description="New-file line number, taken from the number shown before the diff line")
    severity: Literal["high", "medium", "low"]
    issue: str
    suggestion: str
    category: SkipJsonSchema[str] = ""  # set by the reviewer node; hidden from the model's schema


class Review(BaseModel):
    findings: list[Finding] = Field(description="Real problems only; empty if none")


class Plan(BaseModel):
    reviewers: list[Focus] = Field(description="Specialists worth running on this PR")
    reason: str


class State(TypedDict, total=False):
    # input
    pr_url: str
    diff: str
    local_dir: str  # for local diffs: repo root that read_file may read from
    post: bool
    auto_approve: bool
    min_severity: str  # "low" (default: show all), "medium", or "high"
    # derived
    owner: str
    repo: str
    number: int
    head_sha: str
    files: dict[str, FileDiff]
    plan: Plan
    findings: Annotated[list[Finding], operator.add]  # reducer: parallel reviewers append
    final: list[Finding]  # deduped + ranked; what gets posted
    errors: Annotated[list[str], operator.add]  # reviewers that failed after retries; the rest still report
    report: str
    failed: bool  # every reviewer errored: don't post, exit non-zero
    approved: bool
    review_url: str


class ReviewTask(TypedDict):
    focus: str
    path: str
    file_diff: FileDiff
    source: dict | None  # where read_file reads from; None disables the tool


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------- deterministic nodes ----------

def fetch(state: State) -> State:
    def reviewable(diff: str) -> dict[str, FileDiff]:
        return {p: d for p, d in parse_diff(diff).items() if Path(p).name not in LOCKFILES}

    if state.get("diff"):  # local diff supplied; skip the network
        return {"files": reviewable(state["diff"])}
    owner, repo, number = parse_pr_url(state["pr_url"])
    diff = github.get_diff(owner, repo, number)
    sha = github.get_pr(owner, repo, number)["head"]["sha"]
    return {"owner": owner, "repo": repo, "number": number, "head_sha": sha, "diff": diff, "files": reviewable(diff)}


def file_source(state: State) -> dict | None:
    if state.get("owner"):
        return {"kind": "github", "owner": state["owner"], "repo": state["repo"], "ref": state["head_sha"]}
    if state.get("local_dir"):
        return {"kind": "local", "root": state["local_dir"]}
    return None


def read_source(source: dict, path: str) -> str:
    try:
        if source["kind"] == "github":
            text = github.get_file(source["owner"], source["repo"], path, source["ref"])
        else:
            root = Path(source["root"]).resolve()
            target = (root / path).resolve()
            if not target.is_relative_to(root):  # the model picks the path; keep it inside the repo
                return f"Error: {path} is outside the repository"
            text = target.read_text(errors="replace")
    except (httpx.HTTPError, OSError) as e:
        return f"Error reading {path}: {e}"
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n... [truncated: file has {len(text)} chars]"
    return text


MERGE_WITHIN_LINES = 1  # reviewers often flag the same problem one line apart (the def vs. the bad call)


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Merge findings in the same file within MERGE_WITHIN_LINES of a group's first line (e.g. security and
    style both flag the same call) into one. Line-less findings are never merged."""
    groups: list[list[Finding]] = []
    for f in sorted(findings, key=lambda f: (f.file, f.line or 0)):
        g = groups[-1] if groups else None
        if f.line and g and g[0].file == f.file and g[0].line and f.line - g[0].line <= MERGE_WITHIN_LINES:
            g.append(f)
        else:
            groups.append([f])
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        group = sorted(group, key=lambda f: SEVERITY_ORDER[f.severity])  # worst first: the card leads with why it's rated so
        uniq = lambda xs: list(dict.fromkeys(xs))
        bullets = lambda attr: "\n".join(uniq(f"- **{f.category}:** {getattr(f, attr)}" for f in group))
        merged.append(Finding(
            file=group[0].file, line=group[0].line, severity=group[0].severity,
            issue=bullets("issue"), suggestion="\n" + bullets("suggestion"),
            category=", ".join(uniq(f.category for f in group)),
        ))
    return sorted(merged, key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line or 0))


def _inline(text: str) -> str:
    """Merged findings carry one bullet per specialist; nest them under a list item instead of running them on."""
    text = text.strip()
    return text if "\n" not in text else "\n" + "\n".join("     " + ln for ln in text.splitlines())


def render_report(state: State) -> State:
    merged = dedupe(state.get("findings", []))
    floor = state.get("min_severity", "low")
    final = [f for f in merged if SEVERITY_ORDER[f.severity] <= SEVERITY_ORDER[floor]]
    hidden = len(merged) - len(final)
    plan = state.get("plan")
    lines = ["# PR Review", ""]
    if plan:
        lines += [f"**Reviewers run:** {', '.join(plan.reviewers) or 'none'} on {len(state.get('files', {}))} file(s) — {plan.reason}", ""]
    errors = state.get("errors", [])
    planned = len(state.get("files", {})) * len(plan.reviewers) if plan else 0
    failed = planned > 0 and sum(not e.startswith("supervisor:") for e in errors) >= planned
    if failed:  # never let a total outage read as a clean bill of health
        lines += [f"**Review failed:** every reviewer errored ({'; '.join(errors)}). Nothing was reviewed."]
        return {"final": [], "report": "\n".join(lines), "failed": True}
    if errors:
        lines += [f"**Warning: incomplete review.** Failed: {'; '.join(errors)}", ""]
    if not final:
        lines.append("No issues found.")
    for i, f in enumerate(final, 1):
        loc = f"{f.file}:{f.line}" if f.line else f.file
        lines += [f"{i}. **[{f.severity.upper()}] [{f.category}]** `{loc}` — {_inline(f.issue)}", f"   - Fix: {_inline(f.suggestion)}"]
    if hidden:
        lines += ["", f"_{hidden} finding(s) below {floor} severity hidden._"]
    return {"final": final, "report": "\n".join(lines)}


def approve(state: State) -> State:
    # Pauses the graph; the caller resumes with Command(resume={"approved": bool, "drop": [1-based numbers]}).
    decision = interrupt({"report": state["report"], "count": len(state["final"])})
    drop = set(decision.get("drop", []))
    return {"approved": bool(decision.get("approved")), "final": [f for i, f in enumerate(state["final"], 1) if i not in drop]}


def comment_body(f: Finding) -> str:
    return f"**[{f.severity.upper()}] [{f.category}]** {f.issue}\n\n**Suggested fix:** {f.suggestion}"


def post(state: State) -> State:
    files = state.get("files", {})
    inline, general = [], []
    for f in state["final"]:
        (inline if f.line and f.line in files.get(f.file, {}).get("lines", []) else general).append(f)
    plan = state.get("plan")
    body = [f"## Multi-agent review\n\n{len(state['final'])} finding(s), {len(inline)} inline."]
    if plan:
        body.append(f"Reviewers: {', '.join(plan.reviewers) or 'none'} — {plan.reason}")
    if errors := state.get("errors"):
        body.append(f"**Warning: incomplete review.** Failed: {'; '.join(errors)}")
    if not state["final"]:
        body.append("No issues found.")
    if general:
        body.append("### Findings outside the diff\n" + "\n".join(
            f"- `{f.file}{':' + str(f.line) if f.line else ''}` **[{f.severity.upper()}] [{f.category}]** {_inline(f.issue)}\n"
            f"   - Fix: {_inline(f.suggestion)}" for f in general))
    comments = [{"path": f.file, "line": f.line, "side": "RIGHT", "body": comment_body(f)} for f in inline]
    url = github.post_review(state["owner"], state["repo"], state["number"], state["head_sha"], "\n\n".join(body), comments)
    return {"review_url": url}


# ---------- providers ----------

# provider -> (default model, structured-output method)
# anthropic: "function_calling" forces tool_choice, which Opus 5.5 rejects, so it uses native json_schema.
# groq: tool calling is reliable. openrouter: measured on gpt-oss-120b, tool calling came back empty in ~half
# of calls (some hosts ignore tool_choice); json_schema never did, and its occasional 5xx are retried.
PROVIDERS = {
    "gemini": ("gemini-flash-latest", "json_schema"),  # GEMINI_API_KEY or GOOGLE_API_KEY; pro models have no free tier
    "groq": ("openai/gpt-oss-120b", "function_calling"),  # GROQ_API_KEY
    "openrouter": ("openai/gpt-oss-120b", "json_schema"),  # OPENROUTER_API_KEY
    "anthropic": ("claude-opus-5-5", "json_schema"),  # ANTHROPIC_API_KEY
}


def make_llm(provider: str, model: str | None = None):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
    default_model, method = PROVIDERS[provider]
    model = model or default_model
    # Imports are lazy so a missing or broken provider package only matters when that provider is chosen.
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # max_retries=1 disables the Google SDK's own 6 retries (0 means "SDK default"), so they don't multiply ours.
        llm = ChatGoogleGenerativeAI(model=model, max_retries=1)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model)
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        if not (key := os.environ.get("OPENROUTER_API_KEY")):
            raise RuntimeError("Set OPENROUTER_API_KEY")
        llm = ChatOpenAI(model=model, base_url="https://openrouter.ai/api/v1", api_key=key)
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, effort="high", max_tokens=16000)
    return llm, method


# ---------- graph ----------

class NoStructuredOutput(ValueError):
    pass


def _require_output(result):
    if result is None:
        raise NoStructuredOutput("model returned no structured output")
    return result


def build_graph(llm=None, method: str = "json_schema", checkpointer=None):
    """checkpointer is required for the human approval step (state must survive the interrupt)."""
    if llm is None:
        llm, method = make_llm("gemini")
    # With tool-calling structured output, a model that answers in prose yields None instead of raising;
    # turn that into an error so it is retried (and, if it persists, handled like any other failure).
    require = RunnableLambda(_require_output)
    planner = (llm.with_structured_output(Plan, method=method) | require).with_retry(**RETRY)
    reviewer = (llm.with_structured_output(Review, method=method) | require).with_retry(**RETRY)

    def supervisor(state: State) -> State:
        files = state["files"]
        summary = "\n".join(f"{p} (+{d['added']} -{d['removed']})" for p, d in files.items())
        diff = state.get("diff", "")
        body = f"Changed files:\n{summary}"
        if len(diff) <= FULL_DIFF_FOR_SUPERVISOR:
            body += f"\n\nDiff:\n{diff}"
        try:
            plan = planner.invoke([
                ("system", "You are a lead code reviewer. Pick which specialist reviewers "
                           "(security, performance, style) are worth running on this pull request. "
                           "Skip a specialist only if the change clearly has nothing in its area "
                           "(e.g. docs-only changes need no security or performance review)."),
                ("user", body),
            ])
        except Exception as e:  # planning is an optimization; without it, run every specialist
            return {"plan": Plan(reviewers=["security", "performance", "style"], reason="supervisor failed; ran all reviewers"),
                    "errors": [f"supervisor: {type(e).__name__}"]}
        return {"plan": plan}

    def route(state: State):
        if not state["files"]:
            return "report"
        source = file_source(state)
        tasks = [Send("review", {"focus": f, "path": p, "file_diff": d, "source": source})
                 for p, d in state["files"].items() for f in dict.fromkeys(state["plan"].reviewers)]
        return tasks or "report"

    def review(task: ReviewTask) -> State:
        # Progress events for live UIs (stream_mode="custom"); a no-op when nobody is streaming.
        emit = get_stream_writer()
        ids = {"focus": task["focus"], "path": task["path"]}
        emit({"event": "review_start", **ids})
        try:
            out = run_reviewer(task, emit)
        except Exception as e:  # one failed reviewer must not sink the whole review
            err = f"{task['focus']} on {task['path']}: {type(e).__name__}"
            emit({"event": "review_error", **ids, "error": type(e).__name__})
            return {"errors": [err]}
        emit({"event": "review_done", **ids, "findings": len(out["findings"])})
        return out

    def run_reviewer(task: ReviewTask, emit) -> State:
        system = (f"You are a {task['focus']} reviewer. Review ONLY for: {FOCUS_PROMPTS[task['focus']]}. "
                  f"You are reviewing changes to `{task['path']}`. Report concrete problems in the changed (+) lines. "
                  "Each diff line is prefixed with its new-file line number; use that number. "
                  f"{SEVERITY_GUIDE} Only report issues you are confident are real. "
                  "No praise, no speculation, no issues outside your specialty.")
        diff_msg = ("user", f"Diff for `{task['path']}`:\n{task['file_diff']['numbered']}")
        notes = []
        if task["source"]:
            @tool
            def read_file(path: str) -> str:
                """Read a file from the repository at the PR's head commit. `path` is relative to the repo root."""
                return read_source(task["source"], path)

            msgs = [("system", system + " Use read_file when you need surrounding code, callers, or definitions "
                                        "to judge a change. When you have enough context, reply with your analysis."), diff_msg]
            agent = llm.bind_tools([read_file]).with_retry(**RETRY)
            for _ in range(MAX_TOOL_STEPS):
                ai = agent.invoke(msgs)
                msgs.append(ai)
                if not ai.tool_calls:
                    notes.append(f"Your analysis:\n{ai.text}")
                    break
                for call in ai.tool_calls:
                    emit({"event": "read_file", "focus": task["focus"], "path": task["path"], "file": call["args"].get("path")})
                    result = read_file.invoke(call)  # ToolMessage
                    msgs.append(result)
                    notes.append(f"read_file({call['args'].get('path')!r}):\n{result.content}")
        # Final structured call gets a fresh, provider-neutral conversation: some APIs reject tool history
        # that references tools not offered on the current request.
        final_msgs = [("system", system), diff_msg]
        if notes:
            final_msgs.append(("user", "Context you gathered:\n\n" + "\n\n".join(notes)))
        final_msgs.append(("user", "Now list your findings."))
        result = reviewer.invoke(final_msgs)
        # The reviewer only sees this file's diff, so pin the path (models sometimes write "b/x.py" or a bare name).
        # Style is maintainability: real vulnerabilities and slowdowns belong to the other specialists, and live runs
        # showed the style reviewer rating "no try/except" as HIGH, so cap it at medium.
        cap = {"severity": "medium"} if task["focus"] == "style" else {}
        return {"findings": [f.model_copy(update={"category": task["focus"], "file": task["path"],
                                                  **(cap if f.severity == "high" else {})}) for f in result.findings]}

    def after_report(state: State):
        if not state.get("post") or state.get("failed"):
            return END
        return "post" if state.get("auto_approve") else "approve"

    g = StateGraph(State)
    g.add_node("fetch", fetch)
    g.add_node("supervisor", supervisor)
    g.add_node("review", review)
    g.add_node("report", render_report)
    g.add_node("approve", approve)
    g.add_node("post", post)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "supervisor")
    g.add_conditional_edges("supervisor", route, ["review", "report"])
    g.add_edge("review", "report")
    g.add_conditional_edges("report", after_report, ["approve", "post", END])
    g.add_conditional_edges("approve", lambda s: "post" if s["approved"] else END, ["post", END])
    g.add_edge("post", END)
    return g.compile(checkpointer=checkpointer)

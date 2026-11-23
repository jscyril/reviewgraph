"""Multi-agent PR reviewer.

fetch -> supervisor -> Send(review) per (file, specialist) -> report (dedupe + rank)
      -> [approve: human-in-the-loop interrupt] -> post (GitHub review with inline comments)
"""

import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import httpx
from langchain_core.tools import tool
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
MAX_TOOL_STEPS = 4  # read_file rounds per reviewer before it must answer
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
    # derived
    owner: str
    repo: str
    number: int
    head_sha: str
    files: dict[str, FileDiff]
    plan: Plan
    findings: Annotated[list[Finding], operator.add]  # reducer: parallel reviewers append
    final: list[Finding]  # deduped + ranked; what gets posted
    report: str
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


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Merge findings on the same file+line (e.g. security and style both flag it) into one."""
    groups: dict[tuple, list[Finding]] = {}
    for i, f in enumerate(findings):
        groups.setdefault((f.file, f.line) if f.line else (f.file, None, i), []).append(f)
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        uniq = lambda xs: list(dict.fromkeys(xs))
        merged.append(Finding(
            file=group[0].file, line=group[0].line,
            severity=min((f.severity for f in group), key=SEVERITY_ORDER.get),
            issue=" Also: ".join(uniq(f.issue for f in group)),
            suggestion=" Also: ".join(uniq(f.suggestion for f in group)),
            category=", ".join(uniq(f.category for f in group)),
        ))
    return sorted(merged, key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line or 0))


def render_report(state: State) -> State:
    final = dedupe(state.get("findings", []))
    plan = state.get("plan")
    lines = ["# PR Review", ""]
    if plan:
        lines += [f"**Reviewers run:** {', '.join(plan.reviewers) or 'none'} on {len(state.get('files', {}))} file(s) — {plan.reason}", ""]
    if not final:
        lines.append("No issues found.")
    for i, f in enumerate(final, 1):
        loc = f"{f.file}:{f.line}" if f.line else f.file
        lines += [f"{i}. **[{f.severity.upper()}] [{f.category}]** `{loc}` — {f.issue}", f"   - Fix: {f.suggestion}"]
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
    if not state["final"]:
        body.append("No issues found.")
    if general:
        body.append("### Findings outside the diff\n" + "\n".join(
            f"- `{f.file}{':' + str(f.line) if f.line else ''}` " + comment_body(f).replace("\n\n", " ") for f in general))
    comments = [{"path": f.file, "line": f.line, "side": "RIGHT", "body": comment_body(f)} for f in inline]
    url = github.post_review(state["owner"], state["repo"], state["number"], state["head_sha"], "\n\n".join(body), comments)
    return {"review_url": url}


# ---------- providers ----------

# provider -> (default model, structured-output method)
# anthropic: "function_calling" forces tool_choice, which Opus 5.5 rejects, so it uses native json_schema.
# groq/openrouter: json_schema support varies by hosted model; tool calling works across nearly all of them.
PROVIDERS = {
    "gemini": ("gemini-pro-latest", "json_schema"),  # GEMINI_API_KEY or GOOGLE_API_KEY
    "groq": ("openai/gpt-oss-120b", "function_calling"),  # GROQ_API_KEY
    "openrouter": ("openai/gpt-oss-120b", "function_calling"),  # OPENROUTER_API_KEY
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
        llm = ChatGoogleGenerativeAI(model=model)
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

def build_graph(llm=None, method: str = "json_schema", checkpointer=None):
    """checkpointer is required for the human approval step (state must survive the interrupt)."""
    if llm is None:
        llm, method = make_llm("gemini")
    planner = llm.with_structured_output(Plan, method=method)
    reviewer = llm.with_structured_output(Review, method=method)

    def supervisor(state: State) -> State:
        files = state["files"]
        summary = "\n".join(f"{p} (+{d['added']} -{d['removed']})" for p, d in files.items())
        diff = state.get("diff", "")
        body = f"Changed files:\n{summary}"
        if len(diff) <= FULL_DIFF_FOR_SUPERVISOR:
            body += f"\n\nDiff:\n{diff}"
        plan = planner.invoke([
            ("system", "You are a lead code reviewer. Pick which specialist reviewers "
                       "(security, performance, style) are worth running on this pull request. "
                       "Skip a specialist only if the change clearly has nothing in its area "
                       "(e.g. docs-only changes need no security or performance review)."),
            ("user", body),
        ])
        return {"plan": plan}

    def route(state: State):
        if not state["files"]:
            return "report"
        source = file_source(state)
        tasks = [Send("review", {"focus": f, "path": p, "file_diff": d, "source": source})
                 for p, d in state["files"].items() for f in dict.fromkeys(state["plan"].reviewers)]
        return tasks or "report"

    def review(task: ReviewTask) -> State:
        system = (f"You are a {task['focus']} reviewer. Review ONLY for: {FOCUS_PROMPTS[task['focus']]}. "
                  f"You are reviewing changes to `{task['path']}`. Report concrete problems in the changed (+) lines. "
                  "Each diff line is prefixed with its new-file line number; use that number. "
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
            agent = llm.bind_tools([read_file])
            for _ in range(MAX_TOOL_STEPS):
                ai = agent.invoke(msgs)
                msgs.append(ai)
                if not ai.tool_calls:
                    notes.append(f"Your analysis:\n{ai.text}")
                    break
                for call in ai.tool_calls:
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
        return {"findings": [f.model_copy(update={"category": task["focus"], "file": task["path"]}) for f in result.findings]}

    def after_report(state: State):
        if not state.get("post"):
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

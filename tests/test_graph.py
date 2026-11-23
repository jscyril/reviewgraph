import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from reviewer import graph as g
from reviewer.graph import PROVIDERS, Finding, Plan, Review, build_graph, dedupe, make_llm, parse_pr_url, read_source


def new_file_diff(path: str, n: int) -> str:
    body = "".join(f"+line {i}\n" for i in range(1, n + 1))
    return f"diff --git a/{path} b/{path}\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{n} @@\n{body}"


class StubLLM:
    """Stands in for any chat model: canned plan, canned findings per (focus, file), optional scripted tool use."""

    def __init__(self, plan: Plan, findings=None, tool_path: str | None = None, fail_on=()):
        self.plan, self.findings, self.tool_path, self.fail_on = plan, findings or {}, tool_path, set(fail_on)
        self.review_calls, self.final_prompts = [], []

    def with_structured_output(self, schema, method=None):
        stub = self

        def invoke(messages):
            if schema is Plan:
                if "plan" in stub.fail_on:
                    raise ConnectionError("503 high demand")
                return stub.plan
            system = messages[0][1]
            focus = next(f for f in ("security", "performance", "style") if f"a {f} reviewer" in system)
            path = system.split("`")[1]
            stub.review_calls.append((focus, path))
            if focus in stub.fail_on:
                raise ConnectionError("503 high demand")
            stub.final_prompts.append("\n".join(m[1] for m in messages))
            return Review(findings=stub.findings.get((focus, path), []))

        return RunnableLambda(invoke)

    def bind_tools(self, tools):
        stub = self

        def invoke(messages):
            already_called = any(getattr(m, "type", None) == "tool" for m in messages)
            if stub.tool_path and not already_called:
                return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": stub.tool_path}, "id": "c1"}])
            return AIMessage(content="looked fine except noted issues")

        return RunnableLambda(invoke)


def finding(sev, file="app.py", line=1, issue=None):
    return Finding(file=file, line=line, severity=sev, issue=issue or f"{sev} issue", suggestion="fix it")


def test_fans_out_per_file_and_reviewer_and_ranks():
    diff = new_file_diff("a.py", 3) + new_file_diff("b.py", 3)
    llm = StubLLM(Plan(reviewers=["style", "security"], reason="code"),
                  {("security", "a.py"): [finding("high", line=2)], ("style", "b.py"): [finding("low", line=1)]})
    out = build_graph(llm).invoke({"diff": diff})

    assert sorted(llm.review_calls) == [("security", "a.py"), ("security", "b.py"), ("style", "a.py"), ("style", "b.py")]
    assert out["report"].index("[HIGH] [security]** `a.py:2`") < out["report"].index("[LOW] [style]** `b.py:1`")
    assert [f.file for f in out["final"]] == ["a.py", "b.py"]  # file pinned to the reviewed path


def test_lockfiles_are_not_reviewed():
    llm = StubLLM(Plan(reviewers=["security"], reason="deps"))
    build_graph(llm).invoke({"diff": new_file_diff("web/package-lock.json", 5) + new_file_diff("web/index.js", 2)})
    assert llm.review_calls == [("security", "web/index.js")]


def test_failed_reviewer_is_retried_then_reported_without_sinking_the_rest(monkeypatch):
    monkeypatch.setattr(g, "RETRY", {"stop_after_attempt": 3, "wait_exponential_jitter": False})
    llm = StubLLM(Plan(reviewers=["security", "style"], reason="code"),
                  {("style", "a.py"): [finding("low", line=1)]}, fail_on={"security"})
    out = build_graph(llm).invoke({"diff": new_file_diff("a.py", 2)})
    assert llm.review_calls.count(("security", "a.py")) == 3  # retried
    assert out["errors"] == ["security on a.py: ConnectionError"]
    assert len(out["final"]) == 1 and "incomplete review" in out["report"]


def test_total_failure_is_reported_as_failed_and_not_posted(monkeypatch, fake_github):
    monkeypatch.setattr(g, "RETRY", {"stop_after_attempt": 1})
    llm = StubLLM(Plan(reviewers=["security"], reason="code"), fail_on={"security"})
    out = build_graph(llm).invoke({"pr_url": "https://github.com/o/r/pull/7", "post": True, "auto_approve": True})
    assert out["failed"] and "Review failed" in out["report"] and "No issues found" not in out["report"]
    assert not fake_github  # nothing posted


def test_missing_structured_output_is_retried_then_falls_back(monkeypatch):
    monkeypatch.setattr(g, "RETRY", {"stop_after_attempt": 2, "wait_exponential_jitter": False})
    llm = StubLLM(None, {("security", "a.py"): [finding("high", line=1)]})  # planner "answers in prose": returns None
    out = build_graph(llm).invoke({"diff": new_file_diff("a.py", 2)})
    assert out["errors"] == ["supervisor: NoStructuredOutput"]
    assert ("security", "a.py") in llm.review_calls and len(out["final"]) == 1


def test_supervisor_failure_falls_back_to_all_reviewers(monkeypatch):
    monkeypatch.setattr(g, "RETRY", {"stop_after_attempt": 1})
    llm = StubLLM(Plan(reviewers=[], reason="unused"), fail_on={"plan"})
    out = build_graph(llm).invoke({"diff": new_file_diff("a.py", 2)})
    assert sorted(f for f, _ in llm.review_calls) == ["performance", "security", "style"]
    assert out["errors"] == ["supervisor: ConnectionError"]


def test_min_severity_hides_lower_findings():
    llm = StubLLM(Plan(reviewers=["security"], reason="code"),
                  {("security", "a.py"): [finding("high", line=1), finding("medium", line=4), finding("low", line=7)]})
    out = build_graph(llm).invoke({"diff": new_file_diff("a.py", 8), "min_severity": "medium"})
    assert [f.severity for f in out["final"]] == ["high", "medium"]
    assert "1 finding(s) below medium severity hidden" in out["report"]
    assert len(build_graph(llm).invoke({"diff": new_file_diff("a.py", 8)})["final"]) == 3  # default shows all


def test_no_reviewers_goes_straight_to_report():
    llm = StubLLM(Plan(reviewers=[], reason="docs only"))
    out = build_graph(llm).invoke({"diff": new_file_diff("README.md", 1)})
    assert llm.review_calls == [] and "No issues found." in out["report"]


def test_reviewer_tool_results_reach_final_prompt(tmp_path):
    (tmp_path / "helper.py").write_text("def sanitize(x): return x  # SENTINEL")
    llm = StubLLM(Plan(reviewers=["security"], reason="code"), tool_path="helper.py")
    build_graph(llm).invoke({"diff": new_file_diff("a.py", 2), "local_dir": str(tmp_path)})
    assert "SENTINEL" in llm.final_prompts[0] and "looked fine" in llm.final_prompts[0]


def test_streams_progress_events_for_live_uis(tmp_path):
    (tmp_path / "helper.py").write_text("x = 1")
    llm = StubLLM(Plan(reviewers=["security"], reason="code"), {("security", "a.py"): [finding("high")]}, tool_path="helper.py")
    events = [chunk for mode, chunk in build_graph(llm).stream(
        {"diff": new_file_diff("a.py", 2), "local_dir": str(tmp_path)}, stream_mode=["updates", "custom"]) if mode == "custom"]
    assert [e["event"] for e in events] == ["review_start", "read_file", "review_done"]
    assert events[1]["file"] == "helper.py" and events[2]["findings"] == 1


def test_read_source_blocks_path_traversal(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    assert "outside the repository" in read_source({"kind": "local", "root": str(tmp_path / "repo")}, "../secret.txt")
    assert "Error reading" in read_source({"kind": "local", "root": str(tmp_path / "repo")}, "missing.py")


def test_dedupe_merges_adjacent_lines_but_not_distant_ones():
    a = finding("medium", line=13, issue="path traversal").model_copy(update={"category": "security"})
    b = finding("high", line=14, issue="unsanitized filename").model_copy(update={"category": "style"})
    c = finding("low", line=16, issue="unrelated").model_copy(update={"category": "style"})
    out = dedupe([c, b, a])
    assert [(f.line, f.severity) for f in out] == [(14, "high"), (16, "low")]  # merged pair takes the worst finding's line
    assert "path traversal" in out[0].issue and "unsanitized filename" in out[0].issue


def test_style_findings_are_capped_at_medium():
    llm = StubLLM(Plan(reviewers=["style", "security"], reason="code"),
                  {("style", "a.py"): [finding("high", line=1)], ("security", "a.py"): [finding("high", line=9)]})
    out = build_graph(llm).invoke({"diff": new_file_diff("a.py", 10)})
    assert sorted((f.category, f.severity) for f in out["final"]) == [("security", "high"), ("style", "medium")]


def test_dedupe_merges_same_line_keeps_worst_severity():
    a = finding("low", line=5, issue="bad name").model_copy(update={"category": "style"})
    b = finding("high", line=5, issue="sql injection").model_copy(update={"category": "security"})
    c = finding("medium", line=None).model_copy(update={"category": "style"})
    d = finding("medium", line=None).model_copy(update={"category": "performance"})
    out = dedupe([a, b, c, d])
    assert len(out) == 3  # line-less findings are never merged
    assert (out[0].severity, out[0].category) == ("high", "security, style")  # worst first
    assert out[0].issue.splitlines() == ["- **security:** sql injection", "- **style:** bad name"]


@pytest.fixture
def fake_github(monkeypatch):
    posted = {}
    monkeypatch.setattr(g.github, "get_diff", lambda o, r, n: new_file_diff("app.py", 10))
    monkeypatch.setattr(g.github, "get_pr", lambda o, r, n: {"head": {"sha": "abc123"}})
    monkeypatch.setattr(g.github, "get_file", lambda o, r, p, ref: f"contents of {p}@{ref}")

    def post_review(owner, repo, number, sha, body, comments):
        posted.update(owner=owner, number=number, sha=sha, body=body, comments=comments)
        return "https://github.com/o/r/pull/7#review"

    monkeypatch.setattr(g.github, "post_review", post_review)
    return posted


def pr_llm():
    return StubLLM(Plan(reviewers=["security"], reason="code"), {("security", "app.py"): [
        finding("high", line=3, issue="injection"), finding("low", line=99, issue="outside diff"), finding("medium", line=6, issue="drop me")]},
        tool_path="lib.py")


def test_auto_approve_posts_inline_and_body(fake_github):
    llm = pr_llm()
    out = build_graph(llm).invoke({"pr_url": "https://github.com/o/r/pull/7", "post": True, "auto_approve": True})
    assert out["review_url"].endswith("#review")
    assert fake_github["sha"] == "abc123"
    assert [c["line"] for c in fake_github["comments"]] == [3, 6]  # line 99 isn't in the diff
    assert "outside diff" in fake_github["body"]
    assert "contents of lib.py@abc123" in llm.final_prompts[0]  # read_file went through the GitHub API at head sha


def test_human_approval_interrupt_drop_and_post(fake_github):
    graph = build_graph(pr_llm(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"pr_url": "https://github.com/o/r/pull/7", "post": True}, cfg)
    assert out["__interrupt__"] and not fake_github  # paused before posting

    numbers = {f.issue: i for i, f in enumerate(out["final"], 1)}
    out = graph.invoke(Command(resume={"approved": True, "drop": [numbers["drop me"]]}), cfg)
    assert [c["line"] for c in fake_github["comments"]] == [3]
    assert out["review_url"]


def test_human_rejection_does_not_post(fake_github):
    graph = build_graph(pr_llm(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"pr_url": "https://github.com/o/r/pull/7", "post": True}, cfg)
    out = graph.invoke(Command(resume={"approved": False}), cfg)
    assert not fake_github and not out.get("review_url")


def test_parse_pr_url():
    assert parse_pr_url("https://github.com/langchain-ai/langgraph/pull/123") == ("langchain-ai", "langgraph", 123)
    with pytest.raises(ValueError):
        parse_pr_url("https://github.com/foo/bar/issues/1")


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_every_provider_builds_offline(provider, monkeypatch):
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "test-key")
    llm, method = make_llm(provider)
    assert method == PROVIDERS[provider][1]
    llm.with_structured_output(Review, method=method)  # schema conversion must not raise
    llm.bind_tools([convert_to_openai_tool(Review)])  # tool binding must not raise
    build_graph(llm, method)


def test_category_hidden_from_tool_schema():
    # function_calling providers see the schema via this conversion; the model must not fill category
    assert "category" not in str(convert_to_openai_tool(Review)["function"]["parameters"])


def test_unknown_provider():
    with pytest.raises(ValueError):
        make_llm("nope")

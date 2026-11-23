"""ReviewGraph web UI: uv run --extra ui streamlit run app.py

Local use: keys come from the environment / .env, like the CLI. Don't deploy this publicly as is:
every visitor would spend your API keys and post with your GitHub token.
"""

import os
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from reviewer.graph import PROVIDERS, build_graph, make_llm

KEY_VARS = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "groq": ("GROQ_API_KEY",),
            "openrouter": ("OPENROUTER_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY",)}
SEVERITY_STYLE = {"high": ("🔴", "red"), "medium": ("🟠", "orange"), "low": ("🔵", "blue")}
FOCUS_ICON = {"security": "🛡️", "performance": "⚡", "style": "🧹"}
DEMO_DIFF = Path(__file__).parent / "examples" / "demo.diff"
DEMO_REPO = Path(__file__).parent / "examples" / "demo"

st.set_page_config(page_title="ReviewGraph", page_icon="🔎", layout="wide")


def has_key(provider: str) -> bool:
    return any(os.environ.get(v) for v in KEY_VARS[provider])


def snippet(numbered: str, line: int | None, radius: int = 2) -> str | None:
    """The numbered diff lines around `line`, for showing a finding in context."""
    if not line:
        return None
    rows = [r for r in numbered.splitlines() if r[:5].strip().isdigit()]
    idx = next((i for i, r in enumerate(rows) if int(r[:5]) == line), None)
    if idx is None:
        return None
    return "\n".join(rows[max(0, idx - radius): idx + radius + 1])


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    available = [p for p in PROVIDERS if has_key(p)] or list(PROVIDERS)
    provider = st.selectbox("Provider", list(PROVIDERS), index=list(PROVIDERS).index(available[0]),
                            format_func=lambda p: f"{p}{'' if has_key(p) else '  (no key set)'}")
    model = st.text_input("Model", placeholder=PROVIDERS[provider][0], help="Leave empty for the provider default")
    min_severity = st.select_slider("Minimum severity", ["low", "medium", "high"], value="medium")
    concurrency = st.slider("Max reviewers at once", 1, 12, 4, help="Lower it on free tiers with small rate limits")
    st.divider()
    st.caption("Keys are read from the environment (`.env`). Posting needs `GITHUB_TOKEN` with pull-request write access.")

# ---------- header + input ----------
st.title("🔎 ReviewGraph")
st.caption("A LangGraph supervisor picks specialist reviewers, which run in parallel per file, read repo files "
           "when they need context, and report ranked findings you can approve and post to GitHub.")

source = st.radio("Review", ["GitHub pull request", "Diff"], horizontal=True, label_visibility="collapsed")
if source == "GitHub pull request":
    pr_url = st.text_input("Pull request URL", placeholder="https://github.com/owner/repo/pull/123")
    state_in = {"pr_url": pr_url.strip(), "post": True} if pr_url.strip() else None
else:
    c1, c2 = st.columns([3, 1])
    with c2:
        use_demo = st.toggle("Use demo diff", value=True, help="Two files with planted bugs from examples/")
        repo_dir = st.text_input("Repo folder for read_file", value=str(DEMO_REPO) if use_demo else ".")
    with c1:
        diff_text = st.text_area("Unified diff (git diff output)", value=DEMO_DIFF.read_text() if use_demo else "", height=180)
    state_in = {"diff": diff_text, "local_dir": repo_dir} if diff_text.strip() else None

run = st.button("Run review", type="primary", disabled=state_in is None or not has_key(provider))
if not has_key(provider):
    st.info(f"Set {' or '.join(KEY_VARS[provider])} to use {provider}.")

# ---------- run with live progress ----------
if run:
    llm, method = make_llm(provider, model.strip() or None)
    graph = build_graph(llm, method, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "max_concurrency": concurrency,
              "run_name": "pr-review-ui", "tags": ["ui", provider]}
    st.session_state.update(graph=graph, config=config, posted=None)

    with st.status("Reviewing…", expanded=True) as status:
        info = st.container()  # plan lines go above the live board
        progress = st.progress(0.0, text="Fetching changes…")
        board = st.empty()
        tasks: dict[tuple, str] = {}
        total = 0

        def redraw():
            done = sum(s.startswith(("✅", "⚠️")) for s in tasks.values())
            if total:
                progress.progress(done / total, text=f"{done}/{total} reviewers finished")
            board.markdown("\n".join(f"- {FOCUS_ICON[f]} **{f}** · `{p}` — {s}" for (f, p), s in sorted(tasks.items())))

        try:
            for mode, chunk in graph.stream(state_in | {"min_severity": min_severity}, config, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    key = (chunk["focus"], chunk["path"])
                    tasks[key] = {
                        "review_start": "⏳ reviewing…",
                        "read_file": f"📖 reading `{chunk.get('file')}`…",
                        "review_done": f"✅ {chunk.get('findings', 0)} finding(s)",
                        "review_error": f"⚠️ failed ({chunk.get('error')})",
                    }[chunk["event"]]
                    redraw()
                elif "fetch" in chunk:
                    files = chunk["fetch"]["files"]
                    info.write(f"📄 **{len(files)} file(s) to review:** " + ", ".join(f"`{p}`" for p in files))
                elif "supervisor" in chunk:
                    plan = chunk["supervisor"]["plan"]
                    total = len(files) * len(plan.reviewers)
                    info.write(f"🧭 **Supervisor:** {', '.join(FOCUS_ICON[r] + ' ' + r for r in plan.reviewers) or 'no reviewers'} — {plan.reason}")
                    redraw()
        except Exception as e:
            status.update(label="Review crashed", state="error")
            st.exception(e)
            st.stop()
        result = graph.get_state(config).values
        st.session_state["result"] = result
        label = "Review failed" if result.get("failed") else f"Done — {len(result.get('final', []))} finding(s)"
        status.update(label=label, state="error" if result.get("failed") else "complete", expanded=False)

# ---------- results ----------
result = st.session_state.get("result")
if result:
    final, files = result.get("final", []), result.get("files", {})
    if result.get("failed"):
        st.error("Every reviewer failed, so nothing was reviewed. " + "; ".join(result.get("errors", [])))
        st.stop()
    if result.get("errors"):
        st.warning("Incomplete review. Failed: " + "; ".join(result["errors"]))

    counts = {s: sum(f.severity == s for f in final) for s in ("high", "medium", "low")}
    m = st.columns(5)
    m[0].metric("Findings", len(final))
    m[1].metric("🔴 High", counts["high"])
    m[2].metric("🟠 Medium", counts["medium"])
    m[3].metric("Files", len(files))
    m[4].metric("Reviewers", f"{len(result['plan'].reviewers)} × {len(files)} files" if result.get("plan") else "—")

    can_post = bool(result.get("owner")) and st.session_state.get("posted") is None
    if not final:
        st.success("No issues found at this severity.")
    keep = []
    for i, f in enumerate(final, 1):
        icon, color = SEVERITY_STYLE[f.severity]
        with st.container(border=True):
            head, check = st.columns([12, 2])
            loc = f"{f.file}:{f.line}" if f.line else f.file
            cats = " ".join(f"{FOCUS_ICON.get(c.strip(), '')} {c.strip()}" for c in f.category.split(","))
            head.markdown(f"**{i}.** :{color}[**{icon} {f.severity.upper()}**] &nbsp; {cats} &nbsp; `{loc}`")
            if can_post and check.checkbox("Post", value=True, key=f"keep_{i}"):
                keep.append(i)
            st.markdown(f.issue)
            if code := snippet(files.get(f.file, {}).get("numbered", ""), f.line):
                st.code(code, language="diff")
            st.markdown(f"**Fix:** {f.suggestion}")

    b1, b2, _ = st.columns([2, 2, 6])
    b2.download_button("Download report (.md)", result.get("report", ""), file_name="review.md")
    if can_post:
        if b1.button(f"Post {len(keep)} finding(s) to GitHub", type="primary"):
            drop = [i for i in range(1, len(final) + 1) if i not in keep]
            try:
                out = st.session_state["graph"].invoke(Command(resume={"approved": True, "drop": drop}), st.session_state["config"])
                st.session_state["posted"] = out.get("review_url")
            except Exception as e:
                st.error(f"Posting failed: {e}")
            st.rerun()
    if url := st.session_state.get("posted"):
        st.success(f"Posted: {url}")

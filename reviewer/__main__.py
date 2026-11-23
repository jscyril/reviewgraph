import argparse
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from reviewer.graph import PROVIDERS, build_graph, make_llm


def ask_approval(count: int) -> dict:
    prompt = f"\nPost these {count} finding(s) to GitHub? [y]es / [n]o / numbers to drop first (e.g. 2,5): "
    try:
        ans = input(prompt).strip().lower()
    except EOFError:  # no terminal (e.g. CI without --yes): never post silently
        ans = "n"
    if ans in ("y", "yes"):
        return {"approved": True}
    try:
        return {"approved": True, "drop": [int(x) for x in ans.replace(" ", "").split(",")]}
    except ValueError:
        return {"approved": False}


def main() -> None:
    p = argparse.ArgumentParser(prog="pr-reviewer", description="Multi-agent GitHub PR reviewer")
    p.add_argument("target", help="GitHub PR URL or path to a .diff file")
    p.add_argument("--provider", choices=PROVIDERS, default="gemini")
    p.add_argument("--model", help="Override the provider's default model")
    p.add_argument("--post", action="store_true", help="Post the review to the PR (asks for approval first)")
    p.add_argument("--yes", action="store_true", help="With --post: skip the approval prompt (for CI)")
    p.add_argument("--min-severity", choices=["low", "medium", "high"], default="medium",
                   help="Hide findings below this severity (default: medium)")
    p.add_argument("--repo-dir", default=".", help="For local diffs: repo root reviewers may read files from")
    p.add_argument("--max-concurrency", type=int, default=8, help="Max reviewers running at once")
    args = p.parse_args()

    target = Path(args.target)
    if target.is_file():
        if args.post:
            sys.exit("--post needs a GitHub PR URL, not a local diff")
        state = {"diff": target.read_text(), "local_dir": args.repo_dir}
    else:
        state = {"pr_url": args.target, "post": args.post, "auto_approve": args.yes}
    state["min_severity"] = args.min_severity

    llm, method = make_llm(args.provider, args.model)
    graph = build_graph(llm, method, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}, "max_concurrency": args.max_concurrency,
              "run_name": "pr-review", "tags": [args.provider], "metadata": {"target": args.target}}  # LangSmith labels

    out = graph.invoke(state, config)
    print(out["report"])
    if out.get("__interrupt__"):
        out = graph.invoke(Command(resume=ask_approval(len(out["final"]))), config)
        if not out.get("approved"):
            print("Not posted.")
    if out.get("review_url"):
        print(f"\nPosted: {out['review_url']}")
    if out.get("failed"):
        sys.exit(1)  # fail the CI job instead of passing silently


if __name__ == "__main__":
    main()

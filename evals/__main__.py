"""Run the planted-bug eval: uv run python -m evals --provider gemini [--provider groq ...]

Recall: planted bugs a finding hit (same file, within LINE_TOLERANCE lines).
False positives: findings on the clean files.
Results go to evals/results/<provider>.json; a markdown table is printed.
"""

import argparse
import json
import time
from pathlib import Path

from reviewer.graph import PROVIDERS, build_graph, make_llm

from evals.cases import CASES, as_diff, bug_line

LINE_TOLERANCE = 3
RESULTS = Path(__file__).parent / "results"


def score(case: dict, findings: list[dict]) -> dict:
    line = bug_line(case)
    if line is None:
        return {"id": case["id"], "clean": True, "caught": None, "category_match": None, "false_positives": len(findings)}
    hits = [f for f in findings if f["file"] == case["path"] and f["line"] and abs(f["line"] - line) <= LINE_TOLERANCE]
    return {"id": case["id"], "clean": False, "caught": bool(hits),
            "category_match": any(case["category"] in f["category"] for f in hits), "false_positives": None}


def summarize(rows: list[dict]) -> dict:
    bugs = [r for r in rows if not r["clean"]]
    clean = [r for r in rows if r["clean"]]
    caught = sum(r["caught"] for r in bugs)
    return {
        "recall": caught / len(bugs) if bugs else 0.0,
        "caught": caught, "bugs": len(bugs),
        "right_reviewer": sum(r["category_match"] for r in bugs),
        "clean_false_positives": sum(r["false_positives"] for r in clean),
        "clean_files": len(clean),
    }


def run(provider: str, model: str | None) -> dict:
    llm, method = make_llm(provider, model)
    graph = build_graph(llm, method)
    rows, start = [], time.time()
    for case in CASES:
        out = graph.invoke({"diff": as_diff(case)}, {"max_concurrency": 4, "run_name": f"eval:{case['id']}", "tags": ["eval", provider]})
        findings = [f.model_dump() for f in out["final"]]
        rows.append({**score(case, findings), "findings": findings})
        print(f"  {case['id']:<26} {'clean' if rows[-1]['clean'] else ('CAUGHT' if rows[-1]['caught'] else 'missed')}"
              f"  ({len(findings)} finding(s))")
    result = {"provider": provider, "model": model or PROVIDERS[provider][0], "seconds": round(time.time() - start, 1),
              **summarize(rows), "cases": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{provider}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(prog="evals")
    p.add_argument("--provider", action="append", choices=PROVIDERS, required=True, help="Repeat to compare providers")
    p.add_argument("--model", help="Model override (applies to every provider given)")
    args = p.parse_args()

    results = []
    for provider in args.provider:
        print(f"== {provider}")
        results.append(run(provider, args.model))
    print("\n| Provider | Model | Recall | Right reviewer | False positives on clean files | Time |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['provider']} | {r['model']} | {r['caught']}/{r['bugs']} ({r['recall']:.0%}) | {r['right_reviewer']}/{r['bugs']} "
              f"| {r['clean_false_positives']} across {r['clean_files']} files | {r['seconds']}s |")


if __name__ == "__main__":
    main()

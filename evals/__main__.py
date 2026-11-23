"""Run the planted-bug eval: uv run python -m evals --provider groq [--provider gemini ...] [--runs 3]

Each case is a diff plus a small temp repo (so reviewers can read_file other modules, as on a real PR).
Every run is scored twice from the same findings: showing all severities ("low") and with the default
--min-severity filter ("medium"), so you can see how much noise the filter removes and what recall it costs.

Recall: planted bugs a finding hit (same file, within LINE_TOLERANCE lines).
Noise: findings on the clean files (all of them are false positives).
Results go to evals/results/<provider>.json; a markdown table is printed.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from reviewer.graph import PROVIDERS, SEVERITY_ORDER, build_graph, make_llm

from evals.cases import CASES, as_diff, bug_line, write_repo

LINE_TOLERANCE = 3
LEVELS = ("low", "medium")
RESULTS = Path(__file__).parent / "results"


def score(case: dict, findings: list[dict]) -> dict:
    line = bug_line(case)
    if line is None:
        return {"id": case["id"], "clean": True, "caught": None, "category_match": None, "false_positives": len(findings)}
    hits = [f for f in findings if f["file"] == case["path"] and f["line"] and abs(f["line"] - line) <= LINE_TOLERANCE]
    return {"id": case["id"], "clean": False, "caught": bool(hits),
            "category_match": any(case["category"] in f["category"] for f in hits), "false_positives": None}


def summarize(rows: list[dict]) -> dict:
    # A case where every reviewer failed reviewed nothing: it is neither a miss nor a clean pass, so leave it out.
    scored = [r for r in rows if not r.get("failed")]
    bugs = [r for r in scored if not r["clean"]]
    clean = [r for r in scored if r["clean"]]
    caught = sum(r["caught"] for r in bugs)
    return {
        "recall": caught / len(bugs) if bugs else 0.0,
        "caught": caught, "bugs": len(bugs),
        "right_reviewer": sum(r["category_match"] for r in bugs),
        "clean_false_positives": sum(r["false_positives"] for r in clean),
        "clean_files": len(clean),
        "cases_with_errors": sum(bool(r.get("errors")) for r in rows),  # nonzero = scores may understate the model
        "unscored": len(rows) - len(scored),
    }


def at_level(findings: list[dict], level: str) -> list[dict]:
    return [f for f in findings if SEVERITY_ORDER[f["severity"]] <= SEVERITY_ORDER[level]]


def run_once(graph, provider: str, max_concurrency: int) -> list[dict]:
    """One pass over every case; returns per-case rows with findings at every level."""
    rows = []
    for case in CASES:
        with tempfile.TemporaryDirectory() as repo:
            write_repo(case, Path(repo))
            out = graph.invoke({"diff": as_diff(case), "local_dir": repo, "min_severity": "low"},
                               {"max_concurrency": max_concurrency, "run_name": f"eval:{case['id']}", "tags": ["eval", provider]})
        findings = [f.model_dump() for f in out["final"]]
        errors = out.get("errors", [])
        row = {"id": case["id"], "hard": case.get("hard", False), "errors": errors, "findings": findings,
               "failed": bool(out.get("failed"))}
        for level in LEVELS:
            row[level] = score(case, at_level(findings, level))
        rows.append(row)
        s = row["medium"]
        status = "clean" if s["clean"] else ("CAUGHT" if s["caught"] else ("caught@low" if row["low"]["caught"] else "missed"))
        print(f"  {case['id']:<26} {status:<10} medium+: {len(at_level(findings, 'medium'))}  all: {len(findings)}"
              + (f"  ERRORS: {'; '.join(errors)}" if errors else ""), flush=True)
    return rows


def level_summary(runs: list[list[dict]], level: str, hard: bool | None = None) -> dict:
    """Mean over runs of each metric, plus the per-run recall range."""
    per_run = [summarize([{**r[level], "errors": r["errors"], "failed": r.get("failed", False)}
                          for r in rows if hard is None or r["hard"] == hard]) for rows in runs]
    mean = {k: round(statistics.mean(s[k] for s in per_run), 2) for k in per_run[0]}
    mean["recall_range"] = [min(s["recall"] for s in per_run), max(s["recall"] for s in per_run)]
    return mean


def run(provider: str, model: str | None, runs: int, max_concurrency: int) -> dict:
    llm, method = make_llm(provider, model)
    graph = build_graph(llm, method)
    all_runs, start = [], time.time()
    for i in range(runs):
        print(f"-- {provider} run {i + 1}/{runs}", flush=True)
        all_runs.append(run_once(graph, provider, max_concurrency))
    result = {
        "provider": provider, "model": model or PROVIDERS[provider][0], "runs": runs,
        "seconds_per_run": round((time.time() - start) / runs, 1),
        "summary": {level: {"all": level_summary(all_runs, level), "basic": level_summary(all_runs, level, hard=False),
                            "hard": level_summary(all_runs, level, hard=True)} for level in LEVELS},
        "cases": all_runs,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{provider}.json").write_text(json.dumps(result, indent=2))
    return result


def table(results: list[dict]) -> str:
    def pct(s):
        lo, hi = s["recall_range"]
        spread = f" ({lo:.0%}–{hi:.0%})" if lo != hi else ""
        return f"{s['caught']:g}/{s['bugs']} = {s['recall']:.0%}{spread}"

    lines = ["| Provider / model | Runs | Recall, basic | Recall, hard | Recall, all | Noise on clean files (low → medium+) | Cases with a failed reviewer | Not scored (all reviewers failed) | Time/run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        med, low = r["summary"]["medium"], r["summary"]["low"]
        lines.append(f"| {r['provider']} / `{r['model']}` | {r['runs']} | {pct(med['basic'])} | {pct(med['hard'])} | {pct(med['all'])} "
                     f"| {low['all']['clean_false_positives']:g} → **{med['all']['clean_false_positives']:g}** across {med['all']['clean_files']} files "
                     f"| {med['all']['cases_with_errors']:g} | {med['all']['unscored']:g} | {r['seconds_per_run']:.0f}s |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="evals")
    p.add_argument("--provider", action="append", choices=PROVIDERS, required=True, help="Repeat to compare providers")
    p.add_argument("--model", help="Model override (applies to every provider given)")
    p.add_argument("--runs", type=int, default=1, help="Repeat the suite and average (models vary run to run)")
    p.add_argument("--max-concurrency", type=int, default=4, help="Lower for free tiers with small rate limits")
    args = p.parse_args()

    results = []
    for provider in args.provider:
        print(f"== {provider}", flush=True)
        results.append(run(provider, args.model, args.runs, args.max_concurrency))
    print("\nRecall is at the default --min-severity medium. Numbers are means over runs; range shown when runs differ.\n")
    print(table(results))


if __name__ == "__main__":
    main()

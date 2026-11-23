from evals.__main__ import at_level, level_summary, score, summarize, table
from evals.cases import CASES, as_diff, bug_line, write_repo
from reviewer.diff import parse_diff


def test_every_case_diff_parses_and_bug_line_is_commentable():
    for case in CASES:
        files = parse_diff(as_diff(case))
        assert list(files) == [case["path"]], case["id"]
        line = bug_line(case)  # asserts the needle hits exactly one line
        if line is not None:
            assert line in files[case["path"]]["lines"], f"{case['id']}: bug line {line} not in the diff"


def test_suite_has_hard_cases_and_context_files(tmp_path):
    hard = [c for c in CASES if c.get("hard")]
    assert any("before" in c for c in hard) and any("context" in c for c in hard)
    assert any(c["needle"] is None for c in hard)  # clean traps
    case = next(c for c in CASES if "context" in c)
    write_repo(case, tmp_path)
    for rel in [case["path"], *case["context"]]:
        assert (tmp_path / rel).is_file()


def test_score_and_summarize():
    bug = next(c for c in CASES if c["id"] == "sql-injection")
    clean = next(c for c in CASES if c["id"] == "clean-money")
    line = bug_line(bug)
    near = {"file": bug["path"], "line": line + 2, "category": "security, style"}
    far = {"file": bug["path"], "line": line + 10, "category": "security"}
    wrong_file = {"file": "other.py", "line": line, "category": "security"}

    assert score(bug, [near])["caught"] and score(bug, [near])["category_match"]
    assert not score(bug, [far, wrong_file])["caught"]
    rows = [score(bug, [near]), score(bug, []), score(clean, [far, far])]
    s = summarize(rows)
    assert (s["caught"], s["bugs"], s["recall"], s["clean_false_positives"]) == (1, 2, 0.5, 2)


def test_levels_and_multi_run_summary():
    bug = next(c for c in CASES if c["id"] == "sql-injection")
    clean = next(c for c in CASES if c["id"] == "clean-money")
    hit_low = {"file": bug["path"], "line": bug_line(bug), "category": "security", "severity": "low"}
    noise = {"file": clean["path"], "line": 1, "category": "style", "severity": "low"}
    assert at_level([hit_low, noise], "medium") == []

    def row(findings_bug, findings_clean):
        return [{"id": "b", "hard": False, "errors": [], **{lv: score(bug, at_level(findings_bug, lv)) for lv in ("low", "medium")}},
                {"id": "c", "hard": True, "errors": [], **{lv: score(clean, at_level(findings_clean, lv)) for lv in ("low", "medium")}}]

    runs = [row([hit_low], [noise]), row([], [])]  # run 1 caught only at low severity; run 2 caught nothing
    low, med = level_summary(runs, "low"), level_summary(runs, "medium")
    assert (low["recall"], low["recall_range"], low["clean_false_positives"]) == (0.5, [0.0, 1.0], 0.5)
    assert (med["recall"], med["clean_false_positives"]) == (0.0, 0.0)
    result = {"provider": "p", "model": "m", "runs": 2, "seconds_per_run": 1.0,
              "summary": {lv: {"all": level_summary(runs, lv), "basic": level_summary(runs, lv, hard=False),
                               "hard": level_summary(runs, lv, hard=True)} for lv in ("low", "medium")}}
    assert "| p / `m` | 2 |" in table([result])


def test_totally_failed_cases_are_not_scored():
    bug = next(c for c in CASES if c["id"] == "sql-injection")
    clean = next(c for c in CASES if c["id"] == "clean-money")
    rows = [{**score(bug, []), "failed": True}, {**score(clean, []), "failed": True}, score(bug, [{"file": bug["path"], "line": bug_line(bug), "category": "security"}])]
    s = summarize(rows)
    assert (s["caught"], s["bugs"], s["clean_files"], s["unscored"]) == (1, 1, 0, 2)  # failures are neither misses nor clean passes

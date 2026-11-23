from evals.__main__ import score, summarize
from evals.cases import CASES, as_diff, bug_line
from reviewer.diff import parse_diff


def test_every_case_diff_parses_and_needle_is_unique():
    for case in CASES:
        files = parse_diff(as_diff(case))
        assert list(files) == [case["path"]]
        assert files[case["path"]]["lines"] == list(range(1, len(case["code"].splitlines()) + 1))
        bug_line(case)  # asserts the needle hits exactly one line


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

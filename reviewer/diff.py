"""Split a unified diff into per-file chunks and find which new-file lines GitHub can comment on."""

import re
from typing import TypedDict

HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FileDiff(TypedDict):
    diff: str
    numbered: str  # same diff with new-file line numbers in front, so models cite correct lines
    lines: list[int]  # new-file line numbers inside hunks (added or context): valid inline-comment targets
    added: int
    removed: int


def _path(header: str) -> str | None:
    p = header[4:].rstrip("\n").split("\t")[0].strip()
    if p == "/dev/null":
        return None
    return p[2:] if p.startswith("b/") else p


def parse_diff(diff: str) -> dict[str, FileDiff]:
    files: dict[str, FileDiff] = {}
    cur: dict | None = None
    old_left = new_left = 0  # lines still expected in the current hunk

    def flush():
        if cur and cur["path"] and cur["lines"]:  # skips deletions, binaries, pure renames
            files[cur["path"]] = FileDiff(diff="".join(cur["text"]), numbered="".join(cur["num"]),
                                          lines=cur["lines"], added=cur["added"], removed=cur["removed"])

    for line in diff.splitlines(keepends=True):
        if old_left > 0 or new_left > 0:  # inside a hunk: counts, not prefixes, decide where it ends
            cur["text"].append(line)
            tag = line[:1]
            if tag in (" ", "\n", "\r", ""):  # context line (some tools strip its leading space)
                cur["num"].append(f"{cur['next']:>5} {line}")
                cur["lines"].append(cur["next"]); cur["next"] += 1; old_left -= 1; new_left -= 1
            elif tag == "+":
                cur["num"].append(f"{cur['next']:>5} {line}")
                cur["lines"].append(cur["next"]); cur["next"] += 1; new_left -= 1; cur["added"] += 1
            elif tag == "-":
                cur["num"].append(f"{'':>5} {line}")
                old_left -= 1; cur["removed"] += 1
            else:  # "\ No newline at end of file"
                cur["num"].append(f"{'':>5} {line}")
            continue
        if line.startswith("diff --git") or (line.startswith("--- ") and (cur is None or cur["seen_plus"])):
            flush()
            cur = {"path": None, "text": [], "num": [], "lines": [], "next": 0, "seen_plus": False, "added": 0, "removed": 0}
        if cur is None:
            continue  # preamble before the first file
        cur["text"].append(line)
        cur["num"].append(line)
        if line.startswith("+++ "):
            cur["path"], cur["seen_plus"] = _path(line), True
        elif m := HUNK.match(line):
            old_left = int(m[1]) if m[1] is not None else 1
            new_left = int(m[3]) if m[3] is not None else 1
            cur["next"] = int(m[2])
    flush()
    return files

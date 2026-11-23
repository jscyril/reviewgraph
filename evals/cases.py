"""Eval cases: small files with one planted bug each, plus clean files that should get no findings.

A bug's location is given by a `needle` (a substring of the buggy line) rather than a hand-written
line number, so editing a snippet can't silently break the expected line.
"""

import difflib
from pathlib import Path

from evals.hard_cases import HARD_CASES

CASES = [
    {
        "id": "sql-injection", "path": "app/users.py", "category": "security", "needle": 'cur.execute(f"',
        "code": '''import sqlite3


def get_user(db: sqlite3.Connection, username: str):
    cur = db.cursor()
    cur.execute(f"SELECT id, email FROM users WHERE name = '{username}'")
    return cur.fetchone()
''',
    },
    {
        "id": "path-traversal", "path": "app/download.py", "category": "security", "needle": "os.path.join(UPLOAD_DIR, name)",
        "code": '''import os

from flask import Flask, request, send_file

app = Flask(__name__)
UPLOAD_DIR = "/srv/uploads"


@app.get("/download")
def download():
    name = request.args["name"]
    return send_file(os.path.join(UPLOAD_DIR, name))
''',
    },
    {
        "id": "hardcoded-secret", "path": "app/billing.py", "category": "security", "needle": "API_SECRET = ",
        "code": '''import httpx

API_SECRET = "9f8e7d6c5b4a39281706f5e4d3c2b1a0"


def charge(amount_cents: int, customer_id: str) -> dict:
    r = httpx.post(
        "https://billing.internal.example.com/v1/charges",
        headers={"Authorization": f"Bearer {API_SECRET}"},
        json={"amount": amount_cents, "customer": customer_id},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()
''',
    },
    {
        "id": "unsafe-deserialization", "path": "app/settings_api.py", "category": "security", "needle": "pickle.loads(body)",
        "code": '''import pickle

from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/settings/import")
async def import_settings(request: Request):
    body = await request.body()
    settings = pickle.loads(body)
    return {"keys": list(settings)}
''',
    },
    {
        "id": "xss-innerhtml", "path": "web/comments.js", "category": "security", "needle": "el.innerHTML",
        "code": '''export function renderComment(container, comment) {
  const el = document.createElement("div");
  el.className = "comment";
  el.innerHTML = `<b>${comment.author}</b>: ${comment.body}`;
  container.appendChild(el);
}
''',
    },
    {
        "id": "n-plus-one", "path": "app/reports.py", "category": "performance", "needle": "session.query(Book)",
        "code": '''from sqlalchemy.orm import Session

from app.models import Author, Book


def books_by_author(session: Session) -> dict[str, list[str]]:
    result = {}
    for author in session.query(Author).all():
        books = session.query(Book).filter(Book.author_id == author.id).all()
        result[author.name] = [b.title for b in books]
    return result
''',
    },
    {
        "id": "blocking-sleep-in-async", "path": "app/crawler.py", "category": "performance", "needle": "time.sleep(1)",
        "code": '''import time

import httpx


async def fetch_statuses(urls: list[str]) -> list[int]:
    statuses = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            r = await client.get(url)
            statuses.append(r.status_code)
            time.sleep(1)
    return statuses
''',
    },
    {
        "id": "quadratic-dedupe", "path": "app/emails.py", "category": "performance", "needle": "if email in emails[i + 1:]",
        "code": '''def find_duplicates(emails: list[str]) -> list[str]:
    duplicates = []
    for i, email in enumerate(emails):
        if email in emails[i + 1:] and email not in duplicates:
            duplicates.append(email)
    return duplicates
''',
    },
    {
        "id": "swallowed-exception", "path": "app/config.py", "category": "style", "needle": "except Exception:",
        "code": '''import json


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        pass
    return {}
''',
    },
    # Clean cases: any finding here is a false positive.
    {
        "id": "clean-chunked", "path": "app/util.py", "category": None, "needle": None,
        "code": '''def chunked(items: list, size: int) -> list[list]:
    """Split items into consecutive chunks of at most `size` elements."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]
''',
    },
    {
        "id": "clean-money", "path": "app/money.py", "category": None, "needle": None,
        "code": '''from dataclasses import dataclass


@dataclass(frozen=True)
class Money:
    cents: int
    currency: str

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("currency mismatch")
        return Money(self.cents + other.cents, self.currency)
''',
    },
]


CASES += HARD_CASES


def bug_line(case: dict) -> int | None:
    if not case["needle"]:
        return None
    hits = [i for i, line in enumerate(case["code"].splitlines(), 1) if case["needle"] in line]
    assert len(hits) == 1, f"{case['id']}: needle must match exactly one line, got {hits}"
    return hits[0]


def as_diff(case: dict) -> str:
    """Render the case as a git diff: a modification if it has `before`, otherwise a new file."""
    path = case["path"]
    if "before" in case:
        body = "".join(difflib.unified_diff(case["before"].splitlines(keepends=True), case["code"].splitlines(keepends=True),
                                            fromfile=f"a/{path}", tofile=f"b/{path}"))
        return f"diff --git a/{path} b/{path}\n{body}"
    lines = case["code"].splitlines()
    body = "".join(f"+{line}\n" for line in lines)
    return f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"


def write_repo(case: dict, root: Path) -> None:
    """Materialize the case's repo (changed file + context files) so reviewers can read_file it."""
    for rel, text in {case["path"]: case["code"], **case.get("context", {})}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)

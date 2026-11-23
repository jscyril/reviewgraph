"""Hard eval cases.

Modifications to existing files (bug buried among unchanged code), bugs only visible by reading
another file (`context`: the reviewer has to call read_file), subtle bugs, and clean code that
looks risky (false-positive traps).
"""

_DOCS_BEFORE = '''from fastapi import APIRouter, Depends, HTTPException

from app.auth import current_user
from app.db import Session, get_session
from app.models import Document, User

router = APIRouter()


def load_owned(session: Session, doc_id: int, user: User) -> Document:
    doc = session.get(Document, doc_id)
    if doc is None or doc.owner_id != user.id:
        raise HTTPException(status_code=404)
    return doc


@router.get("/documents/{doc_id}")
def read_document(doc_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    return load_owned(session, doc_id, user)


@router.put("/documents/{doc_id}")
def rename_document(doc_id: int, title: str, user: User = Depends(current_user), session: Session = Depends(get_session)):
    doc = load_owned(session, doc_id, user)
    doc.title = title
    session.commit()
    return doc
'''

_USERS_REPO_BEFORE = '''from sqlalchemy.orm import Session

from app.models import User


def get_user(session: Session, user_id: int) -> User | None:
    return session.get(User, user_id)


def users_by_email_domain(session: Session, domain: str) -> list[User]:
    return session.query(User).filter(User.email.endswith("@" + domain)).all()


def count_users(session: Session) -> int:
    return session.query(User).count()
'''

_CLIENT_BEFORE = '''import httpx

BASE_URL = "https://api.example.com"


def get_user(user_id: int) -> dict:
    r = httpx.get(f"{BASE_URL}/users/{user_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def list_orders(user_id: int) -> list[dict]:
    r = httpx.get(f"{BASE_URL}/users/{user_id}/orders", timeout=10)
    r.raise_for_status()
    return r.json()["orders"]
'''

_DB_QUERY = '''import sqlite3

_conn = sqlite3.connect("app.db")


def query(sql: str, *params):
    """Run a query and return all rows."""
    return _conn.execute(sql % params).fetchall()
'''

_SETTINGS = '''import json


def get_settings() -> dict:
    with open("/etc/app/settings.json") as f:
        return json.load(f)
'''

HARD_CASES = [
    {
        "id": "idor-missing-owner-check", "path": "app/documents.py", "category": "security", "needle": "if doc is None:",
        "before": _DOCS_BEFORE,
        "code": _DOCS_BEFORE + '''

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)):
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404)
    session.delete(doc)
    session.commit()
''',
    },
    {
        # Looks like a parameterized query; only app/db.py shows it's %-formatted into the SQL.
        "id": "cross-file-sqli", "path": "app/orders.py", "category": "security", "needle": "return query(",
        "context": {"app/db.py": _DB_QUERY},
        "code": '''from app.db import query


def orders_for(customer_id: str) -> list[tuple]:
    return query("SELECT id, total FROM orders WHERE customer = %s", customer_id)
''',
    },
    {
        # Only app/settings.py shows get_settings() reads and parses a file on every call.
        "id": "cross-file-io-in-loop", "path": "app/pricing.py", "category": "performance", "needle": "rate = get_settings()",
        "context": {"app/settings.py": _SETTINGS},
        "code": '''from app.settings import get_settings


def apply_discounts(prices: list[float]) -> list[float]:
    out = []
    for price in prices:
        rate = get_settings()["discount_rate"]
        out.append(round(price * (1 - rate), 2))
    return out
''',
    },
    {
        "id": "timing-unsafe-compare", "path": "app/webhooks.py", "category": "security", "needle": "return expected == signature",
        "code": '''import hashlib
import hmac
import os


def verify_webhook(body: bytes, signature: str) -> bool:
    secret = os.environ["WEBHOOK_SECRET"].encode()
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return expected == signature
''',
    },
    {
        "id": "full-table-scan", "path": "app/users_repo.py", "category": "performance", "needle": "session.query(User).all()",
        "before": _USERS_REPO_BEFORE,
        "code": _USERS_REPO_BEFORE + '''

def active_users(session: Session) -> list[User]:
    return [u for u in session.query(User).all() if u.is_active]
''',
    },
    {
        "id": "mutable-default-arg", "path": "app/tags.py", "category": "style", "needle": "tags: list[str] = []",
        "code": '''def add_tag(tag: str, tags: list[str] = []) -> list[str]:
    tags.append(tag.strip().lower())
    return tags
''',
    },
    {
        # Bare except also eats KeyboardInterrupt, and after the last attempt it silently returns None.
        "id": "bare-except-retry", "path": "app/fetch.py", "category": "style", "needle": "except:",
        "code": '''import time

import httpx


def fetch_with_retry(url: str, attempts: int = 3) -> dict:
    for _ in range(attempts):
        try:
            return httpx.get(url, timeout=5).json()
        except:
            time.sleep(1)
''',
    },
    # Clean but suspicious-looking.
    {
        "id": "clean-parameterized-sql", "path": "app/order_queries.py", "category": None, "needle": None,
        "code": '''import sqlite3


def find_orders(db: sqlite3.Connection, customer_id: int, status: str) -> list[tuple]:
    cur = db.execute(
        "SELECT id, total FROM orders WHERE customer_id = ? AND status = ?",
        (customer_id, status),
    )
    return cur.fetchall()
''',
    },
    {
        "id": "clean-guarded-path", "path": "app/uploads.py", "category": None, "needle": None,
        "code": '''from pathlib import Path

UPLOAD_ROOT = Path("/srv/uploads").resolve()


def resolve_upload(name: str) -> Path:
    path = (UPLOAD_ROOT / name).resolve()
    if not path.is_relative_to(UPLOAD_ROOT):
        raise ValueError("path escapes upload directory")
    return path
''',
    },
    {
        "id": "clean-refactor", "path": "app/client.py", "category": None, "needle": None,
        "before": _CLIENT_BEFORE,
        "code": _CLIENT_BEFORE.replace('BASE_URL = "https://api.example.com"\n',
                                       'BASE_URL = "https://api.example.com"\nTIMEOUT_SECONDS = 10\n')
                              .replace("timeout=10", "timeout=TIMEOUT_SECONDS"),
    },
]

for _case in HARD_CASES:
    _case["hard"] = True

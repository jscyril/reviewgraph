"""Thin GitHub REST client: PR metadata, diff, file contents at a commit, and posting a review."""

import os
import re
from urllib.parse import quote

import httpx

API = "https://api.github.com"
MAX_BODY = 60_000  # GitHub rejects review bodies over 65,536 chars


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url.strip())
    if not m:
        raise ValueError(f"Not a GitHub PR URL: {url}")
    return m[1], m[2], int(m[3])


def _get(path: str, accept: str = "application/vnd.github+json", **params) -> httpx.Response:
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    r = httpx.get(f"{API}{path}", headers=headers, params=params, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r


def get_pr(owner: str, repo: str, number: int) -> dict:
    return _get(f"/repos/{owner}/{repo}/pulls/{number}").json()


def get_diff(owner: str, repo: str, number: int) -> str:
    return _get(f"/repos/{owner}/{repo}/pulls/{number}", accept="application/vnd.github.diff").text


def get_file(owner: str, repo: str, path: str, ref: str) -> str:
    return _get(f"/repos/{owner}/{repo}/contents/{quote(path)}", accept="application/vnd.github.raw+json", ref=ref).text


def post_review(owner: str, repo: str, number: int, commit_id: str, body: str, comments: list[dict]) -> str:
    """Post one review with inline comments; returns its URL. Falls back to body-only if GitHub rejects a comment."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN to post reviews")
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    url = f"{API}/repos/{owner}/{repo}/pulls/{number}/reviews"
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY] + "\n\n_(review truncated: GitHub's size limit)_"
    payload = {"commit_id": commit_id, "body": body, "event": "COMMENT", "comments": comments}
    r = httpx.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code == 422 and comments:
        # A line GitHub considers outside the diff fails the whole review; keep the findings by moving them to the body.
        moved = "\n".join(f"- `{c['path']}:{c['line']}` {c['body']}" for c in comments)
        payload = {"commit_id": commit_id, "body": f"{body}\n\n### Inline findings\n{moved}"[:MAX_BODY], "event": "COMMENT"}
        r = httpx.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["html_url"]

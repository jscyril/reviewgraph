"""Existing helper (not part of the demo diff). Reviewers must read it to see the injection in orders.py."""

import sqlite3

_conn = sqlite3.connect("shop.db")


def query(sql: str, *params):
    """Run a query and return all rows."""
    return _conn.execute(sql % params).fetchall()

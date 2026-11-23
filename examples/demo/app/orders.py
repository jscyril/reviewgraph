from app.db import query


def orders_for(customer_id: str) -> list[tuple]:
    return query("SELECT id, total FROM orders WHERE customer = %s", customer_id)


def order_summaries(customer_ids: list[str]) -> dict[str, float]:
    totals = {}
    for cid in customer_ids:
        rows = query("SELECT total FROM orders WHERE customer = %s", cid)
        totals[cid] = sum(r[0] for r in rows)
    return totals

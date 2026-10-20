"""
SQLite finance-tracker storage. Deliberately boring/simple — this is a
prototype for a single-user demo, not a production multi-tenant schema.
Fully real, fully testable without any model/network access.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('income', 'expense')),
    amount REAL NOT NULL CHECK (amount > 0),
    category TEXT,
    description TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class Transaction:
    kind: str          # "income" | "expense"
    amount: float
    category: str | None
    description: str | None


def add_transaction(tx: Transaction, db_path: Path = DB_PATH) -> int:
    if tx.kind not in ("income", "expense"):
        raise ValueError(f"kind must be 'income' or 'expense', got {tx.kind!r}")
    if tx.amount <= 0:
        raise ValueError(f"amount must be positive, got {tx.amount}")

    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO transactions (kind, amount, category, description, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (tx.kind, tx.amount, tx.category, tx.description, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_balance(db_path: Path = DB_PATH) -> float:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT "
            "  COALESCE(SUM(CASE WHEN kind='income' THEN amount ELSE 0 END), 0) - "
            "  COALESCE(SUM(CASE WHEN kind='expense' THEN amount ELSE 0 END), 0) AS balance "
            "FROM transactions"
        ).fetchone()
        return row["balance"]


def get_summary(db_path: Path = DB_PATH) -> dict:
    with get_conn(db_path) as conn:
        totals = conn.execute(
            "SELECT kind, COALESCE(SUM(amount), 0) AS total, COUNT(*) AS n "
            "FROM transactions GROUP BY kind"
        ).fetchall()
        by_category = conn.execute(
            "SELECT kind, COALESCE(category, 'uncategorized') AS category, "
            "SUM(amount) AS total FROM transactions GROUP BY kind, category"
        ).fetchall()

    summary = {"income_total": 0.0, "expense_total": 0.0, "income_count": 0,
               "expense_count": 0, "by_category": {}}
    for row in totals:
        summary[f"{row['kind']}_total"] = row["total"]
        summary[f"{row['kind']}_count"] = row["n"]
    for row in by_category:
        summary["by_category"].setdefault(row["kind"], {})[row["category"]] = row["total"]
    summary["balance"] = summary["income_total"] - summary["expense_total"]
    return summary


def list_transactions(limit: int = 20, db_path: Path = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

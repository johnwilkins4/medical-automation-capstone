"""SQLite helpers. Keeps db access in one place so tools don't open their own connections."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import DB_PATH, SCHEMA_SQL


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path: Path | str = DB_PATH, schema_sql: Path = SCHEMA_SQL) -> None:
    """Run schema.sql against the target DB, dropping any existing tables first."""
    sql = Path(schema_sql).read_text()
    with get_conn(db_path) as conn:
        conn.executescript(sql)


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


def log_agent_run(
    user_query: str,
    plan: list[dict],
    trace: list[dict],
    final_answer: str,
    latency_ms: int,
    success: bool = True,
    db_path: Path | str = DB_PATH,
) -> int:
    """Persist an agent run. Returns the new run_id."""
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO agent_runs
                (user_query, plan_json, trace_json, final_answer, latency_ms, success)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_query,
                json.dumps(plan, default=str),
                json.dumps(trace, default=str),
                final_answer,
                latency_ms,
                1 if success else 0,
            ),
        )
        return int(cur.lastrowid)


def recent_runs(limit: int = 25, db_path: Path | str = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows_to_dicts(rows)

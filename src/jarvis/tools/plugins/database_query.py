"""Plugin: query_database — run read-only SQL on local SQLite or CSV files."""
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def handle(tool_input: dict) -> str:
    try:
        source = str(tool_input.get("source", "")).strip()
        query = str(tool_input.get("query", "")).strip()
        limit = int(tool_input.get("limit", 50))

        if not source:
            return "ERROR: 'source' (file path) is required"
        if not query:
            return "ERROR: 'query' (SQL string) is required"

        if _FORBIDDEN.search(query):
            return "ERROR: only SELECT queries are allowed"

        path = Path(source)
        if not path.exists():
            return f"ERROR: file not found — {source}"

        suffix = path.suffix.lower()

        if suffix in {".csv", ".tsv"}:
            return _query_csv(path, query, limit, delimiter="," if suffix == ".csv" else "\t")
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            return _query_sqlite(path, query, limit)
        else:
            return f"ERROR: unsupported file type '{suffix}'. Use .csv, .tsv, .db, .sqlite, or .sqlite3"

    except Exception as e:
        return f"ERROR: query_database failed — {e}"


def _query_sqlite(path: Path, query: str, limit: int) -> str:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(limit)
        if not rows:
            return "Query returned no results."
        cols = rows[0].keys()
        return _to_markdown(cols, [dict(r) for r in rows], limit)
    finally:
        conn.close()


def _query_csv(path: Path, query: str, limit: int, delimiter: str) -> str:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return "ERROR: CSV file has no headers"
        cols = [re.sub(r"\W+", "_", c.strip()) for c in reader.fieldnames]
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        conn.execute(f"CREATE TABLE data ({col_defs})")
        placeholders = ", ".join("?" for _ in cols)
        for row in reader:
            vals = [row.get(orig, "") for orig in reader.fieldnames]
            conn.execute(f"INSERT INTO data VALUES ({placeholders})", vals)

    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(limit)
        if not rows:
            return "Query returned no results."
        result_cols = rows[0].keys()
        return _to_markdown(result_cols, [dict(r) for r in rows], limit)
    finally:
        conn.close()


def _to_markdown(cols, rows: list[dict], limit: int) -> str:
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [str(row.get(c, ""))[:50] for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    suffix = f"\n\n_(showing up to {limit} rows)_" if len(rows) == limit else ""
    return "\n".join(lines) + suffix


SCHEMA: dict = {
    "name": "query_database",
    "description": (
        "Run a read-only SQL SELECT query on a local SQLite database (.db/.sqlite) "
        "or CSV/TSV file. Returns results as a markdown table. "
        "Only SELECT is allowed — INSERT/UPDATE/DELETE/DROP are rejected."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Path to the .db, .sqlite, .csv, or .tsv file.",
            },
            "query": {
                "type": "string",
                "description": "SQL SELECT statement to execute.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return (default 50, max 500).",
                "default": 50,
            },
        },
        "required": ["source", "query"],
    },
}

"""Tool: search_memory — hybrid semantic + BM25 search with forgetting-curve reranking.

Retrieval pipeline:
  1. ChromaDB cosine similarity (semantic)
  2. BM25 keyword score over the fetched candidates (lexical)
  3. Forgetting-curve reranking: recency × frequency × hybrid_score

  hybrid_score    = 0.65 * semantic + 0.35 * bm25_normalized
  recency_factor  = exp(-days_since_last_access / 30)   # half-life ~21 days
  frequency_factor = 1 + log(1 + access_count) * 0.1   # logarithmic boost

Access tracking is persisted in reports_dir/jarvis.db (memory_access table).
"""
from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

_SEMANTIC_WEIGHT = 0.65
_BM25_WEIGHT = 0.35


# ── BM25 (pure Python, no extra deps) ────────────────────────────────────────

def _bm25_score(query: str, doc: str, k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 without corpus-level IDF (treats IDF=1 per term)."""
    query_terms = query.lower().split()
    doc_terms = doc.lower().split()
    doc_len = len(doc_terms) or 1
    avg_doc_len = 400  # approximate average report length in words
    score = 0.0
    for term in set(query_terms):  # deduplicate query terms
        tf = doc_terms.count(term)
        if tf == 0:
            continue
        tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += tf_norm
    return score

_collection = None  # ChromaDB collection, lazily initialised


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_collection(reports_dir: Path):
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    db_path = reports_dir / ".chroma"
    client = chromadb.PersistentClient(path=str(db_path))
    _collection = client.get_or_create_collection(
        name="jarvis_reports",
        metadata={"hnsw:space": "cosine"},
    )
    _index_reports(reports_dir, _collection)
    return _collection


def _index_reports(reports_dir: Path, collection) -> None:
    existing_ids: set[str] = set()
    try:
        existing_ids = set(collection.get()["ids"])
    except Exception:
        pass

    for md_file in sorted(reports_dir.glob("*.md")):
        doc_id = md_file.name
        if doc_id in existing_ids:
            continue
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        collection.add(
            ids=[doc_id],
            documents=[text[:2000]],
            metadatas=[{"filename": md_file.name, "path": str(md_file)}],
        )


def index_new_report(reports_dir: Path, filename: str) -> None:
    """Call this after saving a new report to keep the index up to date."""
    try:
        collection = _get_collection(reports_dir)
        md_file = reports_dir / filename
        if not md_file.exists():
            return
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        collection.upsert(
            ids=[filename],
            documents=[text[:2000]],
            metadatas=[{"filename": filename, "path": str(md_file)}],
        )
    except Exception:
        pass


# ── Forgetting curve (access tracking) ───────────────────────────────────────

def _access_conn(reports_dir: Path) -> sqlite3.Connection:
    db_path = reports_dir / "jarvis.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_access (
            doc_id         TEXT PRIMARY KEY,
            access_count   INTEGER NOT NULL DEFAULT 0,
            last_accessed  REAL    NOT NULL
        )
    """)
    conn.commit()
    return conn


def _rerank(docs, metas, distances, conn: sqlite3.Connection, query: str = "") -> list[tuple]:
    now = time.time()

    # Normalise BM25 scores across the candidate set
    bm25_raw = [_bm25_score(query, doc) for doc in docs] if query else [0.0] * len(docs)
    bm25_max = max(bm25_raw) or 1.0
    bm25_norm = [s / bm25_max for s in bm25_raw]

    scored = []
    for doc, meta, dist, bm25 in zip(docs, metas, distances, bm25_norm):
        doc_id = meta.get("filename", "")
        row = conn.execute(
            "SELECT access_count, last_accessed FROM memory_access WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        access_count = row[0] if row else 0
        last_accessed = row[1] if row else now

        semantic = 1.0 - dist
        hybrid = _SEMANTIC_WEIGHT * semantic + _BM25_WEIGHT * bm25
        days_old = (now - last_accessed) / 86400.0
        recency = math.exp(-days_old / 30.0)
        frequency = 1.0 + math.log1p(access_count) * 0.1
        score = hybrid * recency * frequency
        scored.append((score, semantic, doc, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _record_access(conn: sqlite3.Connection, doc_ids: list[str]) -> None:
    now = time.time()
    for doc_id in doc_ids:
        conn.execute(
            """
            INSERT INTO memory_access (doc_id, access_count, last_accessed) VALUES (?, 1, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                access_count  = access_count + 1,
                last_accessed = ?
            """,
            (doc_id, now, now),
        )
    conn.commit()


# ── Tool handler ──────────────────────────────────────────────────────────────

def handle_search_memory(tool_input: dict, reports_dir: Path) -> str:
    try:
        query = tool_input["query"]
        n_results = int(tool_input.get("n_results", 3))
        collection = _get_collection(reports_dir)

        total = collection.count()
        if total == 0:
            return "No research reports in memory yet. Save a report first with save_report."

        # Fetch 2× candidates so reranking has room to reorder
        fetch_n = min(n_results * 2, total)
        results = collection.query(query_texts=[query], n_results=fetch_n)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return "No matching reports found in memory."

        conn = _access_conn(reports_dir)
        try:
            scored = _rerank(docs, metas, distances, conn, query=query)
            top = scored[:n_results]
            _record_access(conn, [item[3].get("filename", "") for item in top])
        finally:
            conn.close()

        lines = [f"Found {len(top)} relevant report(s) in memory:\n"]
        for i, (score, similarity, doc, meta) in enumerate(top, 1):
            lines.append(
                f"**{i}. {meta.get('filename', 'unknown')}** "
                f"(score: {score:.3f}, similarity: {similarity:.3f})"
            )
            lines.append(doc[:500].strip())
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: search_memory failed — {e}"


SCHEMA: dict = {
    "name": "search_memory",
    "description": (
        "Search JARVIS's long-term memory — all previously saved research reports — "
        "using semantic similarity with forgetting-curve reranking (recent and frequently "
        "accessed reports score higher). Use this before starting new research."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in past reports, e.g. 'RLHF reward modeling'.",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return (default 3, max 10).",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}

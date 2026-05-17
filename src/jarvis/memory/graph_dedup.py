"""Knowledge graph entity deduplication.

Identifies near-duplicate entity names (case variations, whitespace, and
token-overlap similarity) and merges their relationships under a canonical
name — the one that appeared earlier or has more relationships.

Designed to be run as a scheduled job (e.g., nightly) or on demand.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on lowercased word tokens."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


def find_duplicate_pairs(
    db_path: Path,
    user_id: str = "shared",
    similarity_threshold: float = 0.85,
) -> list[tuple[str, str]]:
    """Return pairs of entity names that are likely duplicates.

    Each pair is (canonical, duplicate) where canonical is the older or
    more-connected entity. The duplicate should be merged into the canonical.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT name FROM entities WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        ).fetchall()
        conn.close()
    except Exception:
        return []

    names = [row[0] for row in rows]
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in seen or b in seen:
                continue
            if _normalize(a) == _normalize(b):
                pairs.append((a, b))
                seen.add(b)
            elif _token_overlap(a, b) >= similarity_threshold:
                pairs.append((a, b))
                seen.add(b)

    return pairs


def merge_entities(
    db_path: Path,
    canonical: str,
    duplicate: str,
    user_id: str = "shared",
) -> int:
    """Reassign all relationships from duplicate entity to canonical.

    Returns the number of relationship rows updated.
    """
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        updated = 0
        for col in ("from_entity", "to_entity"):
            cur = conn.execute(
                f"UPDATE relationships SET {col} = ? "
                f"WHERE {col} = ? AND user_id = ?",
                (canonical, duplicate, user_id),
            )
            updated += cur.rowcount
        conn.execute(
            "DELETE FROM entities WHERE name = ? AND user_id = ?",
            (duplicate, user_id),
        )
        conn.commit()
        conn.close()
        return updated
    except Exception:
        return 0


def deduplicate_entities(
    db_path: Path,
    user_id: str = "shared",
    similarity_threshold: float = 0.85,
) -> int:
    """Run a full dedup pass and return the number of merged entity pairs."""
    pairs = find_duplicate_pairs(db_path, user_id, similarity_threshold)
    merged = 0
    for canonical, duplicate in pairs:
        n = merge_entities(db_path, canonical, duplicate, user_id)
        if n >= 0:
            merged += 1
    return merged

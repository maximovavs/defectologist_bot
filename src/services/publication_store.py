from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


def normalize_publication_text(text: str) -> str:
    s = (text or "").lower().replace("ё", "е")
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_jaccard(a: str, b: str) -> float:
    sa = set((a or "").split())
    sb = set((b or "").split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def similarity_score(a: str, b: str) -> float:
    aa = normalize_publication_text(a)
    bb = normalize_publication_text(b)
    if not aa or not bb:
        return 0.0

    seq = SequenceMatcher(None, aa, bb).ratio()
    jac = _token_jaccard(aa, bb)
    return max(seq, jac)


@dataclass
class SimilarPublication:
    canonical_url: str
    similarity: float
    posted_at: str
    audience: str
    rubric_id: str


class PublicationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS publications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_url TEXT NOT NULL UNIQUE,
                    body_hash TEXT NOT NULL,
                    body_norm TEXT NOT NULL,
                    posted_at TEXT NOT NULL,
                    audience TEXT NOT NULL,
                    rubric_id TEXT NOT NULL,
                    rubric_title TEXT NOT NULL,
                    source_domain TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_publications_body_hash ON publications(body_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_publications_posted_at ON publications(posted_at DESC)"
            )

    def has_url(self, canonical_url: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM publications WHERE canonical_url = ? LIMIT 1",
                (canonical_url,),
            ).fetchone()
        return row is not None

    def has_body_hash(self, body_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM publications WHERE body_hash = ? LIMIT 1",
                (body_hash,),
            ).fetchone()
        return row is not None

    def find_similar(
        self,
        body_text: str,
        threshold: float = 0.90,
        since_iso: Optional[str] = None,
        limit: int = 300,
    ) -> Optional[SimilarPublication]:
        body_norm = normalize_publication_text(body_text)
        if not body_norm:
            return None

        sql = """
            SELECT canonical_url, body_norm, posted_at, audience, rubric_id
            FROM publications
        """
        params = []
        if since_iso:
            sql += " WHERE posted_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY posted_at DESC LIMIT ?"
        params.append(limit)

        best: Optional[SimilarPublication] = None
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        for row in rows:
            score = similarity_score(body_norm, row["body_norm"] or "")
            if score < threshold:
                continue
            if best is None or score > best.similarity:
                best = SimilarPublication(
                    canonical_url=row["canonical_url"],
                    similarity=score,
                    posted_at=row["posted_at"],
                    audience=row["audience"],
                    rubric_id=row["rubric_id"],
                )
        return best

    def record_publication(
        self,
        canonical_url: str,
        body_hash: str,
        body_text: str,
        posted_at: str,
        audience: str,
        rubric_id: str,
        rubric_title: str,
        source_domain: str,
    ) -> None:
        body_norm = normalize_publication_text(body_text)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO publications (
                    canonical_url,
                    body_hash,
                    body_norm,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_url,
                    body_hash,
                    body_norm,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain,
                ),
            )

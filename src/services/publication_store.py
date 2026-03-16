from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SEMANTIC_DIM = 384

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

_STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "к", "ко", "о", "об", "обо", "от", "до", "за", "из", "у",
    "а", "но", "или", "либо", "же", "то", "это", "этот", "эта", "эти", "того", "такой", "такая",
    "как", "так", "если", "когда", "чтобы", "что", "чем", "при", "для", "не", "ни", "над", "под",
    "their", "with", "from", "into", "that", "this", "then", "than", "have", "has", "had", "are",
    "was", "were", "for", "and", "the", "you", "your", "they", "them", "his", "her", "our", "not",
}

_RU_SUFFIXES = (
    "иями", "ями", "ами", "иях", "иях", "иях", "ого", "ему", "ому", "ыми", "ими", "ее", "ие", "ые",
    "ое", "ей", "ий", "ый", "ой", "ам", "ям", "ом", "ем", "ах", "ях", "ию", "ью", "ия", "ья", "а",
    "я", "ы", "и", "е", "о", "у",
)

_EN_SUFFIXES = ("ing", "edly", "edly", "edly", "ed", "ly", "es", "s")


def normalize_publication_text(text: str) -> str:
    s = (text or "").lower().replace("ё", "е")
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stem_token(token: str) -> str:
    t = token.lower().replace("ё", "е")
    if len(t) <= 4:
        return t

    for suf in _RU_SUFFIXES:
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]

    for suf in _EN_SUFFIXES:
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]

    return t


def _semantic_tokens(text: str) -> List[str]:
    raw = _TOKEN_RE.findall(normalize_publication_text(text))
    out: List[str] = []
    for token in raw:
        if len(token) <= 1:
            continue
        if token in _STOPWORDS:
            continue
        stemmed = _stem_token(token)
        if stemmed in _STOPWORDS or len(stemmed) <= 1:
            continue
        out.append(stemmed)
    return out


def text_to_embedding(text: str, dim: int = SEMANTIC_DIM) -> List[float]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return [0.0] * dim

    vector = [0.0] * dim

    def add_feature(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vector[idx] += sign * weight

    for tok in tokens:
        add_feature(f"u:{tok}", 1.0)

    for i in range(len(tokens) - 1):
        add_feature(f"b:{tokens[i]}_{tokens[i+1]}", 1.45)

    for i in range(len(tokens) - 2):
        add_feature(f"t:{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}", 1.20)

    for tok in set(tokens):
        if len(tok) >= 6:
            for j in range(len(tok) - 3):
                add_feature(f"c4:{tok[j:j+4]}", 0.25)

    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return [0.0] * dim
    return [v / norm for v in vector]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _vec_to_json(vec: List[float]) -> str:
    return json.dumps(vec, ensure_ascii=False, separators=(",", ":"))


def _vec_from_json(raw: str) -> List[float]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [float(x) for x in data]
    except Exception:
        return []
    return []


@dataclass
class SimilarPublication:
    canonical_url: str
    similarity: float
    posted_at: str
    audience: str
    rubric_id: str
    match_field: str


class PublicationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _existing_columns(self) -> Dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(publications)").fetchall()
        return {row["name"]: row["type"] for row in rows}

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS publications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_url TEXT NOT NULL UNIQUE,
                    body_hash TEXT NOT NULL DEFAULT '',
                    evidence_hash TEXT NOT NULL DEFAULT '',
                    body_norm TEXT NOT NULL DEFAULT '',
                    evidence_norm TEXT NOT NULL DEFAULT '',
                    body_vec_json TEXT NOT NULL DEFAULT '',
                    evidence_vec_json TEXT NOT NULL DEFAULT '',
                    posted_at TEXT NOT NULL DEFAULT '',
                    audience TEXT NOT NULL DEFAULT '',
                    rubric_id TEXT NOT NULL DEFAULT '',
                    rubric_title TEXT NOT NULL DEFAULT '',
                    source_domain TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.commit()

        existing = self._existing_columns()
        wanted = {
            "body_hash": "TEXT NOT NULL DEFAULT ''",
            "evidence_hash": "TEXT NOT NULL DEFAULT ''",
            "body_norm": "TEXT NOT NULL DEFAULT ''",
            "evidence_norm": "TEXT NOT NULL DEFAULT ''",
            "body_vec_json": "TEXT NOT NULL DEFAULT ''",
            "evidence_vec_json": "TEXT NOT NULL DEFAULT ''",
            "posted_at": "TEXT NOT NULL DEFAULT ''",
            "audience": "TEXT NOT NULL DEFAULT ''",
            "rubric_id": "TEXT NOT NULL DEFAULT ''",
            "rubric_title": "TEXT NOT NULL DEFAULT ''",
            "source_domain": "TEXT NOT NULL DEFAULT ''",
        }

        with self._connect() as conn:
            for col, ddl in wanted.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE publications ADD COLUMN {col} {ddl}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_publications_body_hash ON publications(body_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_publications_evidence_hash ON publications(evidence_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_publications_posted_at ON publications(posted_at DESC)")
            conn.commit()

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

    def has_evidence_hash(self, evidence_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM publications WHERE evidence_hash = ? LIMIT 1",
                (evidence_hash,),
            ).fetchone()
        return row is not None

    def find_semantic_duplicate(
        self,
        text: str,
        threshold: float = 0.95,
        since_iso: Optional[str] = None,
        limit: int = 500,
        compare: str = "body",
    ) -> Optional[SimilarPublication]:
        candidate_vec = text_to_embedding(text)
        if not any(candidate_vec):
            return None

        compare = (compare or "body").lower()
        field_map = {
            "body": [("body", "body_vec_json")],
            "evidence": [("evidence", "evidence_vec_json")],
            "both": [("body", "body_vec_json"), ("evidence", "evidence_vec_json")],
        }
        targets = field_map.get(compare, field_map["body"])

        sql = """
            SELECT canonical_url, body_vec_json, evidence_vec_json, posted_at, audience, rubric_id
            FROM publications
        """
        params: List[object] = []
        if since_iso:
            sql += " WHERE posted_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY posted_at DESC LIMIT ?"
        params.append(limit)

        best: Optional[SimilarPublication] = None

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        for row in rows:
            for match_field, col_name in targets:
                vec = _vec_from_json(row[col_name] or "")
                if not vec:
                    continue
                score = cosine_similarity(candidate_vec, vec)
                if score < threshold:
                    continue
                if best is None or score > best.similarity:
                    best = SimilarPublication(
                        canonical_url=row["canonical_url"],
                        similarity=score,
                        posted_at=row["posted_at"],
                        audience=row["audience"],
                        rubric_id=row["rubric_id"],
                        match_field=match_field,
                    )

        return best

    def record_publication(
        self,
        canonical_url: str,
        body_hash: str,
        body_text: str,
        evidence_hash: str,
        evidence_text: str,
        posted_at: str,
        audience: str,
        rubric_id: str,
        rubric_title: str,
        source_domain: str,
    ) -> None:
        body_norm = normalize_publication_text(body_text)
        evidence_norm = normalize_publication_text(evidence_text)
        body_vec = _vec_to_json(text_to_embedding(body_text))
        evidence_vec = _vec_to_json(text_to_embedding(evidence_text))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO publications (
                    canonical_url,
                    body_hash,
                    evidence_hash,
                    body_norm,
                    evidence_norm,
                    body_vec_json,
                    evidence_vec_json,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_url,
                    body_hash,
                    evidence_hash,
                    body_norm,
                    evidence_norm,
                    body_vec,
                    evidence_vec,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain,
                ),
            )
            conn.commit()

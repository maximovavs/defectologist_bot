
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer, util


SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"

_MODEL: Optional[SentenceTransformer] = None


def get_semantic_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
    return _MODEL


def normalize_publication_text(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").split()).strip()


def text_to_embedding(text: str) -> List[float]:
    cleaned = normalize_publication_text(text)
    if not cleaned:
        return []
    model = get_semantic_model()
    vec = model.encode(cleaned, normalize_embeddings=True)
    return [float(x) for x in vec.tolist()]


def text_batch_to_embeddings(texts: Sequence[str]) -> List[List[float]]:
    prepared = [normalize_publication_text(t) for t in texts]
    if not prepared:
        return []
    model = get_semantic_model()
    matrix = model.encode(list(prepared), normalize_embeddings=True)
    return [[float(x) for x in row.tolist()] for row in matrix]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    score = util.pytorch_cos_sim([a], [b])
    return float(score[0][0].item())


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
                    body_embedding_model TEXT NOT NULL DEFAULT '',
                    evidence_embedding_model TEXT NOT NULL DEFAULT '',
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
            "body_embedding_model": "TEXT NOT NULL DEFAULT ''",
            "evidence_embedding_model": "TEXT NOT NULL DEFAULT ''",
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

    def _collect_vectors_for_rows(
        self,
        rows: Sequence[sqlite3.Row],
        targets: Sequence[Tuple[str, str, str]],
    ) -> Dict[Tuple[int, str], List[float]]:
        by_key: Dict[Tuple[int, str], List[float]] = {}
        to_encode_texts: List[str] = []
        to_encode_keys: List[Tuple[int, str]] = []

        for idx, row in enumerate(rows):
            for match_field, vec_col, model_col in targets:
                vec = _vec_from_json(row[vec_col] or "")
                stored_model = (row[model_col] or "").strip()
                if vec and stored_model == SEMANTIC_MODEL_NAME:
                    by_key[(idx, match_field)] = vec
                    continue

                norm_col = "body_norm" if match_field == "body" else "evidence_norm"
                norm_text = normalize_publication_text(row[norm_col] or "")
                if norm_text:
                    to_encode_texts.append(norm_text)
                    to_encode_keys.append((idx, match_field))

        if to_encode_texts:
            for key, vec in zip(to_encode_keys, text_batch_to_embeddings(to_encode_texts)):
                by_key[key] = vec

        return by_key

    def find_semantic_duplicate(
        self,
        text: str,
        threshold: float = 0.85,
        since_iso: Optional[str] = None,
        limit: int = 500,
        compare: str = "body",
    ) -> Optional[SimilarPublication]:
        candidate_vec = text_to_embedding(text)
        if not candidate_vec:
            return None

        compare = (compare or "body").lower()
        field_map = {
            "body": [("body", "body_vec_json", "body_embedding_model")],
            "evidence": [("evidence", "evidence_vec_json", "evidence_embedding_model")],
            "both": [
                ("body", "body_vec_json", "body_embedding_model"),
                ("evidence", "evidence_vec_json", "evidence_embedding_model"),
            ],
        }
        targets = field_map.get(compare, field_map["body"])

        sql = """
            SELECT canonical_url, body_norm, evidence_norm, body_vec_json, evidence_vec_json,
                   body_embedding_model, evidence_embedding_model, posted_at, audience, rubric_id
            FROM publications
        """
        params: List[object] = []
        if since_iso:
            sql += " WHERE posted_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY posted_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        cached_vectors = self._collect_vectors_for_rows(rows, targets)
        best: Optional[SimilarPublication] = None

        for idx, row in enumerate(rows):
            for match_field, _, _ in targets:
                vec = cached_vectors.get((idx, match_field), [])
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
        body_vec, evidence_vec = text_batch_to_embeddings([body_norm, evidence_norm])

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
                    body_embedding_model,
                    evidence_embedding_model,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_url,
                    body_hash,
                    evidence_hash,
                    body_norm,
                    evidence_norm,
                    _vec_to_json(body_vec),
                    _vec_to_json(evidence_vec),
                    SEMANTIC_MODEL_NAME,
                    SEMANTIC_MODEL_NAME,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain,
                ),
            )
            conn.commit()

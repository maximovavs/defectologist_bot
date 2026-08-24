
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer, util


SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"

_MODEL: Optional[SentenceTransformer] = None
_SEMANTIC_DISABLED = False

_DELIVERY_HOOKS_INSTALLED = False
_DELIVERY_ORIGINAL_TG_REQUEST = None
_DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL = None
_ACTIVE_DELIVERY_STORE: Optional["PublicationStore"] = None
_ACTIVE_DELIVERY_ATTEMPT_KEY = ""

_PUBLISHER_HISTORY_NAMES = frozenset({
    "publication_history.sqlite3",
    "publication_history_test.sqlite3",
})


class PublicationDeliveryStateBlocked(RuntimeError):
    """Durable delivery state is missing or contains an unresolved primary send."""


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _is_publisher_history_path(path: Path) -> bool:
    return path.name in _PUBLISHER_HISTORY_NAMES and path.parent.name == ".state"


def _is_production_history_path(path: Path) -> bool:
    return path.name == "publication_history.sqlite3" and path.parent.name == ".state"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_message_id(payload: object) -> Optional[int]:
    result = payload.get("result") if isinstance(payload, dict) else None
    message_id = result.get("message_id") if isinstance(result, dict) else None
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        return None
    return message_id


def get_semantic_model() -> Optional[SentenceTransformer]:
    global _MODEL, _SEMANTIC_DISABLED
    if _SEMANTIC_DISABLED:
        return None
    if _MODEL is None:
        try:
            _MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
        except Exception as e:
            _SEMANTIC_DISABLED = True
            print(
                f"[WARN] semantic_model_unavailable model={SEMANTIC_MODEL_NAME} err={e}",
                flush=True,
            )
            return None
    return _MODEL


def normalize_publication_text(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").split()).strip()


def text_to_embedding(text: str) -> List[float]:
    cleaned = normalize_publication_text(text)
    if not cleaned:
        return []
    model = get_semantic_model()
    if model is None:
        return []
    try:
        vec = model.encode(cleaned, normalize_embeddings=True)
        return [float(x) for x in vec.tolist()]
    except Exception as e:
        print(
            f"[WARN] semantic_encode_failed model={SEMANTIC_MODEL_NAME} err={e}",
            flush=True,
        )
        return []


def text_batch_to_embeddings(texts: Sequence[str]) -> List[List[float]]:
    prepared = [normalize_publication_text(t) for t in texts]
    if not prepared:
        return []
    model = get_semantic_model()
    if model is None:
        return [[] for _ in prepared]
    try:
        matrix = model.encode(list(prepared), normalize_embeddings=True)
        return [[float(x) for x in row.tolist()] for row in matrix]
    except Exception as e:
        print(
            f"[WARN] semantic_batch_encode_failed model={SEMANTIC_MODEL_NAME} err={e}",
            flush=True,
        )
        return [[] for _ in prepared]


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
        self._confirmed_delivery_attempt_key = ""
        self._delivery_hooks_enabled = (
            _is_publisher_history_path(self.db_path) and not _env_truthy("DRY_RUN")
        )

        if _is_production_history_path(self.db_path) and not _env_truthy("DRY_RUN"):
            if os.getenv("PRODUCTION_STATE_RESTORED", "").strip() != "1":
                raise PublicationDeliveryStateBlocked(
                    "production_state_not_restored: refusing production publication"
                )
            if not self.db_path.is_file():
                raise PublicationDeliveryStateBlocked(
                    "production_history_missing: refusing production publication"
                )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        if self._delivery_hooks_enabled:
            if self.has_unresolved_delivery_attempts():
                raise PublicationDeliveryStateBlocked(
                    "unresolved_delivery_quarantine: manual recovery required before publication"
                )
            self._install_publisher_delivery_hooks()

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS delivery_attempts (
                    attempt_key TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    body_hash TEXT NOT NULL DEFAULT '',
                    primary_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT ''
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_attempts_state ON delivery_attempts(state)")
            conn.commit()

    def has_unresolved_delivery_attempts(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM delivery_attempts LIMIT 1").fetchone()
        return row is not None

    def delivery_attempts(self) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT attempt_key, state, body_hash, primary_message_ids_json,
                       started_at, updated_at, last_error
                FROM delivery_attempts
                ORDER BY started_at, attempt_key
                """
            ).fetchall()
        out: List[Dict[str, object]] = []
        for row in rows:
            try:
                message_ids = json.loads(row["primary_message_ids_json"] or "[]")
            except Exception:
                message_ids = []
            out.append(
                {
                    "attempt_key": row["attempt_key"],
                    "state": row["state"],
                    "body_hash": row["body_hash"],
                    "primary_message_ids": message_ids if isinstance(message_ids, list) else [],
                    "started_at": row["started_at"],
                    "updated_at": row["updated_at"],
                    "last_error": row["last_error"],
                }
            )
        return out

    def begin_delivery_attempt(self, attempt_key: str, body_hash: str) -> None:
        if not attempt_key:
            raise PublicationDeliveryStateBlocked("delivery_attempt_key_missing")
        now = _utc_now_iso()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO delivery_attempts (
                        attempt_key, state, body_hash, primary_message_ids_json,
                        started_at, updated_at, last_error
                    ) VALUES (?, 'pending', ?, '[]', ?, ?, '')
                    """,
                    (attempt_key, body_hash, now, now),
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise PublicationDeliveryStateBlocked(
                "delivery_attempt_already_quarantined"
            ) from exc

    def _update_delivery_state(self, attempt_key: str, state: str, last_error: str = "") -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE delivery_attempts
                SET state = ?, updated_at = ?, last_error = ?
                WHERE attempt_key = ?
                """,
                (state, _utc_now_iso(), last_error, attempt_key),
            )
            if cursor.rowcount != 1:
                raise PublicationDeliveryStateBlocked("delivery_attempt_missing")
            conn.commit()

    def mark_delivery_ambiguous(self, attempt_key: str, error_type: str) -> None:
        self._update_delivery_state(attempt_key, "ambiguous", error_type)

    def mark_delivery_confirmed(self, attempt_key: str) -> None:
        self._update_delivery_state(attempt_key, "confirmed", "")
        self._confirmed_delivery_attempt_key = attempt_key

    def clear_delivery_attempt(self, attempt_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM delivery_attempts WHERE attempt_key = ?", (attempt_key,))
            conn.commit()
        if self._confirmed_delivery_attempt_key == attempt_key:
            self._confirmed_delivery_attempt_key = ""

    def add_delivery_message_id(self, attempt_key: str, message_id: int) -> None:
        if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
            return
        with self._connect() as conn:
            row = conn.execute(
                "SELECT primary_message_ids_json FROM delivery_attempts WHERE attempt_key = ?",
                (attempt_key,),
            ).fetchone()
            if row is None:
                raise PublicationDeliveryStateBlocked("delivery_attempt_missing")
            try:
                message_ids = json.loads(row["primary_message_ids_json"] or "[]")
            except Exception:
                message_ids = []
            if not isinstance(message_ids, list):
                message_ids = []
            if message_id not in message_ids:
                message_ids.append(message_id)
            conn.execute(
                """
                UPDATE delivery_attempts
                SET primary_message_ids_json = ?, updated_at = ?
                WHERE attempt_key = ?
                """,
                (json.dumps(message_ids, separators=(",", ":")), _utc_now_iso(), attempt_key),
            )
            conn.commit()

    def remove_delivery_message_id(self, attempt_key: str, message_id: int) -> None:
        if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
            return
        with self._connect() as conn:
            row = conn.execute(
                "SELECT primary_message_ids_json FROM delivery_attempts WHERE attempt_key = ?",
                (attempt_key,),
            ).fetchone()
            if row is None:
                raise PublicationDeliveryStateBlocked("delivery_attempt_missing")
            try:
                message_ids = json.loads(row["primary_message_ids_json"] or "[]")
            except Exception:
                message_ids = []
            if not isinstance(message_ids, list):
                message_ids = []
            message_ids = [value for value in message_ids if value != message_id]
            conn.execute(
                """
                UPDATE delivery_attempts
                SET primary_message_ids_json = ?, updated_at = ?
                WHERE attempt_key = ?
                """,
                (json.dumps(message_ids, separators=(",", ":")), _utc_now_iso(), attempt_key),
            )
            conn.commit()

    def _install_publisher_delivery_hooks(self) -> None:
        global _DELIVERY_HOOKS_INSTALLED
        global _DELIVERY_ORIGINAL_TG_REQUEST
        global _DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL
        global _ACTIVE_DELIVERY_STORE

        publisher = sys.modules.get("src.publisher.run_publisher")
        if publisher is None:
            return
        if not hasattr(publisher, "tg_request") or not hasattr(publisher, "send_post_with_visual"):
            return

        _ACTIVE_DELIVERY_STORE = self

        if _DELIVERY_HOOKS_INSTALLED:
            return

        _DELIVERY_ORIGINAL_TG_REQUEST = publisher.tg_request
        _DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL = publisher.send_post_with_visual

        def tg_request_with_delivery_receipts(method, data, files=None):
            payload = _DELIVERY_ORIGINAL_TG_REQUEST(method, data=data, files=files)
            store = _ACTIVE_DELIVERY_STORE
            attempt_key = _ACTIVE_DELIVERY_ATTEMPT_KEY
            if store is None or not attempt_key:
                return payload

            if method in {"sendPhoto", "sendMessage"}:
                message_id = _payload_message_id(payload)
                if message_id is not None:
                    store.add_delivery_message_id(attempt_key, message_id)
            elif method == "deleteMessage":
                message_id = data.get("message_id") if isinstance(data, dict) else None
                if isinstance(message_id, int) and not isinstance(message_id, bool) and message_id > 0:
                    store.remove_delivery_message_id(attempt_key, message_id)
            return payload

        def send_post_with_durable_state(chat_id, photo_buffer, plain_post, html_full_post):
            global _ACTIVE_DELIVERY_ATTEMPT_KEY

            store = _ACTIVE_DELIVERY_STORE
            if store is None:
                return _DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL(
                    chat_id, photo_buffer, plain_post, html_full_post
                )

            body_hash = hashlib.sha256((plain_post or "").encode("utf-8")).hexdigest()
            attempt_key = hashlib.sha256(
                f"{chat_id}\0{body_hash}".encode("utf-8")
            ).hexdigest()
            ambiguous_type = getattr(publisher, "TelegramDeliveryOutcomeAmbiguous", RuntimeError)

            try:
                store.begin_delivery_attempt(attempt_key, body_hash)
            except Exception as state_error:
                raise ambiguous_type(
                    "telegram_delivery_outcome_ambiguous:delivery_state_begin_failed"
                ) from state_error

            _ACTIVE_DELIVERY_ATTEMPT_KEY = attempt_key
            try:
                result = _DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL(
                    chat_id, photo_buffer, plain_post, html_full_post
                )
            except Exception as send_error:
                if isinstance(send_error, ambiguous_type):
                    try:
                        store.mark_delivery_ambiguous(
                            attempt_key, send_error.__class__.__name__
                        )
                    except Exception:
                        # begin_delivery_attempt committed before Telegram mutation;
                        # keeping "pending" still quarantines the next run.
                        pass
                    raise

                try:
                    store.clear_delivery_attempt(attempt_key)
                except Exception as state_error:
                    raise ambiguous_type(
                        "telegram_delivery_outcome_ambiguous:"
                        "delivery_state_clear_failed_after_deterministic_reject"
                    ) from state_error
                raise
            else:
                try:
                    store.mark_delivery_confirmed(attempt_key)
                except Exception as state_error:
                    raise ambiguous_type(
                        "telegram_delivery_outcome_ambiguous:delivery_state_confirm_failed"
                    ) from state_error
                return result
            finally:
                _ACTIVE_DELIVERY_ATTEMPT_KEY = ""

        publisher.tg_request = tg_request_with_delivery_receipts
        publisher.send_post_with_visual = send_post_with_durable_state
        _DELIVERY_HOOKS_INSTALLED = True

    def deactivate_publisher_delivery_hooks(self) -> None:
        global _ACTIVE_DELIVERY_STORE
        global _ACTIVE_DELIVERY_ATTEMPT_KEY
        if _ACTIVE_DELIVERY_STORE is self:
            _ACTIVE_DELIVERY_STORE = None
            _ACTIVE_DELIVERY_ATTEMPT_KEY = ""

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

    def has_url_since(self, canonical_url: str, since_iso: str) -> bool:
        """True when this canonical URL was published within the cooldown window."""
        if not since_iso:
            return self.has_url(canonical_url)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM publications WHERE canonical_url = ? AND posted_at >= ? LIMIT 1",
                (canonical_url, since_iso),
            ).fetchone()
        return row is not None

    def has_evidence_hash_since(self, evidence_hash: str, since_iso: str) -> bool:
        """True when this evidence hash was published within the cooldown window."""
        if not since_iso:
            return self.has_evidence_hash(evidence_hash)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM publications WHERE evidence_hash = ? AND posted_at >= ? LIMIT 1",
                (evidence_hash, since_iso),
            ).fetchone()
        return row is not None

    def recent_source_domains(self, limit: int = 3) -> List[str]:
        """
        Domains of the last `limit` actual publications, newest first, de-duplicated.

        The window is the rows, not the distinct domains: three publications from
        `d1, d1, d2` yield `{d1, d2}` and must not reach further back for a third
        distinct domain.
        """
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_domain FROM publications
                ORDER BY posted_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        seen: List[str] = []
        for row in rows:
            domain = (row["source_domain"] or "").strip().lower()
            if domain and domain not in seen:
                seen.append(domain)
        return seen

    def find_editorial_core_duplicate(
        self,
        core_text: str,
        threshold: float,
        since_iso: Optional[str] = None,
        limit: int = 200,
        core_extractor=None,
    ) -> Optional[SimilarPublication]:
        """
        Cross-rubric editorial-core freshness check over recent publications.

        The stored body text is reduced with the same deterministic extractor as the
        candidate, so two posts that give the same practical advice match even when
        their wording, rubric or source differ. Nothing is persisted and the schema
        is untouched: cores are derived from `body_norm` on the fly.

        Fails open: when the semantic model is unavailable the candidate embedding is
        empty and this returns None, leaving the exact URL/evidence/body protections
        to do their work.
        """
        cleaned_core = normalize_publication_text(core_text)
        if not cleaned_core:
            return None
        candidate_vec = text_to_embedding(cleaned_core)
        if not candidate_vec:
            return None

        sql = """
            SELECT canonical_url, body_norm, posted_at, audience, rubric_id
            FROM publications
            WHERE body_norm != ''
        """
        params: List[object] = []
        if since_iso:
            sql += " AND posted_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY posted_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return None

        extract = core_extractor or (lambda text: text)
        cores: List[str] = []
        kept_rows: List[sqlite3.Row] = []
        for row in rows:
            stored_core = normalize_publication_text(extract(row["body_norm"] or ""))
            if stored_core:
                cores.append(stored_core)
                kept_rows.append(row)
        if not cores:
            return None

        best: Optional[SimilarPublication] = None
        for row, vec in zip(kept_rows, text_batch_to_embeddings(cores)):
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
                    match_field="editorial_core",
                )
        return best

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
        vectors = text_batch_to_embeddings([body_norm, evidence_norm])
        if len(vectors) >= 2:
            body_vec, evidence_vec = vectors[0], vectors[1]
        else:
            body_vec, evidence_vec = [], []

        body_embedding_model = SEMANTIC_MODEL_NAME if body_vec else ""
        evidence_embedding_model = SEMANTIC_MODEL_NAME if evidence_vec else ""
        confirmed_attempt_key = self._confirmed_delivery_attempt_key

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
                    body_embedding_model,
                    evidence_embedding_model,
                    posted_at,
                    audience,
                    rubric_id,
                    rubric_title,
                    source_domain,
                ),
            )
            if confirmed_attempt_key:
                cursor = conn.execute(
                    "DELETE FROM delivery_attempts WHERE attempt_key = ?",
                    (confirmed_attempt_key,),
                )
                if cursor.rowcount != 1:
                    raise PublicationDeliveryStateBlocked(
                        "confirmed_delivery_attempt_missing_during_record"
                    )
            conn.commit()

        if confirmed_attempt_key:
            self._confirmed_delivery_attempt_key = ""

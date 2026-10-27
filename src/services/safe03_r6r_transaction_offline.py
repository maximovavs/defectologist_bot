"""SAFE03 R6-R: fixture-only source review + single SQLite reservation boundary.

NOT a Telegram sender, trusted-source registry or production integration.  The
existing publisher does not participate in this transaction, so this deliberately
cannot prove production cross-publisher serializability or exactly-once delivery.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from src.services import safe03_r6d_offline_acceptance as frozen
from src.services import safe03_r6n_warning_acceptance_offline as current
from src.services import safe03_r6q_source_sender_offline as q

# Hard-coded real roots intentionally empty; neither owner attestations nor an
# SHA alone is a source verification or a clinician authentication certificate.
PRODUCTION_SOURCE_ROOTS = MappingProxyType({})
PRODUCTION_APPROVAL_ROOTS = MappingProxyType({})
_LOCAL_SEAL_KEY = secrets.token_bytes(32)


def public_source_admit(*_args: object, **_kwargs: object) -> str:
    return 'NO_POST'


def public_reserve(*_args: object, **_kwargs: object) -> str:
    return 'NO_POST'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sign(*args: str) -> str:
    return hmac.new(_LOCAL_SEAL_KEY, '\0'.join(args).encode('utf-8'), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class ClaimExcerpt:
    claim_id: str
    start: int  # byte offsets into frozen source snapshot
    end: int
    quote_sha256: str


@dataclass(frozen=True)
class SourceReview:
    catalog_id: str
    full_entry_sha256: str
    source_url: str
    evidence_sha256: str
    reviewer_id: str
    independent_checker_id: str
    review_receipt_sha256: str
    independent_checker_receipt_sha256: str
    reviewed_at: str
    expires_at: str
    revoked: bool
    excerpts: tuple[ClaimExcerpt, ...]


@dataclass(frozen=True)
class SourceFixtureRoots:
    reviewer_receipts: tuple[tuple[str, str], ...]
    checker_receipts: tuple[tuple[str, str], ...]
    approved_hosts: frozenset[str]


@dataclass(frozen=True)
class SourcePermit:
    status: str
    reason: str
    catalog_id: str = ''
    full_entry_sha256: str = ''
    evidence_sha256: str = ''
    source_url: str = ''
    independent_checker_receipt_sha256: str = ''
    seal: str = ''


def admit_fixture_source(entry: object, finding: object, snapshot: object,
                         review: object, roots: object, *, now_utc: object) -> SourcePermit:
    """Syntax/provenance *fixture* verification, never semantic clinical approval.

    Review hash is only compared with a separately supplied synthetic root; it
    DOES NOT authenticate a real specialist, nor prove the quotes support claims.
    """
    invalid = SourcePermit('NO_POST', 'source_root_or_review_not_trusted')
    if (type(entry) is not frozen.Entry or type(finding) is not q.SourceFinding
        or type(review) is not SourceReview or type(roots) is not SourceFixtureRoots
        or type(snapshot) is not bytes or type(now_utc) is not datetime
        or now_utc.utcoffset() != timedelta(0) or review.revoked is not False):
        return invalid
    try:
        if entry != current.validated_offline_entries()[entry.catalog_id]:
            return invalid
        reviewed = datetime.fromisoformat(review.reviewed_at)
        expires = datetime.fromisoformat(review.expires_at)
        uri = urlsplit(review.source_url)
        source_host = (uri.hostname or '').lower()
        url_port = uri.port  # malformed port is rejected by the enclosing exception guard
    except (ValueError, KeyError, TypeError, OSError):
        return invalid
    if (reviewed.utcoffset() != timedelta(0) or expires.utcoffset() != timedelta(0)
        or not reviewed <= now_utc < expires or expires > reviewed + timedelta(days=365)
        or reviewed > expires or uri.scheme != 'https' or not source_host
        or uri.username is not None or uri.password is not None or url_port not in (None, 443)
        or uri.fragment or source_host not in roots.approved_hosts
        or review.reviewer_id == review.independent_checker_id
        or review.reviewer_id not in dict(roots.reviewer_receipts)
        or review.independent_checker_id not in dict(roots.checker_receipts)
        or dict(roots.reviewer_receipts)[review.reviewer_id] != review.review_receipt_sha256
        or dict(roots.checker_receipts)[review.independent_checker_id] != review.independent_checker_receipt_sha256
        or review.review_receipt_sha256 == review.independent_checker_receipt_sha256
        or len(snapshot) == 0 or len(snapshot) > 500_000
        or review.full_entry_sha256 != frozen.digest(entry)
        or review.catalog_id != entry.catalog_id
        or review.evidence_sha256 != _digest(snapshot)
        or finding.status != 'OFFLINE_FIXTURE_ONLY'
        or finding.catalog_id != entry.catalog_id
        or finding.whole_entry_sha256 != review.full_entry_sha256
        or finding.source_url != review.source_url
        or finding.source_receipt_sha256 != review.review_receipt_sha256):
        return invalid
    # R6-Q finding is itself sealed for the local process. Still fixture only.
    expected_finding_seal = q._seal(finding.catalog_id, finding.whole_entry_sha256,
                                     finding.source_url, finding.source_receipt_sha256)
    if not hmac.compare_digest(finding.local_fixture_seal, expected_finding_seal):
        return invalid
    must = q.REQUIRED_CLAIMS.get(entry.catalog_id)
    if (not must or type(review.excerpts) is not tuple or
        len(review.excerpts) != len(must) or
        {x.claim_id for x in review.excerpts if type(x) is ClaimExcerpt} != must):
        return invalid
    for x in review.excerpts:
        if (type(x) is not ClaimExcerpt or type(x.start) is not int or type(x.end) is not int
            or not 0 <= x.start < x.end <= len(snapshot) or x.end - x.start > 2_000
            or x.quote_sha256 != _digest(snapshot[x.start:x.end])
            or not snapshot[x.start:x.end].strip()):
            return invalid
    return SourcePermit('OFFLINE_FIXTURE_ONLY', 'not_independently_authenticated_or_semantically_reviewed',
                        entry.catalog_id, review.full_entry_sha256,
                        review.evidence_sha256, review.source_url,
                        review.independent_checker_receipt_sha256, _sign(entry.catalog_id, review.full_entry_sha256,
                              review.evidence_sha256, review.source_url,
                              review.independent_checker_receipt_sha256))


@dataclass(frozen=True)
class TxResult:
    status: str
    reason: str
    reservation_key: str = ''


class OfflineSharedTransaction:
    """Transactional *fixture* prototype using actual PublicationStore tables.

    Only for a disposable test database, never a production historical DB. Other
    publishers do not participate in this lock across their pre-send checks.
    """

    def __init__(self, path: Path):
        if (not isinstance(path, Path) or path.name != 'test_history.sqlite3'
            or path.is_symlink() or not path.is_file()):
            raise ValueError('only_existing_disposable_test_history_allowed')
        self.path = path

    def _connect(self):
        c = sqlite3.connect(self.path, timeout=8, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA busy_timeout=8000')
        return c

    @staticmethod
    def _schema(c):
        # Never create the production tables, and refuse partial/malformed schemas.
        names = {row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'publications', 'delivery_attempts'} <= names:
            raise ValueError('production_schema_unavailable')
        pubs = {r[1] for r in c.execute('PRAGMA table_info(publications)')}
        attempts = {r[1] for r in c.execute('PRAGMA table_info(delivery_attempts)')}
        if not {'canonical_url', 'body_hash', 'posted_at'} <= pubs or not {
            'attempt_key', 'state', 'body_hash', 'primary_message_ids_json',
            'started_at', 'updated_at', 'last_error'} <= attempts:
            raise ValueError('production_schema_incomplete')
        c.execute('CREATE TABLE IF NOT EXISTS safe03_r6r_fixture_reservations ('
                  'reservation_key TEXT PRIMARY KEY, catalog_id TEXT NOT NULL, '
                  'full_entry_sha256 TEXT NOT NULL, body_hash TEXT NOT NULL, '
                  'source_sha256 TEXT NOT NULL, media_sha256 TEXT NOT NULL, '
                  'state TEXT NOT NULL, message_id INTEGER, created_utc TEXT NOT NULL)')
        c.execute('CREATE INDEX IF NOT EXISTS safe03_r6r_fixture_hash '
                  'ON safe03_r6r_fixture_reservations(body_hash)')

    def reserve_fixture(self, *, entry: object, permit: object, intent: object,
                        window: str, now_utc: datetime) -> TxResult:
        denied = TxResult('NO_POST', 'invalid_source_intent_or_window')
        if (type(entry) is not frozen.Entry or type(permit) is not SourcePermit
            or type(intent) is not q.PhotoIntent or type(window) is not str
            or type(now_utc) is not datetime or now_utc.utcoffset() != timedelta(0)
            or len(window) != 6 or not window[:4].isdigit() or window[4] != 'Q'
            or window[5] not in '1234' or permit.status != 'OFFLINE_FIXTURE_ONLY'
            or intent.status != 'OFFLINE_FIXTURE_ONLY'):
            return denied
        try:
            known = current.validated_offline_entries()[entry.catalog_id]
        except (ValueError, KeyError, OSError):
            return denied
        full_sha = frozen.digest(entry)
        if (entry != known or entry.catalog_id != permit.catalog_id or permit.full_entry_sha256 != full_sha
            or intent.catalog_id != entry.catalog_id or intent.whole_entry_sha256 != full_sha
            or intent.caption.encode('utf-8') != entry.text.encode('utf-8')
            or permit.source_url != intent.source_button_url
            or not hmac.compare_digest(permit.seal, _sign(permit.catalog_id,permit.full_entry_sha256,
                                                          permit.evidence_sha256,permit.source_url,
                                                          permit.independent_checker_receipt_sha256))
            or not hmac.compare_digest(intent.local_fixture_seal,
                   q._seal(intent.catalog_id,intent.whole_entry_sha256,intent.caption,
                           intent.photo_sha256,intent.source_button_url))
            or intent.method != 'sendPhoto' or intent.caption_parse_mode != 'omitted'
            or intent.attempts != 1):
            return denied
        import re
        body_hash = hashlib.sha1(re.sub(r'\s+', ' ', entry.text.strip()).encode('utf-8')).hexdigest()
        key = hashlib.sha256((entry.catalog_id + '\0' + full_sha + '\0' + window).encode('utf-8')).hexdigest()
        conn = self._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            self._schema(conn)
            if conn.execute('SELECT 1 FROM delivery_attempts LIMIT 1').fetchone():
                conn.execute('ROLLBACK')
                return TxResult('NO_POST','existing_delivery_quarantine')
            if conn.execute('SELECT 1 FROM publications WHERE body_hash=? LIMIT 1', (body_hash,)).fetchone():
                conn.execute('ROLLBACK')
                return TxResult('NO_POST','global_dedup_block')
            if conn.execute('SELECT 1 FROM safe03_r6r_fixture_reservations '
                            "WHERE state IN ('pending','in_flight','unknown') LIMIT 1").fetchone():
                conn.execute('ROLLBACK')
                return TxResult('NO_POST','safe03_unresolved_reservation')
            if conn.execute('SELECT 1 FROM safe03_r6r_fixture_reservations WHERE '
                            'reservation_key=? OR body_hash=? LIMIT 1',(key,body_hash)).fetchone():
                conn.execute('ROLLBACK')
                return TxResult('NO_POST','already_reserved_or_attempted')
            conn.execute('INSERT INTO safe03_r6r_fixture_reservations '
                         '(reservation_key,catalog_id,full_entry_sha256,body_hash,source_sha256,media_sha256,'
                         'state,created_utc) VALUES (?,?,?,?,?,?,?,?)',
                         (key,entry.catalog_id,full_sha,body_hash,permit.evidence_sha256,
                          intent.photo_sha256,'pending',now_utc.isoformat()))
            conn.execute("INSERT INTO delivery_attempts (attempt_key,state,body_hash,primary_message_ids_json,"
                         "started_at,updated_at,last_error) VALUES (?,'pending',?,'[]',?,?,'')",
                         (key,body_hash,now_utc.isoformat(),now_utc.isoformat()))
            conn.execute('COMMIT')
            return TxResult('OFFLINE_RESERVED_ONLY','no_transport_authorized',key)
        except (sqlite3.Error, ValueError):
            if conn.in_transaction: conn.execute('ROLLBACK')
            return TxResult('NO_POST','transaction_schema_or_io_error')
        finally:
            conn.close()

    def _transition(self, key: str, before: str, after: str, *, message_id: int = 0) -> bool:
        if type(key) is not str or len(key) != 64 or type(message_id) is not int:
            return False
        c=self._connect()
        try:
            c.execute('BEGIN IMMEDIATE')
            self._schema(c)
            row = c.execute('SELECT state FROM safe03_r6r_fixture_reservations WHERE reservation_key=?',(key,)).fetchone()
            row2 = c.execute('SELECT state FROM delivery_attempts WHERE attempt_key=?',(key,)).fetchone()
            if row is None or row2 is None or row['state'] != before or row2['state'] != before:
                c.execute('ROLLBACK');return False
            if after == 'in_flight':
                c.execute("UPDATE safe03_r6r_fixture_reservations SET state='in_flight' WHERE reservation_key=?",(key,))
                c.execute("UPDATE delivery_attempts SET state='in_flight' WHERE attempt_key=?",(key,))
            elif after == 'unknown':
                c.execute("UPDATE safe03_r6r_fixture_reservations SET state='unknown' WHERE reservation_key=?",(key,))
                c.execute("UPDATE delivery_attempts SET state='ambiguous',last_error='fixture_unknown' WHERE attempt_key=?",(key,))
            elif after == 'mock_confirmed' and message_id > 0:
                # Mock ACK must make publication durable BEFORE clearing quarantine.
                detail = c.execute('SELECT catalog_id,body_hash,source_sha256 FROM safe03_r6r_fixture_reservations WHERE reservation_key=?',(key,)).fetchone()
                c.execute('INSERT INTO publications (canonical_url,body_hash,posted_at,audience,rubric_id,evidence_hash) '
                          'VALUES (?,?,?,?,?,?)',
                          ('https://example.org/safe03-fixture/'+key,detail['body_hash'],
                           datetime.now(timezone.utc).isoformat(),'parents','SAFE03',detail['source_sha256']))
                c.execute("UPDATE safe03_r6r_fixture_reservations SET state='mock_confirmed',message_id=? WHERE reservation_key=?",(message_id,key))
                c.execute('DELETE FROM delivery_attempts WHERE attempt_key=?',(key,))
            else:
                c.execute('ROLLBACK');return False
            c.execute('COMMIT')
            return True
        except (sqlite3.Error,ValueError):
            if c.in_transaction:c.execute('ROLLBACK')
            return False
        finally:c.close()

    def start_mock(self, key: str) -> bool:
        return self._transition(key,'pending','in_flight')

    def unknown_mock(self, key: str) -> bool:
        return self._transition(key,'in_flight','unknown')

    def confirmed_mock(self, key: str, message_id: int) -> bool:
        return self._transition(key,'in_flight','mock_confirmed',message_id=message_id)

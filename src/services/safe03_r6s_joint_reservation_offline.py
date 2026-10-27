"""R6-S: OFF-by-default, no-transport, dual-lane reservation *fixture*.

This is NOT wired to run_publisher, is NOT a source trust root, and never sends.
The public normal facade is a read-only pass-through marker; SAFE03 is NO_POST.
Fixture adapters allow concurrent SQLite tests on disposable PublicationStore DBs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import urlsplit

from src.services import safe03_r6d_offline_acceptance as frozen
from src.services import safe03_r6n_warning_acceptance_offline as catalog
from src.services import safe03_r6q_source_sender_offline as q
from src.services import safe03_r6r_transaction_offline as r

# No environment variable, API argument or config can enable a production route.
JOINT_RESERVATION_PRODUCTION_ENABLED = False
_LOCAL_TEST_KEY = secrets.token_bytes(32)


def public_normal_reservation(*_args: object, **_kwargs: object) -> str:
    """No-op planning marker. Real normal publisher remains untouched."""
    return 'LEGACY_UNCHANGED'


def public_safe03_reservation(*_args: object, **_kwargs: object) -> str:
    return 'NO_POST'


def _signature(*items: str) -> str:
    return hmac.new(_LOCAL_TEST_KEY, '\0'.join(items).encode(), hashlib.sha256).hexdigest()


def _body_hash(body: str) -> str:
    # Matches production normalize_publication_text and run_publisher.norm_space.
    return hashlib.sha1(' '.join(body.split()).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class FixtureProposal:
    route: str
    ref: str
    caption: str
    body_hash: str
    evidence_hash: str
    source_url: str
    catalog_sha256: str
    seal: str


@dataclass(frozen=True)
class FixtureOutcome:
    status: str
    reason: str
    key: str = ''


def _mint(route: str, ref: str, caption: str, evidence_hash: str,
          source_url: str, catalog_sha256: str) -> FixtureProposal:
    bh = _body_hash(caption)
    return FixtureProposal(route, ref, caption, bh, evidence_hash, source_url,
                           catalog_sha256, _signature(route, ref, caption, bh,
                            evidence_hash, source_url, catalog_sha256))


def normal_fixture(*, caption: str, canonical_url: str, evidence_hash: str) -> FixtureProposal | None:
    """SYNTHETIC only. This factory does not claim real source grounding."""
    if (type(caption) is not str or not caption.strip() or len(caption) > 4000 or
        type(canonical_url) is not str or type(evidence_hash) is not str or
        len(evidence_hash) != 64 or any(ch not in '0123456789abcdef' for ch in evidence_hash)):
        return None
    try:
        u = urlsplit(canonical_url)
        bad_url = (u.scheme != 'https' or not u.hostname or u.username or u.password or
                   u.fragment or u.port not in (None, 443))
    except ValueError:
        return None
    if bad_url:
        return None
    return _mint('NORMAL', canonical_url, caption, evidence_hash, canonical_url, '')


def safe03_fixture(*, entry: object, permit: object, intent: object) -> FixtureProposal | None:
    """Requires existing R6-R *synthetic* source + R6-Q image fixture seals."""
    if (type(entry) is not frozen.Entry or type(permit) is not r.SourcePermit or
        type(intent) is not q.PhotoIntent or permit.status != 'OFFLINE_FIXTURE_ONLY' or
        intent.status != 'OFFLINE_FIXTURE_ONLY'):
        return None
    try:
        known = catalog.validated_offline_entries()[entry.catalog_id]
    except (ValueError, KeyError, OSError):
        return None
    sha = frozen.digest(entry)
    if (entry != known or sha != permit.full_entry_sha256 or
        permit.catalog_id != entry.catalog_id or
        permit.source_url != intent.source_button_url or
        intent.catalog_id != entry.catalog_id or intent.whole_entry_sha256 != sha or
        intent.caption.encode('utf-8') != entry.text.encode('utf-8') or
        intent.method != 'sendPhoto' or intent.caption_parse_mode != 'omitted' or
        intent.attempts != 1 or
        not hmac.compare_digest(permit.seal, r._sign(permit.catalog_id,
             permit.full_entry_sha256, permit.evidence_sha256, permit.source_url,
             permit.independent_checker_receipt_sha256)) or
        not hmac.compare_digest(intent.local_fixture_seal, q._seal(
             intent.catalog_id,intent.whole_entry_sha256,intent.caption,
             intent.photo_sha256,intent.source_button_url))):
        return None
    return _mint('SAFE03', entry.catalog_id, entry.text,
                 permit.evidence_sha256, permit.source_url, sha)


def _valid_proposal(p: object) -> bool:
    if type(p) is not FixtureProposal or p.route not in ('NORMAL', 'SAFE03'):
        return False
    attrs=(p.route,p.ref,p.caption,p.body_hash,p.evidence_hash,p.source_url,p.catalog_sha256)
    if not all(type(a) is str for a in attrs):
        return False
    if (p.body_hash != _body_hash(p.caption) or not p.ref or not p.caption or
        len(p.evidence_hash) != 64 or len(p.caption) > 4000 or
        (p.route=='SAFE03' and len(p.catalog_sha256) != 64) or
        (p.route=='NORMAL' and p.catalog_sha256 != '')):
        return False
    if type(p.seal) is not str:
        return False
    return hmac.compare_digest(p.seal, _signature(*attrs))


class DisabledJointFixtureStore:
    """No production path; both lanes share a single disposable SQLite write lock."""
    def __init__(self, db_path: Path):
        if (not isinstance(db_path, Path) or db_path.name != 'test_history.sqlite3' or
            not db_path.is_file() or db_path.is_symlink() or
            any(p.is_symlink() for p in db_path.parents)):
            raise ValueError('disposable_non_symlink_db_required')
        self.db_path = db_path

    def _connection(self):
        c = sqlite3.connect(self.db_path, isolation_level=None, timeout=5)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA busy_timeout=5000')
        return c

    @staticmethod
    def _schema(c: sqlite3.Connection) -> None:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'publications','delivery_attempts'} <= tables:
            raise ValueError('missing_publisher_schema')
        pub = {x[1] for x in c.execute('PRAGMA table_info(publications)')}
        delivery = {x[1] for x in c.execute('PRAGMA table_info(delivery_attempts)')}
        if not {'canonical_url','body_hash','posted_at','evidence_hash'} <= pub or not {
            'attempt_key','state','body_hash','primary_message_ids_json','started_at','updated_at','last_error'} <= delivery:
            raise ValueError('publisher_schema_mismatch')
        c.execute('CREATE TABLE IF NOT EXISTS safe03_r6s_fixture_claims('
                  'key TEXT PRIMARY KEY, route TEXT NOT NULL, ref TEXT NOT NULL, '
                  'body_hash TEXT NOT NULL, evidence_hash TEXT NOT NULL, '
                  'state TEXT NOT NULL, created TEXT NOT NULL, message_id INTEGER)')
        c.execute('CREATE INDEX IF NOT EXISTS safe03_r6s_fixture_bodyhash '
                  'ON safe03_r6s_fixture_claims(body_hash)')

    def reserve_fixture(self, proposal: object, *, window: str, now: datetime) -> FixtureOutcome:
        deny = FixtureOutcome('NO_POST', 'invalid_fixture_or_window')
        if (not _valid_proposal(proposal) or type(window) is not str or
            len(window)!=6 or not window[:4].isdigit() or window[4]!='Q' or
            window[5] not in '1234' or type(now) is not datetime or
            now.utcoffset()!=timedelta(0)):
            return deny
        p = proposal
        key = hashlib.sha256((p.route+'\0'+p.ref+'\0'+p.body_hash+'\0'+window).encode()).hexdigest()
        c = self._connection()
        try:
            c.execute('BEGIN IMMEDIATE')
            self._schema(c)
            # Existing normal publisher fail-closed rule: ANY outstanding attempt.
            if c.execute('SELECT 1 FROM delivery_attempts LIMIT 1').fetchone():
                c.execute('ROLLBACK'); return FixtureOutcome('NO_POST', 'global_delivery_quarantine')
            if c.execute('SELECT 1 FROM publications WHERE body_hash=? OR canonical_url=? LIMIT 1',
                         (p.body_hash,p.ref if p.route=='NORMAL' else '\0SAFE03')).fetchone():
                c.execute('ROLLBACK'); return FixtureOutcome('NO_POST', 'global_duplicate')
            if c.execute('SELECT 1 FROM safe03_r6s_fixture_claims WHERE '
                         'body_hash=? OR key=? LIMIT 1',(p.body_hash,key)).fetchone():
                c.execute('ROLLBACK'); return FixtureOutcome('NO_POST','fixture_already_attempted')
            # Existing R6-R fixture claims are separate: not a release migration.
            if 'safe03_r6r_fixture_reservations' in {
                x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            } and c.execute('SELECT 1 FROM safe03_r6r_fixture_reservations LIMIT 1').fetchone():
                c.execute('ROLLBACK'); return FixtureOutcome('NO_POST','prior_fixture_attempt_present')
            stamp = now.isoformat()
            c.execute('INSERT INTO safe03_r6s_fixture_claims '
                      '(key,route,ref,body_hash,evidence_hash,state,created) VALUES (?,?,?,?,?,?,?)',
                      (key,p.route,p.ref,p.body_hash,p.evidence_hash,'pending',stamp))
            c.execute("INSERT INTO delivery_attempts (attempt_key,state,body_hash,primary_message_ids_json,"
                      "started_at,updated_at,last_error) VALUES (?,'pending',?,'[]',?,?,'')",
                      (key,p.body_hash,stamp,stamp))
            c.execute('COMMIT')
            return FixtureOutcome('OFFLINE_RESERVED_ONLY','no_transport_or_activation',key)
        except (sqlite3.Error, ValueError):
            if c.in_transaction: c.execute('ROLLBACK')
            return FixtureOutcome('NO_POST','transaction_failed')
        finally:
            c.close()

    def _transition(self, key: str, before: str, after: str, *, message_id: int=0) -> bool:
        if (type(key) is not str or len(key)!=64 or any(ch not in '0123456789abcdef' for ch in key)
            or type(message_id) is not int):
            return False
        c = self._connection()
        try:
            c.execute('BEGIN IMMEDIATE'); self._schema(c)
            claim = c.execute('SELECT * FROM safe03_r6s_fixture_claims WHERE key=?',(key,)).fetchone()
            attempt = c.execute('SELECT * FROM delivery_attempts WHERE attempt_key=?',(key,)).fetchone()
            if claim is None or attempt is None or claim['state'] != before or attempt['state'] != before:
                c.execute('ROLLBACK'); return False
            stamp=datetime.now().astimezone().isoformat()
            if after == 'in_flight':
                c.execute("UPDATE safe03_r6s_fixture_claims SET state='in_flight' WHERE key=?",(key,))
                c.execute("UPDATE delivery_attempts SET state='in_flight',updated_at=? WHERE attempt_key=?",(stamp,key))
            elif after == 'unknown':
                c.execute("UPDATE safe03_r6s_fixture_claims SET state='unknown' WHERE key=?",(key,))
                c.execute("UPDATE delivery_attempts SET state='ambiguous',last_error='fixture_response_unknown',updated_at=? "
                          "WHERE attempt_key=?",(stamp,key))
            elif after == 'mock_ack' and message_id > 0:
                url = ('https://example.org/fixture-normal/'+key if claim['route']=='NORMAL'
                       else 'https://example.org/fixture-safe03/'+key)
                c.execute('INSERT INTO publications '
                          '(canonical_url,body_hash,posted_at,evidence_hash,audience,rubric_id) '
                          'VALUES (?,?,?,?,?,?)',
                          (url, claim['body_hash'], stamp, claim['evidence_hash'],
                           'parents', 'SAFE03' if claim['route']=='SAFE03' else 'OFFLINE_NORMAL'))
                c.execute("UPDATE safe03_r6s_fixture_claims SET state='mock_ack',message_id=? WHERE key=?",(message_id,key))
                c.execute('DELETE FROM delivery_attempts WHERE attempt_key=?',(key,))
            else:
                c.execute('ROLLBACK'); return False
            c.execute('COMMIT'); return True
        except (ValueError,sqlite3.Error):
            if c.in_transaction: c.execute('ROLLBACK')
            return False
        finally:
            c.close()

    def start_mock(self, key: str) -> bool:
        return self._transition(key,'pending','in_flight')

    def unknown_mock(self, key: str) -> bool:
        return self._transition(key,'in_flight','unknown')

    def acknowledge_mock(self, key: str, message_id: int) -> bool:
        return self._transition(key,'in_flight','mock_ack',message_id=message_id)

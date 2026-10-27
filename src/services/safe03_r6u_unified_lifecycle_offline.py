"""R6-U: offline-only unified reservation/delivery lifecycle.

This module does not import the Telegram provider or publish anything. Its only
DB target is an existing disposable ``test_history.sqlite3`` file. Both roles
share ONE immediate SQLite transaction for checks and reservation. External
Telegram exactly-once semantics are not claimed; unknown outcomes quarantine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import hmac
import secrets
import sqlite3

from src.services import safe03_r6s_joint_reservation_offline as previous

LIVE_ENABLED = False
_SECRET = secrets.token_bytes(32)


def public_safe03(*_args: object, **_kwargs: object) -> str:
    return 'NO_POST'


def public_normal(*_args: object, **_kwargs: object) -> str:
    return 'LEGACY_UNCHANGED'


@dataclass(frozen=True)
class Result:
    status: str
    reason: str
    key: str = ''


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class OfflineLifecycle:
    """One-use mock sender lifecycle; no transport, no production DB access."""

    def __init__(self, path: Path):
        if (not isinstance(path, Path) or path.name != 'test_history.sqlite3'
                or not path.is_file() or path.is_symlink()
                or any(p.is_symlink() for p in path.parents)):
            raise ValueError('disposable_test_history_required')
        self.path = path

    def _connection(self):
        c = sqlite3.connect(self.path, isolation_level=None, timeout=8)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA busy_timeout=8000')
        return c

    @staticmethod
    def _schema(c):
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'publications', 'delivery_attempts'} <= names:
            raise ValueError('publication_store_schema_required')
        required = {
            'publications': {'canonical_url', 'body_hash'},
            'delivery_attempts': {'attempt_key','body_hash','state','primary_message_ids_json',
                                  'started_at','updated_at','last_error'},
        }
        for name, cols in required.items():
            observed = {r[1] for r in c.execute(f'PRAGMA table_info({name})')}
            if not cols <= observed:
                raise ValueError('publication_store_schema_mismatch')
        c.execute('''CREATE TABLE IF NOT EXISTS safe03_r6u_fixture_claims (
            key TEXT PRIMARY KEY,
            route TEXT NOT NULL,
            ref TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            normalized_sha1 TEXT NOT NULL UNIQUE,
            display_sha256 TEXT NOT NULL,
            evidence_hash TEXT NOT NULL,
            state TEXT NOT NULL,
            receipt_message_id INTEGER NOT NULL DEFAULT 0,
            seal TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS safe03_r6u_display_hash_idx '
                  'ON safe03_r6u_fixture_claims(display_sha256)')

    @staticmethod
    def _seal(key, route, ref, chat, norm, display, evidence):
        data = '\0'.join((key,route,ref,chat,norm,display,evidence)).encode()
        return hmac.new(_SECRET, data, hashlib.sha256).hexdigest()

    @classmethod
    def _valid_claim(cls, record):
        expected = cls._seal(record['key'],record['route'],record['ref'],record['chat_id'],
                            record['normalized_sha1'],record['display_sha256'],record['evidence_hash'])
        return hmac.compare_digest(expected,record['seal'])

    def reserve(self, proposal: object, *, display_plain: str, chat_id: str,
                now: datetime) -> Result:
        deny = Result('NO_POST','invalid_or_untrusted_fixture')
        if (not previous._valid_proposal(proposal) or type(display_plain) is not str
                or not display_plain or len(display_plain) > 4000 or
                type(chat_id) is not str or not chat_id.isdecimal() or
                type(now) is not datetime or now.utcoffset() != timezone.utc.utcoffset(now)):
            return deny
        # SAFE03 caption is immutable; only the NORMAL route may carry an engagement footer.
        if proposal.route == 'SAFE03' and display_plain != proposal.caption:
            return Result('NO_POST','safe03_exact_caption_required')
        # Source permission is a *synthetic* R6-S fixture, never an approval root.
        norm = proposal.body_hash  # SHA-1(normalized original post), legacy DB key.
        shown = _digest(display_plain)  # SHA-256(actual caption), delivery-hook key.
        key = _digest('\0'.join((proposal.route,proposal.ref,chat_id,norm,shown)))
        seal = self._seal(key,proposal.route,proposal.ref,chat_id,norm,shown,proposal.evidence_hash)
        c = self._connection()
        try:
            c.execute('BEGIN IMMEDIATE')
            self._schema(c)
            if c.execute('SELECT 1 FROM delivery_attempts LIMIT 1').fetchone():
                c.execute('ROLLBACK');return Result('NO_POST','global_delivery_quarantine')
            if c.execute('SELECT 1 FROM publications WHERE body_hash=? OR canonical_url=? LIMIT 1',
                         (norm,proposal.ref if proposal.route=='NORMAL' else '\0SAFE03')).fetchone():
                c.execute('ROLLBACK');return Result('NO_POST','published_duplicate')
            if c.execute('SELECT 1 FROM safe03_r6u_fixture_claims LIMIT 1').fetchone():
                c.execute('ROLLBACK');return Result('NO_POST','previous_claim_requires_review')
            # Earlier R6-R/S local reservations are not a supported migration.
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for older in ('safe03_r6s_fixture_claims','safe03_r6r_fixture_reservations'):
                if older in names and c.execute(f'SELECT 1 FROM {older} LIMIT 1').fetchone():
                    c.execute('ROLLBACK');return Result('NO_POST','prior_fixture_quarantine')
            stamp = now.isoformat()
            c.execute('INSERT INTO safe03_r6u_fixture_claims '
                      '(key,route,ref,chat_id,normalized_sha1,display_sha256,evidence_hash,state,seal,created_at) '
                      'VALUES (?,?,?,?,?,?,?,?,?,?)',
                      (key,proposal.route,proposal.ref,chat_id,norm,shown,proposal.evidence_hash,'pending',seal,stamp))
            c.execute('INSERT INTO delivery_attempts '
                      '(attempt_key,state,body_hash,primary_message_ids_json,started_at,updated_at,last_error) '
                      "VALUES (?,'pending',?,'[]',?,?,'')",(key,shown,stamp,stamp))
            c.execute('COMMIT')
            return Result('OFFLINE_RESERVED_ONLY','one_claim_and_one_attempt',key)
        except (sqlite3.Error,ValueError):
            if c.in_transaction:c.execute('ROLLBACK')
            return Result('NO_POST','sqlite_or_schema_failure')
        finally:c.close()

    def _transition(self,key: str,expected: str,new_state: str,*,chat_id: str='',display_plain: str='',
                    message_id: int=0) -> Result:
        if (type(key) is not str or len(key)!=64 or
                not all(x in '0123456789abcdef' for x in key)):
            return Result('NO_POST','invalid_attempt_key')
        c=self._connection()
        try:
            c.execute('BEGIN IMMEDIATE');self._schema(c)
            claim=c.execute('SELECT * FROM safe03_r6u_fixture_claims WHERE key=?',(key,)).fetchone()
            attempt=c.execute('SELECT * FROM delivery_attempts WHERE attempt_key=?',(key,)).fetchone()
            if (claim is None or attempt is None or not self._valid_claim(claim) or
                    claim['state']!=expected or attempt['state']!=expected or
                    attempt['body_hash']!=claim['display_sha256']):
                c.execute('ROLLBACK');return Result('NO_POST','state_or_seal_mismatch')
            stamp=datetime.now(timezone.utc).isoformat()
            if new_state=='in_flight':
                if (type(chat_id) is not str or chat_id!=claim['chat_id'] or
                        type(display_plain) is not str or _digest(display_plain)!=claim['display_sha256']):
                    c.execute('ROLLBACK');return Result('NO_POST','actual_payload_changed')
                # Hook consumes the *existing* attempt; never inserts a second.
                c.execute("UPDATE safe03_r6u_fixture_claims SET state='in_flight' WHERE key=?",(key,))
                c.execute("UPDATE delivery_attempts SET state='in_flight',updated_at=? WHERE attempt_key=?",
                          (stamp,key))
            elif new_state=='ambiguous':
                c.execute("UPDATE safe03_r6u_fixture_claims SET state='ambiguous' WHERE key=?",(key,))
                c.execute("UPDATE delivery_attempts SET state='ambiguous',updated_at=?,last_error='offline_unknown' "
                          'WHERE attempt_key=?',(stamp,key))
            elif new_state=='mock_ack':
                if type(message_id) is not int or message_id<=0:
                    c.execute('ROLLBACK');return Result('NO_POST','invalid_mock_ack')
                # Equivalent to record_publication + deleting attempt in one TX;
                # mock ID is NOT evidence of Telegram delivery.
                fake_url=('https://example.org/safe03-r6u/offline/'+key if claim['route']=='SAFE03'
                          else claim['ref'])
                c.execute('INSERT INTO publications '
                          '(canonical_url,body_hash,evidence_hash,posted_at,audience,rubric_id) '
                          'VALUES (?,?,?,?,?,?)',
                          (fake_url,claim['normalized_sha1'],claim['evidence_hash'],stamp,'parents',
                           'SAFE03_OFFLINE' if claim['route']=='SAFE03' else 'NORMAL_OFFLINE'))
                c.execute("UPDATE safe03_r6u_fixture_claims SET state='mock_ack',receipt_message_id=? WHERE key=?",
                          (message_id,key))
                c.execute('DELETE FROM delivery_attempts WHERE attempt_key=?',(key,))
            else:
                c.execute('ROLLBACK');return Result('NO_POST','invalid_transition')
            c.execute('COMMIT')
            return Result('OFFLINE_TRANSITION_ONLY',new_state,key)
        except (sqlite3.Error,ValueError):
            if c.in_transaction:c.execute('ROLLBACK')
            return Result('NO_POST','sqlite_transition_failed')
        finally:c.close()

    def mock_hook_begin(self,key: str,*,chat_id: str,display_plain: str)->Result:
        return self._transition(key,'pending','in_flight',chat_id=chat_id,display_plain=display_plain)

    def mock_unknown(self,key: str)->Result:
        return self._transition(key,'in_flight','ambiguous')

    def mock_ack(self,key: str,message_id: int)->Result:
        return self._transition(key,'in_flight','mock_ack',message_id=message_id)

"""SAFE03 R6-H offline SQLite atomic reservation + conservative receipt model.

No Telegram calls. No global dedup bypass. This cannot guarantee Telegram
exactly-once delivery: a lost response is quarantined, NEVER retried blindly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import re
import sqlite3
from pathlib import Path

from src.services import safe03_r6d_offline_acceptance as catalog
from src.services.safe03_r6g_optin_offline import Plan, _signature, _key, _utf16_units
import hmac


@dataclass(frozen=True)
class RepeatPermit:
    """Fixture-only editorial authority: NOT self-attested or production trusted."""
    catalog_id: str
    entry_digest: str
    window_id: str
    verified_permit_sha: str


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    key: str = ''


class OfflineLedger:
    """Durable test model. NEVER point at production publication_history.sqlite."""
    def __init__(self, sqlite_path: Path):
        if not isinstance(sqlite_path, Path) or sqlite_path == Path(':memory:'):
            raise ValueError('local_test_file_required')
        self.path = sqlite_path
        with self._connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS safe03_trials ('
                      'reservation_key TEXT PRIMARY KEY, catalog_id TEXT NOT NULL, '
                      'entry_digest TEXT NOT NULL, content_hash TEXT NOT NULL, '
                      'kind TEXT NOT NULL, state TEXT NOT NULL, '
                      'created_utc TEXT NOT NULL, confirmed_utc TEXT, message_id INTEGER)')
            c.execute('CREATE INDEX IF NOT EXISTS safe03_trials_hash_idx ON safe03_trials(content_hash)')

    def _connect(self):
        c = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    def reserve(
        self, *, plan: Plan, send_window: str, external_exact_hash_seen: bool,
        external_history_known: bool, external_unresolved_delivery: bool,
        external_last_confirmed_utc: datetime | None = None,
        repeat_permit: RepeatPermit | None = None,
        verified_permits: frozenset[str] = frozenset(),
        now_utc: datetime | None = None, min_repeat_days: int = 90,
    ) -> Decision:
        if (type(plan) is not Plan or plan.status != 'OFFLINE_SIMULATED_ALLOW'
            or plan.card_type != 'static_text_card' or plan.delivery_mode != 'single_photo_exact_caption'
            or plan.caption_parse_mode != 'none' or plan.lane not in ('exercise','safety_warning')
            or not re.fullmatch('[0-9a-f]{64}',plan.entry_digest)
            or not re.fullmatch('[0-9a-f]{40}',plan.content_hash)):
            return Decision(False, 'invalid_offline_plan')
        entry = catalog.CATALOG.get(plan.catalog_id)
        if (entry is None or not isinstance(plan.signature, str)
            or not hmac.compare_digest(plan.signature, _signature(plan))
            or plan.entry_digest != catalog.digest(entry)
            or plan.full_text.encode('utf-8') != entry.text.encode('utf-8')
            or plan.idempotency_key != _key(entry)
            or plan.content_hash != hashlib.sha1(re.sub(r'\s+', ' ', entry.text.strip()).encode('utf-8')).hexdigest()
            or (plan.lane == 'safety_warning') != (entry.kind == 'warning')
            or not (entry.age_min_months <= plan.age_months
                    and (entry.age_max_months is None or plan.age_months <= entry.age_max_months))
            or _utf16_units(entry.text) > 1000):
            return Decision(False, 'forged_or_mutated_plan')
        if (type(send_window) is not str or not re.fullmatch(r'[0-9]{4}Q[1-4]', send_window)
            or type(external_history_known) is not bool or not external_history_known
            or type(external_unresolved_delivery) is not bool or external_unresolved_delivery
            or type(external_exact_hash_seen) is not bool):
            return Decision(False,'unknown_or_unresolved_global_state')
        if now_utc is None or not isinstance(now_utc, datetime) or now_utc.utcoffset() != timedelta(0):
            return Decision(False, 'trusted_utc_clock_missing')
        if type(min_repeat_days) is not int or min_repeat_days < 90:
            return Decision(False, 'repeat_cooldown_too_short')
        is_repeat = external_exact_hash_seen
        if is_repeat:
            if (not isinstance(external_last_confirmed_utc, datetime)
                or external_last_confirmed_utc.utcoffset() != timedelta(0)
                or external_last_confirmed_utc > now_utc
                or now_utc < external_last_confirmed_utc + timedelta(days=min_repeat_days)):
                return Decision(False,'external_repeat_cooldown_or_timestamp_untrusted')
        elif external_last_confirmed_utc is not None:
            return Decision(False, 'external_history_inconsistent')
        permit_ok = (type(repeat_permit) is RepeatPermit
                     and plan.lane == 'safety_warning'
                     and repeat_permit.catalog_id == plan.catalog_id
                     and repeat_permit.entry_digest == plan.entry_digest
                     and repeat_permit.window_id == send_window
                     and bool(re.fullmatch('[0-9a-f]{64}',repeat_permit.verified_permit_sha))
                     and repeat_permit.verified_permit_sha in verified_permits)
        if is_repeat and not permit_ok:
            return Decision(False,'global_dedup_not_overridden')
        # The repeated warning is a PROPOSED namespaced exemption only. It does
        # NOT change or bypass the publisher's existing global body-hash check.
        key_src = f'{plan.idempotency_key}:{send_window}'
        key = hashlib.sha256(key_src.encode('utf-8')).hexdigest()
        conn = self._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute("SELECT 1 FROM safe03_trials WHERE state IN ('reserved','in_flight','unknown') LIMIT 1").fetchone():
                conn.execute('ROLLBACK')
                return Decision(False,'unresolved_or_reserved_attempt')
            if conn.execute('SELECT 1 FROM safe03_trials WHERE reservation_key=?',(key,)).fetchone():
                conn.execute('ROLLBACK')
                return Decision(False,'window_already_used')
            prev = conn.execute('SELECT confirmed_utc FROM safe03_trials WHERE content_hash=? AND state=\'confirmed\' '
                                'ORDER BY confirmed_utc DESC LIMIT 1', (plan.content_hash,)).fetchone()
            if prev:
                latest = datetime.fromisoformat(prev['confirmed_utc'])
                if now_utc < latest + timedelta(days=min_repeat_days):
                    conn.execute('ROLLBACK')
                    return Decision(False,'warning_repeat_cooldown' if plan.lane == 'safety_warning'
                                    else 'exercise_duplicate')
                if not permit_ok:
                    conn.execute('ROLLBACK')
                    return Decision(False,'prior_content_requires_new_permission')
            elif is_repeat and not permit_ok:
                conn.execute('ROLLBACK')
                return Decision(False,'global_dedup_not_overridden')
            conn.execute('INSERT INTO safe03_trials (reservation_key,catalog_id,entry_digest,content_hash,kind,state,created_utc) '
                         'VALUES (?,?,?,?,?,?,?)', (key,plan.catalog_id,plan.entry_digest,plan.content_hash,
                                                    plan.lane,'reserved',now_utc.isoformat()))
            conn.execute('COMMIT')
            return Decision(True,'reserved_offline_only',key)
        except BaseException:
            if conn.in_transaction: conn.execute('ROLLBACK')
            raise
        finally:
            conn.close()

    def _transition(self, key: str, expected: str, new: str, *, msg_id: int | None = None):
        conn=self._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            cur=conn.execute('UPDATE safe03_trials SET state=?, message_id=?, confirmed_utc=CASE WHEN ?="confirmed" '
                             'THEN strftime("%Y-%m-%dT%H:%M:%S+00:00","now") ELSE confirmed_utc END '
                             'WHERE reservation_key=? AND state=?',
                             (new,msg_id,new,key,expected))
            if cur.rowcount!=1:
                conn.execute('ROLLBACK')
                return False
            conn.execute('COMMIT')
            return True
        finally:
            conn.close()

    def mark_inflight(self,key:str)->bool:
        return self._transition(key,'reserved','in_flight')

    def mark_unknown(self,key:str)->bool:
        return self._transition(key,'in_flight','unknown')

    def mark_confirmed(self,key:str,message_id:int)->bool:
        if type(message_id) is not int or message_id<=0:
            return False
        return self._transition(key,'in_flight','confirmed',msg_id=message_id)

    def get(self,key:str)->dict|None:
        with self._connect() as c:
            row=c.execute('SELECT * FROM safe03_trials WHERE reservation_key=?',(key,)).fetchone()
            return dict(row) if row else None

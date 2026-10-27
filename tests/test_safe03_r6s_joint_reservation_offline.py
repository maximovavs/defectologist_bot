"""Dual-lane R6-S seam: tests exercise disposable real-schema sqlite, never send."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import hashlib
import os
import sqlite3
import sys
import tempfile
import types

from src.services import safe03_r6s_joint_reservation_offline as s
from src.services import safe03_r6r_transaction_offline as r
from tests.test_safe03_r6r_transaction_offline import R6RSourceTests, NOW

EVIDENCE = 'c'*64


class JointReservationTests(TestCase):
    @staticmethod
    def db():
        from src.services import publication_store
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name)/'test_history.sqlite3'
        with patch.dict(os.environ, {'DRY_RUN':'1'}):
            store = publication_store.PublicationStore(path)
        return tmp,path,store

    @staticmethod
    def normal(body='Один и тот же проверочный текст', url='https://example.org/offline'):
        return s.normal_fixture(caption=body,canonical_url=url,evidence_hash=EVIDENCE)

    @staticmethod
    def safe():
        from tests.test_safe03_r6r_transaction_offline import R6RSourceTests
        e,f,snap,review,roots,intent=R6RSourceTests().fixture('warning_small_magnets')
        permit=r.admit_fixture_source(e,f,snap,review,roots,now_utc=NOW)
        return e,s.safe03_fixture(entry=e,permit=permit,intent=intent)

    def test_public_facades_always_disabled_and_zero_side_effects(self):
        d,path,_=self.db()
        try:
            before=path.read_bytes()
            for env in ['1','true','enabled']:
                with patch.dict(os.environ,{'SAFE03_ENABLED':env,'SAFE03_SHARED_RESERVATION_ENABLED':env}):
                    self.assertFalse(s.JOINT_RESERVATION_PRODUCTION_ENABLED)
                    self.assertEqual(s.public_normal_reservation(), 'LEGACY_UNCHANGED')
                    self.assertEqual(s.public_safe03_reservation(), 'NO_POST')
            self.assertEqual(path.read_bytes(),before)
            with sqlite3.connect(path) as c:
                self.assertNotIn('safe03_r6s_fixture_claims', {a[0] for a in c.execute("SELECT name FROM sqlite_master")})
        finally:d.cleanup()

    def test_real_production_publisher_files_untouched_and_not_routed(self):
        root=Path(__file__).resolve().parents[1]
        live=(root/'src/publisher/run_publisher.py').read_text()
        self.assertIn('store.has_body_hash(body_hash)', live)
        self.assertNotIn('safe03_r6s_joint_reservation_offline',live)
        self.assertIn('send_post_with_visual(',live)
        self.assertNotIn('safe03_r6s_joint_reservation_offline', (root/'.github/workflows/post.yml').read_text())

    def test_fixture_normal_and_safe_contract(self):
        n=self.normal();e,t=self.safe()
        self.assertEqual(n.route,'NORMAL')
        self.assertEqual(t.route,'SAFE03')
        self.assertEqual(t.caption.encode(),e.text.encode())
        self.assertEqual(n.body_hash,hashlib.sha1(' '.join(n.caption.split()).encode()).hexdigest())
        self.assertTrue(s._valid_proposal(n));self.assertTrue(s._valid_proposal(t))

    def test_invalid_normal_url_and_semantics(self):
        for bad in ['http://example.org/x','https://x@host.com','https://example.org/#frag',
                    'https://example.org:999/a','https://example.org:invalid','', 'mailto:hi@example.org']:
            with self.subTest(bad=bad):
                self.assertIsNone(s.normal_fixture(caption='Hello',canonical_url=bad,evidence_hash=EVIDENCE))
        self.assertIsNone(s.normal_fixture(caption=' ',canonical_url='https://example.org',evidence_hash=EVIDENCE))
        self.assertIsNone(s.normal_fixture(caption='safe',canonical_url='https://example.org',evidence_hash='bad'))

    def test_refuse_forged_facade_tokens(self):
        n=self.normal()
        d,path,_=self.db()
        try:
            bad=[replace(n,seal='0'*64),replace(n,body_hash='0'*40),replace(n,caption='X'),
                 replace(n,route='SAFE03'),replace(n,source_url='https://evil.example'),
                 replace(n,ref='https://evil.example'),replace(n,seal=123)]
            for p in bad:
                with self.subTest(p=p):
                    self.assertEqual(s.DisabledJointFixtureStore(path).reserve_fixture(p,window='2026Q4',now=NOW).status,'NO_POST')
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],0)
        finally:d.cleanup()

    def test_safe_requires_source_review_fixture(self):
        e,f,snap,review,roots,intent=R6RSourceTests().fixture('warning_small_magnets')
        permit=r.admit_fixture_source(e,f,snap,review,roots,now_utc=NOW)
        self.assertEqual(permit.status,'OFFLINE_FIXTURE_ONLY')
        self.assertIsNone(s.safe03_fixture(entry=e,permit=replace(permit,seal='0'*64),intent=intent))
        self.assertIsNone(s.safe03_fixture(entry=e,permit=permit,intent=replace(intent,caption='tampered')))
        self.assertIsNone(s.safe03_fixture(entry=e,permit=permit,intent=replace(intent,method='sendMessage')))

    def test_fixture_requires_disposable_file_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'publication_history.sqlite3';path.touch()
            with self.assertRaises(ValueError):s.DisabledJointFixtureStore(path)
            link=Path(td)/'test_history.sqlite3';link.symlink_to(path)
            with self.assertRaises(ValueError):s.DisabledJointFixtureStore(link)
            with self.assertRaises(ValueError):s.DisabledJointFixtureStore(Path(td)/'test_history.sqlite3'/ 'no')

    def test_one_normal_reservation_real_delivery_quarantine(self):
        d,path,store=self.db()
        try:
            x=s.DisabledJointFixtureStore(path).reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertEqual(x.status,'OFFLINE_RESERVED_ONLY')
            self.assertTrue(store.has_unresolved_delivery_attempts())
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT state FROM delivery_attempts').fetchone()[0],'pending')
                self.assertEqual(c.execute('SELECT route FROM safe03_r6s_fixture_claims').fetchone()[0],'NORMAL')
        finally:d.cleanup()

    def test_one_safe_reservation_same_shared_tables(self):
        d,path,_=self.db()
        try:
            e,proposal=self.safe()
            x=s.DisabledJointFixtureStore(path).reserve_fixture(proposal,window='2026Q4',now=NOW)
            self.assertEqual(x.status,'OFFLINE_RESERVED_ONLY',x.reason)
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT route FROM safe03_r6s_fixture_claims').fetchone()[0],'SAFE03')
        finally:d.cleanup()

    def test_dual_lane_race_same_text_only_one(self):
        d,path,_=self.db()
        try:
            e,p=self.safe();n=self.normal(body=e.text)
            def one(k):
                return s.DisabledJointFixtureStore(path).reserve_fixture(p if k%2 else n,window='2026Q4',now=NOW)
            with ThreadPoolExecutor(max_workers=16) as pool:
                out=list(pool.map(one,range(24)))
            self.assertEqual(sum(v.status=='OFFLINE_RESERVED_ONLY' for v in out),1)
            self.assertEqual(sum(v.status=='NO_POST' for v in out),23)
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT count(*) FROM safe03_r6s_fixture_claims').fetchone()[0],1)
        finally:d.cleanup()

    def test_dual_lane_race_different_text_only_one_pending(self):
        d,path,_=self.db()
        try:
            _,p=self.safe();n=self.normal()
            def one(k):
                return s.DisabledJointFixtureStore(path).reserve_fixture(p if k%2 else n,window='2026Q4',now=NOW)
            with ThreadPoolExecutor(max_workers=12) as pool:
                out=list(pool.map(one,range(24)))
            self.assertEqual(sum(v.status=='OFFLINE_RESERVED_ONLY' for v in out),1)
        finally:d.cleanup()

    def test_existing_normal_publication_hash_blocks_safe(self):
        d,path,_=self.db()
        try:
            e,p=self.safe()
            with sqlite3.connect(path) as c:
                c.execute('INSERT INTO publications(canonical_url,body_hash,posted_at) VALUES (?,?,?)',
                          ('https://example.org/older',p.body_hash,NOW.isoformat()))
            x=s.DisabledJointFixtureStore(path).reserve_fixture(p,window='2026Q4',now=NOW)
            self.assertEqual(x.reason,'global_duplicate')
        finally:d.cleanup()

    def test_existing_safe_publication_hash_blocks_normal(self):
        d,path,_=self.db()
        try:
            n=self.normal()
            with sqlite3.connect(path) as c:
                c.execute('INSERT INTO publications(canonical_url,body_hash,posted_at) VALUES (?,?,?)',
                          ('https://example.org/older',n.body_hash,NOW.isoformat()))
            x=s.DisabledJointFixtureStore(path).reserve_fixture(n,window='2026Q4',now=NOW)
            self.assertEqual(x.reason,'global_duplicate')
        finally:d.cleanup()

    def test_existing_legacy_delivery_attempt_in_any_state_blocks_both(self):
        for st in ('pending','confirmed','ambiguous','in_flight'):
            d,path,_=self.db()
            try:
                with sqlite3.connect(path) as c:
                    c.execute('INSERT INTO delivery_attempts(attempt_key,state,body_hash) VALUES (?,?,?)',('x',st,'different'))
                e,p=self.safe()
                for prop in (self.normal(),p):
                    x=s.DisabledJointFixtureStore(path).reserve_fixture(prop,window='2026Q4',now=NOW)
                    self.assertEqual(x.reason,'global_delivery_quarantine')
            finally:d.cleanup()

    def test_unknown_outcome_cross_lane_block_and_never_retry(self):
        d,path,store=self.db()
        try:
            g=s.DisabledJointFixtureStore(path)
            x=g.reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertTrue(g.start_mock(x.key))
            self.assertTrue(g.unknown_mock(x.key))
            self.assertFalse(g.start_mock(x.key))
            self.assertFalse(g.acknowledge_mock(x.key,999))
            self.assertEqual(g.reserve_fixture(self.safe()[1],window='2026Q4',now=NOW).reason,'global_delivery_quarantine')
            self.assertTrue(store.has_unresolved_delivery_attempts())
        finally:d.cleanup()

    def test_crash_before_transport_no_automatic_recovery(self):
        d,path,_=self.db()
        try:
            g=s.DisabledJointFixtureStore(path)
            x=g.reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertEqual(s.DisabledJointFixtureStore(path).reserve_fixture(self.safe()[1],window='2026Q4',now=NOW).status,'NO_POST')
            self.assertFalse(g.unknown_mock(x.key))
        finally:d.cleanup()

    def test_ack_persists_normal_publication_then_clears_quarantine(self):
        d,path,store=self.db()
        try:
            g=s.DisabledJointFixtureStore(path)
            x=g.reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertTrue(g.start_mock(x.key))
            self.assertFalse(g.acknowledge_mock(x.key,0))
            self.assertTrue(g.acknowledge_mock(x.key,555))
            self.assertFalse(store.has_unresolved_delivery_attempts())
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT count(*) FROM publications').fetchone()[0],1)
                self.assertEqual(c.execute('SELECT route,state FROM safe03_r6s_fixture_claims').fetchone(),('NORMAL','mock_ack'))
            self.assertEqual(g.reserve_fixture(self.normal(),window='2026Q4',now=NOW).status,'NO_POST')
        finally:d.cleanup()

    def test_ack_safe_then_new_normal_nonduplicate_can_reserve(self):
        d,path,_=self.db()
        try:
            g=s.DisabledJointFixtureStore(path)
            x=g.reserve_fixture(self.safe()[1],window='2026Q4',now=NOW)
            self.assertTrue(g.start_mock(x.key))
            self.assertTrue(g.acknowledge_mock(x.key,42))
            y=g.reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertEqual(y.status,'OFFLINE_RESERVED_ONLY')
        finally:d.cleanup()

    def test_rollback_if_delivery_insert_fails_both_routes(self):
        for prop in (self.normal(),self.safe()[1]):
            d,path,_=self.db()
            try:
                with sqlite3.connect(path) as c:
                    c.execute("CREATE TRIGGER block_fixture BEFORE INSERT ON delivery_attempts BEGIN SELECT RAISE(ABORT,'blocked'); END")
                g=s.DisabledJointFixtureStore(path)
                x=g.reserve_fixture(prop,window='2026Q4',now=NOW)
                self.assertEqual(x.reason,'transaction_failed')
                with sqlite3.connect(path) as c:
                    tabs={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if 'safe03_r6s_fixture_claims' in tabs:
                        self.assertEqual(c.execute('SELECT count(*) FROM safe03_r6s_fixture_claims').fetchone()[0],0)
            finally:d.cleanup()

    def test_wrong_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test_history.sqlite3'
            sqlite3.connect(path).close()
            x=s.DisabledJointFixtureStore(path).reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertEqual(x.reason,'transaction_failed')

    def test_replay_and_window_does_not_override_existing_attempt(self):
        d,path,_=self.db()
        try:
            g=s.DisabledJointFixtureStore(path)
            x=g.reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertEqual(g.reserve_fixture(self.normal(),window='2027Q1',now=NOW).status,'NO_POST')
            self.assertTrue(g.start_mock(x.key));self.assertTrue(g.acknowledge_mock(x.key,10))
            self.assertEqual(g.reserve_fixture(self.normal(),window='2027Q1',now=NOW).reason,'global_duplicate')
        finally:d.cleanup()

    def test_prior_r6r_reservations_block_r6s(self):
        d,path,_=self.db()
        try:
            with sqlite3.connect(path) as c:
                c.execute('CREATE TABLE safe03_r6r_fixture_reservations(reservation_key TEXT PRIMARY KEY)')
                c.execute("INSERT INTO safe03_r6r_fixture_reservations VALUES ('old')")
            result=s.DisabledJointFixtureStore(path).reserve_fixture(self.normal(),window='2026Q4',now=NOW)
            self.assertEqual(result.reason,'prior_fixture_attempt_present')
        finally:d.cleanup()

    def test_no_real_transport_secrets_or_runtime_flag(self):
        src=Path(s.__file__).read_text()
        self.assertNotIn('requests.',src)
        self.assertNotIn('send_post_with_visual(',src)
        self.assertNotIn('tg_request(',src)
        self.assertNotIn('TELEGRAM_BOT_TOKEN',src)
        self.assertNotIn('os.getenv(',src)

"""Offline R6-U: no provider/Telegram transport, only disposable SQLite fixtures."""
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
import tempfile
import unittest

from src.services import safe03_r6u_unified_lifecycle_offline as u
from src.services import safe03_r6s_joint_reservation_offline as s
from src.services.publication_store import PublicationStore
from tests.test_safe03_r6s_joint_reservation_offline import JointReservationTests

NOW=datetime(2026,10,26,tzinfo=timezone.utc)
EVIDENCE='a'*64
CHAT='12345678'
TEXT='Ребёнок слушает и играет'


class R6UUnifiedLifecycleTests(TestCase):
    @staticmethod
    def fixture():
        tmp=tempfile.TemporaryDirectory()
        path=Path(tmp.name)/'test_history.sqlite3'
        with patch.dict(os.environ,{'DRY_RUN':'1'}):
            db=PublicationStore(path)
        return tmp,path,db,u.OfflineLifecycle(path)

    @staticmethod
    def normal(text=TEXT,url='https://example.org/normal'):
        return s.normal_fixture(caption=text,canonical_url=url,evidence_hash=EVIDENCE)

    def test_public_paths_always_off_and_no_network_arming(self):
        from src.publisher import run_publisher as publisher
        from src.services import publication_store as store_module
        self.assertIs(u.LIVE_ENABLED,False)
        self.assertIs(publisher._R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED,False)
        self.assertIs(store_module._R6U_UNIFIED_DELIVERY_HOOK_ENABLED,False)
        self.assertEqual(u.public_safe03(),'NO_POST')
        self.assertEqual(u.public_normal(),'LEGACY_UNCHANGED')
        with patch.dict(os.environ,{'SAFE03_ENABLED':'1','SAFE03_R6U_ENABLED':'1'}):
            self.assertFalse(publisher._R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED)
            self.assertFalse(store_module._R6U_UNIFIED_DELIVERY_HOOK_ENABLED)

    def test_fixture_refuses_production_path_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/'publication_history.sqlite3').touch()
            with self.assertRaises(ValueError):u.OfflineLifecycle(d/'publication_history.sqlite3')
            (d/'test_history.sqlite3').symlink_to(d/'publication_history.sqlite3')
            with self.assertRaises(ValueError):u.OfflineLifecycle(d/'test_history.sqlite3')

    def test_reserve_once_two_different_hashes_and_real_schema(self):
        tmp,path,db,lc=self.fixture()
        try:
            p=self.normal()
            intent=lc.reserve(p,display_plain=TEXT+'\n\n#логопед',chat_id=CHAT,now=NOW)
            self.assertEqual(intent.status,'OFFLINE_RESERVED_ONLY',intent.reason)
            rows=db.delivery_attempts()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['attempt_key'],intent.key)
            self.assertEqual(rows[0]['body_hash'],hashlib.sha256((TEXT+'\n\n#логопед').encode()).hexdigest())
            with sqlite3.connect(path) as c:
                norm=c.execute('SELECT normalized_sha1 FROM safe03_r6u_fixture_claims').fetchone()[0]
                self.assertEqual(norm,p.body_hash)
                self.assertEqual(len(norm),40)
                self.assertEqual(len(rows[0]['body_hash']),64)
            self.assertEqual(lc.reserve(p,display_plain=TEXT,chat_id=CHAT,now=NOW).status,'NO_POST')
        finally:tmp.cleanup()

    def test_hook_reuses_pending_attempt_without_double_reservation(self):
        tmp,path,db,lc=self.fixture()
        try:
            p=self.normal()
            key=lc.reserve(p,display_plain=TEXT,chat_id=CHAT,now=NOW).key
            self.assertEqual(lc.mock_hook_begin(key,chat_id=CHAT,display_plain=TEXT).reason,'in_flight')
            self.assertEqual(lc.mock_hook_begin(key,chat_id=CHAT,display_plain=TEXT).status,'NO_POST')
            self.assertEqual(len(db.delivery_attempts()),1)
            self.assertEqual(db.delivery_attempts()[0]['state'],'in_flight')
            with self.assertRaisesRegex(Exception,'already_quarantined'):
                db.begin_delivery_attempt(key,hashlib.sha256(TEXT.encode()).hexdigest())
        finally:tmp.cleanup()

    def test_hook_rejects_mismatched_payload_or_chat_without_state_change(self):
        tmp,path,db,lc=self.fixture()
        try:
            key=lc.reserve(self.normal(),display_plain=TEXT,chat_id=CHAT,now=NOW).key
            self.assertEqual(lc.mock_hook_begin(key,chat_id='999',display_plain=TEXT).reason,'actual_payload_changed')
            self.assertEqual(lc.mock_hook_begin(key,chat_id=CHAT,display_plain=TEXT+' ').reason,'actual_payload_changed')
            self.assertEqual(db.delivery_attempts()[0]['state'],'pending')
        finally:tmp.cleanup()

    def test_unknown_outcome_persists_and_blocks_all_lanes(self):
        tmp,path,db,lc=self.fixture()
        try:
            key=lc.reserve(self.normal(),display_plain=TEXT,chat_id=CHAT,now=NOW).key
            lc.mock_hook_begin(key,chat_id=CHAT,display_plain=TEXT)
            self.assertEqual(lc.mock_unknown(key).reason,'ambiguous')
            self.assertEqual(db.delivery_attempts()[0]['state'],'ambiguous')
            self.assertEqual(lc.mock_ack(key,123).status,'NO_POST')
            self.assertEqual(lc.reserve(self.normal('другое','https://example.org/2'),
                                         display_plain='другое',chat_id=CHAT,now=NOW).reason,'global_delivery_quarantine')
        finally:tmp.cleanup()

    def test_crash_between_reserve_and_mock_hook_is_quarantined(self):
        tmp,path,db,lc=self.fixture()
        try:
            key=lc.reserve(self.normal(),display_plain=TEXT,chat_id=CHAT,now=NOW).key
            reopened=u.OfflineLifecycle(path)
            self.assertEqual(reopened.reserve(self.normal('новое','https://example.org/new'),
                        display_plain='новое',chat_id=CHAT,now=NOW).reason,'global_delivery_quarantine')
            self.assertEqual(db.delivery_attempts()[0]['state'],'pending')
        finally:tmp.cleanup()

    def test_ack_mock_commits_publication_and_clears_same_tx(self):
        tmp,path,db,lc=self.fixture()
        try:
            p=self.normal()
            key=lc.reserve(p,display_plain=TEXT,chat_id=CHAT,now=NOW).key
            lc.mock_hook_begin(key,chat_id=CHAT,display_plain=TEXT)
            self.assertEqual(lc.mock_ack(key,123).reason,'mock_ack')
            self.assertFalse(db.has_unresolved_delivery_attempts())
            self.assertTrue(db.has_body_hash(p.body_hash))
            self.assertEqual(lc.reserve(self.normal(),display_plain=TEXT,chat_id=CHAT,now=NOW).reason,
                             'published_duplicate')
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT receipt_message_id FROM safe03_r6u_fixture_claims').fetchone()[0],123)
        finally:tmp.cleanup()

    def test_sqlite_rollback_if_publication_insert_fails(self):
        tmp,path,db,lc=self.fixture()
        try:
            key=lc.reserve(self.normal(),display_plain=TEXT,chat_id=CHAT,now=NOW).key
            lc.mock_hook_begin(key,chat_id=CHAT,display_plain=TEXT)
            with sqlite3.connect(path) as c:
                c.execute('CREATE TRIGGER block_ack BEFORE INSERT ON publications BEGIN SELECT RAISE(ABORT,"blocked"); END')
                c.commit()
            self.assertEqual(lc.mock_ack(key,77).reason,'sqlite_transition_failed')
            self.assertEqual(db.delivery_attempts()[0]['state'],'in_flight')
            with sqlite3.connect(path) as c:
                self.assertEqual(c.execute('SELECT count(*) FROM publications').fetchone()[0],0)
        finally:tmp.cleanup()

    def test_sqlite_rollback_if_delivery_insert_fails(self):
        tmp,path,db,lc=self.fixture()
        try:
            with sqlite3.connect(path) as c:
                c.execute('CREATE TRIGGER block_pending BEFORE INSERT ON delivery_attempts BEGIN SELECT RAISE(ABORT,"blocked"); END')
                c.commit()
            self.assertEqual(lc.reserve(self.normal(),display_plain=TEXT,chat_id=CHAT,now=NOW).status,'NO_POST')
            with sqlite3.connect(path) as c:
                if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='safe03_r6u_fixture_claims'").fetchone():
                    self.assertEqual(c.execute('SELECT count(*) FROM safe03_r6u_fixture_claims').fetchone()[0],0)
                self.assertEqual(c.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],0)
        finally:tmp.cleanup()

    def test_safe03_rejects_caption_footer_change_without_db_write(self):
        tmp,path,db,lc=self.fixture()
        try:
            _,safe=JointReservationTests.safe()
            result=lc.reserve(safe,display_plain=safe.caption+'#footer',chat_id=CHAT,now=NOW)
            self.assertEqual(result.reason,'safe03_exact_caption_required')
            self.assertFalse(db.has_unresolved_delivery_attempts())
        finally:tmp.cleanup()

    def test_competing_routes_only_one_reservation(self):
        tmp,path,db,lc=self.fixture()
        try:
            _,safe=JointReservationTests.safe()
            self.assertIsNotNone(safe)
            def run(i):
                prop=self.normal(TEXT,'https://example.org/normal') if i%2==0 else safe
                return lc.reserve(prop,display_plain=prop.caption,chat_id=CHAT,now=NOW)
            with ThreadPoolExecutor(max_workers=12) as pool:
                outcomes=list(pool.map(run,range(30)))
            self.assertEqual(sum(x.status=='OFFLINE_RESERVED_ONLY' for x in outcomes),1)
            self.assertEqual(len(db.delivery_attempts()),1)
            self.assertTrue(all(x.status in ('OFFLINE_RESERVED_ONLY','NO_POST') for x in outcomes))
        finally:tmp.cleanup()

    def test_legacy_normal_path_is_unaltered_when_off(self):
        from src.publisher import run_publisher as pub
        import ast
        content=Path(pub.__file__).read_text()
        guard='                    if _R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED:\n                        raise RuntimeError("r6u_shared_lifecycle_not_wired_stop_before_send")\n'
        self.assertEqual(content.count(guard),1)
        self.assertTrue(content.index(guard)<content.index('post_message_id = send_post_with_visual('))
        tree=ast.parse(content)
        tests=[n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name)
               and n.test.id=='_R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED']
        self.assertEqual(len(tests),1)
        exec(compile(ast.fix_missing_locations(ast.Module(body=tests,type_ignores=[])),'guard','exec'),
             {'_R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED':False})
        with self.assertRaisesRegex(RuntimeError,'stop_before_send'):
            exec(compile(ast.fix_missing_locations(ast.Module(body=tests,type_ignores=[])),'guard','exec'),
                 {'_R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED':True})

    def test_source_approval_fixtures_do_not_become_public(self):
        from src.services import safe03_r6r_transaction_offline as r
        from src.services import safe03_r6s_joint_reservation_offline as s2
        self.assertFalse(s2.JOINT_RESERVATION_PRODUCTION_ENABLED)
        self.assertFalse(u.LIVE_ENABLED)
        self.assertEqual(r.public_source_admit(),'NO_POST')
        self.assertEqual(u.public_safe03(),'NO_POST')


if __name__=='__main__':unittest.main()

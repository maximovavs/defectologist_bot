"""Offline-only checks: no real sends, no provider, no publication changes."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.services import safe03_r6t_publisher_seam_offline as bridge
from src.services import safe03_r6s_joint_reservation_offline as shared
from tests.test_safe03_r6s_joint_reservation_offline import JointReservationTests

NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)
EVIDENCE = 'a' * 64


class R6TSeamTests(unittest.TestCase):
    def test_public_route_must_be_hard_off(self):
        from src.publisher import run_publisher as pub
        self.assertIs(pub._R6T_SHARED_RESERVATION_PRODUCTION_ENABLED, False)
        self.assertIs(bridge.PRODUCTION_ENABLED, False)
        self.assertEqual(bridge.public_safe03(), 'NO_POST')
        for flag in ('1','on','true'):
            with patch.dict(os.environ, {'SAFE03_ENABLED':flag,
                  'SAFE03_SHARED_RESERVATION_ENABLED':flag,
                  'R6T_SHARED_RESERVATION_ENABLED':flag}):
                self.assertFalse(pub._R6T_SHARED_RESERVATION_PRODUCTION_ENABLED)
                self.assertFalse(bridge.PRODUCTION_ENABLED)
                self.assertEqual(bridge.public_safe03(), 'NO_POST')

    def test_live_on_stops_before_network_no_override(self):
        for _ in range(4):
            with self.assertRaisesRegex(bridge.R6TProductionBlock, 'before_send'):
                bridge.stop_live_normal_route()

    def test_publisher_contains_only_guarded_insertion(self):
        import ast
        path = Path(__file__).resolve().parents[1]/'src/publisher/run_publisher.py'
        text = path.read_text()
        guard = '''                    if _R6T_SHARED_RESERVATION_PRODUCTION_ENABLED:\n                        raise RuntimeError("r6t_unwired_delivery_hook_must_stop_before_send")\n'''
        self.assertEqual(text.count(guard),1)
        self.assertLess(text.index(guard),text.index('post_message_id = send_post_with_visual('))
        self.assertEqual(text.count('_R6T_SHARED_RESERVATION_PRODUCTION_ENABLED = False'),1)
        self.assertNotIn('SAFE03_SHARED_RESERVATION_ENABLED = os.getenv',text)
        ast.parse(text)

    def test_off_source_restores_original_publisher_bytes(self):
        root = Path(__file__).resolve().parents[1]
        text = (root/'src/publisher/run_publisher.py').read_text()
        import hashlib
        a = "\n# SAFE03 R6-T: reviewed local integration seam. Hard OFF in production.\n" \
            "# Never enable until the existing delivery hook and reservation share one lifecycle.\n" \
            "_R6T_SHARED_RESERVATION_PRODUCTION_ENABLED = False\n"
        b = "                    # R6-T: OFF leaves the historical path untouched.  Should a\n" \
            "                    # future change flip this constant before the delivery\n" \
            "                    # hook is integrated, fail closed BEFORE any Telegram call.\n" \
            "                    if _R6T_SHARED_RESERVATION_PRODUCTION_ENABLED:\n" \
            "                        raise RuntimeError(\"r6t_unwired_delivery_hook_must_stop_before_send\")\n"
        self.assertEqual(text.count(a), 1)
        self.assertEqual(text.count(b), 1)
        # R6-U adds only two hard-OFF guards; strip these as well before parity.
        u_const = "# R6-U: a future shared lifecycle needs one delivery-hook owner; never allow live ON.\n_R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED = False\n"
        u_guard = '                    if _R6U_UNIFIED_DELIVERY_PRODUCTION_ENABLED:\n                        raise RuntimeError("r6u_shared_lifecycle_not_wired_stop_before_send")\n'
        self.assertEqual(text.count(u_const),1)
        self.assertEqual(text.count(u_guard),1)
        legacy = text.replace(a, '').replace(b, '').replace(u_const, '').replace(u_guard, '')
        self.assertEqual(hashlib.sha256(legacy.encode()).hexdigest(),
                         '57f60192692b6db81f7e3aa72d743ac101d0bed0fc23698f4d5d9bba1318c657')

    def test_ast_on_guard_stops_before_a_send(self):
        import ast
        from src.publisher import run_publisher as pub
        text = (Path(__file__).resolve().parents[1]/'src/publisher/run_publisher.py').read_text()
        tree = ast.parse(text)
        guards = [n for n in ast.walk(tree) if isinstance(n, ast.If) and
                  isinstance(n.test, ast.Name) and n.test.id == '_R6T_SHARED_RESERVATION_PRODUCTION_ENABLED']
        self.assertEqual(len(guards), 1)
        code = compile(ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])),
                       '<r6t-guard-only>', 'exec')
        exec(code, {'_R6T_SHARED_RESERVATION_PRODUCTION_ENABLED':False})
        with self.assertRaisesRegex(RuntimeError, 'must_stop_before_send'):
            exec(code, {'_R6T_SHARED_RESERVATION_PRODUCTION_ENABLED':True})

    @staticmethod
    def local_db():
        import types, sys
        from src.services.publication_store import PublicationStore
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name)/'test_history.sqlite3'
        with patch.dict(os.environ, {'DRY_RUN':'1'}):
            PublicationStore(path)
        return temp,path

    @staticmethod
    def normal(path, plain='Ребёнок слушает и играет', url='https://example.org/n'):
        return bridge.simulate_pre_send_normal(
            db_path=path, plain=plain, canonical_url=url,
            evidence_hash=EVIDENCE, quarter='2026Q4', now=NOW,local_offline=True)

    def test_off_denies_fixture_and_does_not_write(self):
        temp,path=self.local_db()
        try:
            pre=path.read_bytes()
            response=bridge.simulate_pre_send_normal(db_path=path,plain='x',
               canonical_url='https://example.org/n',evidence_hash=EVIDENCE,
               now=NOW,quarter='2026Q4')
            self.assertEqual(response.status,'NO_POST')
            self.assertEqual(path.read_bytes(),pre)
        finally:temp.cleanup()

    def test_normal_simulation_writes_same_delivery_schema(self):
        temp,path=self.local_db()
        try:
            outcome=self.normal(path)
            self.assertEqual(outcome.status,'OFFLINE_RESERVED_ONLY')
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],1)
                self.assertEqual(db.execute('SELECT route FROM safe03_r6s_fixture_claims').fetchone()[0],'NORMAL')
            self.assertEqual(self.normal(path).status,'NO_POST')
        finally:temp.cleanup()

    def test_safe_and_normal_mock_compete_same_db(self):
        temp,path=self.local_db()
        try:
            _,p=JointReservationTests.safe()
            def run(i):
                if i%2:
                    return bridge.simulate_pre_send_safe03(db_path=path,proposal=p,
                                    now=NOW,quarter='2026Q4',local_offline=True)
                return self.normal(path)
            with ThreadPoolExecutor(max_workers=12) as pool:
                outcomes=list(pool.map(run,range(28)))
            self.assertEqual(sum(r.status=='OFFLINE_RESERVED_ONLY' for r in outcomes),1)
            self.assertEqual(sum(r.status=='NO_POST' for r in outcomes),27)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],1)
                self.assertEqual(db.execute('SELECT count(*) FROM safe03_r6s_fixture_claims').fetchone()[0],1)
        finally:temp.cleanup()

    def test_unknown_outcome_never_retries_between_lanes(self):
        temp,path=self.local_db()
        try:
            a=self.normal(path)
            self.assertEqual(a.status,'OFFLINE_RESERVED_ONLY')
            store=shared.DisabledJointFixtureStore(path)
            self.assertTrue(store.start_mock(a.reservation_key))
            self.assertTrue(store.unknown_mock(a.reservation_key))
            _,p=JointReservationTests.safe()
            self.assertEqual(bridge.simulate_pre_send_safe03(db_path=path,proposal=p,
                  quarter='2026Q4',now=NOW,local_offline=True).status,'NO_POST')
            self.assertEqual(self.normal(path).status,'NO_POST')
        finally:temp.cleanup()

    def test_global_dedup_across_lanes(self):
        temp,path=self.local_db()
        try:
            _,p=JointReservationTests.safe()
            with sqlite3.connect(path) as db:
                db.execute('INSERT INTO publications(canonical_url,body_hash,posted_at) VALUES (?,?,?)',
                           ('https://example.org/previous',p.body_hash,NOW.isoformat()))
            res=bridge.simulate_pre_send_safe03(db_path=path,proposal=p,
                       now=NOW,quarter='2026Q4',local_offline=True)
            self.assertEqual(res.status,'NO_POST')
            self.assertEqual(res.reason,'global_duplicate')
        finally:temp.cleanup()

    def test_invalid_fields_fail_closed(self):
        temp,path=self.local_db()
        try:
            for h in ('bad','','f'*63):
                r=bridge.simulate_pre_send_normal(db_path=path,plain='x',
                     canonical_url='https://example.org/n',evidence_hash=h,
                     quarter='2026Q4',now=NOW,local_offline=True)
                self.assertEqual(r.status,'NO_POST')
            self.assertEqual(bridge.simulate_pre_send_safe03(db_path=path,proposal=object(),
                     quarter='2026Q4',now=NOW,local_offline=True).status,'NO_POST')
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],0)
        finally:temp.cleanup()

    def test_refuse_live_db_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            prod=Path(tmp)/'publication_history.sqlite3';prod.touch()
            r=bridge.simulate_pre_send_normal(db_path=prod,plain='x',
                   canonical_url='https://example.org/n',evidence_hash=EVIDENCE,
                   quarter='2026Q4',now=NOW,local_offline=True)
            self.assertEqual(r.status,'NO_POST')


if __name__=='__main__':
    unittest.main()

"""R6-R fixture-only shared-DB source/dedup/delivery transaction probes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from src.services import safe03_r6d_offline_acceptance as frozen
from src.services import safe03_r6h_trust_offline as trust
from src.services import safe03_r6n_warning_acceptance_offline as current
from src.services import safe03_r6q_source_sender_offline as q
from src.services import safe03_r6r_transaction_offline as tx

NOW = datetime(2026,10,26,18,0,tzinfo=timezone.utc)


class R6RSourceTests(unittest.TestCase):
    def fixture(self, catalog_id='warning_small_magnets'):
        entry=current.validated_offline_entries()[catalog_id]
        digest=frozen.digest(entry)
        host='www.rch.org.au' if catalog_id=='warning_small_magnets' else 'www.poison.org'
        url=f'https://{host}/clinicalguide' if 'magnets' in catalog_id else f'https://{host}/battery/guideline'
        required=sorted(q.REQUIRED_CLAIMS[catalog_id])
        # Deliberately synthetic source text: NOT a clinically validated snapshot.
        parts=[f'fixture_{i}: {claim}\n'.encode('utf-8') for i,claim in enumerate(required)]
        snapshot=b''.join(parts)
        excerpts=[]; offset=0
        for claim,part in zip(required,parts):
            excerpts.append(tx.ClaimExcerpt(claim,offset,offset+len(part),hashlib.sha256(part).hexdigest()))
            offset+=len(part)
        ev_hash=hashlib.sha256(snapshot).hexdigest()
        reviewer=trust.ReviewerReceipt(catalog_id,digest,'ped_fixture','pediatrician_child_safety','a'*64)
        source=trust.SourceReceipt(catalog_id,digest,url,ev_hash,host,'reviewed_guidance','2026-10-26','c'*64)
        roots=trust.VerifiedRoots({'ped_fixture':'a'*64},{catalog_id:'c'*64},frozenset({host}))
        claims=q.ClaimMap(catalog_id,digest,ev_hash,q.REQUIRED_CLAIMS[catalog_id],'c'*64)
        finding=q._offline_source_fixture(entry,reviewer=reviewer,source=source,roots=roots,claims=claims)
        self.assertEqual(finding.status,'OFFLINE_FIXTURE_ONLY')
        review=tx.SourceReview(catalog_id,digest,url,ev_hash,'ped_fixture','second_reviewer',
                              'c'*64,'e'*64,'2026-10-26T00:00:00+00:00','2026-12-01T00:00:00+00:00',False,tuple(excerpts))
        synth_roots=tx.SourceFixtureRoots((('ped_fixture','c'*64),),(('second_reviewer','e'*64),),frozenset({host}))
        png=BytesIO();Image.new('RGB',(512,512),'white').save(png,format='PNG')
        photo=png.getvalue()
        intent=q._offline_photo_fixture(entry,finding,photo,hashlib.sha256(photo).hexdigest())
        self.assertEqual(intent.status,'OFFLINE_FIXTURE_ONLY')
        return entry,finding,snapshot,review,synth_roots,intent

    def admit(self, info=None):
        e,f,s,r,roots,i=info or self.fixture()
        return tx.admit_fixture_source(e,f,s,r,roots,now_utc=NOW)

    def test_public_routes_permanently_stop(self):
        self.assertEqual(dict(tx.PRODUCTION_SOURCE_ROOTS),{})
        self.assertEqual(dict(tx.PRODUCTION_APPROVAL_ROOTS),{})
        self.assertEqual(tx.public_source_admit(),'NO_POST')
        self.assertEqual(tx.public_reserve(),'NO_POST')

    def test_two_clinical_warning_source_fixtures(self):
        for cid in ('warning_button_batteries','warning_small_magnets'):
            with self.subTest(catalog_id=cid):
                self.assertEqual(self.admit(self.fixture(cid)).status,'OFFLINE_FIXTURE_ONLY')

    def test_rejects_missing_independent_root_and_same_reviewer(self):
        e,f,s,r,roots,i=self.fixture()
        for bad_r,bad_roots in ((replace(r,independent_checker_id=r.reviewer_id),roots),
                                (r,replace(roots,checker_receipts=())),
                                (r,replace(roots,reviewer_receipts=())),
                                (replace(r,review_receipt_sha256='d'*64),roots),
                                (replace(r,independent_checker_receipt_sha256='c'*64),roots)):
            with self.subTest(bad=bad_r):
                self.assertEqual(tx.admit_fixture_source(e,f,s,bad_r,bad_roots,now_utc=NOW).status,'NO_POST')

    def test_rejects_drift_claim_binders_and_snapshot_change(self):
        e,f,s,r,roots,i=self.fixture()
        failures=[(s+b'tamper',r,roots),
                  (s,replace(r,full_entry_sha256='e'*64),roots),
                  (s,replace(r,excerpts=r.excerpts[:-1]),roots),
                  (s,replace(r,excerpts=(replace(r.excerpts[0],quote_sha256='a'*64),)+r.excerpts[1:]),roots),
                  (s,replace(r,source_url='https://evil.example/clinicalguide'),roots),
                  (s,replace(r,source_url='https://www.rch.org.au:garbage/clinicalguide'),roots),
                  (s,replace(r,revoked=True),roots),
                  (s,replace(r,expires_at='2026-10-25T00:00:00+00:00'),roots),
                  (s,replace(r,reviewed_at='2027-01-01T00:00:00+00:00'),roots)]
        for snap,review,root in failures:
            with self.subTest(review=review):
                self.assertEqual(tx.admit_fixture_source(e,f,snap,review,root,now_utc=NOW).status,'NO_POST')

    def test_source_hash_and_quote_offsets_are_bytes_exact(self):
        e,f,s,r,roots,i=self.fixture()
        self.assertTrue(s.startswith(b'fixture_0'))
        self.assertEqual(hashlib.sha256(s).hexdigest(),r.evidence_sha256)
        self.assertEqual(self.admit((e,f,s,r,roots,i)).status,'OFFLINE_FIXTURE_ONLY')
        wrong=replace(r,excerpts=(replace(r.excerpts[0],start=r.excerpts[0].start+1),)+r.excerpts[1:])
        self.assertEqual(tx.admit_fixture_source(e,f,s,wrong,roots,now_utc=NOW).status,'NO_POST')


class R6RTransactionTests(R6RSourceTests):
    def db(self):
        td=tempfile.TemporaryDirectory(); p=Path(td.name)/'test_history.sqlite3'
        # Actual existing PublicationStore schema, not hand-crafted approximation.
        try:
            from src.services import publication_store
        except ModuleNotFoundError as exc:
            if exc.name != 'sentence_transformers':raise
            stub=types.ModuleType('sentence_transformers')
            stub.SentenceTransformer=lambda *_a,**_kw: (_ for _ in ()).throw(RuntimeError('offline'))
            stub.util=types.SimpleNamespace(pytorch_cos_sim=lambda *_a,**_kw: (_ for _ in ()).throw(RuntimeError('offline')))
            with patch.dict(sys.modules,{'sentence_transformers':stub}):
                from src.services import publication_store
        with patch.dict(os.environ,{'DRY_RUN':'1'}):
            store=publication_store.PublicationStore(p)
        return td,p,store

    def case(self,cid='warning_small_magnets'):
        e,f,s,r,roots,intent=self.fixture(cid)
        permit=tx.admit_fixture_source(e,f,s,r,roots,now_utc=NOW)
        self.assertEqual(permit.status,'OFFLINE_FIXTURE_ONLY')
        return e,permit,intent

    def test_one_reservation_writes_shared_tables_atomically(self):
        d,p,store=self.db()
        try:
            e,permit,intent=self.case(); unit=tx.OfflineSharedTransaction(p)
            res=unit.reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            self.assertEqual(res.status,'OFFLINE_RESERVED_ONLY',res.reason)
            with sqlite3.connect(p) as c:
                self.assertEqual(c.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],1)
                self.assertEqual(c.execute('SELECT count(*) FROM safe03_r6r_fixture_reservations').fetchone()[0],1)
                self.assertEqual(c.execute('SELECT count(*) FROM publications').fetchone()[0],0)
            self.assertTrue(store.has_unresolved_delivery_attempts())
        finally:d.cleanup()

    def test_parallel_same_and_different_entry_only_one_reserved(self):
        d,p,_=self.db()
        try:
            infos=[self.case('warning_small_magnets'), self.case('warning_button_batteries')]
            def race(i):
                e,permit,intent=infos[i%2]
                return tx.OfflineSharedTransaction(p).reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            with ThreadPoolExecutor(max_workers=12) as pool:
                results=list(pool.map(race,range(12)))
            self.assertEqual(sum(x.status=='OFFLINE_RESERVED_ONLY' for x in results),1)
            self.assertEqual(sum(x.status=='NO_POST' for x in results),11)
        finally:d.cleanup()

    def test_published_global_hash_blocks(self):
        d,p,_=self.db()
        try:
            e,permit,intent=self.case()
            bh=hashlib.sha1(' '.join(e.text.split()).encode()).hexdigest()
            with sqlite3.connect(p) as c:
                c.execute('INSERT INTO publications (canonical_url,body_hash,posted_at) VALUES (?,?,?)',
                          ('https://example.org/preexisting',bh,NOW.isoformat()))
            r=tx.OfflineSharedTransaction(p).reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            self.assertEqual(r.reason,'global_dedup_block')
            with sqlite3.connect(p) as c:self.assertEqual(c.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],0)
        finally:d.cleanup()

    def test_global_quarantine_blocks_regardless_state(self):
        for state in ('pending','confirmed','ambiguous'):
            with self.subTest(state=state):
                d,p,_=self.db()
                try:
                    e,permit,intent=self.case()
                    with sqlite3.connect(p) as c:
                        c.execute("INSERT INTO delivery_attempts(attempt_key,state,body_hash) VALUES (?,?,?)",('existing',state,'x'))
                    out=tx.OfflineSharedTransaction(p).reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
                    self.assertEqual(out.reason,'existing_delivery_quarantine')
                finally:d.cleanup()

    def test_unknown_quarantine_blocks_all_future_attempts(self):
        d,p,_=self.db()
        try:
            e,permit,intent=self.case()
            unit=tx.OfflineSharedTransaction(p)
            r=unit.reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            self.assertTrue(unit.start_mock(r.reservation_key))
            self.assertTrue(unit.unknown_mock(r.reservation_key))
            self.assertFalse(unit.start_mock(r.reservation_key))
            e2,p2,i2=self.case('warning_button_batteries')
            retry=unit.reserve_fixture(entry=e2,permit=p2,intent=i2,window='2026Q4',now_utc=NOW)
            self.assertEqual(retry.status,'NO_POST')
            with sqlite3.connect(p) as c:
                self.assertEqual(c.execute('SELECT state FROM delivery_attempts').fetchone()[0],'ambiguous')
        finally:d.cleanup()

    def test_mock_confirmation_persists_before_clearing_delivery_row(self):
        d,p,store=self.db()
        try:
            e,permit,intent=self.case()
            unit=tx.OfflineSharedTransaction(p)
            r=unit.reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            self.assertTrue(unit.start_mock(r.reservation_key))
            self.assertFalse(unit.confirmed_mock(r.reservation_key,-1))
            self.assertTrue(unit.confirmed_mock(r.reservation_key,121))
            with sqlite3.connect(p) as c:
                self.assertEqual(c.execute('SELECT count(*) FROM publications').fetchone()[0],1)
                self.assertEqual(c.execute('SELECT count(*) FROM delivery_attempts').fetchone()[0],0)
                self.assertEqual(c.execute('SELECT message_id FROM safe03_r6r_fixture_reservations').fetchone()[0],121)
            self.assertFalse(store.has_unresolved_delivery_attempts())
            retry=unit.reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            self.assertEqual(retry.status,'NO_POST')
        finally:d.cleanup()

    def test_transaction_rollback_leaves_no_orphan_reservation(self):
        d,p,_=self.db()
        try:
            e,permit,intent=self.case()
            with sqlite3.connect(p) as c:
                c.execute('CREATE TRIGGER deny_delivery BEFORE INSERT ON delivery_attempts '
                          "BEGIN SELECT RAISE(ABORT,'fixture_failure'); END;")
            out=tx.OfflineSharedTransaction(p).reserve_fixture(entry=e,permit=permit,intent=intent,window='2026Q4',now_utc=NOW)
            self.assertEqual(out.reason,'transaction_schema_or_io_error')
            with sqlite3.connect(p) as c:
                tables={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'safe03_r6r_fixture_reservations' in tables:
                    self.assertEqual(c.execute('SELECT count(*) FROM safe03_r6r_fixture_reservations').fetchone()[0],0)
        finally:d.cleanup()

    def test_rejects_forged_intent_and_permit(self):
        d,p,_=self.db()
        try:
            e,permit,intent=self.case()
            for bad_permit,bad_intent in ((replace(permit,seal='f'*64),intent),
                                          (permit,replace(intent,caption=intent.caption+' ')),
                                          (permit,replace(intent,source_button_url='https://evil.example')),
                                          (permit,replace(intent,attempts=2)),
                                          (permit,replace(intent,caption_parse_mode='Markdown'))):
                with self.subTest(bad_intent=bad_intent):
                    self.assertEqual(tx.OfflineSharedTransaction(p).reserve_fixture(
                        entry=e,permit=bad_permit,intent=bad_intent,window='2026Q4',now_utc=NOW).status,'NO_POST')
        finally:d.cleanup()

    def test_production_store_filename_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'publication_history.sqlite3';p.touch()
            with self.assertRaises(ValueError):tx.OfflineSharedTransaction(p)
            link=Path(td)/'test_history.sqlite3';link.symlink_to(p)
            with self.assertRaises(ValueError):tx.OfflineSharedTransaction(link)

    def test_actual_publisher_not_yet_participating_in_shared_reservation(self):
        source=Path(__file__).resolve().parents[1]/'src/publisher/run_publisher.py'
        code=source.read_text(encoding='utf-8')
        self.assertIn('store.has_body_hash(body_hash)',code)
        self.assertIn('send_post_with_visual(',code)
        self.assertNotIn('safe03_r6r_fixture_reservations',code)

    def test_no_transport_or_prod_mutation_dependencies(self):
        src=Path(tx.__file__).read_text(encoding='utf-8')
        for banned in ('send_post_with_visual(', 'tg_request(', 'requests.post(', 'import telegram'):
            self.assertNotIn(banned,src)


if __name__=='__main__':unittest.main()

"""Fail-closed trust and exactly-one-send architecture, WITHOUT a network sender."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from PIL import Image
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.services import safe03_r6d_offline_acceptance as old
from src.services import safe03_r6h_trust_offline as trust
from src.services import safe03_r6n_warning_acceptance_offline as catalog
from src.services import safe03_r6q_source_sender_offline as q


class R6QContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = catalog.validated_offline_entries()

    def card(self):
        buff=BytesIO()
        Image.new('RGB',(1080,1080),'white').save(buff,format='PNG')
        return buff.getvalue()

    def source_pair(self, catalog_id='warning_small_magnets'):
        entry = self.entries[catalog_id]
        digest = old.digest(entry)
        host = 'www.rch.org.au' if catalog_id == 'warning_small_magnets' else 'www.poison.org'
        reviewer = trust.ReviewerReceipt(catalog_id,digest,'fixture_ped','pediatrician_child_safety','a'*64)
        source = trust.SourceReceipt(catalog_id,digest,'https://'+host+'/clinicalguide',
                      'b'*64,host,'reviewed_guidance','2026-10-26','c'*64)
        roots = trust.VerifiedRoots({'fixture_ped':'a'*64},{catalog_id:'c'*64},frozenset({host}))
        claims = q.ClaimMap(catalog_id,digest,'b'*64,q.REQUIRED_CLAIMS[catalog_id],'c'*64)
        return entry,reviewer,source,roots,claims

    def approved_fixture(self,catalog_id='warning_small_magnets'):
        entry,r,s,roots,claims = self.source_pair(catalog_id)
        result=q._offline_source_fixture(entry,reviewer=r,source=s,roots=roots,claims=claims)
        self.assertEqual(result.status,'OFFLINE_FIXTURE_ONLY')
        return entry,result

    def test_strict_roots_empty_and_public_interfaces_never_authorize(self):
        self.assertEqual(dict(q.PRODUCTION_TRUST_ROOTS),{})
        self.assertEqual(dict(q.PRODUCTION_SEND_CAPABILITIES),{})
        self.assertEqual(q.public_source().status,'NO_POST')
        self.assertEqual(q.public_sender().status,'NO_POST')
        self.assertEqual(catalog.public_candidate().accepted_for,'NO_POST')

    def test_full_entry_sha_and_claims_required(self):
        entry,r,s,roots,claims=self.source_pair()
        for altered in (q.ClaimMap(claims.catalog_id,'0'*64,claims.evidence_sha256,claims.claim_ids,claims.clinical_review_receipt_sha256),
                        q.ClaimMap(claims.catalog_id,claims.whole_entry_sha256,claims.evidence_sha256,frozenset(),claims.clinical_review_receipt_sha256),
                        q.ClaimMap(claims.catalog_id,claims.whole_entry_sha256,'1'*64,claims.claim_ids,claims.clinical_review_receipt_sha256)):
            with self.subTest(altered=altered):
                self.assertEqual(q._offline_source_fixture(entry,reviewer=r,source=s,roots=roots,claims=altered).status,'NO_POST')

    def test_wrong_identity_or_source_host_rejected(self):
        entry,r,s,roots,claims=self.source_pair()
        self.assertEqual(q._offline_source_fixture(entry,reviewer=r,source=s,roots=trust.VerifiedRoots({}, {}, frozenset()),claims=claims).status,'NO_POST')
        evil=trust.SourceReceipt(s.catalog_id,s.whole_entry_digest,'https://evil.example/clinicalguide',s.evidence_digest,s.vetted_host,s.source_category,s.reviewed_version,s.source_review_sha256)
        self.assertEqual(q._offline_source_fixture(entry,reviewer=r,source=evil,roots=roots,claims=claims).status,'NO_POST')

    def test_warning_synthetic_source_binding_both(self):
        for key in ('warning_small_magnets','warning_button_batteries'):
            with self.subTest(key=key): self.approved_fixture(key)

    def test_single_photo_exact_caption_no_parse_mode(self):
        entry,reviewed=self.approved_fixture()
        png=self.card()
        intent=q._offline_photo_fixture(entry,reviewed,png,hashlib.sha256(png).hexdigest())
        self.assertEqual(intent.status,'OFFLINE_FIXTURE_ONLY')
        self.assertEqual(intent.caption.encode(),entry.text.encode())
        self.assertEqual(intent.method,'sendPhoto')
        self.assertEqual(intent.caption_parse_mode,'omitted')
        self.assertEqual(intent.attempts,1)
        self.assertEqual(intent.source_button_url,reviewed.source_url)
        self.assertIn('unverified',intent.reason)

    def test_tampered_caption_image_or_source_not_published(self):
        entry,reviewed=self.approved_fixture()
        png=self.card()
        altered=q.SourceFinding('OFFLINE_FIXTURE_ONLY','fake',reviewed.catalog_id,'f'*64,reviewed.source_url,reviewed.source_receipt_sha256)
        self.assertEqual(q._offline_photo_fixture(entry,altered,png,hashlib.sha256(png).hexdigest()).status,'NO_POST')
        self.assertEqual(q._offline_photo_fixture(entry,reviewed,png,'e'*64).status,'NO_POST')
        self.assertEqual(q._offline_photo_fixture(entry,reviewed,b'NOTPNG',hashlib.sha256(b'NOTPNG').hexdigest()).status,'NO_POST')
        forged=q.SourceFinding('OFFLINE_FIXTURE_ONLY','fake',reviewed.catalog_id,reviewed.whole_entry_sha256,'https://evil.example/source',reviewed.source_receipt_sha256)
        self.assertEqual(q._offline_photo_fixture(entry,forged,png,hashlib.sha256(png).hexdigest()).status,'NO_POST')

    def fixture_db(self,root):
        path=Path(root)/'test_history.sqlite3'
        con=sqlite3.connect(path)
        con.execute('CREATE TABLE publications (body_hash TEXT NOT NULL, posted_at TEXT NOT NULL)')
        con.execute('CREATE TABLE delivery_attempts (attempt_key TEXT PRIMARY KEY, state TEXT NOT NULL)')
        con.commit();con.close()
        return path

    def test_snapshot_empty_and_missing_db_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture_db(d)
            s=q.inspect_fixture_publication_store(p,self.entries['warning_small_magnets'].text)
            self.assertTrue(s.known);self.assertFalse(s.existing_body_hash);self.assertFalse(s.unresolved_delivery)
            self.assertFalse(q.inspect_fixture_publication_store(p.with_name('missing.db'),'caption').known)
            self.assertFalse(q.inspect_fixture_publication_store(p.with_name('publication_history.sqlite3'),'caption').known)

    def test_snapshot_global_dedup_exact_and_unresolved_attempt(self):
        entry=self.entries['warning_small_magnets']
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture_db(d)
            bh=hashlib.sha1(' '.join(entry.text.split()).encode()).hexdigest()
            with sqlite3.connect(p) as con:
                con.execute('INSERT INTO publications VALUES (?,?)',(bh,'2026-10-26T18:00:00+00:00'))
            s=q.inspect_fixture_publication_store(p,entry.text)
            self.assertTrue(s.existing_body_hash)
            self.assertEqual(s.last_posted_utc,datetime(2026,10,26,18,tzinfo=timezone.utc))
            with sqlite3.connect(p) as con:
                con.execute('INSERT INTO delivery_attempts VALUES (?,?)',('attempt','ambiguous'))
            self.assertTrue(q.inspect_fixture_publication_store(p,entry.text).unresolved_delivery)

    def test_sender_simulation_rejects_dup_timeout_unknown(self):
        entry,reviewed=self.approved_fixture()
        png=self.card()
        intent=q._offline_photo_fixture(entry,reviewed,png,hashlib.sha256(png).hexdigest())
        clear=q.StoreInspection(True,'fixture',False,False)
        self.assertEqual(q.simulate_one_attempt(intent,clear,outcome='timeout'),'UNKNOWN_QUARANTINE_NO_AUTORETRY')
        self.assertEqual(q.simulate_one_attempt(intent,clear,outcome='mock_rejected'),'DETERMINISTIC_REJECT_NO_FALLBACK')
        self.assertEqual(q.simulate_one_attempt(intent,clear,outcome='mock_accepted'),'SIMULATED_ONLY_NOT_DELIVERY_PROOF')
        self.assertEqual(q.simulate_one_attempt(intent,q.StoreInspection(True,'fixture',True,False),outcome='mock_accepted'),'NO_POST')
        self.assertEqual(q.simulate_one_attempt(intent,q.StoreInspection(True,'fixture',False,True),outcome='mock_accepted'),'NO_POST')
        self.assertEqual(q.simulate_one_attempt(intent,q.StoreInspection(False,'broken'),outcome='mock_accepted'),'NO_POST')

    def test_no_telegram_or_publisher_dependency(self):
        src=Path(q.__file__).read_text()
        self.assertNotIn('tg_request(',src)
        self.assertNotIn('requests.post',src)
        self.assertNotIn('send_post_with_visual(',src)
        self.assertNotIn('send_message(',src)
        self.assertNotIn('import telegram',src)


if __name__ == '__main__':
    unittest.main()

class R6QActualPublicationStore(unittest.TestCase):
    def test_actual_store_schema_and_read_only_snapshot(self):
        from unittest.mock import patch
        import sys, types
        try:
            from src.services import publication_store as mod
        except ModuleNotFoundError as error:
            if error.name != 'sentence_transformers':
                raise
            stub=types.ModuleType('sentence_transformers')
            class UnavailableModel:
                def __init__(self,*a,**kw): raise RuntimeError('offline disabled')
            stub.SentenceTransformer=UnavailableModel
            stub.util=types.SimpleNamespace(pytorch_cos_sim=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('disabled')))
            with patch.dict(sys.modules,{'sentence_transformers':stub}):
                from src.services import publication_store as mod
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'test_history.sqlite3'
            with patch.dict(os.environ, {'DRY_RUN':'1'}):
                store=mod.PublicationStore(p)
            try:
                original_data=p.read_bytes()
                entry=catalog.validated_offline_entries()['warning_small_magnets']
                view=q.inspect_fixture_publication_store(p,entry.text)
                self.assertTrue(view.known, view.reason)
                self.assertFalse(view.existing_body_hash)
                self.assertFalse(view.unresolved_delivery)
                self.assertEqual(original_data,p.read_bytes())
                body_hash=hashlib.sha1(' '.join(entry.text.split()).encode()).hexdigest()
                self.assertFalse(store.has_body_hash(body_hash))
                store.begin_delivery_attempt('fixture-123', body_hash)
                view=q.inspect_fixture_publication_store(p,entry.text)
                self.assertTrue(view.unresolved_delivery)
                self.assertTrue(store.has_unresolved_delivery_attempts())
                store.mark_delivery_ambiguous('fixture-123','simulated_lost_reply')
                self.assertTrue(q.inspect_fixture_publication_store(p,entry.text).unresolved_delivery)
                self.assertEqual(store.delivery_attempts()[0]['state'],'ambiguous')
            finally:
                store.deactivate_publisher_delivery_hooks()

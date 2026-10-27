"""Local acceptance tests. No providers, Telegram, or network."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import re
import socket
from tempfile import TemporaryDirectory
from pathlib import Path
from types import MappingProxyType
from unittest import TestCase, mock

from src.services import safe03_r6n_warning_acceptance_offline as r6n
from src.services import safe03_r6d_offline_acceptance as original
from src.services import safe03_r6g_optin_offline as route
from src.services import safe03_r6h_trust_offline as trust
from src.services.safe03_r6h_delivery_offline import OfflineLedger
from src.services.safe03_r6g_text_card_offline import render_static_card_offline

MAIN_SHA='0e96a0ae494f1b8c80d098f65d7c8c824712c8b0'

class OfflineR6N(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_socket = socket.socket
        socket.socket = mock.Mock(side_effect=AssertionError('network forbidden'))

    @classmethod
    def tearDownClass(cls):
        socket.socket=cls.old_socket

    def setUp(self):
        self.spec,self.legacy = r6n._load()
        self.entries = r6n.validated_offline_entries()
        self.specific = ('warning_button_batteries','warning_small_magnets')

    def test_exact_scope_and_versions(self):
        self.assertEqual(len(self.entries),5)
        self.assertEqual({k:self.entries[k].version for k in self.specific},
                         {'warning_button_batteries':4,'warning_small_magnets':3})

    def test_5_digest_and_immutable_content(self):
        for cid,e in self.entries.items():
            with self.subTest(cid=cid):
                if cid in r6n.NEW_WARNING_DIGESTS:
                    self.assertEqual((hashlib.sha256(e.text.encode()).hexdigest(),original.digest(e)),
                                     r6n.NEW_WARNING_DIGESTS[cid])
                else:
                    self.assertEqual(e,original.CATALOG[cid])

    def test_frozen_catalog_unmodified(self):
        self.assertEqual(hashlib.sha256((Path(r6n.__file__).resolve().parents[2]/'SAFE03_R6M_FROZEN_SOURCE_UNMODIFIED.json').read_bytes()).hexdigest(),
                         r6n.FROZEN_CATALOG_SHA256)
        for cid in self.specific:
            self.assertNotEqual(self.entries[cid].text,original.CATALOG[cid].text)
            self.assertEqual(original.CATALOG[cid].version,self.entries[cid].version-1)

    def test_owner_report_is_not_independent_authority(self):
        self.assertEqual(self.spec['trusted_approvals'],{})
        self.assertEqual(self.spec['trusted_sources'],{})
        self.assertEqual(dict(r6n.PRODUCTION_APPROVALS),{})
        self.assertEqual(dict(trust.PRODUCTION_REVIEWER_KEYS),{})
        self.assertEqual(dict(trust.PRODUCTION_SOURCE_RECORDS),{})
        self.assertEqual(r6n.public_candidate('anything').accepted_for,'NO_POST')
        for cid in self.specific:
            e = r6n.offline_editorial_statement(cid)
            self.assertEqual(e.accepted_for,'OFFLINE_EDITORIAL_ONLY')
            self.assertEqual(e.reported_approval,'USER_ATTESTED_CLINICAL_OK')

    def test_legacy_normal_mode_is_unchanged(self):
        normal={'post':'обычное безопасное сообщение','params':{'day':'MON'}}
        result = route.route(normal)
        self.assertEqual(result.route,'LEGACY_UNTOUCHED')
        self.assertIs(result.payload,normal)
        self.assertEqual(route.route({'safe03_mode':'catalog','proposal':'x'}).route,'NO_POST')
        self.assertEqual(trust.public_candidate(catalog_id='warning_button_batteries').status,'NO_POST')

    def test_age_hearing_and_warning_guard(self):
        self.assertEqual((self.entries['picture_naming_24_48'].age_min_months,self.entries['picture_naming_24_48'].age_max_months),(24,48))
        self.assertEqual((self.entries['soft_ball_roll_36_60'].age_min_months,self.entries['soft_ball_roll_36_60'].age_max_months),(36,60))
        self.assertEqual((self.entries['clap_rhythm_24_48'].age_min_months,self.entries['clap_rhythm_24_48'].age_max_months),(24,48))
        for cid in self.specific:
            e=self.entries[cid]
            self.assertEqual(e.kind,'warning')
            self.assertEqual((e.age_min_months,e.age_max_months),(0,None))
            self.assertEqual(e.actor,'adult_caregiver')
            self.assertIn('no_handling_exercise',e.constraints)
        self.assertIn('не является проверкой слуха',self.entries['clap_rhythm_24_48'].text)

    def test_clinical_constraints_battery(self):
        t=self.entries['warning_button_batteries'].text
        for term in ['экстренной медицинской помощью','Не вызывайте рвоту','менее 12 часов','12 месяцев','10 мл','каждые 10 минут','максимум 6 доз','точно не из слухового аппарата','не заменяет экстренную медицинскую помощь','При сомнениях']:
            with self.subTest(term=term):self.assertIn(term,t)
        self.assertNotIn('варенье',t)

    def test_clinical_constraints_magnets(self):
        t=self.entries['warning_small_magnets'].text
        for term in ['необходимо рентгенологическое обследование','срочно обращайтесь','Не ждите появления симптомов','нескольких магнитов']:
            with self.subTest(term=term):self.assertIn(term,t)

    def test_caption_utf16_and_no_footer(self):
        for cid in self.specific:
            text=self.entries[cid].text
            units=len(text.encode('utf-16-le'))//2
            self.assertLessEqual(units,1000)
            self.assertEqual(units,955 if cid=='warning_button_batteries' else 501)
            self.assertNotIn('https://',text)

    def test_approved_receipt_binding(self):
        expected=set(self.specific)
        self.assertEqual({r['catalog_id'] for r in self.spec['editorial_receipts']},expected)
        for r in self.spec['editorial_receipts']:
            cid=r['catalog_id']
            self.assertEqual(r['report_date'],'2026-10-26')
            self.assertEqual(r['independent_identity_verified'],False)
            self.assertEqual(r['independent_signature_verified'],False)
            self.assertEqual(r['entry_sha256'],original.digest(self.entries[cid]))
            self.assertEqual(r['text_sha256'],hashlib.sha256(self.entries[cid].text.encode()).hexdigest())

    def test_tampered_whole_entry_and_attestation_fail_closed(self):
        for cid in self.specific:
            for field,bad in [('text','поддельное'),('version',99),('action','give_child_battery'),('age_max_months',120)]:
                altered=copy.deepcopy(self.spec)
                for item in altered['entries']:
                    if item['catalog_id']==cid: item[field]=bad
                self._expect_bad(altered)
            for field,bad in [('evidence_grade','INDEPENDENT_SIGNED'),('report_date','2026-10-24'),('entry_sha256','0'*64),('independent_identity_verified',True)]:
                altered=copy.deepcopy(self.spec)
                for item in altered['editorial_receipts']:
                    if item['catalog_id']==cid:item[field]=bad
                self._expect_bad(altered)

    def _expect_bad(self,bad):
        with mock.patch.object(r6n,'_load',return_value=(bad,self.legacy)):
            self.assertEqual(r6n.public_candidate('warning_button_batteries').accepted_for,'NO_POST')
            with self.assertRaises(ValueError):r6n.validated_offline_entries()
            self.assertEqual(r6n.offline_editorial_statement('warning_button_batteries').accepted_for,'NO_POST')

    def test_tampered_5_game_scope_fail_closed(self):
        test=copy.deepcopy(self.spec)
        test['entries'][0]['text']='нет данных'
        self._expect_bad(test)

    def test_offline_synthetic_plan_static_visual_and_sender_contract(self):
        from src.services.safe03_r6h_trust_offline import ReviewerReceipt,SourceReceipt,VerifiedRoots
        from src.services.safe03_r6g_optin_offline import StoreSnapshot
        source_urls={ 'warning_button_batteries':('https://www.poison.org/battery/guideline','www.poison.org'),
                      'warning_small_magnets':('https://www.rch.org.au/clinicalguide/guideline_index/Foreign_body_ingestion/','www.rch.org.au')}
        # Here roots are FAKE and injected in tests; no real user/reviewer/source is trusted.
        with mock.patch.object(original,'CATALOG',self.entries):
            for cid in self.specific:
                with self.subTest(cid=cid):
                    e=self.entries[cid];url,host=source_urls[cid]
                    rec=ReviewerReceipt(cid,original.digest(e),'fixture-key','pediatrician_child_safety','a'*64)
                    src=SourceReceipt(cid,original.digest(e),url,'b'*64,host,'reviewed_guidance','offline','c'*64)
                    roots=VerifiedRoots({'fixture-key':'a'*64},{cid:'c'*64},frozenset({host}))
                    history=StoreSnapshot(frozenset(),frozenset(),False,True)
                    result=trust.preview_from_synthetic_verified_roots(catalog_id=cid,age_months=30,
                                        reviewer=rec,source=src,roots=roots,history=history)
                    self.assertEqual(result.status,'OFFLINE_SIMULATED_ALLOW',result.reason)
                    self.assertEqual(result.caption,e.text)
                    self.assertEqual(result.source_keyboard,(('Источник',url),))
                    self.assertEqual(result.plan.delivery_mode,'single_photo_exact_caption')
                    self.assertEqual(result.plan.caption_parse_mode,'none')
                    self.assertEqual(result.plan.content_hash,hashlib.sha1(re.sub(r'\s+',' ',e.text.strip()).encode('utf-8')).hexdigest())
                    png=render_static_card_offline(result.plan)
                    self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
                    with TemporaryDirectory() as td:
                        ledger=OfflineLedger(Path(td)/'trial.sqlite3')
                        now=datetime(2026,10,26,18,0,0,tzinfo=timezone.utc)
                        args=dict(plan=result.plan,send_window='2026Q4',external_exact_hash_seen=False,
                                  external_history_known=True,external_unresolved_delivery=False,now_utc=now)
                        d=ledger.reserve(**args)
                        self.assertTrue(d.allowed,d.reason)
                        self.assertFalse(ledger.reserve(**args).allowed)
                        self.assertTrue(ledger.mark_inflight(d.key))
                        self.assertTrue(ledger.mark_unknown(d.key))
                        self.assertFalse(ledger.reserve(**{**args,'send_window':'2027Q1'}).allowed)
                        self.assertFalse(ledger.mark_confirmed(d.key,19))

    def test_no_publishing_code_imports(self):
        # No imported existing publisher / workflow code is mutated by candidate.
        import sys
        self.assertNotIn('src.publisher.run_publisher',r6n.__dict__)
        self.assertFalse(hasattr(r6n,'send_photo'))
        self.assertFalse(hasattr(r6n,'post_to_telegram'))

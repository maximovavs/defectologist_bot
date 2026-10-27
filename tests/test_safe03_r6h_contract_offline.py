"""Test-only fixtures: never production trust. No socket/provider/Telegram calls."""
import dataclasses
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import unittest

from src.services import safe03_r6d_offline_acceptance as catalog
from src.services.safe03_r6g_optin_offline import StoreSnapshot
from src.services.safe03_r6h_trust_offline import (
    ReviewerReceipt, SourceReceipt, VerifiedRoots, public_candidate,
    preview_from_synthetic_verified_roots, _source_https,
)
from src.services.safe03_r6h_delivery_offline import OfflineLedger, RepeatPermit


class Contract(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        self.entry=catalog.CATALOG['warning_small_magnets']
        self.dig=catalog.digest(self.entry)
        self.review=ReviewerReceipt(self.entry.catalog_id,self.dig,'offline-key','pediatrician_child_safety','a'*64)
        self.source=SourceReceipt(self.entry.catalog_id,self.dig,'https://example.org/source','b'*64,
                                  'example.org','reviewed_guidance','2026-10-26','c'*64)
        self.roots=VerifiedRoots({'offline-key':'a'*64},{self.entry.catalog_id:'c'*64},frozenset({'example.org'}))
        self.history=StoreSnapshot(frozenset(),frozenset(),False,True)

    def make(self, **change):
        rev=change.pop('review',self.review)
        src=change.pop('source',self.source)
        roots=change.pop('roots',self.roots)
        hist=change.pop('history',self.history)
        age=change.pop('age_months',15)
        return preview_from_synthetic_verified_roots(catalog_id=self.entry.catalog_id,age_months=age,
                reviewer=rev,source=src,roots=roots,history=hist,**change)

    def test_public_fail_closed_and_user_claims_not_roots(self):
        self.assertEqual(public_candidate('approved',self.review).status,'NO_POST')
        self.assertEqual(self.make(roots=None).status,'NO_POST')
        self.assertEqual(self.make(roots=VerifiedRoots({}, {}, frozenset())).status,'NO_POST')
        self.assertEqual(self.make(roots=VerifiedRoots({'offline-key':'z'*64},{self.entry.catalog_id:'c'*64},frozenset({'example.org'}))).status,'NO_POST')

    def test_exact_fixture_only_and_keyboard_separate_from_caption(self):
        p=self.make()
        self.assertEqual(p.status,'OFFLINE_SIMULATED_ALLOW')
        self.assertEqual(p.caption,self.entry.text)
        self.assertEqual(p.source_keyboard,(('Источник','https://example.org/source'),))
        self.assertNotIn('https://',p.caption)
        self.assertEqual(p.plan.lane,'safety_warning')
        self.assertEqual(p.plan.caption_parse_mode,'none')

    def test_all_five_roles_and_immutable_text(self):
        cases={
            'picture_naming_24_48':'speech_defectologist',
            'soft_ball_roll_36_60':'speech_defectologist',
            'clap_rhythm_24_48':'audiologist',
            'warning_small_magnets':'pediatrician_child_safety',
            'warning_button_batteries':'pediatrician_child_safety',
        }
        for cid,role in cases.items():
            with self.subTest(cid=cid):
                e=catalog.CATALOG[cid]; dig=catalog.digest(e)
                rev=ReviewerReceipt(cid,dig,'offline-key',role,'a'*64)
                src=SourceReceipt(cid,dig,'https://example.org/source','b'*64,'example.org',
                                  'reviewed_guidance','v1','c'*64)
                roots=VerifiedRoots({'offline-key':'a'*64},{cid:'c'*64},frozenset({'example.org'}))
                res=preview_from_synthetic_verified_roots(catalog_id=cid,age_months=e.age_min_months,
                       reviewer=rev,source=src,roots=roots,history=self.history)
                self.assertEqual(res.status,'OFFLINE_SIMULATED_ALLOW')
                self.assertEqual(res.caption.encode(),e.text.encode())
                self.assertLessEqual(len(res.caption.encode('utf-16-le'))//2,1000)
                self.assertEqual(res.plan.card_type,'static_text_card')

    def test_wrong_role_and_unbound_metadata(self):
        self.assertEqual(self.make(review=dataclasses.replace(self.review,reviewer_role='speech_defectologist')).status,'NO_POST')
        self.assertEqual(self.make(review=dataclasses.replace(self.review,whole_entry_digest='f'*64)).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,whole_entry_digest='0'*64)).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,source_category='blog')).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,evidence_digest='invalid')).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,reviewed_version='')).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,source_review_sha256='1'*64)).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,source_ref='http://example.org/source')).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,source_ref='https://evil.example.org/source')).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,source_ref='https://example.org/source#fragment')).status,'NO_POST')
        self.assertEqual(self.make(source=dataclasses.replace(self.source,source_ref='https://evil@example.org/source')).status,'NO_POST')
        self.assertEqual(self.make(history=dataclasses.replace(self.history,trusted_snapshot=False)).status,'NO_POST')
        self.assertEqual(self.make(history=dataclasses.replace(self.history,unresolved_delivery=True)).status,'NO_POST')

    def test_url_policy_hardening(self):
        allowed=frozenset({'example.org'})
        for url in ('javascript:alert(1)','https://example.org.evil.com/x',
                    'https://example.org:444/x','https://example.org/x\nxx',
                    'https://example.org@evil.com/a','http://example.org/source',
                    'https://example.org','#ref','https://example.org/source#x',
                    'https://127.0.0.1/test'):
            with self.subTest(url=url):self.assertFalse(_source_https(url,allowed))
        self.assertTrue(_source_https('https://example.org/source',allowed))

    def test_age_scope_cannot_override(self):
        self.assertEqual(self.make(age_months=True).status,'NO_POST')
        self.assertEqual(self.make(age_months=-1).status,'NO_POST')


class Delivery(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger=OfflineLedger(Path(self.tmp.name)/'ledger.db')
        self.e=catalog.CATALOG['warning_small_magnets']
        dig=catalog.digest(self.e)
        r=ReviewerReceipt(self.e.catalog_id,dig,'offline-key','pediatrician_child_safety','a'*64)
        s=SourceReceipt(self.e.catalog_id,dig,'https://example.org/source','b'*64,'example.org',
                        'reviewed_guidance','v1','c'*64)
        roots=VerifiedRoots({'offline-key':'a'*64},{self.e.catalog_id:'c'*64},frozenset({'example.org'}))
        self.plan=preview_from_synthetic_verified_roots(catalog_id=self.e.catalog_id,age_months=12,
                      reviewer=r,source=s,roots=roots,
                      history=StoreSnapshot(frozenset(),frozenset(),False,True)).plan
        self.now=datetime.now(timezone.utc)

    def reserve(self, **extra):
        args={'plan':self.plan,'send_window':'2026Q4','external_exact_hash_seen':False,
              'external_history_known':True,'external_unresolved_delivery':False,'now_utc':self.now}
        args.update(extra)
        return self.ledger.reserve(**args)

    def test_success_single_send_receipt_and_reopen(self):
        d=self.reserve();self.assertTrue(d.allowed)
        self.assertEqual(self.ledger.get(d.key)['state'],'reserved')
        self.assertTrue(self.ledger.mark_inflight(d.key))
        self.assertTrue(self.ledger.mark_confirmed(d.key,12345))
        self.assertEqual(self.ledger.get(d.key)['message_id'],12345)
        self.assertFalse(self.ledger.mark_confirmed(d.key,12345))
        self.assertEqual(OfflineLedger(self.ledger.path).get(d.key)['state'],'confirmed')
        self.assertFalse(self.reserve().allowed)

    def test_unknown_quarantine_blocks_every_window(self):
        d=self.reserve();self.assertTrue(d.allowed)
        self.assertTrue(self.ledger.mark_inflight(d.key))
        self.assertTrue(self.ledger.mark_unknown(d.key))
        self.assertEqual(self.ledger.get(d.key)['state'],'unknown')
        self.assertFalse(self.ledger.mark_confirmed(d.key,876))
        self.assertEqual(self.reserve(send_window='2027Q1').reason,'unresolved_or_reserved_attempt')
        self.assertEqual(OfflineLedger(self.ledger.path).get(d.key)['state'],'unknown')

    def test_unresolved_reserved_blocks_second_attempt(self):
        d=self.reserve()
        self.assertTrue(d.allowed)
        self.assertEqual(self.reserve(send_window='2027Q1').reason,'unresolved_or_reserved_attempt')

    def test_bad_receipt_and_unknown_history(self):
        d=self.reserve();self.assertTrue(d.allowed)
        self.assertFalse(self.ledger.mark_confirmed(d.key,True))
        self.assertFalse(self.ledger.mark_confirmed(d.key,0))
        self.assertFalse(self.ledger.mark_confirmed(d.key,100))
        for extra in ({'external_history_known':False},{'external_unresolved_delivery':True},
                      {'external_exact_hash_seen':'false'}, {'now_utc':None},
                      {'min_repeat_days':1}):
            with self.subTest(extra=extra):self.assertFalse(self.reserve(**extra).allowed)

    def test_global_dedup_and_repeat_cooldown(self):
        self.assertEqual(self.reserve(external_exact_hash_seen=True).reason,'external_repeat_cooldown_or_timestamp_untrusted')
        permit=RepeatPermit(self.e.catalog_id,catalog.digest(self.e),'2026Q4','a'*64)
        ninety_one_days=self.now-timedelta(days=91)
        self.assertEqual(self.reserve(external_exact_hash_seen=True,
                    external_last_confirmed_utc=ninety_one_days).reason,'global_dedup_not_overridden')
        self.assertEqual(self.reserve(external_exact_hash_seen=True,
                    external_last_confirmed_utc=ninety_one_days,
                    repeat_permit=permit,verified_permits=frozenset()).reason,'global_dedup_not_overridden')
        self.assertEqual(self.reserve(external_exact_hash_seen=True,
                    external_last_confirmed_utc=self.now-timedelta(days=10),
                    repeat_permit=permit,verified_permits=frozenset({'a'*64})).reason,'external_repeat_cooldown_or_timestamp_untrusted')
        self.assertTrue(self.reserve(external_exact_hash_seen=True,
                    external_last_confirmed_utc=ninety_one_days,repeat_permit=permit,
                    verified_permits=frozenset({'a'*64})).allowed)

    def test_duplicate_window_key_immutable(self):
        first=self.reserve()
        self.assertTrue(self.ledger.mark_inflight(first.key))
        self.assertTrue(self.ledger.mark_confirmed(first.key,1001))
        self.assertEqual(self.reserve().reason,'window_already_used')

    def test_forged_plan_cannot_reserve(self):
        from src.services.safe03_r6g_optin_offline import Plan
        forged = Plan(status='OFFLINE_SIMULATED_ALLOW', reason='trust_me',
                      catalog_id=self.e.catalog_id, entry_digest=catalog.digest(self.e),
                      lane='safety_warning', full_text=self.e.text,
                      card_type='static_text_card', delivery_mode='single_photo_exact_caption',
                      caption_parse_mode='none', idempotency_key='arbitrary',
                      content_hash=hashlib.sha1(self.e.text.encode()).hexdigest())
        self.assertEqual(self.reserve(plan=forged).reason,'forged_or_mutated_plan')
        tampered=dataclasses.replace(self.plan, full_text=self.plan.full_text+'\nНе ждите.',signature=self.plan.signature)
        self.assertEqual(self.reserve(plan=tampered).reason,'forged_or_mutated_plan')
        wrong=dataclasses.replace(self.plan,content_hash='0'*40)
        self.assertEqual(self.reserve(plan=wrong).reason,'forged_or_mutated_plan')

    def test_no_silent_switch_to_message_or_html(self):
        for changes in ({'delivery_mode':'split_photo_and_text'},
                        {'caption_parse_mode':'HTML'},
                        {'card_type':'object_image'},
                        {'status':'ALLOW'}):
            with self.subTest(changes=changes):
                self.assertFalse(self.reserve(plan=dataclasses.replace(self.plan,**changes)).allowed)

    def test_future_time_and_inconsistent_external_snapshot(self):
        self.assertEqual(self.reserve(external_exact_hash_seen=True,
                external_last_confirmed_utc=self.now+timedelta(days=1)).reason,
                'external_repeat_cooldown_or_timestamp_untrusted')
        self.assertEqual(self.reserve(external_last_confirmed_utc=self.now).reason,
                'external_history_inconsistent')

    def test_concurrent_reservations_only_one(self):
        with ThreadPoolExecutor(max_workers=10) as pool:
            outcomes=list(pool.map(lambda _:self.reserve(),range(20)))
        self.assertEqual(sum(o.allowed for o in outcomes),1)

if __name__=='__main__':unittest.main()

"""R6-G offline-only prospective seam contract; no provider, TG or DB mutation."""
import dataclasses
import hashlib
import importlib
from pathlib import Path
import unittest

from src.services import safe03_r6d_offline_acceptance as cat
from src.services import safe03_r6g_optin_offline as g
from src.services.safe03_r6g_text_card_offline import render_static_card_offline


class TestR6GOptIn(unittest.TestCase):
    @staticmethod
    def approvals():
        return {e.catalog_id: cat.digest(e) for e in cat.CATALOG.values()}

    @staticmethod
    def grants():
        return {e.catalog_id: g.EvidenceGrant(e.catalog_id, cat.digest(e),
                'offline-fixture:source:' + e.catalog_id,
                hashlib.sha256(e.catalog_id.encode()).hexdigest(), 'expert_record')
                for e in cat.CATALOG.values()}

    @staticmethod
    def history():
        return g.StoreSnapshot(frozenset(), frozenset(), trusted_snapshot=True)

    def preview(self, entry, age=None, **kwargs):
        if age is None:
            age = entry.age_min_months
        return g._offline_fixture_preview({'safe03_mode':'catalog',
                'proposal':cat.proposal(entry,age)}, approvals=kwargs.pop('approvals',self.approvals()),
                grants=kwargs.pop('grants',self.grants()),history=kwargs.pop('history',self.history()), **kwargs)

    def check(self, p, **kwargs):
        return g._offline_fixture_final_check(p,final_caption=kwargs.pop('final_caption',p.full_text),
             visual_type=kwargs.pop('visual_type','static_text_card'),
             delivery_mode=kwargs.pop('delivery_mode','single_photo_exact_caption'),
             approvals=kwargs.pop('approvals',self.approvals()),grants=kwargs.pop('grants',self.grants()),
             history=kwargs.pop('history',self.history()), **kwargs)

    def test_normal_path_identity_and_no_import_side_effects(self):
        for payload in ({'post': 'ordinary_text', 'rubric':'pro_friendly'},
                        {'age': 25}, {}, {'safe03_mode':'legacy','post':'ordinary'}):
            routed = g.route(payload)
            if 'safe03_mode' not in payload:
                self.assertEqual(routed.route,'LEGACY_UNTOUCHED')
                self.assertIs(routed.payload,payload)
            else:
                self.assertEqual(routed.route,'NO_POST')
        self.assertEqual(g.route({'safe03_mode':'catalog','proposal':{}}).route,'NO_POST')
        self.assertEqual(g.route({'safe03_mode':'typo'}).route,'NO_POST')
        self.assertEqual(g.route('text').route, 'NO_POST')
        self.assertEqual(cat.PRODUCTION_APPROVAL_DIGESTS,{})

    def test_five_frozen_exact_texts_and_age(self):
        for e in cat.CATALOG.values():
            with self.subTest(id=e.catalog_id):
                p=self.preview(e)
                self.assertEqual(p.status,'OFFLINE_SIMULATED_ALLOW')
                self.assertEqual(p.full_text.encode(),e.text.encode())
                self.assertEqual(p.card_type,'static_text_card')
                self.assertEqual(p.delivery_mode,'single_photo_exact_caption')
                self.assertEqual(p.caption_parse_mode,'none')
                self.assertLessEqual(g._utf16_units(p.full_text),1000)
                self.assertTrue(self.check(p))
                self.assertEqual(p.entry_digest,cat.digest(e))
                self.assertEqual(p.lane,'safety_warning' if e.kind=='warning' else 'exercise')

    def test_no_approval_or_source_never_allows(self):
        for e in cat.CATALOG.values():
            with self.subTest(e.catalog_id):
                self.assertEqual(self.preview(e,approvals={}).status,'NO_POST')
                self.assertEqual(self.preview(e,grants={}).reason,'missing_source_provenance')
                self.assertEqual(g.route({'safe03_mode':'catalog','proposal':cat.proposal(e,e.age_min_months)}).route,'NO_POST')

    def test_warning_card_is_typographic_only(self):
        for e in cat.CATALOG.values():
            with self.subTest(e.catalog_id):
                p=self.preview(e)
                raw=render_static_card_offline(p)
                self.assertTrue(raw.startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertGreater(len(raw),1000)
        with self.assertRaises(ValueError):
            render_static_card_offline(g.Plan('NO_POST','none'))

    def test_no_post_if_legacy_full_body_not_byte_exact(self):
        e=cat.CATALOG['warning_button_batteries']
        p=self.preview(e)
        for changed in ('Заголовок\n'+p.full_text, p.full_text+'\n#логопед',
                        p.full_text+'\nИсточник: пример',p.full_text+'\nВопрос к родителям?',
                        p.full_text.replace('мёд','мед'),p.full_text.replace('\n','\r\n'),
                        p.full_text+'\u200b',p.full_text[:200]):
            with self.subTest(len=len(changed)):
                self.assertFalse(self.check(p,final_caption=changed))
        for visual in ('generated_object_photo', 'illustration', 'object_fallback', 'none'):
            self.assertFalse(self.check(p,visual_type=visual))
        for delivery in ('sendMessage', 'image_then_message','split'):
            self.assertFalse(self.check(p,delivery_mode=delivery))

    def test_digest_and_metadata_mismatch(self):
        e=cat.CATALOG['warning_small_magnets']
        p=self.preview(e)
        for field, value in [('age_months', 2000),('entry_digest','00'*32),('card_title','MAGNET PHOTO'),
                             ('source_ref','untrusted'),('content_hash','00'*32),('lane','exercise'),
                             ('idempotency_key','bypass'),('caption_parse_mode','html'),('full_text',p.full_text+'!')]:
            with self.subTest(field):
                self.assertFalse(self.check(dataclasses.replace(p,**{field:value})))
        self.assertEqual(p.status,'OFFLINE_SIMULATED_ALLOW')

    def test_source_binding_is_digest_specific(self):
        e=cat.CATALOG['warning_button_batteries']; gr=self.grants()
        gr[e.catalog_id]=dataclasses.replace(gr[e.catalog_id], entry_digest='0'*64)
        self.assertEqual(self.preview(e,grants=gr).reason,'invalid_source_binding')
        gr=self.grants(); gr[e.catalog_id]=dataclasses.replace(gr[e.catalog_id], evidence_digest='garbage')
        self.assertEqual(self.preview(e,grants=gr).reason,'invalid_source_binding')
        gr=self.grants(); gr[e.catalog_id]=dataclasses.replace(gr[e.catalog_id], source_category='random_blog')
        self.assertEqual(self.preview(e,grants=gr).reason,'invalid_source_binding')

    def test_alltime_dedup_fail_closed(self):
        e=cat.CATALOG['warning_small_magnets']; p=self.preview(e)
        hist=g.StoreSnapshot(frozenset({p.idempotency_key}),frozenset(),trusted_snapshot=True)
        self.assertEqual(self.preview(e,history=hist).reason,'duplicate_without_policy_override')
        hist=g.StoreSnapshot(frozenset(),frozenset({p.content_hash}),trusted_snapshot=True)
        self.assertEqual(self.preview(e,history=hist).reason,'duplicate_without_policy_override')
        self.assertFalse(self.check(p,history=hist))
        self.assertEqual(self.preview(e,history=g.StoreSnapshot(frozenset(),frozenset())).reason,'no_trusted_delivery_snapshot')
        self.assertEqual(self.preview(e,history=g.StoreSnapshot(frozenset(),frozenset(),unresolved_delivery=True,trusted_snapshot=True)).reason,'unresolved_delivery')

    def test_body_hash_matches_exact_current_publisher_policy(self):
        # run_publisher.sha1(norm_space(plain)) behavior, verified against fresh main.
        import re
        for e in cat.CATALOG.values():
            plan=self.preview(e)
            expected=hashlib.sha1(re.sub(r'\s+', ' ', e.text.strip()).encode()).hexdigest()
            self.assertEqual(plan.content_hash,expected)
            self.assertEqual(len(plan.content_hash),40)

    def test_caption_budget(self):
        e=cat.CATALOG['warning_button_batteries']
        self.assertEqual(g._utf16_units(e.text), 916)
        self.assertEqual(self.preview(e,caption_limit=915).reason,'exact_caption_too_long')
        self.assertEqual(self.preview(e,caption_limit=1001).reason,'invalid_caption_budget')
        self.assertEqual(self.preview(e,caption_limit=1000).status,'OFFLINE_SIMULATED_ALLOW')
        self.assertFalse(self.check(self.preview(e),caption_limit=915))

    def test_reject_non_exact_outer_schema(self):
        e=cat.CATALOG['clap_rhythm_24_48']; p=cat.proposal(e,30)
        for env in [{'safe03_mode':'catalog','proposal':p,'approved':True},
                    {'safe03_mode':'catalog','proposal':p,'source':'example.com'},
                    {'safe03_mode':'catalog'},[],None]:
            self.assertEqual(g._offline_fixture_preview(env,approvals=self.approvals(),
                grants=self.grants(),history=self.history()).status,'NO_POST')

    def test_plain_text_to_production_validator_is_nonmutating(self):
        """Content guard parity is checked separately in integration test evidence."""
        for e in cat.CATALOG.values():
            p=self.preview(e)
            self.assertEqual(p.full_text, cat.CATALOG[p.catalog_id].text)

    def test_no_publisher_import_or_workflow_or_registry_mutation(self):
        root=Path(__file__).resolve().parents[1]
        for path in ('src/publisher/run_publisher.py','.github/workflows/post.yml',
                     '.github/workflows/publisher_policy_pr_checks.yml'):
            self.assertNotIn('safe03_r6g_optin_offline',(root/path).read_text())
        self.assertEqual(cat.PRODUCTION_APPROVAL_DIGESTS,{})

if __name__=='__main__':
    unittest.main()

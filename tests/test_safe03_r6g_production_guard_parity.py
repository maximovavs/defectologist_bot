"""Grounding/safety compatibility ONLY; not proof of clinical efficacy or live send."""
import dataclasses
import hashlib
import random
import unittest

from src.services.safe03_r6d_offline_acceptance import CATALOG, digest, proposal
from src.services import llm_generator as llm
from src.services import safe03_r6g_optin_offline as g
from src.services.safe03_r6g_text_card_offline import render_static_card_offline

class R6GProductionGuardParity(unittest.TestCase):
    @staticmethod
    def context():
        approved={e.catalog_id: digest(e) for e in CATALOG.values()}
        grants={e.catalog_id: g.EvidenceGrant(e.catalog_id,digest(e),
             'offline-expert-record:'+e.catalog_id,
             hashlib.sha256(e.text.encode()).hexdigest(),'expert_record') for e in CATALOG.values()}
        history=g.StoreSnapshot(frozenset(),frozenset(),trusted_snapshot=True)
        return approved,grants,history

    def test_existing_plain_text_safety_guards_on_exact_five(self):
        for e in CATALOG.values():
            with self.subTest(e.catalog_id):
                for fn in (llm._validate_parent_oral_safety_output,
                           llm._validate_parent_safety_output,
                           llm._validate_parent_hearing_inference_output,
                           llm._validate_parent_age_action_fit):
                    result=fn(e.text)
                    self.assertTrue(result[0],(e.catalog_id,fn.__name__,result))
                ok,reason=llm._validate_parent_hearing_guidance_output(
                    e.text,topic_id='hearing_and_speech' if e.catalog_id=='clap_rhythm_24_48' else '')
                self.assertTrue(ok,(e.catalog_id,reason))

    def test_publisher_body_sha1_parity(self):
        # Production body_hash = sha1(norm_space(plain)) - same exact impl.
        from src.publisher.run_publisher import sha1, norm_space, _telegram_utf16_units
        approvals,grants,history=self.context()
        for e in CATALOG.values():
            plan=g._offline_fixture_preview({'safe03_mode':'catalog','proposal':proposal(e,e.age_min_months)},
                                              approvals=approvals,grants=grants,history=history)
            self.assertEqual(plan.content_hash,sha1(norm_space(e.text)))
            self.assertEqual(_telegram_utf16_units(e.text),g._utf16_units(e.text))
            self.assertTrue(g._offline_fixture_final_check(plan,final_caption=e.text,
                visual_type='static_text_card',delivery_mode='single_photo_exact_caption',
                approvals=approvals,grants=grants,history=history))

    def test_adversarial_mutations_frozen_rng(self):
        approvals,grants,history=self.context()
        rng=random.Random(20261026)
        entries=list(CATALOG.values())
        bad=0
        for _ in range(500):
            e=rng.choice(entries)
            value=proposal(e,e.age_min_months)
            field=rng.choice(['text','object_id','action','actor','recipient','version','constraints','age_months','kind'])
            if field=='text': value['text'] += rng.choice([' ', '\u200b','!','#безопасность','\n'])
            elif field=='age_months': value[field]=-1
            elif field=='version': value[field]+=1
            elif field=='constraints': value[field]=value[field]+['dangerous_small_part']
            else: value[field]='WRONG'
            plan=g._offline_fixture_preview({'safe03_mode':'catalog','proposal':value},
                 approvals=approvals,grants=grants,history=history)
            self.assertEqual(plan.status,'NO_POST',(e.catalog_id,field))
            bad+=1
        self.assertEqual(bad,500)

    def test_card_spoof_does_not_generate(self):
        approvals,grants,history=self.context()
        e=CATALOG['warning_button_batteries']
        plan=g._offline_fixture_preview({'safe03_mode':'catalog','proposal':proposal(e,20)},
             approvals=approvals,grants=grants,history=history)
        with self.assertRaises(ValueError):
            render_static_card_offline(dataclasses.replace(plan,card_title='БАТАРЕЙКИ СЪЕДОБНЫ'))
        with self.assertRaises(ValueError):
            render_static_card_offline(dataclasses.replace(plan,full_text=plan.full_text+'X'))

if __name__=='__main__':unittest.main()

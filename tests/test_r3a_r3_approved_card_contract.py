"""Synthetic fixtures only. No clinical or production source is approved here."""
import copy
import hashlib
import json
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "reference"))
sys.path.insert(0, str(HERE / "research"))
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from r3a_r3_approved_card_contract import (
    SCHEMA, SignedPack, SourceSnapshot, OfflineAuthority, canonical_payload,
    evaluate_last_boundary,
)


def digest(b):
    return hashlib.sha256(b).hexdigest()


class R3OfflineWholeCard(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()
        self.pub = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.source = SourceSnapshot(
            source_id="SYNTHETIC_NOT_CLINICAL", canonical_url="https://example.invalid/source",
            raw_html=b"<html>Entirely synthetic laboratory fixture</html>",
            evidence_text="Synthetic test: one cube might be blue, and another cube red. No developmental conclusions.",
            topic_id="SYNTHETIC_TOPIC", audience="parents", age_scope="TEST_AGE_NOT_CLINICAL")
        self.auth = OfflineAuthority(
            reviewer_public_keys={"test-reviewer":self.pub},
            reviewer_roles={"test-reviewer":frozenset({"clinical_reviewer"})},
            approved_channel_id="TEST_CHANNEL_NOT_REAL",
            canonical_source_urls={self.source.source_id:self.source.canonical_url},
            revoked_reviewer_ids=frozenset(), revoked_pack_ids=frozenset(),
            revoked_source_ids=frozenset(), active_packs={}, historical_source_ids=frozenset({
            "healthychildren_bilingual_myths","nationwide_bilingual_speech_delay",
            "asha_speech_sound_multilingual_influence","nlg_nhs_tongue_tie_speech_myth",
            "asha_newborn_hearing_screening","nationwide_school_hearing_screening",
            "nationwide_newborn_hearing_screening","healthychildren_one_year_talking",
            "nationwide_aac_myths","healthychildren_crawling_reading_myth",
            "healthychildren_learning_disorders_dyslexia"}),
            now_utc=datetime(2026,10,29,tzinfo=timezone.utc))
        self.record={
            "schema":SCHEMA,"pack_id":"TEST_ONLY_P1","revision":1,
            "reviewer_id":"test-reviewer",
            "reviewer_role":"clinical_reviewer", "channel_id":"TEST_CHANNEL_NOT_REAL",
            "rubric_id":"myth_fact", "valid_from_utc":"2026-10-01T00:00:00Z",
            "valid_until_utc":"2026-11-30T00:00:00Z",
            "source_id":self.source.source_id,"canonical_url":self.source.canonical_url,
            "raw_html_sha256":digest(self.source.raw_html),
            "evidence_sha256":digest(self.source.evidence_text.encode("utf-8")),
            "evidence_refutation":{"start":0,"text":"Synthetic test: one cube might be blue, and another cube red."},
            "source_qualifiers":[{"start":62,"text":"No developmental conclusions."}],
            "topic_id":self.source.topic_id,"audience":"parents","age_scope":self.source.age_scope,
            "cards":[
                {"card_id":"a","myth":"Все тестовые кубики всегда синие.",
                 "fact":"У тестовых кубиков бывают разные цвета.",
                 "qualifier":"Пример вымышленный; это не медицинская информация.",
                 "action":"Посмотрите на два условных кубика."},
                {"card_id":"b","myth":"Тестовый красный кубик никогда не бывает другим цветом.",
                 "fact":"Цвет условного кубика может различаться.",
                 "qualifier":"Этот тест не описывает развитие ребёнка.",
                 "action":"Сравните два нарисованных кубика."}
            ]}
        self.auth = replace(self.auth, active_packs={
            self.scope_for(self.record): self.identity_for(self.record)
        })
    def scope_for(self, rec):
        return (rec["source_id"], rec["topic_id"], rec["channel_id"], rec["rubric_id"])
    def identity_for(self, rec):
        return (rec["revision"], rec["pack_id"], digest(canonical_payload(rec)))
    def registered(self, rec):
        return replace(self.auth, active_packs={self.scope_for(rec): self.identity_for(rec)})
    def sign(self, rec=None, *, key=None):
        data = canonical_payload(self.record if rec is None else rec)
        return SignedPack(data,(key or self.key).sign(data))
    def run_decision(self, **kwargs):
        args=dict(signed_pack=self.sign(),source=self.source,authority=self.auth,
                  card_id="a",simulation_only=True)
        args.update(kwargs)
        return evaluate_last_boundary(**args)
    def refuse(self, **kwargs):
        v=self.run_decision(**kwargs)
        self.assertFalse(v.allowed,v.reason)
        self.assertIsNone(v.send_plan)
        self.assertTrue(v.reason.startswith("NO_POST:"))
        return v

    def test_two_whole_card_variants_without_resigning(self):
        a=self.run_decision(card_id="a");b=self.run_decision(card_id="b")
        self.assertTrue(a.allowed);self.assertTrue(b.allowed)
        self.assertNotEqual(a.send_plan["text"],b.send_plan["text"])
        self.assertEqual(a.send_plan["parse_mode"],None)
        self.assertEqual(a.send_plan["method"],"sendMessage")
        self.assertTrue(all("TEST_AGE_NOT_CLINICAL" in x.send_plan["text"] for x in (a,b)))
    def test_fails_closed_when_not_explicitly_offline(self): self.refuse(simulation_only=False)
    def test_simulation_integer_is_rejected(self): self.refuse(simulation_only=1)
    def test_unapproved_variant_rejected(self): self.refuse(card_id="injected")
    def test_variant_subclass_rejected(self):
        class N(str):
            def __eq__(self, other):return True
        self.refuse(card_id=N("injected"))
    def test_model_free_text_not_an_approval(self): self.refuse(card_id="У моего ребёнка гарантированно всё хорошо")
    def test_altered_final_text_rejected(self):
        original=self.run_decision().send_plan
        modified=dict(original,text=original["text"].replace("бывают разные", "всегда одинаковые"))
        self.refuse(proposed_transport=modified)
    def test_missing_qualifier_rejected(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport=dict(original,text=original["text"].replace(
            "Пример вымышленный; это не медицинская информация.","")))
    def test_missing_age_rejected(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport=dict(original,text=original["text"].replace("👶 Возраст: TEST_AGE_NOT_CLINICAL", "")))
    def test_edited_source_url_rejected_at_send(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport=dict(original,text=original["text"].replace("example.invalid","evil.invalid")))
    def test_html_delivery_rejected(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport=dict(original,parse_mode="HTML"))
    def test_markdown_delivery_rejected(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport=dict(original,parse_mode="Markdown"))
    def test_caption_delivery_rejected(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport=dict(original,method="sendPhoto"))
    def test_transport_split_rejected(self):
        original=self.run_decision().send_plan
        self.refuse(proposed_transport={**original,"extra_message":"More text"})
    def test_transport_subclass_rejected(self):
        class Fake(dict):pass
        self.refuse(proposed_transport=Fake(self.run_decision().send_plan))
    def test_proposed_transport_exact_allowed(self):
        original=self.run_decision().send_plan
        self.assertTrue(self.run_decision(proposed_transport=dict(original)).allowed)
    def test_source_html_change(self):
        self.refuse(source=replace(self.source,raw_html=b"different"))
    def test_evidence_change(self):
        self.refuse(source=replace(self.source,evidence_text=self.source.evidence_text+" changed"))
    def test_source_family_change(self):
        self.refuse(source=replace(self.source,topic_id="different"))
    def test_source_population_change(self):
        self.refuse(source=replace(self.source,audience="pros"))
    def test_source_age_change(self):
        self.refuse(source=replace(self.source,age_scope="TEST_ADULT"))
    def test_url_trust_registry_change(self):
        auth=replace(self.auth,canonical_source_urls={self.source.source_id:"https://wrong.invalid/"})
        self.refuse(authority=auth)
    def test_wrong_channel(self):
        self.refuse(authority=replace(self.auth, approved_channel_id="other_channel"))
    def test_wrong_rubric(self):
        rec=copy.deepcopy(self.record);rec["rubric_id"]="other_rubric"
        self.refuse(signed_pack=self.sign(rec))
    def test_missing_clinical_reviewer_role(self):
        self.refuse(authority=replace(self.auth,reviewer_roles={"test-reviewer":frozenset({"editor"})}))
    def test_wrong_signed_role(self):
        rec=copy.deepcopy(self.record);rec["reviewer_role"]="editor"
        self.refuse(signed_pack=self.sign(rec))
    def test_missing_authority_key(self):
        self.refuse(authority=replace(self.auth,reviewer_public_keys={}))
    def test_attacker_signature_rejected_by_original_root(self):
        attack_key=Ed25519PrivateKey.generate()
        self.refuse(signed_pack=self.sign(key=attack_key))
    def test_root_injection_is_not_a_production_guarantee(self):
        # Documented GAP: caller can replace whole offline authority; this test
        # never certifies production reviewer authority or credentials.
        attack_key=Ed25519PrivateKey.generate()
        attacker_pub=attack_key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
        replacement=replace(self.auth,reviewer_public_keys={"test-reviewer":attacker_pub})
        self.assertTrue(self.run_decision(signed_pack=self.sign(key=attack_key),authority=replacement).allowed)
    def test_low_order_identity_key_refused(self):
        self.refuse(authority=replace(self.auth,reviewer_public_keys={"test-reviewer":b"\x01"+b"\0"*31}))
    def test_revoke_reviewer(self):
        self.refuse(authority=replace(self.auth,revoked_reviewer_ids=frozenset({"test-reviewer"})))
    def test_revoke_pack(self):
        self.refuse(authority=replace(self.auth,revoked_pack_ids=frozenset({"TEST_ONLY_P1"})))
    def test_revoke_source(self):
        self.refuse(authority=replace(self.auth,revoked_source_ids=frozenset({self.source.source_id})))
    def test_expired_pack(self):
        self.refuse(authority=replace(self.auth,now_utc=datetime(2027,1,1,tzinfo=timezone.utc)))
    def test_premature_pack(self):
        self.refuse(authority=replace(self.auth,now_utc=datetime(2026,9,1,tzinfo=timezone.utc)))
    def test_historical_source_unconditionally_no_post(self):
        self.refuse(source=replace(self.source,source_id="nationwide_aac_myths"))
    def test_all_eleven_historical_sources_stay_no_post(self):
        for source_id in self.auth.historical_source_ids:
            with self.subTest(source_id=source_id):
                self.refuse(source=replace(self.source, source_id=source_id))
    def test_frozen_r1_negative_text_is_not_appendable(self):
        text=self.run_decision().send_plan["text"]
        for negative in ("Skipping crawling can cause reading difficulties.",
                         "Shared reading must take place only at night."):
            with self.subTest(negative=negative):
                original=dict(self.run_decision().send_plan)
                original["text"]=text+"\n"+negative
                self.refuse(proposed_transport=original)
    def test_history_inventory_is_11(self):
        self.assertEqual(len(self.auth.historical_source_ids),11)
    def test_reordered_object_canonical_json_rejected(self):
        payload=json.dumps(self.record,ensure_ascii=False).encode("utf-8")
        self.refuse(signed_pack=SignedPack(payload,self.key.sign(payload)))
    def test_double_json_key_rejected(self):
        payload=self.sign().payload.replace(b'"pack_id":',b'"pack_id":"spoof","pack_id":',1)
        self.refuse(signed_pack=SignedPack(payload,self.key.sign(payload)))
    def test_unknown_signed_field_rejected(self):
        rec=copy.deepcopy(self.record);rec["unsigned_note"]="dummy"
        self.refuse(signed_pack=self.sign(rec))
    def test_bool_revision_rejected(self):
        rec=copy.deepcopy(self.record);rec["revision"]=True
        self.refuse(signed_pack=self.sign(rec))
    def test_empty_cards_rejected(self):
        rec=copy.deepcopy(self.record);rec["cards"]=[]
        self.refuse(signed_pack=self.sign(rec))
    def test_duplicate_card_ids_rejected(self):
        rec=copy.deepcopy(self.record);rec["cards"][1]["card_id"]="a"
        self.refuse(signed_pack=self.sign(rec))
    def test_unapproved_card_myth_in_signed_payload_requires_new_signature(self):
        rec=copy.deepcopy(self.record);rec["cards"][0]["myth"]="ALL CHILDREN MUST..."
        data=canonical_payload(rec)
        self.refuse(signed_pack=SignedPack(data,self.sign().signature))
    def test_attacker_invalid_text_subclass_rejected(self):
        class Attack(str):
            def __eq__(self,other):return True
        self.refuse(source=replace(self.source,source_id=Attack("SYNTHETIC_NOT_CLINICAL")))
    def test_signed_pack_subclass_rejected(self):
        class Attack(SignedPack):
            def __getattribute__(self,name):
                if name=="payload":return super().__getattribute__(name)
                return super().__getattribute__(name)
        self.refuse(signed_pack=Attack(self.sign().payload,self.sign().signature))
    def test_newline_in_approved_card_rejected(self):
        rec=copy.deepcopy(self.record);rec["cards"][0]["fact"]="Safe\nHidden claim"
        self.refuse(signed_pack=self.sign(rec))
    def test_placeholders_rejected(self):
        rec=copy.deepcopy(self.record);rec["cards"][0]["fact"]="Answer {{ arbitrary_llm_text }}"
        self.refuse(signed_pack=self.sign(rec))
    def test_oversized_transport_rejected(self):
        rec=copy.deepcopy(self.record);rec["cards"][0]["action"]="К"*4000
        self.refuse(signed_pack=self.sign(rec))
    def test_no_html_or_photo_in_success(self):
        plan=self.run_decision().send_plan
        self.assertEqual(set(plan),{"method","text","parse_mode","disable_web_page_preview"})
        self.assertNotIn("<b>",plan["text"])
        self.assertEqual(plan["parse_mode"],None)
    def test_source_refutation_missing(self):
        rec=copy.deepcopy(self.record);rec["evidence_refutation"]={"start":0,"text":"Unrelated evidence."}
        self.refuse(signed_pack=self.sign(rec))
    def test_source_qualifier_missing(self):
        rec=copy.deepcopy(self.record);rec["source_qualifiers"]=[]
        self.refuse(signed_pack=self.sign(rec))
    def test_source_qualifier_mismatch(self):
        rec=copy.deepcopy(self.record);rec["source_qualifiers"][0]["start"]=1
        self.refuse(signed_pack=self.sign(rec))
    def test_refutation_truncation_rejected(self):
        rec=copy.deepcopy(self.record);rec["evidence_refutation"]["text"]="Synthetic test: one cube might be blue"
        self.refuse(signed_pack=self.sign(rec))
    def test_refutation_start_mismatch(self):
        rec=copy.deepcopy(self.record);rec["evidence_refutation"]["start"]=10
        self.refuse(signed_pack=self.sign(rec))
    def test_immutable_send_plan(self):
        plan=self.run_decision().send_plan
        with self.assertRaises(TypeError):plan["text"]="tampered"
    def test_source_url_change_inside_permit(self):
        rec=copy.deepcopy(self.record);rec["canonical_url"]="https://wrong.invalid"
        self.refuse(signed_pack=self.sign(rec))
    def test_pack_key_must_be_prime_order(self):
        self.assertEqual(self.refuse(authority=replace(self.auth,reviewer_public_keys={"test-reviewer":b"\x01"+b"\0"*31})).reason,"NO_POST:signed_approval_invalid")
    def test_do_not_allow_freeform_admission_even_if_pack_contains_two_versions(self):
        # Model cannot mix fact variant a and myth variant b.
        a=self.run_decision(card_id="a").send_plan
        b=self.run_decision(card_id="b").send_plan
        crossed=a["text"].replace("Все тестовые кубики всегда синие.","Тестовый красный кубик никогда не бывает другим цветом.")
        self.refuse(proposed_transport=dict(a,text=crossed))

if __name__ == '__main__': unittest.main()

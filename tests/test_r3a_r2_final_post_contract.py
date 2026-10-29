"""Only frozen R3-A bilingualism positive fixture; all variants are mutations.

The Ed25519 private key is ephemeral and ONLY in this test. It is never a real
clinical signature, and may not become an approval key for any production path.
"""

from __future__ import annotations

from dataclasses import replace
import unittest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from research.r3a_r2_final_post_contract import (
    SourceSnapshot, Span, ReviewPermit, SignedReview,
    assess_finalized_post, digest, PERMIT_SCHEMA, REJECTION_REASON,
)

# Copied from tests/test_myth_fact_fidelity.py at main c92adc1e (not source HTML).
BILINGUAL_EVIDENCE = (
    "A common myth is that bilingualism causes language delay. "
    "Bilingualism does not cause language delay, and using two languages is not itself a language disorder. "
    "Families can keep using the home language during books, meals, and play. "
    "Children can participate in ordinary family conversations while learning the community language too. "
    "There is no evidence that two languages by themselves create a speech or language disorder."
)
VALID_BILINGUAL_CARD = (
    "Два языка не вызывают задержку сами по себе\n\n"
    "👶 Возраст: 3–6 лет\n\n"
    "🔴 Миф: Два языка вызывают задержку речи.\n\n"
    "Двуязычие само по себе не является причиной задержки речи. В семье можно продолжать использовать "
    "домашний язык в обычных разговорах, чтении и игре, не превращая общение в проверку ребёнка.\n\n"
    "🧩 Что попробовать сегодня:\n"
    "Прочитайте знакомую книгу на домашнем языке и обсудите две картинки короткими фразами.\n\n"
    "💡 Что это дает: Ребёнок участвует в семейном разговоре и отвечает доступным ему способом."
)
MYTH = "Два языка вызывают задержку речи."
FACT = (
    "Двуязычие само по себе не является причиной задержки речи. В семье можно продолжать использовать "
    "домашний язык в обычных разговорах, чтении и игре, не превращая общение в проверку ребёнка."
)
REFUTATION = "Bilingualism does not cause language delay, and using two languages is not itself a language disorder."
QUALIFIER = "There is no evidence that two languages by themselves create a speech or language disorder."


class R3AR2FinalPostContractTest(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()
        self.other_key = Ed25519PrivateKey.generate()
        pub = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.pub = pub
        # Existing old synthetic evidence bytes used as test-only stand-in for
        # the raw response. NOT a fabricated original HTML capture.
        self.source = SourceSnapshot(
            source_id="fixture:bilingual", canonical_url="fixture://bilingual",
            raw_response_bytes=BILINGUAL_EVIDENCE.encode("utf-8"),
            evidence_text=BILINGUAL_EVIDENCE, topic_id="bilingualism",
            population="children", age_scope="3–6 лет",
        )
        self.permit = ReviewPermit(
            permit_id="fixture:existing_bilingual", reviewer_id="test-ephemeral-reviewer",
            source_id=self.source.source_id, canonical_url=self.source.canonical_url,
            raw_html_sha256=digest(self.source.raw_response_bytes),
            evidence_sha256=digest(BILINGUAL_EVIDENCE.encode("utf-8")),
            topic_id="bilingualism", population="children", age_scope="3–6 лет",
            myth_line=MYTH, fact_text=FACT,
            refutation=Span(BILINGUAL_EVIDENCE.index(REFUTATION), REFUTATION),
            mandatory_source_qualifiers=(Span(BILINGUAL_EVIDENCE.index(QUALIFIER), QUALIFIER),),
            mandatory_post_qualifiers=("само по себе", "не превращая общение в проверку ребёнка"),
            approved_final_post_sha256=digest(VALID_BILINGUAL_CARD.encode("utf-8")),
        )
        self.trusted_urls = {self.source.source_id: self.source.canonical_url}
        self.trusted_keys = {self.permit.reviewer_id: pub}

    def sign(self, permit=None, key=None):
        p = permit or self.permit
        return SignedReview(p, (key or self.key).sign(p.signed_bytes()))

    def assess(self, post=VALID_BILINGUAL_CARD, *, source=None, permit=None,
               signed_reviews=None, simulation_only=True, trusted_keys=None,
               trusted_urls=None):
        src = source if source is not None else self.source
        if signed_reviews is None:
            signed_reviews = {src.source_id: (self.sign(permit),)}
        return assess_finalized_post(
            src, post, signed_reviews=signed_reviews,
            trusted_reviewer_keys=(self.trusted_keys if trusted_keys is None else trusted_keys),
            trusted_source_urls=(self.trusted_urls if trusted_urls is None else trusted_urls),
            simulation_only=simulation_only,
        )

    def assert_denied(self, d, detail=None):
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, REJECTION_REASON)
        if detail is not None:
            self.assertEqual(d.detail, detail)

    def test_no_simulation_never_admits_even_signed_fixture(self):
        self.assert_denied(self.assess(simulation_only=False), "not_wired_to_a_trusted_production_authority")

    def test_empty_registry_denies(self):
        self.assert_denied(self.assess(signed_reviews={}))

    def test_empty_review_keys_denies(self):
        self.assert_denied(self.assess(trusted_keys={}))

    def test_empty_trusted_source_urls_denies(self):
        self.assert_denied(self.assess(trusted_urls={}))

    def test_existing_positive_with_ephemeral_trust_only(self):
        d = self.assess()
        self.assertTrue(d.allowed)
        self.assertEqual(d.detail, "simulated_review_only")

    def test_fact_polarity_mutated_after_review_denies(self):
        mutated = VALID_BILINGUAL_CARD.replace("не является причиной", "является причиной")
        self.assert_denied(self.assess(mutated), "no_unique_permitted_final_post")

    def test_fact_removed_after_review_denies(self):
        self.assert_denied(self.assess(VALID_BILINGUAL_CARD.replace(FACT, "")))

    def test_visible_caveat_removed_after_review_denies(self):
        self.assert_denied(self.assess(VALID_BILINGUAL_CARD.replace("само по себе ", "")))

    def test_fact_relocated_to_footer_denies(self):
        x = VALID_BILINGUAL_CARD.replace(FACT, "") + "\n" + FACT
        self.assert_denied(self.assess(x))

    def test_myth_mutated_denies(self):
        self.assert_denied(self.assess(VALID_BILINGUAL_CARD.replace(MYTH, MYTH[:-1])))

    def test_parent_age_mutated_denies(self):
        self.assert_denied(self.assess(VALID_BILINGUAL_CARD.replace("3–6 лет", "4–7 лет")))

    def test_finalizer_changes_single_space_denies(self):
        self.assert_denied(self.assess(VALID_BILINGUAL_CARD.replace("\n\n", "\n", 1)))

    def test_zero_width_character_denies(self):
        self.assert_denied(self.assess(VALID_BILINGUAL_CARD.replace("Два языка", "Два\u200b языка", 1)))

    def test_wrong_source_id_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, source_id="fixture:other")))

    def test_wrong_source_url_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, canonical_url="fixture://other")))

    def test_wrong_raw_bytes_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, raw_response_bytes=b"altered")))

    def test_wrong_evidence_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, evidence_text=BILINGUAL_EVIDENCE + " ")))

    def test_wrong_topic_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, topic_id="preliteracy")))

    def test_wrong_population_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, population="adults")))

    def test_wrong_age_scope_denies(self):
        self.assert_denied(self.assess(source=replace(self.source, age_scope="other")))

    def test_replaced_refutation_receipt_cannot_be_reused(self):
        p = replace(self.permit, refutation=Span(self.permit.refutation.start, "There is no evidence"))
        self.assert_denied(self.assess(permit=p, signed_reviews={p.source_id:(SignedReview(p, self.sign().signature),)}),
                           "review_signature_invalid")

    def test_truncated_refutation_with_new_test_signature_still_denies(self):
        p = replace(self.permit, refutation=Span(self.permit.refutation.start, "Bilingualism does not cause"))
        self.assert_denied(self.assess(permit=p), "missing_complete_refutation")

    def test_refutation_position_changed_new_signature_denies(self):
        p = replace(self.permit, refutation=Span(0, REFUTATION))
        self.assert_denied(self.assess(permit=p), "missing_complete_refutation")

    def test_missing_source_qualifier_with_new_signature_denies(self):
        p = replace(self.permit, mandatory_source_qualifiers=())
        self.assert_denied(self.assess(permit=p), "source_qualifier_missing_or_incomplete")

    def test_source_qualifier_wrong_position_with_new_signature_denies(self):
        p = replace(self.permit, mandatory_source_qualifiers=(Span(0, QUALIFIER),))
        self.assert_denied(self.assess(permit=p), "source_qualifier_missing_or_incomplete")

    def test_unapproved_post_qualifier_with_new_signature_denies(self):
        p = replace(self.permit, mandatory_post_qualifiers=("not in fact",))
        self.assert_denied(self.assess(permit=p), "post_qualifier_not_in_fact")

    def test_empty_post_qualifier_list_denies(self):
        p = replace(self.permit, mandatory_post_qualifiers=())
        self.assert_denied(self.assess(permit=p), "post_qualifier_not_in_fact")

    def test_wrong_fact_anchor_even_signed_denies(self):
        p = replace(self.permit, fact_text="not in post")
        self.assert_denied(self.assess(permit=p), "reviewed_fact_missing_or_ambiguous")

    def test_forged_signature_from_other_key_denies(self):
        s = self.sign(key=self.other_key)
        self.assert_denied(self.assess(signed_reviews={self.source.source_id:(s,)}), "review_signature_invalid")

    def test_unsigned_permit_altered_after_sign_denies(self):
        s = self.sign()
        altered = SignedReview(replace(s.permit, population="adults"), s.signature)
        self.assert_denied(self.assess(signed_reviews={self.source.source_id:(altered,)}), "review_signature_invalid")

    def test_stale_schema_denies_even_new_signature(self):
        p = replace(self.permit, schema="r3a-r2-old")
        self.assert_denied(self.assess(permit=p), "stale_or_unknown_contract")

    def test_duplicate_same_post_permits_are_ambiguous(self):
        p = replace(self.permit, permit_id="fixture:duplicate")
        registry = {self.source.source_id:(self.sign(),self.sign(p))}
        self.assert_denied(self.assess(signed_reviews=registry), "no_unique_permitted_final_post")

    def test_changed_registered_url_denies(self):
        self.assert_denied(self.assess(trusted_urls={self.source.source_id:"fixture://other"}))

    def test_approved_full_post_digest_is_required(self):
        p = replace(self.permit, approved_final_post_sha256="0" * 64)
        self.assert_denied(self.assess(permit=p), "no_unique_permitted_final_post")

    def test_shortened_source_qualifier_with_new_signature_denies(self):
        span = self.permit.mandatory_source_qualifiers[0]
        short = Span(span.start, "There is no evidence")
        p = replace(self.permit, mandatory_source_qualifiers=(short,))
        self.assert_denied(self.assess(permit=p), "source_qualifier_missing_or_incomplete")

    def test_missing_age_line_even_with_new_signed_post_denies(self):
        modified = VALID_BILINGUAL_CARD.replace("👶 Возраст: 3–6 лет\n\n", "")
        p = replace(self.permit, approved_final_post_sha256=digest(modified.encode("utf-8")))
        self.assert_denied(self.assess(modified, permit=p), "post_age_scope_changed")

    def test_changed_age_line_even_new_signed_post_denies(self):
        modified = VALID_BILINGUAL_CARD.replace("👶 Возраст: 3–6 лет", "👶 Возраст: 4–7 лет")
        p = replace(self.permit, approved_final_post_sha256=digest(modified.encode("utf-8")))
        self.assert_denied(self.assess(modified, permit=p), "post_age_scope_changed")

    def test_second_myth_line_even_new_signed_post_denies(self):
        modified = VALID_BILINGUAL_CARD.replace("🔴 Миф: " + MYTH, "🔴 Миф: " + MYTH + "\n🔴 Миф: " + MYTH)
        p = replace(self.permit, approved_final_post_sha256=digest(modified.encode("utf-8")))
        self.assert_denied(self.assess(modified, permit=p), "myth_line_changed")

    def test_other_frozen_r1_negative_myths_fail(self):
        # Exact strings copied from the existing R1 negatives, no new clinical cases.
        negatives = (
            "Shared reading must take place only at night.",
            "Books cause babies to skip crawling.",
            "Skipping crawling can cause reading difficulties.",
        )
        for old_negative in negatives:
            with self.subTest(old_negative=old_negative):
                modified = VALID_BILINGUAL_CARD.replace(MYTH, old_negative)
                self.assert_denied(self.assess(modified))

    def test_no_fake_delivery_on_refusal(self):
        # This is a toy sink proving ordering of the *offline* adapter ONLY,
        # not a claim about the live publisher or Telegram behavior.
        events = []
        final_post = VALID_BILINGUAL_CARD.replace("само по себе ", "")
        decision = self.assess(final_post)
        if decision.allowed:
            events.append("mock_delivery")
        self.assertFalse(decision.allowed)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()

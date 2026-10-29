"""Regression for Claude-confirmed A1/A2 using existing frozen R1 negative strings.

No new clinical statements or real approvals. Everything uses ephemeral test keys.
"""
from dataclasses import replace
import unittest

from tests.test_r3a_r2_final_post_contract import VALID_BILINGUAL_CARD
from research.r3a_r2_final_post_contract import (
    ReviewPermit, SignedReview, SourceSnapshot, Span, digest, REJECTION_REASON,
)


class BindingAdversarialTest(unittest.TestCase):
    """Use established fixture without rerunning inherited tests."""

    def setUp(self):
        from tests import test_r3a_r2_final_post_contract as original_tests
        self.fixture = original_tests.R3AR2FinalPostContractTest("test_existing_positive_with_ephemeral_trust_only")
        self.fixture.setUp()
        self.key = self.fixture.key
        self.permit = self.fixture.permit
        self.source = self.fixture.source
        self.unreviewed_post = (
            VALID_BILINGUAL_CARD
            + "\nSkipping crawling can cause reading difficulties."
            + "\nShared reading must take place only at night."
        )

    def assess(self, *args, **kwargs):
        return self.fixture.assess(*args, **kwargs)

    def assert_denied(self, *args, **kwargs):
        return self.fixture.assert_denied(*args, **kwargs)

    def sign(self, *args, **kwargs):
        return self.fixture.sign(*args, **kwargs)

    def test_A1_evil_string_equality_is_not_a_signed_permission(self):
        class EvilStr(str):
            def __eq__(self, other):
                return True
        signed_as_A = replace(
            self.permit,
            approved_final_post_sha256=EvilStr(self.permit.approved_final_post_sha256),
        )
        actual_signature = self.key.sign(ReviewPermit.signed_bytes(signed_as_A))
        signed = SignedReview(signed_as_A, actual_signature)
        decision = self.assess(
            self.unreviewed_post,
            signed_reviews={self.source.source_id: (signed,)},
        )
        self.assert_denied(decision, "invalid_review_object")

    def test_A2_overridden_signed_bytes_cannot_sign_post_B_with_post_A_signature(self):
        original_signed_bytes = ReviewPermit.signed_bytes(self.permit)

        class ForgedPermit(ReviewPermit):
            def signed_bytes(self):
                return original_signed_bytes

        b_permit = ForgedPermit(**{
            **self.permit.__dict__,
            "approved_final_post_sha256": digest(self.unreviewed_post.encode("utf-8")),
        })
        signature_for_A = self.key.sign(original_signed_bytes)
        signed = SignedReview(b_permit, signature_for_A)
        self.assert_denied(self.assess(
            self.unreviewed_post,
            signed_reviews={self.source.source_id: (signed,)},
        ), "invalid_review_object")

    def test_signed_review_subclass_denied(self):
        class CustomSignedReview(SignedReview):
            pass
        s = self.sign()
        self.assert_denied(self.assess(signed_reviews={self.source.source_id: (
            CustomSignedReview(s.permit, s.signature),
        )}), "invalid_review_object")

    def test_source_snapshot_subclass_denied(self):
        class CustomSource(SourceSnapshot):
            pass
        changed_source = CustomSource(**self.source.__dict__)
        self.assert_denied(self.assess(source=changed_source), "untrusted_source_input")

    def test_nested_span_subclass_denied(self):
        class CustomSpan(Span):
            pass
        fake_span = CustomSpan(self.permit.refutation.start, self.permit.refutation.text)
        p = replace(self.permit, refutation=fake_span)
        self.assert_denied(self.assess(permit=p), "invalid_review_object")

    def test_subclassed_text_cannot_bypass_exact_post_digest(self):
        class CustomPost(str):
            def __eq__(self, other):
                return True
        self.assert_denied(self.assess(CustomPost(self.unreviewed_post)), "empty_final_post")

    def test_signed_post_A_cannot_admit_mutated_B_with_plain_types(self):
        # Genuine reviewer signature for A, all plain schema types.
        self.assert_denied(self.assess(
            self.unreviewed_post,
            signed_reviews={self.source.source_id: (self.sign(),)},
        ), "no_unique_permitted_final_post")

    def test_subclassed_digest_on_source_denied(self):
        class CustomEvidence(str):
            def __eq__(self, other):
                return True
        self.assert_denied(self.assess(source=replace(
            self.source, evidence_text=CustomEvidence(self.source.evidence_text)
        )), "missing_source_provenance")

    def test_unsigned_review_derived_from_subclassed_signature_denied(self):
        class CustomBytes(bytes):
            pass
        s = self.sign()
        altered = SignedReview(s.permit, CustomBytes(s.signature))
        self.assert_denied(self.assess(
            signed_reviews={self.source.source_id:(altered,)}
        ), "invalid_review_object")

    def test_reject_malicious_registry_key_subtype(self):
        class CustomKey(str):
            def __eq__(self, other):
                return True
            __hash__ = str.__hash__
        bad_key = CustomKey(self.source.source_id)
        self.assert_denied(self.assess(signed_reviews={
            bad_key: (self.sign(),),
        }), "invalid_registry_keys")

    def test_invalid_utf8_final_post_fails_closed(self):
        invalid = VALID_BILINGUAL_CARD + "\ud800"
        self.assert_denied(self.assess(invalid), "invalid_text_encoding")

    def test_invalid_utf8_evidence_fails_closed(self):
        source = replace(self.source, evidence_text=self.source.evidence_text + "\ud800")
        self.assert_denied(self.assess(source=source), "invalid_text_encoding")


if __name__ == '__main__':
    unittest.main()

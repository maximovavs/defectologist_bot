"""H2 independent key validation regressions (offline only, zero production approvals).

All clinical text is reused from frozen existing R1/R2 fixture; signatures and
key material here are test-only. This test explicitly reproduces Claude's F case.
"""
from dataclasses import replace
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from research.r3a_r2_final_post_contract import (
    SignedReview, digest, _ed25519_prime_order_public_key,
)
from tests import test_r3a_r2_final_post_contract as base_tests

_IDENTITY = bytes.fromhex('01' + '00' * 31)
# Complete 8-element Ed25519 torsion subgroup (canonical encodings).
# Generated independently in a finite-field offline fixture, never as review keys.
_TORSION_KEYS = tuple(bytes.fromhex(x) for x in (
    '0100000000000000000000000000000000000000000000000000000000000000',
    'c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa',
    '0000000000000000000000000000000000000000000000000000000000000000',
    '26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85',
    'ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
    '26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05',
    '0000000000000000000000000000000000000000000000000000000000000080',
    'c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a',
))
_MIXED_SUBGROUP = bytes.fromhex('b2a4b8a3b47d22ea8d86603ab90b393fc1b87399ab55b3806ba4cb815cd509f2')
_RFC_PUBLIC_KEYS = (
    bytes.fromhex('d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a'),
    bytes.fromhex('3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c'),
)


class H2SmallOrderRegression(unittest.TestCase):
    def setUp(self):
        self.fixture = base_tests.R3AR2FinalPostContractTest('test_existing_positive_with_ephemeral_trust_only')
        self.fixture.setUp()

    def test_f_claude_forgery_does_not_admit_unreviewed_post(self):
        # This WAS ACCEPTED under H1 (cryptography 46.0.4); unrelated
        # negative sentences are the exact frozen R1 examples.
        unreviewed = (
            base_tests.VALID_BILINGUAL_CARD + '\nSkipping crawling can cause reading difficulties.'
            + '\nShared reading must take place only at night.'
        )
        permit = replace(
            self.fixture.permit,
            approved_final_post_sha256=digest(unreviewed.encode('utf-8')),
        )
        signed = SignedReview(permit, _IDENTITY + bytes(32))
        decision = self.fixture.assess(
            unreviewed,
            signed_reviews={self.fixture.source.source_id: (signed,)},
            trusted_keys={permit.reviewer_id: _IDENTITY},
        )
        self.fixture.assert_denied(decision, 'reviewer_not_trusted')

    def test_low_order_keys_cannot_admit_even_original_fixture(self):
        signed = SignedReview(self.fixture.permit, _IDENTITY + bytes(32))
        self.fixture.assert_denied(self.fixture.assess(
            signed_reviews={self.fixture.source.source_id: (signed,)},
            trusted_keys={self.fixture.permit.reviewer_id: _IDENTITY},
        ), 'reviewer_not_trusted')

    def test_all_eight_canonical_torsion_public_keys_rejected(self):
        self.assertEqual(len(set(_TORSION_KEYS)), 8)
        for key in _TORSION_KEYS:
            with self.subTest(public_key=key.hex()):
                self.assertFalse(_ed25519_prime_order_public_key(key))

    def test_mixed_subgroup_key_rejected(self):
        self.assertFalse(_ed25519_prime_order_public_key(_MIXED_SUBGROUP))

    def test_noncanonical_y_encodings_rejected(self):
        p = 2**255 - 19
        for y in (p, p+1, p+2, p+18):
            with self.subTest(y=y):
                self.assertFalse(_ed25519_prime_order_public_key(y.to_bytes(32, 'little')))

    def test_noncanonical_identity_with_negative_zero_x_rejected(self):
        self.assertFalse(_ed25519_prime_order_public_key(
            bytes.fromhex('01' + '00'*30 + '80')))

    def test_malformed_sizes_rejected(self):
        for bad in (b'', b'\x01'*31, b'\x01'*33):
            with self.subTest(size=len(bad)):
                self.assertFalse(_ed25519_prime_order_public_key(bad))

    def test_other_types_rejected(self):
        for bad in (None, bytearray(_RFC_PUBLIC_KEYS[0]), memoryview(_RFC_PUBLIC_KEYS[0])):
            with self.subTest(type=str(type(bad))):
                self.assertFalse(_ed25519_prime_order_public_key(bad))

    def test_public_key_subclass_rejected(self):
        class CustomBytes(bytes):
            pass
        self.assertFalse(_ed25519_prime_order_public_key(CustomBytes(_RFC_PUBLIC_KEYS[0])))

    def test_known_rfc8032_vectors_accepted(self):
        for key in _RFC_PUBLIC_KEYS:
            with self.subTest(public_key=key.hex()):
                self.assertTrue(_ed25519_prime_order_public_key(key))

    def test_ephemeral_real_ed25519_keys_accepted(self):
        for _ in range(32):
            pub = Ed25519PrivateKey.generate().public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            self.assertTrue(_ed25519_prime_order_public_key(pub))

    def test_valid_review_still_accepted_only_in_simulation(self):
        d = self.fixture.assess()
        self.assertTrue(d.allowed)
        self.assertEqual(d.detail, 'simulated_review_only')
        d = self.fixture.assess(simulation_only=False)
        self.fixture.assert_denied(d, 'not_wired_to_a_trusted_production_authority')


if __name__ == '__main__':
    unittest.main()

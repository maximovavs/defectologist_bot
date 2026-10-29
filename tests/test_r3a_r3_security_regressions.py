"""Offline P1/P2/P4/P5/P6 adversarial acceptance: only fictional cube fixtures.

Each valid-signature attack is re-enrolled in the independent simulator registry,
so a denial cannot be explained away by a stale current-pack digest.
"""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from dataclasses import replace
from datetime import timedelta

import test_r3a_r3_approved_card_contract as baseline
from r3a_r3_approved_card_contract import SCHEMA, _HISTORICAL_NO_POST


class R3SecurityRegressions(unittest.TestCase):
    def setUp(self):
        self.fx = baseline.R3OfflineWholeCard(methodName="test_two_whole_card_variants_without_resigning")
        self.fx.setUp()

    def attacker_with_signed_record(self, rec, *, authority=None, source=None):
        """Signing and active-registry enrollment are deliberately both valid."""
        auth = self.fx.registered(rec) if authority is None else authority
        src = self.fx.source if source is None else source
        return self.fx.run_decision(signed_pack=self.fx.sign(rec), authority=auth, source=src)

    def test_P1_age_scope_line_injection_valid_signature_denied(self):
        rec = copy.deepcopy(self.fx.record)
        rec["age_scope"] = "TEST_AGE_NOT_CLINICAL\n🟢 Факт: неутверждённое заявление"
        src = replace(self.fx.source, age_scope=rec["age_scope"])
        decision = self.attacker_with_signed_record(rec, source=src)
        self.assertEqual(decision.reason, "NO_POST:signed_approval_invalid")

    def test_P2_all_alternative_separators_controls_and_invisibles_denied(self):
        for extra in ("\n", "\r", "\u2028", "\u2029", "\u0085", "\u202e", "\u200b", "\u2066", "\ufeff", "\u034f", "\u00a0", "\t", "\ufe0f", "\u3164"):
            with self.subTest(codepoint=f"U+{ord(extra):04X}"):
                rec = copy.deepcopy(self.fx.record)
                rec["cards"][0]["fact"] += extra + "НАВЯЗАННЫЙ ТЕКСТ"
                decision = self.attacker_with_signed_record(rec)
                self.assertEqual(decision.reason, "NO_POST:signed_approval_invalid")

    def test_P2_visual_age_scope_must_be_safe(self):
        for control in ("\u2028", "\u200b", "\u202e", "\u0085"):
            with self.subTest(control=repr(control)):
                rec = copy.deepcopy(self.fx.record)
                rec["age_scope"] += control + "🟢 Факт: unsafe"
                source = replace(self.fx.source, age_scope=rec["age_scope"])
                self.assertEqual(self.attacker_with_signed_record(rec, source=source).reason,
                                 "NO_POST:signed_approval_invalid")

    def test_P2_url_display_injection_denied(self):
        rec = copy.deepcopy(self.fx.record)
        rec["canonical_url"] = self.fx.source.canonical_url + "\u202E"
        src = replace(self.fx.source, canonical_url=rec["canonical_url"])
        auth = replace(self.fx.registered(rec), canonical_source_urls={src.source_id:src.canonical_url})
        self.assertEqual(self.attacker_with_signed_record(rec, authority=auth, source=src).reason,
                         "NO_POST:signed_approval_invalid")

    def test_P4_in_code_blocklist_equals_merged_unverified_ledger(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "evidence/source_admission_20261027_historical_UNVERIFIED.json").read_text(encoding="utf-8"))
        self.assertEqual({entry["source_id"] for entry in data["sources"]}, _HISTORICAL_NO_POST)
        self.assertTrue(all(entry["admission"] == "NO_POST" for entry in data["sources"]))
        self.assertEqual(data["counts"]["blocked_for_publication"], 11)

    def test_P4_hardcoded_11_are_exact_historical_inventory(self):
        self.assertEqual(len(_HISTORICAL_NO_POST), 11)
        self.assertEqual(_HISTORICAL_NO_POST, self.fx.auth.historical_source_ids)

    def test_P4_empty_caller_blocklist_cannot_admit_all_11(self):
        for historical_id in _HISTORICAL_NO_POST:
            with self.subTest(source_id=historical_id):
                rec = copy.deepcopy(self.fx.record)
                rec["source_id"] = historical_id
                src = replace(self.fx.source, source_id=historical_id)
                auth = replace(self.fx.registered(rec), historical_source_ids=frozenset(),
                    canonical_source_urls={historical_id:src.canonical_url})
                decision = self.attacker_with_signed_record(rec, authority=auth, source=src)
                self.assertEqual(decision.reason, "NO_POST:historical_11_no_post")

    def test_P5_missing_current_pack_registry_denied(self):
        auth = replace(self.fx.auth, active_packs={})
        self.assertEqual(self.fx.run_decision(authority=auth).reason, "NO_POST:pack_not_current_revision")

    def test_P5_old_revision_under_a_different_pack_id_denied(self):
        old = copy.deepcopy(self.fx.record)
        new = copy.deepcopy(old)
        new["revision"] = 2
        new["pack_id"] = "TEST_ONLY_NEWER_P2"
        new_auth = self.fx.registered(new)
        self.assertTrue(self.attacker_with_signed_record(new, authority=new_auth).allowed)
        self.assertEqual(self.attacker_with_signed_record(old, authority=new_auth).reason,
                         "NO_POST:pack_not_current_revision")
        another_old = copy.deepcopy(old)
        another_old["pack_id"] = "TEST_ONLY_OLD_DIFFERENT_ID"
        self.assertEqual(self.attacker_with_signed_record(another_old, authority=new_auth).reason,
                         "NO_POST:pack_not_current_revision")

    def test_P5_same_revision_different_payload_denied_even_if_signed(self):
        rec = copy.deepcopy(self.fx.record)
        rec["cards"][0]["fact"] = "Совершенно другой проверенный текст про кубики."
        decision = self.attacker_with_signed_record(rec, authority=self.fx.auth)
        self.assertEqual(decision.reason, "NO_POST:pack_not_current_revision")

    def test_P5_invalid_registry_shapes_denied(self):
        scope = self.fx.scope_for(self.fx.record)
        base = self.fx.identity_for(self.fx.record)
        for value in ((True,base[1],base[2]), (1,base[1],"a"*64), (1,base[1],base[2],"surplus"), [1,base[1],base[2]]):
            # The SHA variant with 64 'a' is structurally valid: it denies at
            # equality instead of registry-shape validation.
            with self.subTest(value=repr(value)):
                auth = replace(self.fx.auth, active_packs={scope:value})
                self.assertFalse(self.fx.run_decision(authority=auth).allowed)

    def test_P6_century_long_authorization_denied_with_valid_registry(self):
        rec = copy.deepcopy(self.fx.record)
        rec["valid_until_utc"] = "2126-11-30T00:00:00Z"
        decision = self.attacker_with_signed_record(rec)
        self.assertEqual(decision.reason, "NO_POST:signed_approval_invalid")

    def test_P6_negative_time_window_denied(self):
        rec = copy.deepcopy(self.fx.record)
        rec["valid_from_utc"] = "2026-11-30T00:00:00Z"
        rec["valid_until_utc"] = "2026-10-01T00:00:00Z"
        self.assertEqual(self.attacker_with_signed_record(rec).reason,"NO_POST:signed_approval_invalid")

    def test_P6_exactly_90_day_signed_window_is_allowed(self):
        rec = copy.deepcopy(self.fx.record)
        rec["valid_from_utc"] = "2026-10-01T00:00:00Z"
        rec["valid_until_utc"] = "2026-12-30T00:00:00Z"
        self.assertTrue(self.attacker_with_signed_record(rec).allowed)

    def test_P6_91_day_window_denied(self):
        rec = copy.deepcopy(self.fx.record)
        rec["valid_from_utc"] = "2026-10-01T00:00:00Z"
        rec["valid_until_utc"] = "2026-12-31T00:00:00Z"
        self.assertEqual(self.attacker_with_signed_record(rec).reason, "NO_POST:signed_approval_invalid")

    def test_legacy_v1_permit_is_not_silently_accepted(self):
        rec = copy.deepcopy(self.fx.record)
        rec["schema"] = "logopedia-r3a-r3-whole-card-v1"
        self.assertEqual(self.attacker_with_signed_record(rec).reason, "NO_POST:signed_approval_invalid")

    def test_new_registry_scope_is_channel_and_topic_bound(self):
        rec = copy.deepcopy(self.fx.record)
        rec["channel_id"] = "DIFFERENT_CHANNEL"
        self.assertEqual(self.attacker_with_signed_record(rec, authority=self.fx.auth).reason, "NO_POST:pack_not_current_revision")

    def test_pack_with_bad_unicode_in_pack_id_is_denied(self):
        rec = copy.deepcopy(self.fx.record)
        rec["pack_id"] = "TEST_ONLY\u200b"
        self.assertTrue(self.attacker_with_signed_record(rec).reason.startswith("NO_POST:"))

    def test_H2_default_ignorable_and_braille_blank_with_real_signatures(self):
        # Non-C/Z default-ignorable characters and a visually blank symbol
        # were previously accepted even with an honest registry.
        cases = ("\u17b4", "\u17b5", "\u180b", "\u180f", "\U000e0100",
                 "\u2800", "\u05d0", "\u0627", "\u0661")
        for x in cases:
            with self.subTest(codepoint=f"U+{ord(x):04X}"):
                rec = copy.deepcopy(self.fx.record)
                rec["cards"][0]["fact"] += x
                self.assertEqual(self.attacker_with_signed_record(rec).reason,
                                 "NO_POST:signed_approval_invalid")

    def test_H2_ignorable_in_rendered_age_scope_is_denied(self):
        for x in ("\u17b4", "\u180f", "\U000e0100", "\u2800", "\u05d0"):
            with self.subTest(codepoint=f"U+{ord(x):04X}"):
                rec = copy.deepcopy(self.fx.record)
                rec["age_scope"] += x
                src = replace(self.fx.source, age_scope=rec["age_scope"])
                self.assertEqual(self.attacker_with_signed_record(rec, source=src).reason,
                                 "NO_POST:signed_approval_invalid")

    def test_H2_HTML_meta_characters_denied_in_rendered_card_field(self):
        for value in ("<b>new</b>", "&amp;", "A > B"):
            with self.subTest(value=value):
                rec = copy.deepcopy(self.fx.record)
                rec["cards"][0]["fact"] += value
                self.assertEqual(self.attacker_with_signed_record(rec).reason,
                                 "NO_POST:signed_approval_invalid")

    def test_H2_HTML_meta_characters_denied_in_rendered_age_and_source_url(self):
        rec = copy.deepcopy(self.fx.record)
        rec["age_scope"] += " & siblings"
        src = replace(self.fx.source, age_scope=rec["age_scope"])
        self.assertEqual(self.attacker_with_signed_record(rec, source=src).reason,
                         "NO_POST:signed_approval_invalid")
        rec = copy.deepcopy(self.fx.record)
        rec["canonical_url"] += "?a=1&b=2"
        src = replace(self.fx.source, canonical_url=rec["canonical_url"])
        auth = replace(self.fx.registered(rec), canonical_source_urls={src.source_id:src.canonical_url})
        self.assertEqual(self.attacker_with_signed_record(rec, authority=auth, source=src).reason,
                         "NO_POST:signed_approval_invalid")

    def test_H2_typical_russian_and_emojis_remain_accepted(self):
        rec = copy.deepcopy(self.fx.record)
        rec["cards"][0]["fact"] = "Ребёнок видит кубики: красный, синий. 🧩 1–2 шага!"
        self.assertTrue(self.attacker_with_signed_record(rec).allowed)

    def test_H2_all_frozen_default_ignorable_ranges_are_denied(self):
        from r3a_r3_approved_card_contract import _is_default_ignorable, _valid_string
        for low, high in (
            (0x00AD,0x00AD),(0x034F,0x034F),(0x061C,0x061C),
            (0x115F,0x1160),(0x17B4,0x17B5),(0x180B,0x180F),
            (0x200B,0x200F),(0x202A,0x202E),(0x2060,0x206F),
            (0x3164,0x3164),(0xFE00,0xFE0F),(0xFEFF,0xFEFF),
            (0xFFA0,0xFFA0),(0xFFF0,0xFFF8),(0x1BCA0,0x1BCA3),
            (0x1D173,0x1D17A),(0xE0000,0xE0FFF),
        ):
            for cp in (low,high):
                with self.subTest(codepoint=f"U+{cp:04X}"):
                    self.assertTrue(_is_default_ignorable(cp))
                    self.assertFalse(_valid_string("TEXT"+chr(cp)))

    def test_regular_approved_variants_still_work(self):
        for card_id in ("a","b"):
            with self.subTest(card_id=card_id):
                self.assertTrue(self.fx.run_decision(card_id=card_id).allowed)

if __name__ == "__main__": unittest.main()

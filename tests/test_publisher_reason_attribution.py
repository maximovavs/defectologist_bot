from __future__ import annotations

import inspect
import unittest
from datetime import datetime
from unittest.mock import patch

from src.publisher import run_publisher as publisher


OWNERSHIP_REASONS = (
    "myth_claim_not_grounded",
    "parent_age_not_grounded",
    "parent_modality_not_grounded",
    "parent_diagnostic_role_violation",
    "parent_false_hearing_inference",
    "parent_risky_oral_manipulation",
    "exercise_coherence_violation",
    "parent_professional_role_violation",
)


class PublisherReasonOwnershipTest(unittest.TestCase):
    def test_all_policy_ownership_reasons_are_explicit_soft_validation_reasons(self) -> None:
        self.assertEqual(set(publisher.POLICY_OWNERSHIP_REASONS), set(OWNERSHIP_REASONS))
        for reason in OWNERSHIP_REASONS:
            with self.subTest(reason=reason):
                self.assertIn(reason, publisher.SOFT_SKIP_REASONS)
                self.assertIn(reason, publisher.VALIDATION_SKIP_REASONS)
                self.assertEqual(publisher._skip_kind(reason), "soft")

    def test_existing_hard_transport_reasons_remain_hard(self) -> None:
        for reason in (
            "source_fetch_failed",
            "evidence_fetch_failed",
            "llm_timeout",
            "llm_failed",
            "visual_build_failed",
            "telegram_send_failed",
            "max_run_seconds",
        ):
            with self.subTest(reason=reason):
                self.assertIn(reason, publisher.HARD_SKIP_REASONS)
                self.assertEqual(publisher._skip_kind(reason), "hard")


class PublisherValidationEnvelopeTest(unittest.TestCase):
    def test_all_four_invalid_provider_envelopes_preserve_every_ownership_reason(self) -> None:
        envelopes = (
            "invalid_groq:{reason}",
            "invalid_groq_retry:{reason}",
            "invalid_gemini:{reason}",
            "invalid_gemini_retry:{reason}",
        )
        for reason in OWNERSHIP_REASONS:
            for envelope in envelopes:
                with self.subTest(reason=reason, envelope=envelope):
                    note = envelope.format(reason=reason)
                    self.assertEqual(publisher._extract_validation_skip_reason(note), reason)
                    self.assertEqual(publisher._resolve_llm_skip(note), (reason, "llm_validation"))

    def test_nested_provider_note_preserves_inner_policy_reason(self) -> None:
        note = (
            "gemini_failed:temporary provider error | "
            "groq=invalid_groq_retry:parent_age_not_grounded"
        )
        self.assertEqual(
            publisher._extract_validation_skip_reason(note),
            "parent_age_not_grounded",
        )
        self.assertEqual(
            publisher._resolve_llm_skip(note),
            ("parent_age_not_grounded", "llm_validation"),
        )

    def test_p2d_fail_closed_wrapper_with_nested_invalid_reason_preserves_owner(self) -> None:
        note = (
            "p2d_fail_closed:exercise_coherence_violation:"
            "invalid_groq_retry:exercise_coherence_violation"
        )
        self.assertEqual(
            publisher._extract_validation_skip_reason(note),
            "exercise_coherence_violation",
        )
        self.assertEqual(
            publisher._resolve_llm_skip(note),
            ("exercise_coherence_violation", "llm_validation"),
        )

    def test_p2d_fail_closed_wrapper_without_nested_invalid_reason_preserves_owner(self) -> None:
        note = "p2d_fail_closed:parent_professional_role_violation:provider fallback blocked"
        self.assertEqual(
            publisher._extract_validation_skip_reason(note),
            "parent_professional_role_violation",
        )
        self.assertEqual(
            publisher._resolve_llm_skip(note),
            ("parent_professional_role_violation", "llm_validation"),
        )

    def test_legacy_modality_repair_exception_maps_in_one_canonical_place(self) -> None:
        note = "groq_failed_after_modality_repair:repair failed"
        self.assertEqual(
            publisher._extract_validation_skip_reason(note),
            "parent_modality_not_grounded",
        )
        self.assertEqual(
            publisher._resolve_llm_skip(note),
            ("parent_modality_not_grounded", "llm_validation"),
        )

    def test_legacy_diagnostic_repair_exception_maps_in_one_canonical_place(self) -> None:
        note = "groq_failed_after_diagnostic_repair:repair failed"
        self.assertEqual(
            publisher._extract_validation_skip_reason(note),
            "parent_diagnostic_role_violation",
        )
        self.assertEqual(
            publisher._resolve_llm_skip(note),
            ("parent_diagnostic_role_violation", "llm_validation"),
        )

    def test_unknown_unregistered_validation_text_stays_generic(self) -> None:
        note = "invalid_groq_retry:invented_policy_reason"
        self.assertEqual(publisher._extract_validation_skip_reason(note), "")
        self.assertEqual(publisher._resolve_llm_skip(note), ("llm_invalid_output", ""))

    def test_quota_resolution_is_unchanged(self) -> None:
        self.assertEqual(
            publisher._resolve_llm_skip("gemini_quota_exhausted_cached"),
            ("gemini_quota_exhausted_cached", ""),
        )
        self.assertEqual(
            publisher._resolve_llm_skip("gemini_quota_exhausted"),
            ("gemini_quota_exhausted", ""),
        )

    def test_exact_policy_reason_beats_quota_fallback(self) -> None:
        note = (
            "invalid_groq_retry:parent_modality_not_grounded | "
            "gemini_quota_exhausted"
        )
        self.assertEqual(
            publisher._resolve_llm_skip(note),
            ("parent_modality_not_grounded", "llm_validation"),
        )

    def test_existing_prefixed_validation_reason_is_preserved(self) -> None:
        note = "invalid_gemini_retry:unsupported_mechanism_claim:brain activation"
        self.assertEqual(
            publisher._extract_validation_skip_reason(note),
            "unsupported_mechanism_claim:brain activation",
        )


class PublisherStageTelemetryTest(unittest.TestCase):
    def _containers(self):
        return {}, {}, [], {}

    def test_same_canonical_reason_can_have_distinct_stage_provenance(self) -> None:
        soft, hard, samples, stages = self._containers()
        publisher._record_skip(
            "myth_topic_mismatch",
            "https://example.org/pre",
            soft,
            hard,
            samples,
            stage_skip_reasons=stages,
            stage="pre_llm",
        )
        publisher._record_skip(
            "myth_topic_mismatch",
            "https://example.org/post",
            soft,
            hard,
            samples,
            stage_skip_reasons=stages,
            stage="llm_validation",
        )

        self.assertEqual(soft, {"myth_topic_mismatch": 2})
        self.assertEqual(hard, {})
        self.assertEqual(stages["pre_llm"], {"myth_topic_mismatch": 1})
        self.assertEqual(stages["llm_validation"], {"myth_topic_mismatch": 1})

    def test_required_stage_mappings_are_recorded_without_changing_reason(self) -> None:
        soft, hard, samples, stages = self._containers()
        cases = (
            ("dup_url_recent", "url_cooldown"),
            ("no_evidence_short", "evidence"),
            ("source_authority_required", "source_authority"),
            ("myth_evidence_missing_refutation_anchor", "pre_llm"),
            ("parent_professional_role_violation", "llm_validation"),
        )
        for index, (reason, stage) in enumerate(cases):
            publisher._record_skip(
                reason,
                f"https://example.org/{index}",
                soft,
                hard,
                samples,
                stage_skip_reasons=stages,
                stage=stage,
            )

        self.assertEqual(sum(soft.values()), len(cases))
        self.assertEqual(hard, {})
        for reason, stage in cases:
            self.assertEqual(stages[stage][reason], 1)

    def test_stage_tracking_does_not_change_soft_or_hard_totals(self) -> None:
        soft, hard, samples, stages = self._containers()
        publisher._record_skip(
            "dup_url_recent",
            "https://example.org/soft",
            soft,
            hard,
            samples,
            stage_skip_reasons=stages,
            stage="url_cooldown",
        )
        publisher._record_skip(
            "llm_timeout",
            "https://example.org/hard",
            soft,
            hard,
            samples,
            stage_skip_reasons=stages,
        )
        self.assertEqual(sum(soft.values()), 1)
        self.assertEqual(sum(hard.values()), 1)

    def test_unknown_stage_is_not_fabricated(self) -> None:
        soft, hard, samples, stages = self._containers()
        publisher._record_skip(
            "dup_url_recent",
            "https://example.org/soft",
            soft,
            hard,
            samples,
            stage_skip_reasons=stages,
            stage="invented_stage",
        )
        self.assertEqual(soft, {"dup_url_recent": 1})
        self.assertEqual(stages, {})


class PublisherDiagnosticRenderingTest(unittest.TestCase):
    def _render(self, *, stages=None):
        with patch.object(publisher, "gemini_text_provider_status", return_value="unavailable"):
            return publisher._build_posted_zero_alert_plain(
                now=datetime(2026, 8, 19, 12, 0, 0),
                day="WE",
                week_key="2026-W34",
                audience="both",
                provider="auto",
                soft_skip_reasons={"myth_topic_mismatch": 2, "dup_url_recent": 1},
                hard_skip_reasons={"llm_timeout": 1},
                samples=[],
                state_scope="prod",
                db_name="publication_history.sqlite3",
                attempted_rubrics=["myth_fact"],
                topic_preference="auto",
                stage_skip_reasons=stages,
            )

    def test_diagnostic_preserves_existing_reason_counts_and_adds_stage_breakdown(self) -> None:
        text = self._render(
            stages={
                "url_cooldown": {"dup_url_recent": 1},
                "pre_llm": {"myth_topic_mismatch": 1},
                "llm_validation": {"myth_topic_mismatch": 1},
            }
        )
        self.assertIn("Soft skips: 3 | Hard skips: 1", text)
        self.assertIn("• myth_topic_mismatch: 2", text)
        self.assertIn("• dup_url_recent: 1", text)
        self.assertIn("• llm_timeout: 1", text)
        self.assertIn("Skip stage attribution:", text)
        self.assertIn("• url_cooldown | dup_url_recent: 1", text)
        self.assertIn("• pre_llm | myth_topic_mismatch: 1", text)
        self.assertIn("• llm_validation | myth_topic_mismatch: 1", text)

    def test_diagnostic_omits_stage_section_when_no_stage_data_exists(self) -> None:
        text = self._render(stages={})
        self.assertNotIn("Skip stage attribution:", text)

    def test_stage_diagnostic_does_not_render_provider_exception_payloads(self) -> None:
        text = self._render(stages={"llm_validation": {"parent_modality_not_grounded": 1}})
        self.assertNotIn("repair failed", text)
        self.assertNotIn("API_KEY", text)


class PublisherP2EWiringTest(unittest.TestCase):
    def test_candidate_loop_uses_only_required_stage_attribution_points(self) -> None:
        source = inspect.getsource(publisher.amain)
        self.assertIn('note("dup_url_recent", canon, stage="url_cooldown")', source)
        self.assertIn('note("no_evidence_short", canon, stage="evidence")', source)
        self.assertIn('note("source_authority_required", canon, stage="source_authority")', source)
        self.assertIn('note(myth_evidence_reason, canon, stage="pre_llm")', source)
        self.assertIn('note(pro_evidence_reason, canon, stage="pre_llm")', source)
        self.assertIn("skip_reason, skip_stage = _resolve_llm_skip(llm_note)", source)
        self.assertIn("kind = note(skip_reason, canon, stage=skip_stage)", source)

    def test_legacy_envelope_aliases_are_centralized_outside_candidate_loop(self) -> None:
        source = inspect.getsource(publisher.amain)
        self.assertNotIn("groq_failed_after_modality_repair", source)
        self.assertNotIn("groq_failed_after_diagnostic_repair", source)
        self.assertEqual(
            publisher.LEGACY_VALIDATION_NOTE_ALIASES,
            {
                "groq_failed_after_modality_repair:": "parent_modality_not_grounded",
                "groq_failed_after_diagnostic_repair:": "parent_diagnostic_role_violation",
            },
        )


if __name__ == "__main__":
    unittest.main()

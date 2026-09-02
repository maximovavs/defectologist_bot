from io import BytesIO
import json
import os
import unittest
from unittest.mock import Mock, patch

from src.services.visual_pipeline import (
    VISUAL_QA_HARD_REASONS,
    VISUAL_STYLE_RETRY_MARKER,
    VISUAL_STYLE_TAIL,
    _enforce_object_visual_qa,
    _parse_visual_qa_response,
    _prepare_pollinations_prompt,
    _safe_visual_qa,
    build_post_visual,
    build_visual_retry_prompt,
    evaluate_visual_quality,
)


def _qa_response(payload):
    response = Mock(status_code=200, text="")
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    return response


def _human_qa(**overrides):
    payload = {
        "pass": True,
        "reason": "ok",
        "people_count": 2,
        "adult_count": 1,
        "child_count": 1,
        "ppe_detected": False,
        "text_detected": False,
        "ui_artifact_detected": False,
        "illustration_style_match": True,
        "character_roles_match": True,
        "action_match": True,
    }
    payload.update(overrides)
    return payload


def _object_qa(**overrides):
    payload = {
        "status": "pass",
        "pass": True,
        "reason": "ok",
        "people_count": 0,
        "adult_count": 0,
        "child_count": 0,
        "ppe_detected": False,
        "text_detected": False,
        "illustration_style_match": True,
    }
    payload.update(overrides)
    return payload


class HumanTextUiStyleVisualQaTest(unittest.TestCase):
    def _production_result(self, payload):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "OFFLINE_TEST_KEY", "GEMINI_VISUAL_QA_API_KEY": ""},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            return_value=_qa_response(payload),
        ) as request:
            result = _safe_visual_qa(
                evaluate_visual_quality,
                BytesIO(b"offline-image"),
                rubric_id="tip_of_day",
                audience="parents",
                expected_prompt=(
                    "Expected roles: exactly one adult parent and one child.\n"
                    "Expected action: the parent and child point to a picture card together."
                ),
            )
        self.assertEqual(request.call_count, 1)
        return result, request

    def test_human_qa_prompt_requires_surface_fields_and_taxonomy(self):
        result, request = self._production_result(_human_qa())
        prompt_text = request.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"].lower()

        self.assertEqual(result["status"], "pass")
        for phrase in (
            "text_detected",
            "ui_artifact_detected",
            "illustration_style_match",
            "gibberish",
            "pseudo-text",
            "malformed letters or glyphs",
            "pseudo-cyrillic",
            "pseudo-latin",
            "captions",
            "labels",
            "logos",
            "brand marks",
            "watermarks",
            "app or browser frame",
            "screenshot-like chrome",
            "2d hand-painted watercolor and gouache",
        ):
            with self.subTest(required=phrase):
                self.assertIn(phrase, prompt_text)

    def test_human_provider_prompts_block_surface_artifacts(self):
        prompt = f"An adult and child point to a picture card together. {VISUAL_STYLE_TAIL}"
        retry = build_visual_retry_prompt(
            prompt,
            rubric_id="tip_of_day",
            qa_reason="style_mismatch",
            expected_action="the adult and child point to a picture card together",
        )
        provider_prompts = (_prepare_pollinations_prompt(prompt), _prepare_pollinations_prompt(retry))

        self.assertIn(VISUAL_STYLE_RETRY_MARKER, retry)
        for provider_prompt in provider_prompts:
            lower = provider_prompt.lower()
            self.assertNotIn(VISUAL_STYLE_RETRY_MARKER, provider_prompt)
            for phrase in (
                "no readable text",
                "gibberish",
                "pseudo-text",
                "malformed glyph-like writing",
                "captions",
                "labels",
                "logos",
                "brand marks",
                "watermarks",
                "ui/interface elements",
                "browser/app/screenshot chrome",
            ):
                with self.subTest(prompt=provider_prompt[:20], blocked=phrase):
                    self.assertIn(phrase, lower)

    def test_text_detected_is_hard_failure(self):
        result, _ = self._production_result(_human_qa(text_detected=True))

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "human_text_detected")
        self.assertIn("human_text_detected", VISUAL_QA_HARD_REASONS)

    def test_missing_and_malformed_text_verdicts_fail_closed(self):
        missing = _human_qa()
        missing.pop("text_detected")
        for label, payload in (("missing", missing), ("malformed", _human_qa(text_detected="maybe"))):
            with self.subTest(verdict=label):
                result, _ = self._production_result(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], "human_text_unknown")

    def test_ui_artifact_detected_is_hard_failure(self):
        result, _ = self._production_result(_human_qa(ui_artifact_detected=True))

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "human_ui_artifact_detected")
        self.assertIn("human_ui_artifact_detected", VISUAL_QA_HARD_REASONS)

    def test_missing_and_malformed_ui_verdicts_fail_closed(self):
        missing = _human_qa()
        missing.pop("ui_artifact_detected")
        for label, payload in (
            ("missing", missing),
            ("malformed", _human_qa(ui_artifact_detected="uncertain")),
        ):
            with self.subTest(verdict=label):
                result, _ = self._production_result(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], "human_ui_artifact_unknown")

    def test_style_mismatch_is_hard_failure(self):
        result, _ = self._production_result(_human_qa(illustration_style_match=False))

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "style_mismatch")
        self.assertIn("style_mismatch", VISUAL_QA_HARD_REASONS)

    def test_missing_and_malformed_style_verdicts_fail_closed(self):
        missing = _human_qa()
        missing.pop("illustration_style_match")
        for label, payload in (
            ("missing", missing),
            ("malformed", _human_qa(illustration_style_match="uncertain")),
        ):
            with self.subTest(verdict=label):
                result, _ = self._production_result(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], "style_match_unknown")

    def test_exact_safe_surface_verdicts_pass(self):
        result, _ = self._production_result(_human_qa())

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertFalse(result["text_detected"])
        self.assertFalse(result["ui_artifact_detected"])
        self.assertTrue(result["illustration_style_match"])

    def test_existing_count_ppe_and_anatomical_reasons_keep_priority(self):
        cases = (
            (
                "character_counts_unknown",
                _human_qa(
                    people_count="unknown",
                    adult_count="unknown",
                    child_count="unknown",
                    text_detected=True,
                ),
            ),
            ("unexpected_ppe", _human_qa(ppe_detected=True, text_detected=True)),
            (
                "deformed_hands",
                _human_qa(
                    **{
                        "pass": False,
                        "reason": "deformed_hands",
                        "text_detected": True,
                        "ui_artifact_detected": True,
                        "illustration_style_match": False,
                    }
                ),
            ),
        )
        for expected_reason, payload in cases:
            with self.subTest(reason=expected_reason):
                result, _ = self._production_result(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], expected_reason)

    def test_p3a_and_p3b_reasons_keep_priority(self):
        missing_role = _human_qa(text_detected=True)
        missing_role.pop("character_roles_match")
        missing_action = _human_qa(text_detected=True)
        missing_action.pop("action_match")
        cases = (
            ("wrong_character_roles", _human_qa(character_roles_match=False, text_detected=True)),
            ("character_roles_unknown", missing_role),
            ("action_mismatch", _human_qa(action_match=False, text_detected=True)),
            ("action_match_unknown", missing_action),
        )
        for expected_reason, payload in cases:
            with self.subTest(reason=expected_reason):
                result, _ = self._production_result(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], expected_reason)

    def test_parser_preserves_surface_booleans_and_unknowns(self):
        parsed = _parse_visual_qa_response(json.dumps(_human_qa()))
        self.assertFalse(parsed["text_detected"])
        self.assertFalse(parsed["ui_artifact_detected"])
        self.assertTrue(parsed["illustration_style_match"])

        incomplete = _human_qa()
        incomplete.pop("text_detected")
        incomplete.pop("ui_artifact_detected")
        incomplete.pop("illustration_style_match")
        unknown = _parse_visual_qa_response(json.dumps(incomplete))
        self.assertEqual(unknown["text_detected"], "unknown")
        self.assertEqual(unknown["ui_artifact_detected"], "unknown")
        self.assertEqual(unknown["illustration_style_match"], "unknown")

    def test_object_semantics_remain_unchanged(self):
        cases = (
            ("ok", True, _object_qa(ui_artifact_detected=True)),
            ("object_contains_text", False, _object_qa(text_detected=True)),
            ("object_style_mismatch", False, _object_qa(illustration_style_match=False)),
        )
        for expected_reason, expected_pass, payload in cases:
            with self.subTest(reason=expected_reason):
                result = _enforce_object_visual_qa(payload)
                self.assertEqual(result["pass"], expected_pass)
                self.assertEqual(result["reason"], expected_reason)

    def test_style_mismatch_uses_one_targeted_human_retry(self):
        qa_results = iter(
            [
                {
                    "status": "fail",
                    **_human_qa(**{"pass": False, "reason": "style_mismatch"}),
                },
                {"status": "pass", **_human_qa()},
            ]
        )
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"human-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            buffer, meta = build_post_visual(
                title="Совместная игра",
                day_key="2026-09-02",
                image_prompt="the parent and child point to a picture card together",
                rubric_id="tip_of_day",
                audience="parents",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )

        self.assertEqual(len(prompts), 2)
        self.assertNotIn(VISUAL_STYLE_RETRY_MARKER, prompts[0])
        self.assertIn(VISUAL_STYLE_RETRY_MARKER, prompts[1])
        self.assertEqual(buffer.getvalue(), b"human-2")
        self.assertEqual(meta["mode"], "ai_human_retry")
        self.assertEqual(meta["fallback_stage"], "human_retry")
        self.assertEqual(meta["visual_qa_attempts"], "2")
        self.assertEqual(meta["object_generation_status"], "not_run")

    def test_text_and_ui_failures_keep_bounded_object_and_text_fallback_ladder(self):
        for human_reason in ("human_text_detected", "human_ui_artifact_detected"):
            with self.subTest(reason=human_reason):
                human_failure = {
                    "status": "fail",
                    **_human_qa(**{"pass": False, "reason": human_reason}),
                }
                qa_results = iter(
                    [
                        human_failure,
                        human_failure,
                        _object_qa(text_detected=True),
                        _object_qa(text_detected=True),
                    ]
                )
                downloads = [
                    (BytesIO(b"human-1"), {"attempts_used": "1"}),
                    (BytesIO(b"human-2"), {"attempts_used": "1"}),
                    (BytesIO(b"object-1"), {"attempts_used": "1"}),
                    (BytesIO(b"object-2"), {"attempts_used": "1"}),
                ]
                with patch(
                    "src.services.visual_pipeline.download_pollinations_image_with_meta",
                    side_effect=downloads,
                ) as download, patch(
                    "src.services.visual_pipeline.build_fallback_cover_buffer",
                    return_value=BytesIO(b"text-fallback"),
                ) as text_fallback:
                    buffer, meta = build_post_visual(
                        title="Книги и новые слова",
                        day_key="2026-09-02",
                        image_prompt="the parent and child point to a picture card together",
                        rubric_id="tip_of_day",
                        audience="parents",
                        visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
                    )

                self.assertEqual(download.call_count, 4)
                text_fallback.assert_called_once()
                self.assertEqual(buffer.getvalue(), b"text-fallback")
                self.assertEqual(meta["mode"], "text_fallback")
                self.assertEqual(meta["fallback_stage"], "text")
                self.assertEqual(meta["visual_qa_attempts"], "2")
                self.assertEqual(meta["object_generation_attempts"], "2")
                self.assertEqual(meta["final_reason"], "object_fallback_rejected")


if __name__ == "__main__":
    unittest.main()

from io import BytesIO
import json
import os
import unittest
from unittest.mock import Mock, patch

from src.services.visual_pipeline import (
    GEMINI_VISUAL_QA_TIMEOUT_SECONDS,
    VISUAL_STYLE_TAIL,
    _enforce_object_visual_qa,
    _object_scene_category,
    _prepare_pollinations_prompt,
    build_object_only_visual_prompt,
    evaluate_visual_quality,
)


def _qa_response(payload):
    response = Mock(status_code=200)
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps(payload)}
                    ]
                }
            }
        ]
    }
    return response


class VisualPromptResilienceTest(unittest.TestCase):
    def test_provider_human_prompt_removes_specific_ppe_tokens(self):
        internal = f"Adult and child speech activity. {VISUAL_STYLE_TAIL}"
        provider = _prepare_pollinations_prompt(internal)
        lower = provider.lower()

        self.assertLess(len(provider), len(internal))
        for phrase in ("surgical masks", "face shields", "high-vis vests", "hard hats", "respirators"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lower)
        self.assertIn("no medical or industrial context or equipment", lower)
        self.assertIn("watercolor", lower)
        self.assertIn("gouache", lower)

    def test_provider_object_prompt_removes_ppe_catalogue_and_keeps_scene(self):
        internal = build_object_only_visual_prompt(
            "Включайте речь в обычные дела",
            "question_week",
            variation_key="2026-08-07",
        )
        provider = _prepare_pollinations_prompt(internal)
        lower = provider.lower()

        self.assertIn("object-only still life", lower)
        self.assertIn("laundry basket", lower)
        for phrase in ("surgical mask", "face shield", "reflective vest", "hard hat", "respirator"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lower)

    def test_recent_titles_map_to_specific_object_categories(self):
        self.assertEqual(
            _object_scene_category("Учимся слушать", "bilingual_corner"),
            "hearing_sounds_music",
        )
        self.assertEqual(
            _object_scene_category("Включайте речь в обычные дела", "question_week"),
            "household_routines",
        )
        self.assertEqual(
            _object_scene_category("Раннее понимание слов в действии", "myth_fact"),
            "books_vocab_phrases_stories",
        )

    def test_visual_qa_default_timeout_is_at_least_twenty_seconds(self):
        self.assertGreaterEqual(GEMINI_VISUAL_QA_TIMEOUT_SECONDS, 20)

    def test_production_explicit_dedicated_key_prefers_general_key(self):
        payload = {
            "pass": True,
            "reason": "ok",
            "people_count": 2,
            "adult_count": 1,
            "child_count": 1,
            "ppe_detected": False,
            "text_detected": False,
        }
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            return_value=_qa_response(payload),
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                gemini_api_key="VISUAL_SECRET",
            )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["qa_key_source"], "general")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.kwargs["headers"]["x-goog-api-key"], "GENERAL_SECRET")

    def test_object_text_is_hard_reject_when_reported(self):
        result = _enforce_object_visual_qa(
            {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 0,
                "adult_count": 0,
                "child_count": 0,
                "ppe_detected": False,
                "text_detected": True,
            }
        )

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "object_contains_text")

    def test_object_qa_uses_short_dedicated_prompt(self):
        payload = {
            "pass": True,
            "reason": "ok",
            "people_count": 0,
            "adult_count": 0,
            "child_count": 0,
            "ppe_detected": False,
            "text_detected": False,
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            return_value=_qa_response(payload),
        ) as request:
            result = evaluate_visual_quality(BytesIO(b"image"), qa_mode="object")

        qa_prompt = request.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertEqual(result["status"], "pass")
        self.assertIn("do not evaluate the educational topic or artistic style", qa_prompt.lower())
        self.assertNotIn("for parent rubrics", qa_prompt.lower())
        self.assertLess(len(qa_prompt), 1000)


if __name__ == "__main__":
    unittest.main()

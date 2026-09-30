from io import BytesIO
from contextlib import redirect_stdout
import hashlib
import os
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

from src.services import visual_pipeline
from src.services.visual_pipeline import (
    DEFAULT_GEMINI_VISUAL_QA_FALLBACK_MODEL,
    DEFAULT_GEMINI_VISUAL_QA_MODEL,
    OBJECT_PROVIDER_COMPOSITIONS,
    OBJECT_PROVIDER_NEGATIVES,
    OBJECT_SCENE_CATEGORIES,
    OBJECT_SCENE_MARKER_RE,
    OBJECT_TEXT_FAILURE_REASON,
    OBJECT_TEXT_SAFE_COMPOSITION_BANNED_TOKENS,
    OBJECT_TEXT_SAFE_PROVIDER_COMPOSITIONS,
    OBJECT_SCENE_CONTEXT_STYLE_MARKER,
    OBJECT_TEXT_SAFE_RETRY_CATEGORY,
    OBJECT_TEXT_SAFE_RETRY_SOURCE_CATEGORY,
    _VisualQABuildCircuit,
    VISUAL_QA_HARD_REASONS,
    VISUAL_STYLE_TAIL,
    VisualBrief,
    _clean_cover_title,
    _compile_visual_prompt,
    _enforce_object_visual_qa,
    _object_scene_category,
    _parse_compiled_visual_prompt,
    _object_provider_composition,
    _object_scene_context_semantics,
    _prepare_pollinations_prompt,
    _text_safe_object_retry_category,
    build_object_provider_prompt,
    _visual_qa_model_candidates,
    _visual_qa_key_candidates,
    build_object_only_visual_prompt,
    build_post_visual,
    build_visual_retry_prompt,
    build_visual_role_rule,
    evaluate_visual_quality,
)


def _qa_response(status_code, payload=None, text=""):
    response = Mock(status_code=status_code)
    response.text = text
    response.json.return_value = payload or {
        "candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1, "ppe_detected": false, "text_detected": false, "ui_artifact_detected": false, "illustration_style_match": true, "character_roles_match": true, "action_match": true}'}]}}]
    }
    return response


def _object_pass():
    return {
        "status": "pass",
        "pass": True,
        "reason": "ok",
        "people_count": 0,
        "adult_count": 0,
        "child_count": 0,
        "ppe_detected": False,
        # Object QA only passes an image that is verifiably in the channel's
        # watercolor/gouache illustration style.
        "illustration_style_match": True,
    }


class VisualFallbackPolicyTest(unittest.TestCase):
    def test_clean_cover_title_unwraps_only_matching_outer_quotes(self):
        cases = (
            ("Игра «Найди медведя»", "Игра «Найди медведя»"),
            ("«Найди медведя»", "Найди медведя"),
            ("Найди медведя", "Найди медведя"),
            ("Игра «Найди медведя", "Игра «Найди медведя"),
            ("«Найди медведя» дома", "«Найди медведя» дома"),
            ("Игра «Найди медведя» дома", "Игра «Найди медведя» дома"),
            ('Игра "Найди медведя"', 'Игра "Найди медведя"'),
            ('"Найди медведя"', "Найди медведя"),
            ("Игра 'Найди медведя'", "Игра 'Найди медведя'"),
            ("'Найди медведя'", "Найди медведя"),
            ("Игра “Найди медведя”", "Игра “Найди медведя”"),
            ("“Найди медведя”", "“Найди медведя”"),
        )

        for raw_title, expected in cases:
            with self.subTest(raw_title=raw_title):
                self.assertEqual(
                    _clean_cover_title(raw_title, fallback="Fallback"),
                    expected,
                )

    def test_text_fallback_preserves_balanced_embedded_cover_quotes(self):
        title = "Игра «Найди медведя»"
        qa_results = iter(
            [
                {
                    "status": "fail",
                    "pass": False,
                    "reason": "object_contains_text",
                    "people_count": 0,
                    "adult_count": 0,
                    "child_count": 0,
                    "ppe_detected": False,
                    "text_detected": True,
                    "illustration_style_match": True,
                },
                {
                    "status": "fail",
                    "pass": False,
                    "reason": "object_contains_text",
                    "people_count": 0,
                    "adult_count": 0,
                    "child_count": 0,
                    "ppe_detected": False,
                    "text_detected": True,
                    "illustration_style_match": True,
                },
            ]
        )

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"object-1"), {}),
                (BytesIO(b"object-2"), {}),
            ],
        ) as download, patch(
            "src.services.visual_pipeline.build_fallback_cover_buffer",
            return_value=BytesIO(b"text-fallback"),
        ) as fallback:
            buffer, meta = build_post_visual(
                title=title,
                day_key="MO",
                image_prompt="",
                rubric_id="play_and_speak",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(buffer.getvalue(), b"text-fallback")
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["title"], title)
        self.assertEqual(meta["visual_title"], title)
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["title"], title)

    def test_gemini_37_visual_qa_payload_omits_legacy_sampling_controls(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=_qa_response(200)
        ) as request:
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(DEFAULT_GEMINI_VISUAL_QA_MODEL, "gemini-3.7-flash")
        self.assertEqual(result["status"], "pass")
        self.assertIn(
            "/models/gemini-3.7-flash:generateContent",
            request.call_args.args[0],
        )
        generation_config = request.call_args.kwargs["json"]["generationConfig"]
        self.assertEqual(generation_config, {"responseMimeType": "application/json"})
        self.assertNotIn("temperature", generation_config)
        self.assertNotIn("topP", generation_config)
        self.assertNotIn("topK", generation_config)

    def test_gemini_visual_qa_uses_25_fallback_when_primary_model_is_unavailable(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[
                _qa_response(404, text="model not found"),
                _qa_response(200),
            ],
        ) as request:
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(DEFAULT_GEMINI_VISUAL_QA_MODEL, "gemini-3.7-flash")
        self.assertEqual(DEFAULT_GEMINI_VISUAL_QA_FALLBACK_MODEL, "gemini-2.5-flash")
        self.assertEqual(
            _visual_qa_model_candidates(),
            ("gemini-3.7-flash", "gemini-2.5-flash"),
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            ],
        )
        for call in request.call_args_list:
            generation_config = call.kwargs["json"]["generationConfig"]
            self.assertEqual(generation_config, {"responseMimeType": "application/json"})
            self.assertNotIn("temperature", generation_config)
            self.assertNotIn("topP", generation_config)
            self.assertNotIn("topK", generation_config)

    def test_gemini_visual_qa_503_uses_fallback_model_on_same_key_before_key_fallback(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(503), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                gemini_api_key="VISUAL_SECRET",
            )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["human_qa_key_source"], "general")
        self.assertEqual(result["human_qa_key_attempts"], "1")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            ],
        )
        self.assertEqual(
            [call.kwargs["headers"]["x-goog-api-key"] for call in request.call_args_list],
            ["GENERAL_SECRET", "GENERAL_SECRET"],
        )

    def test_gemini_visual_qa_503_exhausts_model_then_uses_next_key_without_repeating_pair(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(503), _qa_response(503), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                gemini_api_key="VISUAL_SECRET",
            )

        urls = [call.args[0] for call in request.call_args_list]
        keys = [call.kwargs["headers"]["x-goog-api-key"] for call in request.call_args_list]
        pairs = list(zip(urls, keys))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["human_qa_key_source"], "explicit")
        self.assertEqual(result["human_qa_key_attempts"], "2")
        self.assertEqual(result["human_qa_key_fallback_used"], "True")
        self.assertEqual(result["human_qa_key_fallback_trigger"], "http_503")
        self.assertEqual(
            urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            ],
        )
        self.assertEqual(keys, ["GENERAL_SECRET", "GENERAL_SECRET", "VISUAL_SECRET"])
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_gemini_visual_qa_workflow_pins_primary_and_fallback_models(self):
        workflow = (
            Path(__file__).resolve().parents[1] / ".github/workflows/post.yml"
        ).read_text(encoding="utf-8")

        self.assertIn('GEMINI_VISUAL_QA_MODEL: "gemini-3.7-flash"', workflow)
        self.assertIn(
            'GEMINI_VISUAL_QA_FALLBACK_MODEL: "gemini-2.5-flash"',
            workflow,
        )

    def test_gemini_human_qa_prompt_requires_character_roles_match_and_exact_child_age(self):
        expected = (
            "Expected roles: Exactly one adult parent and exactly one 2-year-old toddler, "
            "visibly different in age and height, no other people.\n"
            "Expected action: the parent rolls a ball while the child names it\n"
            "Allowed props: ball"
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=_qa_response(200)
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                expected_prompt=expected,
            )

        qa_prompt = request.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertEqual(result["status"], "pass")
        self.assertIn("character_roles_match (boolean)", qa_prompt)
        self.assertIn("counts alone are not enough", qa_prompt)
        self.assertIn("adult parent from an adult speech specialist", qa_prompt)
        self.assertIn("child age descriptor", qa_prompt)
        self.assertIn("2-year-old toddler", qa_prompt)

    def test_gemini_character_roles_match_true_keeps_pass(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": true, "action_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertIs(result["character_roles_match"], True)

    def test_gemini_character_roles_match_false_forces_wrong_character_roles(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": false, "action_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "wrong_character_roles")
        self.assertIs(result["character_roles_match"], False)

    def test_gemini_missing_character_roles_match_fails_closed(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "action_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "character_roles_unknown")
        self.assertEqual(result["character_roles_match"], "unknown")
        self.assertIn("character_roles_unknown", VISUAL_QA_HARD_REASONS)

    def test_gemini_malformed_character_roles_match_fails_closed(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": "maybe", "action_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "character_roles_unknown")
        self.assertEqual(result["character_roles_match"], "unknown")

    def test_gemini_wrong_character_roles_retry_strengthens_role_only(self):
        action = "The parent rolls a ball while the child names it"
        prompt = _compile_visual_prompt(
            VisualBrief(
                rubric_id="tip_of_day",
                role_rule=build_visual_role_rule("tip_of_day", age_descriptor="2-year-old toddler"),
                age_descriptor="2-year-old toddler",
                setting="simple home play area",
                action=action,
                props=("ball",),
            )
        )
        retry = build_visual_retry_prompt(
            prompt,
            rubric_id="tip_of_day",
            qa_reason="wrong_character_roles",
        )
        brief = _parse_compiled_visual_prompt(retry, rubric_id="tip_of_day")

        self.assertIsNotNone(brief)
        self.assertEqual(brief.action, action)
        self.assertIn("2-year-old toddler", brief.role_rule)
        self.assertIn("unmistakably mature", brief.role_rule.lower())

    def test_gemini_character_roles_unknown_retry_strengthens_role_only(self):
        action = "The parent rolls a ball while the child names it"
        prompt = _compile_visual_prompt(
            VisualBrief(
                rubric_id="tip_of_day",
                role_rule=build_visual_role_rule("tip_of_day", age_descriptor="2-year-old toddler"),
                age_descriptor="2-year-old toddler",
                setting="simple home play area",
                action=action,
                props=("ball",),
            )
        )
        retry = build_visual_retry_prompt(
            prompt,
            rubric_id="tip_of_day",
            qa_reason="character_roles_unknown",
        )
        brief = _parse_compiled_visual_prompt(retry, rubric_id="tip_of_day")

        self.assertIsNotNone(brief)
        self.assertEqual(brief.action, action)
        self.assertIn("2-year-old toddler", brief.role_rule)
        self.assertIn("unmistakably mature", brief.role_rule.lower())

    def test_gemini_human_qa_prompt_requires_structured_action_match(self):
        expected = (
            "Expected roles: Exactly one adult parent and exactly one 2-year-old toddler, no other people.\n"
            "Expected action: the parent rolls a ball toward the child while the child reaches for it\n"
            "Allowed props: ball"
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=_qa_response(200)
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                expected_prompt=expected,
            )

        qa_prompt = request.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertEqual(result["status"], "pass")
        self.assertIn("action_match (boolean)", qa_prompt)
        self.assertIn("main visible action materially matches the exact Expected action", qa_prompt)
        self.assertIn("actor, visible action, target/object/prop relationship", qa_prompt)
        self.assertIn("exact spoken word", qa_prompt)
        self.assertIn("reading, drawing, or ordinary conversation", qa_prompt)
        self.assertIn("the parent rolls a ball toward the child while the child reaches for it", qa_prompt)

    def test_gemini_action_match_true_keeps_pass(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": true, "action_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertIs(result["action_match"], True)

    def test_gemini_action_match_false_forces_action_mismatch(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": true, "action_match": false}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "action_mismatch")
        self.assertIs(result["action_match"], False)

    def test_gemini_missing_action_match_fails_closed(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "action_match_unknown")
        self.assertEqual(result["action_match"], "unknown")
        self.assertIn("action_match_unknown", VISUAL_QA_HARD_REASONS)

    def test_gemini_malformed_action_match_fails_closed(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": true, "action_match": "maybe"}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "action_match_unknown")
        self.assertEqual(result["action_match"], "unknown")

    def test_gemini_action_match_unknown_retry_keeps_exact_expected_action(self):
        action = "The parent rolls a ball while the child reaches for it"
        prompt = _compile_visual_prompt(
            VisualBrief(
                rubric_id="tip_of_day",
                role_rule=build_visual_role_rule("tip_of_day", age_descriptor="2-year-old toddler"),
                age_descriptor="2-year-old toddler",
                setting="simple home play area",
                action=action,
                props=("ball",),
            )
        )
        retry = build_visual_retry_prompt(
            prompt,
            rubric_id="tip_of_day",
            qa_reason="action_match_unknown",
            expected_action=action,
        )
        brief = _parse_compiled_visual_prompt(retry, rubric_id="tip_of_day")

        self.assertIsNotNone(brief)
        self.assertEqual(brief.action, action)
        self.assertEqual(brief.role_rule, build_visual_role_rule("tip_of_day", age_descriptor="2-year-old toddler"))

    def test_gemini_character_role_failure_precedes_action_failure(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, '
                '"child_count": 1, "ppe_detected": false, "character_roles_match": false, "action_match": false}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "wrong_character_roles")
        self.assertIs(result["character_roles_match"], False)
        self.assertIs(result["action_match"], False)

    def test_gemini_object_qa_does_not_require_action_match(self):
        response = _qa_response(200, {
            "candidates": [{"content": {"parts": [{"text": (
                '{"pass": true, "reason": "ok", "people_count": 0, "adult_count": 0, '
                '"child_count": 0, "ppe_detected": false, "text_detected": false, '
                '"illustration_style_match": true, "object_topic_match": true}'
            )}]}}]
        })
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=response
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                qa_mode="object",
                expected_prompt=(
                    "Expected roles: zero people, zero adults, zero children; object-only still life.\n"
                    "Expected action: show one recognizable ball.\n"
                    "Allowed props: ball"
                ),
            )

        qa_prompt = request.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertNotIn("action_match (boolean)", qa_prompt)
        self.assertNotIn("action_match", result)

    def test_visual_key_403_then_general_pass_keeps_human_image(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(403), _qa_response(200)],
        ) as request, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(BytesIO(b"human"), {"attempts_used": "1"}),
        ):
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
            )

        self.assertEqual(buffer.getvalue(), b"human")
        self.assertEqual(meta["mode"], "ai_human")
        self.assertEqual(meta["human_qa_key_source"], "general")
        self.assertEqual(meta["human_qa_key_attempts"], "2")
        self.assertEqual(meta["human_qa_key_fallback_used"], "True")
        self.assertEqual(meta["human_qa_key_fallback_trigger"], "http_403")
        self.assertEqual(request.call_count, 2)

    def test_duplicate_key_is_not_tried_twice(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "SAME_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post", return_value=_qa_response(200)
        ) as request:
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")
            candidates = _visual_qa_key_candidates("SAME_SECRET")

        self.assertEqual(candidates, (("explicit", "SAME_SECRET"), ("general", "GENERAL_SECRET")))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["human_qa_key_source"], "visual_qa")
        self.assertEqual(result["human_qa_key_attempts"], "1")

    def test_two_403_responses_use_qa_checked_object_fallback(self):
        qa_results = iter([
            {"status": "skipped", "pass": True, "reason": "qa_http_403"},
            _object_pass(),
        ])

        def qa(*_args, **_kwargs):
            return next(qa_results)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {}), (BytesIO(b"object"), {})],
        ) as download:
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=qa,
            )

        self.assertEqual(buffer.getvalue(), b"object")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["fallback_trigger"], "qa_unavailable_for_required_rubric")
        self.assertEqual(meta["object_qa_status"], "pass")
        self.assertEqual(meta["object_qa_people_count"], "0")
        self.assertEqual(download.call_count, 2)

    def test_401_and_429_can_use_next_key_once(self):
        for first_status in (401, 429):
            with self.subTest(first_status=first_status), patch.dict(
                os.environ,
                {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
                clear=True,
            ), patch(
                "src.services.visual_pipeline.requests.post",
                side_effect=[_qa_response(first_status), _qa_response(200)],
            ) as request:
                result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["human_qa_key_source"], "general")
            self.assertEqual(request.call_count, 2)

    def test_timeout_uses_fallback_model_on_same_general_key_before_key_fallback(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                gemini_api_key="VISUAL_SECRET",
            )

        urls = [call.args[0] for call in request.call_args_list]
        keys = [call.kwargs["headers"]["x-goog-api-key"] for call in request.call_args_list]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["human_qa_key_source"], "general")
        self.assertEqual(result["human_qa_key_attempts"], "1")
        self.assertEqual(result["human_qa_key_fallback_used"], "False")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(
            urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            ],
        )
        self.assertEqual(keys, ["GENERAL_SECRET", "GENERAL_SECRET"])

    def test_timeout_exhausts_model_then_uses_next_key_without_repeating_pair(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), requests.Timeout(), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                gemini_api_key="VISUAL_SECRET",
            )

        urls = [call.args[0] for call in request.call_args_list]
        keys = [call.kwargs["headers"]["x-goog-api-key"] for call in request.call_args_list]
        pairs = list(zip(urls, keys))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["human_qa_key_source"], "explicit")
        self.assertEqual(result["human_qa_key_attempts"], "2")
        self.assertEqual(result["human_qa_key_fallback_used"], "True")
        self.assertEqual(result["human_qa_key_fallback_trigger"], "timeout")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(
            urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            ],
        )
        self.assertEqual(keys, ["GENERAL_SECRET", "GENERAL_SECRET", "VISUAL_SECRET"])
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_timeout_on_fallback_model_uses_existing_next_key_fallback(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="tip_of_day",
                model=DEFAULT_GEMINI_VISUAL_QA_FALLBACK_MODEL,
            )

        urls = [call.args[0] for call in request.call_args_list]
        keys = [call.kwargs["headers"]["x-goog-api-key"] for call in request.call_args_list]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["human_qa_key_source"], "general")
        self.assertEqual(result["human_qa_key_attempts"], "2")
        self.assertEqual(result["human_qa_key_fallback_used"], "True")
        self.assertEqual(result["human_qa_key_fallback_trigger"], "timeout")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(
            urls,
            [
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            ],
        )
        self.assertEqual(keys, ["VISUAL_SECRET", "GENERAL_SECRET"])

    def test_both_keys_are_bounded_to_two_requests_and_logs_hide_keys(self):
        visual_key = "VISUAL_SECRET_DO_NOT_LOG"
        general_key = "GENERAL_SECRET_DO_NOT_LOG"
        output = StringIO()
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": visual_key, "GEMINI_API_KEY": general_key},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(403), _qa_response(403)],
        ) as request, redirect_stdout(output):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(request.call_count, 2)
        self.assertNotIn(visual_key, output.getvalue())
        self.assertNotIn(general_key, output.getvalue())
        self.assertNotIn(visual_key, repr(result))
        self.assertNotIn(general_key, repr(result))

    def test_missing_visual_key_uses_general_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=_qa_response(200)
        ) as request:
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")
        self.assertEqual(result["human_qa_key_source"], "general")
        self.assertEqual(result["qa_key_source"], "general")
        self.assertEqual(result["qa_key_attempts"], "1")
        self.assertEqual(request.call_count, 1)

    def test_key_fallback_does_not_weaken_hard_failure(self):
        hard = _qa_response(200, {"candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ghosted_figure", "people_count": 2, "adult_count": 1, "child_count": 1, "ppe_detected": false}'}]}}]})
        with patch.dict(os.environ, {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=hard
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "ghosted_figure")

    def test_unexpected_ppe_is_hard_failure(self):
        ppe = _qa_response(200, {"candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1, "ppe_detected": true}'}]}}]})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=ppe
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "unexpected_ppe")

    def test_object_prompt_is_people_free_styled_and_does_not_include_raw_title(self):
        prompt = build_object_only_visual_prompt(
            "Русская игра с мячом и ребёнком", "play_and_speak", "raw Russian prompt"
        )
        self.assertIn("Object-only educational still life", prompt)
        self.assertIn("No people", prompt)
        self.assertIn("No faces", prompt)
        self.assertIn("No hands", prompt)
        self.assertIn("No PPE", prompt)
        self.assertIn("watercolor", prompt.lower())
        self.assertIn("gouache", prompt.lower())
        self.assertIn("Not photorealistic", prompt)
        self.assertIn("16:9 landscape", prompt)
        self.assertNotIn("Русская игра", prompt)
        self.assertNotIn("raw Russian prompt", prompt)

    def test_human_style_and_role_prompt_have_watercolor_and_anti_ppe(self):
        self.assertIn("watercolor", VISUAL_STYLE_TAIL.lower())
        self.assertIn("gouache", VISUAL_STYLE_TAIL.lower())
        self.assertIn("surgical masks", VISUAL_STYLE_TAIL.lower())
        self.assertIn("high-vis vests", VISUAL_STYLE_TAIL.lower())
        self.assertIn("not photorealistic", VISUAL_STYLE_TAIL.lower())
        role = build_visual_role_rule("method_piggybank")
        self.assertIn("speech specialist", role.lower())
        self.assertIn("ordinary casual professional indoor clothing", VISUAL_STYLE_TAIL.lower())
        self.assertIn("medical/industrial ppe", VISUAL_STYLE_TAIL.lower())

    def test_object_prompt_varies_by_publication_key_and_stays_deterministic(self):
        first = build_object_only_visual_prompt(
            "Мелодии и слова",
            "bilingual_corner",
            variation_key="2026-07-30",
        )
        repeated = build_object_only_visual_prompt(
            "Мелодии и слова",
            "bilingual_corner",
            variation_key="2026-07-30",
        )
        next_day = build_object_only_visual_prompt(
            "Мелодии и слова",
            "bilingual_corner",
            variation_key="2026-07-31",
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_day)
        self.assertNotIn("Internal visual variation cue", first)
        self.assertRegex(first, r"\[object_scene:[a-z_]+\|[0-9a-f]+\]$")
        self.assertNotIn("Мелодии и слова", first)

    def test_empty_prompt_object_fallback_varies_between_days_and_is_qa_checked(self):
        prompts = []
        qa_calls = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(b"object"), {"attempts_used": "1"}

        def qa(*_args, **kwargs):
            qa_calls.append(kwargs)
            return _object_pass()

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            _, first_meta = build_post_visual(
                title="Мелодии и слова",
                day_key="2026-07-30",
                image_prompt="",
                rubric_id="bilingual_corner",
                visual_qa_fn=qa,
            )
            _, second_meta = build_post_visual(
                title="Пойте и разговаривайте",
                day_key="2026-07-31",
                image_prompt="",
                rubric_id="question_week",
                visual_qa_fn=qa,
            )

        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(qa_calls), 2)
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertEqual(first_meta["object_scene_category"], "hearing_sounds_music")
        self.assertEqual(second_meta["object_scene_category"], "hearing_sounds_music")
        self.assertEqual(first_meta["object_qa_status"], "pass")

    def test_object_qa_rejects_detected_human_and_retries(self):
        qa_results = iter([
            {"status": "pass", "pass": True, "reason": "ok", "people_count": 1, "adult_count": 1, "child_count": 0, "ppe_detected": False},
            _object_pass(),
        ])
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"object-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch("src.services.visual_pipeline.download_pollinations_image_with_meta", side_effect=download):
            buffer, meta = build_post_visual(
                title="Разговоры во время бытовых дел",
                day_key="2026-08-01",
                image_prompt="",
                rubric_id="method_piggybank",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        self.assertEqual(buffer.getvalue(), b"object-2")
        self.assertEqual(len(prompts), 2)
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertEqual(meta["object_qa_status"], "pass")
        self.assertEqual(meta["object_generation_attempts"], "2")

    def test_object_qa_rejects_ppe_and_retries(self):
        qa_results = iter([
            {"status": "pass", "pass": True, "reason": "ok", "people_count": 0, "adult_count": 0, "child_count": 0, "ppe_detected": True},
            _object_pass(),
        ])
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"bad-ppe"), {}), (BytesIO(b"safe-object"), {})],
        ):
            buffer, meta = build_post_visual(
                title="Разговоры во время бытовых дел",
                day_key="2026-08-01",
                image_prompt="",
                rubric_id="method_piggybank",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )
        self.assertEqual(buffer.getvalue(), b"safe-object")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["object_qa_ppe_detected"], "False")

    def test_two_object_qa_failures_use_text_fallback(self):
        qa_results = iter([
            {"status": "pass", "pass": True, "reason": "ok", "people_count": 1, "adult_count": 1, "child_count": 0, "ppe_detected": False},
            {"status": "fail", "pass": False, "reason": "unexpected_ppe", "people_count": 0, "adult_count": 0, "child_count": 0, "ppe_detected": True},
        ])
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"object-1"), {}), (BytesIO(b"object-2"), {})],
        ), patch(
            "src.services.visual_pipeline.build_fallback_cover_buffer",
            return_value=BytesIO(b"text-fallback"),
        ):
            buffer, meta = build_post_visual(
                title="Разговоры во время бытовых дел",
                day_key="2026-08-01",
                image_prompt="",
                rubric_id="method_piggybank",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )
        self.assertEqual(buffer.getvalue(), b"text-fallback")
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["final_reason"], "object_fallback_rejected")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["object_qa_reason"], "unexpected_ppe")

    def test_skipped_human_qa_requires_object_qa_before_publish(self):
        qa_results = iter([
            {"status": "skipped", "pass": True, "reason": "qa_http_429"},
            _object_pass(),
        ])
        qa_calls = []

        def qa(*args, **kwargs):
            qa_calls.append(kwargs)
            return next(qa_results)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human"), {"attempts_used": "1"}),
                (BytesIO(b"object"), {"attempts_used": "1"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Игра со звуками",
                day_key="TU",
                image_prompt="adult and child play with a bell",
                rubric_id="tip_of_day",
                visual_qa_fn=qa,
            )

        self.assertEqual(buffer.getvalue(), b"object")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["visual_source"], "object_ai")
        self.assertEqual(meta["object_generation_status"], "generated")
        self.assertEqual(meta["human_qa_first_reason"], "qa_http_429")
        self.assertEqual(meta["object_qa_status"], "pass")
        self.assertEqual(len(qa_calls), 2)
        self.assertEqual(download.call_count, 2)

    def test_two_human_failures_then_two_object_failures_use_text_fallback(self):
        qa_results = iter(
            [
                {"status": "fail", "pass": False, "reason": "ghosted_figure", "people_count": 2, "adult_count": 1, "child_count": 1, "ppe_detected": False},
                {"status": "fail", "pass": False, "reason": "action_mismatch", "people_count": 2, "adult_count": 1, "child_count": 1, "ppe_detected": False},
                {"status": "pass", "pass": True, "reason": "ok", "people_count": 1, "adult_count": 1, "child_count": 0, "ppe_detected": False},
                {"status": "fail", "pass": False, "reason": "unexpected_ppe", "people_count": 0, "adult_count": 0, "child_count": 0, "ppe_detected": True},
            ]
        )
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human"), {}),
                (BytesIO(b"retry"), {}),
                (BytesIO(b"object-1"), {}),
                (BytesIO(b"object-2"), {}),
            ],
        ):
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )
        self.assertNotIn(buffer.getvalue(), {b"human", b"retry", b"object-1", b"object-2"})
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["human_qa_first_reason"], "ghosted_figure")
        self.assertEqual(meta["human_qa_retry_reason"], "action_mismatch")

    def test_method_piggybank_human_retry_exhaustion_skips_object_fallback(self):
        qa_results = iter([
            {
                "status": "fail",
                "pass": False,
                "reason": "action_mismatch",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
                "ppe_detected": False,
            },
            {
                "status": "fail",
                "pass": False,
                "reason": "photorealistic_imagery",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
                "ppe_detected": False,
            },
        ])

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human-1"), {"attempts_used": "1"}),
                (BytesIO(b"human-2"), {"attempts_used": "1"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Нейропсихологическое упражнение «Кулак-ребро-ладонь»",
                day_key="2026-08-15",
                image_prompt=(
                    "the speech specialist demonstrates a fist-edge-palm hand sequence "
                    "while the child copies the movements"
                ),
                rubric_id="method_piggybank",
                audience="pros",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )

        self.assertEqual(download.call_count, 2)
        self.assertNotIn(buffer.getvalue(), {b"human-1", b"human-2"})
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["object_prompt_used"], "False")
        self.assertEqual(meta["object_generation_status"], "not_run")
        self.assertEqual(meta["object_generation_attempts"], "0")
        self.assertEqual(meta["final_reason"], "method_piggybank_object_fallback_not_allowed")

    def test_method_piggybank_retry_qa_skipped_goes_directly_to_text(self):
        qa_results = iter([
            {
                "status": "fail",
                "pass": False,
                "reason": "action_mismatch",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
                "ppe_detected": False,
            },
            {
                "status": "skipped",
                "pass": True,
                "reason": "qa_timeout",
                "people_count": "unknown",
                "adult_count": "unknown",
                "child_count": "unknown",
                "ppe_detected": "unknown",
            },
        ])

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human-1"), {"attempts_used": "1"}),
                (BytesIO(b"human-2"), {"attempts_used": "1"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Нейропсихологическое упражнение «Кулак-ребро-ладонь»",
                day_key="2026-08-15",
                image_prompt=(
                    "the speech specialist demonstrates a fist-edge-palm hand sequence "
                    "while the child copies the movements"
                ),
                rubric_id="method_piggybank",
                audience="pros",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )

        self.assertEqual(download.call_count, 2)
        self.assertNotIn(buffer.getvalue(), {b"human-1", b"human-2"})
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["object_prompt_used"], "False")
        self.assertEqual(meta["object_generation_status"], "not_run")
        self.assertEqual(meta["object_generation_attempts"], "0")
        self.assertEqual(meta["human_qa_first_reason"], "action_mismatch")
        self.assertEqual(meta["human_qa_retry_status"], "skipped")
        self.assertEqual(meta["human_qa_retry_reason"], "qa_timeout")
        self.assertEqual(meta["final_reason"], "method_piggybank_object_fallback_not_allowed")

    def test_object_fallback_categories_follow_title_topic(self):
        cases = (
            ("Положение языка при произнесении звука", "tip_of_day", "articulation_speech"),
            ("Игра для двух языков дома", "tip_of_day", "bilingual_languages"),
            ("Реакция малыша на колокольчик", "tip_of_day", "hearing_sounds_music"),
            ("Язык находится за верхними зубами", "tip_of_day", "articulation_speech"),
            ("Развитие домашнего языка в двуязычной семье", "tip_of_day", "bilingual_languages"),
            ("Реакция на колокольчик", "bilingual_corner", "hearing_sounds_music"),
            ("Положение языка при произнесении звука", "bilingual_corner", "articulation_speech"),
            ("Два языка дома", "bilingual_corner", "bilingual_languages"),
            ("Игра с мячом дома", "bilingual_corner", "games_everyday_communication"),
            ("Положение языка при произнесении звука", "speech_sounds", "articulation_speech"),
            ("Реакция малыша на колокольчик", "hearing_and_speech", "hearing_sounds_music"),
            ("Мелодии и слова", "bilingual_corner", "hearing_sounds_music"),
            ("Пойте и разговаривайте", "question_week", "hearing_sounds_music"),
            ("Разговоры во время бытовых дел", "method_piggybank", "household_routines"),
            ("Стирка и новые слова", "tip_of_day", "household_routines"),
        )
        for title, rubric_id, expected in cases:
            with self.subTest(title=title, rubric_id=rubric_id):
                self.assertEqual(_object_scene_category(title, rubric_id), expected)

    def test_household_category_can_use_safe_brief_context_without_leaking_it(self):
        self.assertEqual(
            _object_scene_category("Новые слова", "method_piggybank", "placing a T-shirt into the washing machine"),
            "household_routines",
        )
        prompt = build_object_only_visual_prompt(
            "Новые слова",
            "method_piggybank",
            context_hint="placing a T-shirt into the washing machine",
            variation_key="2026-08-01",
        )
        self.assertIn("Scene category: household_routines", prompt)
        self.assertNotIn("washing machine", prompt.lower())
        self.assertNotIn("Новые слова", prompt)

    def test_object_enforcement_rejects_people_ppe_and_unknown_counts(self):
        person = _enforce_object_visual_qa({
            "status": "pass", "pass": True, "reason": "ok",
            "people_count": 1, "adult_count": 1, "child_count": 0, "ppe_detected": False,
        })
        self.assertEqual(person["reason"], "object_contains_person")
        self.assertFalse(person["pass"])

        ppe = _enforce_object_visual_qa({
            "status": "pass", "pass": True, "reason": "ok",
            "people_count": 0, "adult_count": 0, "child_count": 0, "ppe_detected": True,
        })
        self.assertEqual(ppe["reason"], "unexpected_ppe")
        self.assertFalse(ppe["pass"])

        unknown = _enforce_object_visual_qa({
            "status": "pass", "pass": True, "reason": "ok",
            "people_count": "unknown", "adult_count": 0, "child_count": 0, "ppe_detected": False,
        })
        self.assertEqual(unknown["reason"], "object_counts_unknown")
        self.assertFalse(unknown["pass"])

    def test_legacy_bilingual_rubric_does_not_override_neutral_title(self):
        self.assertNotEqual(
            _object_scene_category("Речь в разных ситуациях", "bilingual_corner"),
            "bilingual_languages",
        )

    def test_lone_language_word_does_not_select_bilingual_category(self):
        self.assertNotEqual(_object_scene_category("Положение языка", "tip_of_day"), "bilingual_languages")


class VisualFallbackLadderTest(unittest.TestCase):
    def test_ladder_is_human_human_retry_object_object_text(self):
        """human -> human retry -> object #1 -> object #2 -> text, no extra attempts."""
        human_fail = {
            "status": "fail",
            "pass": False,
            "reason": "photorealistic_imagery",
            "people_count": 2,
            "adult_count": 1,
            "child_count": 1,
            "ppe_detected": False,
        }
        object_fail = {
            "status": "fail",
            "pass": False,
            "reason": "object_style_mismatch",
            "people_count": 0,
            "adult_count": 0,
            "child_count": 0,
            "ppe_detected": False,
            "text_detected": False,
            "illustration_style_match": False,
        }
        qa_results = iter([human_fail, human_fail, object_fail, object_fail])
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"image-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            buffer, meta = build_post_visual(
                title="Книги и новые слова",
                day_key="2026-08-10",
                image_prompt="an adult and child looking at a picture book together",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        self.assertEqual(len(prompts), 4)
        self.assertEqual(meta["visual_qa_attempts"], "2")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertNotIn(buffer.getvalue(), {b"image-1", b"image-2", b"image-3", b"image-4"})

    def test_object_attempts_use_different_variation_and_seed(self):
        qa_results = iter([
            {
                "status": "fail",
                "pass": False,
                "reason": "wrong_character_roles",
                "people_count": 2,
                "adult_count": 2,
                "child_count": 0,
                "ppe_detected": False,
            },
            {
                "status": "fail",
                "pass": False,
                "reason": "wrong_character_roles",
                "people_count": 2,
                "adult_count": 2,
                "child_count": 0,
                "ppe_detected": False,
            },
            {
                "status": "fail",
                "pass": False,
                "reason": "object_style_mismatch",
                "people_count": 0,
                "adult_count": 0,
                "child_count": 0,
                "ppe_detected": False,
                "text_detected": False,
                "illustration_style_match": False,
            },
            _object_pass(),
        ])
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"image-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            _, meta = build_post_visual(
                title="Книги и новые слова",
                day_key="2026-08-10",
                image_prompt="an adult and child looking at a picture book together",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        object_prompts = prompts[2:]
        self.assertEqual(len(object_prompts), 2)
        self.assertNotEqual(object_prompts[0], object_prompts[1])
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["object_generation_attempts"], "2")




_PRIMARY_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent"
_FALLBACK_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def _qa_fail_response(reason):
    """A technically healthy QA response that rejects the image on content."""

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"pass": false, "reason": "%s", "people_count": 3, '
                                '"adult_count": 1, "child_count": 2, "ppe_detected": false, '
                                '"text_detected": false, "ui_artifact_detected": false, '
                                '"illustration_style_match": true, "character_roles_match": false, '
                                '"action_match": false}' % reason
                            )
                        }
                    ]
                }
            }
        ]
    }
    return _qa_response(200, payload=payload)


class VisualQABuildCircuitTest(unittest.TestCase):
    """Per-build circuit breaker for the primary visual QA model."""

    def _urls(self, request):
        return [call.args[0] for call in request.call_args_list]

    # --- 1 / 2: healthy primary, then technical fallback --------------------

    def test_healthy_first_primary_qa_uses_37(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", side_effect=[_qa_response(200)]
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(self._urls(request), [_PRIMARY_URL])
        self.assertFalse(circuit.primary_unavailable)

    def test_primary_timeout_uses_25_fallback_and_opens_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(self._urls(request), [_PRIMARY_URL, _FALLBACK_URL])
        self.assertTrue(circuit.primary_unavailable)
        self.assertEqual(circuit.primary_unavailable_reason, "timeout")

    # --- 3 / 4 / 20: same build skips 3.7; a new build retries it -----------

    def test_later_qa_cycle_in_same_build_skips_37(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_response(200), _qa_response(200), _qa_response(200)],
        ) as request:
            evaluate_visual_quality(BytesIO(b"a"), rubric_id="tip_of_day", _build_circuit=circuit)
            evaluate_visual_quality(BytesIO(b"b"), rubric_id="tip_of_day", _build_circuit=circuit)
            evaluate_visual_quality(BytesIO(b"c"), rubric_id="tip_of_day", _build_circuit=circuit)
        # Only the first cycle pays the primary timeout; later cycles start on 2.5.
        self.assertEqual(
            self._urls(request), [_PRIMARY_URL, _FALLBACK_URL, _FALLBACK_URL, _FALLBACK_URL]
        )

    def test_next_build_tries_37_again(self):
        first = _VisualQABuildCircuit()
        second = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_response(200), _qa_response(200)],
        ) as request:
            evaluate_visual_quality(BytesIO(b"a"), rubric_id="tip_of_day", _build_circuit=first)
            evaluate_visual_quality(BytesIO(b"b"), rubric_id="tip_of_day", _build_circuit=second)
        self.assertEqual(self._urls(request), [_PRIMARY_URL, _FALLBACK_URL, _PRIMARY_URL])
        self.assertTrue(first.primary_unavailable)
        self.assertFalse(second.primary_unavailable)

    def test_circuit_state_is_build_local_not_module_global(self):
        import ast

        source = Path("src/services/visual_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        # No ContextVar is imported or used (prose in docstrings does not count).
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.update(a.name for a in node.names)
                if node.module:
                    imported.add(node.module)
        self.assertNotIn("ContextVar", imported)
        self.assertNotIn("contextvars", imported)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn("ContextVar", used)

        # The circuit is only ever constructed inside build_post_visual.
        builders = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(call.func, ast.Name) and call.func.id == "_VisualQABuildCircuit"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
        }
        self.assertEqual(builders, {"build_post_visual"})

        # No module-level instance exists.
        module_level = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("build_circuit", module_level)

        fresh = _VisualQABuildCircuit()
        self.assertFalse(fresh.primary_unavailable)
        self.assertEqual(fresh.primary_unavailable_reason, "")

    # --- 5 / 6 / 7: qualifying technical triggers open the circuit ----------

    def test_404_opens_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(404, text="model not found"), _qa_response(200)],
        ) as request:
            evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit)
        self.assertTrue(circuit.primary_unavailable)
        self.assertEqual(circuit.primary_unavailable_reason, "http_404")
        self.assertEqual(self._urls(request), [_PRIMARY_URL, _FALLBACK_URL])

    def test_503_opens_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(503), _qa_response(200)],
        ):
            evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit)
        self.assertTrue(circuit.primary_unavailable)
        self.assertEqual(circuit.primary_unavailable_reason, "http_503")

    def test_400_with_model_unavailable_marker_opens_circuit(self):
        for marker in ("model", "not found", "decommissioned", "unsupported", "does not exist"):
            with self.subTest(marker=marker):
                circuit = _VisualQABuildCircuit()
                with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
                    "src.services.visual_pipeline.requests.post",
                    side_effect=[_qa_response(400, text=f"the {marker} is gone"), _qa_response(200)],
                ):
                    evaluate_visual_quality(
                        BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
                    )
                self.assertTrue(circuit.primary_unavailable)
                self.assertEqual(circuit.primary_unavailable_reason, "http_400")

    # --- 8 / 9 / 10 / 11: non-qualifying failures never open the circuit ----

    def test_429_does_not_open_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(429), _qa_response(200)],
        ) as request:
            evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit)
        self.assertFalse(circuit.primary_unavailable)
        # 429 stays a key-level failure: same model, next key.
        self.assertEqual(self._urls(request), [_PRIMARY_URL, _PRIMARY_URL])

    def test_generic_5xx_does_not_open_circuit(self):
        for status in (500, 502, 504):
            with self.subTest(status=status):
                circuit = _VisualQABuildCircuit()
                with patch.dict(
                    os.environ,
                    {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
                    clear=True,
                ), patch(
                    "src.services.visual_pipeline.requests.post",
                    side_effect=[_qa_response(status), _qa_response(200)],
                ) as request:
                    evaluate_visual_quality(
                        BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
                    )
                self.assertFalse(circuit.primary_unavailable)
                self.assertEqual(self._urls(request), [_PRIMARY_URL, _PRIMARY_URL])

    def test_generic_request_exception_does_not_open_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.ConnectionError(), _qa_response(200)],
        ) as request:
            evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit)
        self.assertFalse(circuit.primary_unavailable)
        self.assertEqual(self._urls(request), [_PRIMARY_URL, _PRIMARY_URL])

    def test_ordinary_400_does_not_open_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(400, text="malformed request payload")],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        self.assertFalse(circuit.primary_unavailable)
        self.assertEqual(self._urls(request), [_PRIMARY_URL])
        self.assertEqual(result["status"], "skipped")

    # --- 12 / 13: content rejections are never circuit triggers -------------

    def test_semantic_rejection_by_37_does_not_open_circuit(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_fail_response("too_many_people")],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        self.assertFalse(circuit.primary_unavailable)
        self.assertEqual(self._urls(request), [_PRIMARY_URL])
        self.assertFalse(result["pass"])

    def test_semantic_rejection_by_25_cannot_bypass_fallback_ladder(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_fail_response("action_mismatch")],
        ) as request:
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        self.assertTrue(circuit.primary_unavailable)
        self.assertEqual(self._urls(request), [_PRIMARY_URL, _FALLBACK_URL])
        # An open circuit never turns a rejection into an acceptance.
        self.assertFalse(result["pass"])
        self.assertEqual(result["status"], "fail")

    # --- 14 / 15: fail-closed -----------------------------------------------

    def test_technical_primary_failure_plus_valid_fallback_pass_works(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(503), _qa_response(200)],
        ):
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertTrue(circuit.primary_unavailable)

    def test_both_models_unavailable_remain_fail_closed(self):
        circuit = _VisualQABuildCircuit()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), requests.Timeout()],
        ):
            result = evaluate_visual_quality(
                BytesIO(b"image"), rubric_id="tip_of_day", _build_circuit=circuit
            )
        # "skipped" is the existing unavailable verdict; build_post_visual then
        # routes required rubrics into the object/text ladder rather than
        # publishing an unverified human image.
        self.assertEqual(result["status"], "skipped")
        self.assertTrue(circuit.primary_unavailable)

    def test_open_circuit_still_requires_a_real_verdict_for_required_rubric(self):
        """An open circuit must not let an unverified human image through."""

        qa_results = iter([
            {"status": "skipped", "pass": True, "reason": "qa_timeout"},
            _object_pass(),
        ])

        def qa(*_args, **_kwargs):
            return next(qa_results)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {}), (BytesIO(b"object"), {})],
        ):
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=qa,
            )
        self.assertEqual(buffer.getvalue(), b"object")
        self.assertEqual(meta["mode"], "ai_object_fallback")

    # --- 16 / 17 / 18 / 19: the ladder is unchanged -------------------------

    def test_object_contains_text_remains_rejected(self):
        qa_results = iter([
            {"status": "fail", "pass": False, "reason": "action_mismatch"},
            {"status": "fail", "pass": False, "reason": "action_mismatch"},
            {"status": "fail", "pass": False, "reason": "object_contains_text",
             "people_count": 0, "adult_count": 0, "child_count": 0,
             "ppe_detected": False, "text_detected": True, "illustration_style_match": False},
            {"status": "fail", "pass": False, "reason": "object_contains_text",
             "people_count": 0, "adult_count": 0, "child_count": 0,
             "ppe_detected": False, "text_detected": True, "illustration_style_match": False},
        ])

        def qa(*_args, **_kwargs):
            return next(qa_results)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human"), {}),
                (BytesIO(b"retry"), {}),
                (BytesIO(b"object1"), {}),
                (BytesIO(b"object2"), {}),
            ],
        ) as download:
            _buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=qa,
            )
        # exactly one human retry, exactly two object attempts, then text card
        self.assertEqual(download.call_count, 4)
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["object_generation_status"], "rejected")

    def test_model_ids_and_order_remain_unchanged(self):
        self.assertEqual(DEFAULT_GEMINI_VISUAL_QA_MODEL, "gemini-3.7-flash")
        self.assertEqual(DEFAULT_GEMINI_VISUAL_QA_FALLBACK_MODEL, "gemini-2.5-flash")
        self.assertEqual(
            _visual_qa_model_candidates(), ("gemini-3.7-flash", "gemini-2.5-flash")
        )

    def test_method_piggybank_special_fallback_remains_unchanged(self):
        qa_results = iter([
            {"status": "fail", "pass": False, "reason": "wrong_character_roles"},
            {"status": "fail", "pass": False, "reason": "wrong_character_roles"},
        ])

        def qa(*_args, **_kwargs):
            return next(qa_results)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {}), (BytesIO(b"retry"), {})],
        ) as download:
            _buffer, meta = build_post_visual(
                title="Метод дня",
                day_key="SA",
                image_prompt="speech therapist demonstrates articulation",
                rubric_id="method_piggybank",
                visual_qa_fn=qa,
            )
        # method_piggybank goes straight to the text card after the human retry,
        # without the two object attempts.
        self.assertEqual(download.call_count, 2)
        self.assertEqual(meta["mode"], "text_fallback")

    def test_injected_qa_fn_is_not_forced_to_accept_the_private_parameter(self):
        """Custom evaluators keep their existing signature."""

        seen = []

        def strict_qa(image_buffer, rubric_id="", audience="", expected_prompt="", gemini_api_key=""):
            seen.append(rubric_id)
            return _object_pass()

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {})],
        ):
            _buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=strict_qa,
            )
        self.assertTrue(seen)
        self.assertNotEqual(meta["mode"], "")


def _human_content_reject(reason):
    """HTTP-200 human QA verdict that rejects on content, not on transport."""

    return _qa_response(200, payload={
        "candidates": [{"content": {"parts": [{"text":
            '{"pass": false, "reason": "%s", "people_count": 3, "adult_count": 1, '
            '"child_count": 2, "ppe_detected": false, "text_detected": false, '
            '"ui_artifact_detected": false, "illustration_style_match": true, '
            '"character_roles_match": false, "action_match": false}' % reason
        }]}}]
    })


def _object_content_reject():
    """HTTP-200 object QA verdict rejecting a card that contains text."""

    return _qa_response(200, payload={
        "candidates": [{"content": {"parts": [{"text":
            '{"pass": false, "reason": "object_contains_text", "people_count": 0, '
            '"adult_count": 0, "child_count": 0, "ppe_detected": false, '
            '"text_detected": true, "ui_artifact_detected": false, '
            '"illustration_style_match": false, "object_topic_match": true}'
        }]}}]
    })


class VisualQABuildCircuitIntegrationTest(unittest.TestCase):
    """The circuit must be threaded through the real build path.

    These tests drive `build_post_visual` with the production
    `evaluate_visual_quality` (no injected `visual_qa_fn`), so a missing
    `_build_circuit` argument at any call site shows up as a repeated
    primary-model request.
    """

    def _urls(self, request):
        return [call.args[0] for call in request.call_args_list]

    def _models(self, request):
        return [u.rsplit("/models/", 1)[1].split(":", 1)[0] for u in self._urls(request)]

    def test_primary_model_is_tried_once_per_build_across_every_qa_cycle(self):
        # human QA: 3.7 times out, 2.5 rejects on content -> human retry
        # human retry QA: starts on 2.5, rejects on content -> object ladder
        # object #1 QA: starts on 2.5, rejects (object_contains_text)
        # object #2 QA: starts on 2.5, rejects -> terminal text card
        responses = [
            requests.Timeout(),
            _human_content_reject("too_many_people"),
            _human_content_reject("action_mismatch"),
            _object_content_reject(),
            _object_content_reject(),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", side_effect=responses
        ) as request, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human"), {}),
                (BytesIO(b"retry"), {}),
                (BytesIO(b"object1"), {}),
                (BytesIO(b"object2"), {}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_api_key="GENERAL_SECRET",
            )

        models = self._models(request)
        # The whole build pays the primary timeout exactly once.
        self.assertEqual(models.count("gemini-3.7-flash"), 1)
        self.assertEqual(models[0], "gemini-3.7-flash")
        self.assertEqual(
            models,
            [
                "gemini-3.7-flash",
                "gemini-2.5-flash",
                "gemini-2.5-flash",
                "gemini-2.5-flash",
                "gemini-2.5-flash",
            ],
        )
        # Every later QA cycle started directly on the fallback model.
        self.assertTrue(all(m == "gemini-2.5-flash" for m in models[1:]))
        # The ladder itself is untouched: human, one retry, two objects.
        self.assertEqual(download.call_count, 4)
        # Content rejections still drive the ladder to the terminal text card,
        # and no unverified image was accepted.
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["object_generation_status"], "rejected")
        self.assertNotIn(buffer.getvalue(), (b"human", b"retry", b"object1", b"object2"))

    def test_object_cycles_alone_also_reuse_the_open_circuit(self):
        """Circuit threading through _build_object_visual_fallback specifically."""

        # human QA passes on 2.5 after a primary timeout would end the build, so
        # instead reject on content twice to reach the object ladder, then check
        # that both object cycles skip the primary model.
        responses = [
            requests.Timeout(),
            _human_content_reject("too_many_people"),
            _human_content_reject("action_mismatch"),
            _object_content_reject(),
            _object_content_reject(),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", side_effect=responses
        ) as request, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human"), {}),
                (BytesIO(b"retry"), {}),
                (BytesIO(b"object1"), {}),
                (BytesIO(b"object2"), {}),
            ],
        ):
            build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_api_key="GENERAL_SECRET",
            )

        # requests 4 and 5 are the two object QA cycles.
        object_models = self._models(request)[3:]
        self.assertEqual(object_models, ["gemini-2.5-flash", "gemini-2.5-flash"])

    def test_a_second_build_tries_the_primary_model_again(self):
        """Build-local state: a separate invocation must retry 3.7."""

        responses = [
            # build #1
            requests.Timeout(),
            _qa_response(200),
            # build #2
            requests.Timeout(),
            _qa_response(200),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", side_effect=responses
        ) as request, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"first"), {}), (BytesIO(b"second"), {})],
        ):
            first_buffer, _first_meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_api_key="GENERAL_SECRET",
            )
            second_buffer, _second_meta = build_post_visual(
                title="Speech activity",
                day_key="TU",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_api_key="GENERAL_SECRET",
            )

        # Each build independently pays one primary attempt: no leaked state.
        self.assertEqual(
            self._models(request),
            [
                "gemini-3.7-flash",
                "gemini-2.5-flash",
                "gemini-3.7-flash",
                "gemini-2.5-flash",
            ],
        )
        self.assertEqual(first_buffer.getvalue(), b"first")
        self.assertEqual(second_buffer.getvalue(), b"second")

    def test_healthy_build_never_touches_the_fallback_model(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "GENERAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", side_effect=[_qa_response(200)]
        ) as request, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {})],
        ):
            buffer, _meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_api_key="GENERAL_SECRET",
            )
        self.assertEqual(self._models(request), ["gemini-3.7-flash"])
        self.assertEqual(buffer.getvalue(), b"human")


# --- genuine reading_prep: object_contains_text -> text-safe final attempt ----
#
# What this block validates: for a publication whose CLEAN semantic category is
# `reading_prep`, a first bounded object attempt rejected with
# `object_contains_text` redirects the second/final attempt to an intrinsically
# text-safe scene, for that one reason only.
#
# Run #500 (run 36324830249, post #500, rubric age_norms) was historically the
# trigger for investigating this text-prone retry path: human QA rejected twice,
# both bounded object attempts were rendered from the `reading_prep` props
# (picture cards, letter-like blocks, children's book, pencil and blank paper)
# and both were rejected with `object_contains_text`, so the publication fell
# through to the text card. Run #500 reached those props only through classifier
# contamination, though — its own clean category is
# `books_vocab_phrases_stories`. Its corrected semantics are covered separately
# by ObjectSceneCompiledContextContaminationTest below; the fixtures here stand
# for a genuinely reading publication.

# The reading-prep props that must not survive into the text-safe retry.
_READING_PREP_RISKY_PHRASES = (
    "picture cards",
    "letter-like blocks",
    "children’s book",
    "pencil and blank paper",
    "book",
    "card",
    "pencil",
)

def _compiled_context(action, props, title_rubric="age_norms", setting="simple uncluttered home play area"):
    """Compile a brief exactly the way the production visual path does."""

    return _compile_visual_prompt(
        VisualBrief(
            rubric_id=title_rubric,
            role_rule=build_visual_role_rule(title_rubric),
            age_descriptor="",
            setting=setting,
            action=action,
            props=tuple(props),
        )
    )


# The reading-prep fixture must carry GENUINE reading semantics in the compiled
# brief's action/props. A raw image_prompt does not: `build_post_visual` rebuilds
# the action through the role normalizer, which replaces it with a generic
# phrase, so a raw fixture used to reach reading_prep only through the compiled
# style tail ("No readable text ... letters") that the classifier now excludes.
_READING_PREP_TITLE = "Готовим ребенка к чтению"
_READING_PREP_PROMPT = _compiled_context(
    "The parent and child look at letter cards and learn to read",
    ("letter cards",),
)

# A different derived category that also carries picture cards, so it is
# text-prone in the same way. The switch must not reach it: articulation is
# outside the two approved categories. (It takes precedence over reading_prep.)
_NON_READING_TITLE = "Как поставить артикуляцию звука С"
_NON_READING_PROMPT = "the parent and child practice the sound in front of a small mirror"
_NON_READING_CATEGORY = "articulation_speech"


def _human_fail(reason):
    return {
        "status": "fail",
        "pass": False,
        "reason": reason,
        "people_count": 2,
        "adult_count": 1,
        "child_count": 1,
        "ppe_detected": False,
        "text_detected": False,
        "illustration_style_match": True,
    }


def _object_text_fail():
    """Object QA rejecting the rendered scene for readable text."""

    return {
        "status": "fail",
        "pass": False,
        "reason": OBJECT_TEXT_FAILURE_REASON,
        "people_count": 0,
        "adult_count": 0,
        "child_count": 0,
        "ppe_detected": False,
        "text_detected": True,
        "illustration_style_match": True,
    }


def _object_non_text_fail():
    """Object QA rejecting for a reason unrelated to rendered text."""

    return {
        "status": "fail",
        "pass": False,
        "reason": "object_style_mismatch",
        "people_count": 0,
        "adult_count": 0,
        "child_count": 0,
        "ppe_detected": False,
        "text_detected": False,
        "illustration_style_match": False,
    }


class ObjectContainsTextSafeRetryTest(unittest.TestCase):
    """The bounded object retry after `object_contains_text` for reading prep."""

    def _run_ladder(self, object_qa_results, title=None, image_prompt=None):
        """Drive the full ladder: human fail, human retry fail, then objects."""

        qa_results = iter(
            [_human_fail("action_mismatch"), _human_fail("object_contains_text")]
            + list(object_qa_results)
        )
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"image-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ), patch(
            "src.services.visual_pipeline.build_fallback_cover_buffer",
            return_value=BytesIO(b"text-card"),
        ):
            buffer, meta = build_post_visual(
                title=_READING_PREP_TITLE if title is None else title,
                day_key="2026-09-25",
                image_prompt=_READING_PREP_PROMPT if image_prompt is None else image_prompt,
                rubric_id="age_norms",
                audience="parents",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        return buffer, meta, prompts

    @staticmethod
    def _marker_categories(prompts):
        return [
            OBJECT_SCENE_MARKER_RE.search(prompt).group(1)
            for prompt in prompts
            if OBJECT_SCENE_MARKER_RE.search(prompt)
        ]

    def test_reading_prep_is_the_derived_category_for_this_publication(self):
        self.assertEqual(
            _object_scene_category(
                _READING_PREP_TITLE, "age_norms", context_hint=_READING_PREP_PROMPT
            ),
            "reading_prep",
        )

    def test_first_object_attempt_still_uses_reading_prep(self):
        """Regression 1: the first bounded object attempt is unchanged."""

        _buffer, meta, prompts = self._run_ladder([_object_pass()])

        object_prompts = prompts[2:]
        self.assertEqual(len(object_prompts), 1)
        self.assertEqual(self._marker_categories(object_prompts), ["reading_prep"])
        self.assertEqual(meta["object_scene_category"], "reading_prep")
        self.assertEqual(meta["object_generation_attempts"], "1")
        self.assertIn(
            OBJECT_SCENE_CATEGORIES["reading_prep"],
            _prepare_pollinations_prompt(object_prompts[0]),
        )

    def test_object_contains_text_switches_only_the_final_object_attempt(self):
        """Regression 2: attempt 1 keeps reading_prep, attempt 2 goes text-safe."""

        _buffer, meta, prompts = self._run_ladder([_object_text_fail(), _object_pass()])

        object_prompts = prompts[2:]
        self.assertEqual(len(object_prompts), 2)
        self.assertEqual(
            self._marker_categories(object_prompts),
            ["reading_prep", OBJECT_TEXT_SAFE_RETRY_CATEGORY],
        )
        self.assertEqual(meta["object_scene_category"], OBJECT_TEXT_SAFE_RETRY_CATEGORY)

    def test_text_safe_retry_prompt_drops_the_reading_prep_props(self):
        """Regression 3: the risky reading-prep vocabulary is gone from attempt 2."""

        _buffer, _meta, prompts = self._run_ladder([_object_text_fail(), _object_pass()])

        first = _prepare_pollinations_prompt(prompts[2]).lower()
        retry = _prepare_pollinations_prompt(prompts[3]).lower()

        self.assertIn(OBJECT_SCENE_CATEGORIES["reading_prep"].lower(), first)
        self.assertNotIn(OBJECT_SCENE_CATEGORIES["reading_prep"].lower(), retry)
        for phrase in _READING_PREP_RISKY_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, retry)
        # The props themselves carry no printable surface at all, so the retry
        # cannot regress by re-adding one through the category constant.
        safe_props = OBJECT_SCENE_CATEGORIES[OBJECT_TEXT_SAFE_RETRY_CATEGORY].lower()
        for token in ("book", "card", "letter", "pencil", "paper", "page", "sign", "label", "screen"):
            with self.subTest(token=token):
                self.assertNotIn(token, safe_props)

    def test_object_attempt_count_stays_exactly_two(self):
        """Regression 4: the bounded object budget is unchanged."""

        _buffer, meta, prompts = self._run_ladder(
            [_object_text_fail(), _object_text_fail()]
        )

        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(prompts[2:]), 2)
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["object_qa_attempts"], "2")

    def test_human_visual_retry_count_stays_unchanged(self):
        """Regression 5: the human stage still gets exactly one retry."""

        _buffer, meta, prompts = self._run_ladder(
            [_object_text_fail(), _object_text_fail()]
        )

        human_prompts = prompts[:2]
        self.assertEqual(len(human_prompts), 2)
        # Human prompts carry no object-scene marker: the human stage is untouched.
        self.assertEqual(self._marker_categories(human_prompts), [])
        self.assertEqual(meta["visual_qa_attempts"], "2")
        self.assertEqual(meta["human_qa_first_reason"], "action_mismatch")
        self.assertEqual(meta["human_qa_retry_reason"], "object_contains_text")

    def test_passing_text_safe_retry_is_accepted_through_the_object_path(self):
        """Regression 6: a passing safe retry publishes as an object fallback."""

        buffer, meta, prompts = self._run_ladder([_object_text_fail(), _object_pass()])

        self.assertEqual(buffer.getvalue(), b"image-4")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["visual_source"], "object_ai")
        self.assertEqual(meta["fallback_stage"], "object")
        self.assertEqual(meta["final_reason"], "object_fallback_success")
        self.assertEqual(meta["object_prompt_used"], "True")
        self.assertEqual(meta["text_fallback_used"], "False")

    def test_failing_text_safe_retry_still_reaches_the_text_fallback(self):
        """Regression 7: the terminal text card is unchanged."""

        buffer, meta, _prompts = self._run_ladder(
            [_object_text_fail(), _object_text_fail()]
        )

        self.assertEqual(buffer.getvalue(), b"text-card")
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["visual_source"], "text_card")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["final_reason"], "object_fallback_rejected")
        self.assertEqual(meta["text_fallback_used"], "True")
        self.assertEqual(meta["object_scene_category"], OBJECT_TEXT_SAFE_RETRY_CATEGORY)

    def test_object_contains_text_stays_fail_closed(self):
        """Regression 8: the QA enforcement itself is untouched."""

        normalized = _enforce_object_visual_qa(
            {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 0,
                "adult_count": 0,
                "child_count": 0,
                "ppe_detected": False,
                "text_detected": True,
                "illustration_style_match": True,
            }
        )

        self.assertEqual(normalized["status"], "fail")
        self.assertFalse(normalized["pass"])
        self.assertEqual(normalized["reason"], OBJECT_TEXT_FAILURE_REASON)

    def test_non_text_object_failure_keeps_the_derived_category(self):
        """Regression 9: unrelated QA reasons keep today's behavior."""

        self.assertEqual(
            _text_safe_object_retry_category(
                OBJECT_TEXT_SAFE_RETRY_SOURCE_CATEGORY, OBJECT_TEXT_FAILURE_REASON
            ),
            OBJECT_TEXT_SAFE_RETRY_CATEGORY,
        )
        for reason in ("object_style_mismatch", "object_contains_person", "object_topic_mismatch", "", None):
            with self.subTest(reason=reason):
                self.assertEqual(
                    _text_safe_object_retry_category(
                        OBJECT_TEXT_SAFE_RETRY_SOURCE_CATEGORY, reason
                    ),
                    "",
                )

        _buffer, meta, prompts = self._run_ladder(
            [_object_non_text_fail(), _object_non_text_fail()]
        )

        self.assertEqual(
            self._marker_categories(prompts[2:]), ["reading_prep", "reading_prep"]
        )
        self.assertEqual(meta["object_scene_category"], "reading_prep")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["mode"], "text_fallback")

    def test_object_contains_text_on_a_non_reading_category_keeps_that_category(self):
        """Categories outside reading prep and books retain their route."""

        # The helper refuses unrelated categories, text reason or not.
        for category in (
            _NON_READING_CATEGORY,
            "articulation_speech",
            "household_routines",
            "default",
            "",
            None,
        ):
            with self.subTest(category=category):
                self.assertEqual(
                    _text_safe_object_retry_category(category, OBJECT_TEXT_FAILURE_REASON),
                    "",
                )

        _buffer, meta, prompts = self._run_ladder(
            [_object_text_fail(), _object_text_fail()],
            title=_NON_READING_TITLE,
            image_prompt=_NON_READING_PROMPT,
        )

        object_prompts = prompts[2:]
        # First attempt is the originally derived non-reading category ...
        self.assertEqual(
            self._marker_categories(object_prompts),
            [_NON_READING_CATEGORY, _NON_READING_CATEGORY],
        )
        # ... and so is the second, despite `object_contains_text` on the first.
        self.assertNotIn(
            OBJECT_TEXT_SAFE_RETRY_CATEGORY, self._marker_categories(object_prompts)
        )
        self.assertEqual(meta["object_scene_category"], _NON_READING_CATEGORY)
        self.assertEqual(meta["object_qa_reason"], OBJECT_TEXT_FAILURE_REASON)
        # No extra object attempt was added.
        self.assertEqual(len(object_prompts), 2)
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["mode"], "text_fallback")

    # The artistic substrate the style block is allowed to name, and the global
    # negatives sentence: neither depicts a printable object, so both are stripped
    # before the provider prompt is scanned for depicted printable surfaces.
    _ALLOWED_SUBSTRATE_PHRASE = "subtle watercolor paper texture"

    @staticmethod
    def _text_safe_variation_ids(count=256):
        return ["%012x" % index for index in range(count)]

    def test_text_safe_provider_composition_pool_excludes_printable_surfaces(self):
        """No text-safe composition variant may depict a printable surface."""

        # The unsafe shared variant exists, so this pool is a real narrowing.
        self.assertIn(
            "painted arrangement with generous blank paper space",
            OBJECT_PROVIDER_COMPOSITIONS,
        )
        self.assertNotIn(
            "painted arrangement with generous blank paper space",
            OBJECT_TEXT_SAFE_PROVIDER_COMPOSITIONS,
        )
        self.assertTrue(OBJECT_TEXT_SAFE_PROVIDER_COMPOSITIONS)

        for composition in OBJECT_TEXT_SAFE_PROVIDER_COMPOSITIONS:
            for token in OBJECT_TEXT_SAFE_COMPOSITION_BANNED_TOKENS:
                with self.subTest(composition=composition, token=token):
                    self.assertNotIn(token, composition.lower())

    def test_every_reachable_text_safe_composition_variant_is_text_safe(self):
        """Exhaustive over the selection rule, not one chosen variation id."""

        reached = set()
        for variation_id in self._text_safe_variation_ids():
            composition = _object_provider_composition(
                OBJECT_TEXT_SAFE_RETRY_CATEGORY, variation_id
            )
            with self.subTest(variation_id=variation_id):
                self.assertIn(composition, OBJECT_TEXT_SAFE_PROVIDER_COMPOSITIONS)
            reached.add(composition)

        # Every safe variant is actually reachable, so the coverage above is total.
        self.assertEqual(reached, set(OBJECT_TEXT_SAFE_PROVIDER_COMPOSITIONS))

    def test_text_safe_provider_prompts_never_depict_a_printable_surface(self):
        """The assembled provider-facing prompt carries no printable surface."""

        for variation_id in self._text_safe_variation_ids():
            prompt = build_object_provider_prompt(
                OBJECT_TEXT_SAFE_RETRY_CATEGORY, variation_id
            ).lower()
            with self.subTest(variation_id=variation_id):
                self.assertNotIn("blank paper space", prompt)
                # Strip the allowed artistic substrate and the global negatives,
                # then nothing depicting a printable surface may remain.
                scanned = prompt.replace(self._ALLOWED_SUBSTRATE_PHRASE, " ").replace(
                    OBJECT_PROVIDER_NEGATIVES.lower(), " "
                )
                for token in OBJECT_TEXT_SAFE_COMPOSITION_BANNED_TOKENS:
                    self.assertNotIn(token, scanned, msg=f"token={token!r}")

    def test_other_categories_keep_the_shared_composition_pool(self):
        """The narrowing must not change any pre-existing category."""

        for category in OBJECT_SCENE_CATEGORIES:
            if category == OBJECT_TEXT_SAFE_RETRY_CATEGORY:
                continue
            for variation_id in self._text_safe_variation_ids(48):
                digest = hashlib.sha256(
                    f"{category}|{variation_id}".encode("utf-8")
                ).hexdigest()
                expected = OBJECT_PROVIDER_COMPOSITIONS[
                    int(digest[:8], 16) % len(OBJECT_PROVIDER_COMPOSITIONS)
                ]
                with self.subTest(category=category, variation_id=variation_id):
                    self.assertEqual(
                        _object_provider_composition(category, variation_id), expected
                    )

    def test_generation_failure_on_first_attempt_does_not_trigger_the_switch(self):
        """A missing QA result is not an `object_contains_text` rejection."""

        qa_results = iter(
            [
                _human_fail("action_mismatch"),
                _human_fail("object_contains_text"),
                _object_pass(),
            ]
        )
        prompts = []
        calls = {"n": 0}

        def download(*, prompt, token):
            calls["n"] += 1
            prompts.append(prompt)
            if calls["n"] == 3:
                raise visual_pipeline.PollinationsImageError("object generation failed")
            return BytesIO(f"image-{calls['n']}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            _buffer, meta = build_post_visual(
                title=_READING_PREP_TITLE,
                day_key="2026-09-25",
                image_prompt=_READING_PREP_PROMPT,
                rubric_id="age_norms",
                audience="parents",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        self.assertEqual(self._marker_categories(prompts[2:]), ["reading_prep", "reading_prep"])
        self.assertEqual(meta["object_scene_category"], "reading_prep")
        self.assertEqual(meta["object_generation_attempts"], "2")

    def test_visual_pipeline_holds_no_telegram_or_publication_side_effects(self):
        """Regression 10: the patched module cannot touch delivery or state."""

        source = Path(visual_pipeline.__file__).read_text(encoding="utf-8")
        for token in (
            "api.telegram.org",
            "sendPhoto",
            "sendMessage",
            "mark_published",
            "save_state",
            "update_state",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


# --- compiled-context contamination of the object category (post-PR #78) ------
#
# `build_post_visual` hands the COMPILED visual prompt to the object fallback as
# `context_hint`, and every compiled prompt ends with a fixed style tail saying
# "No readable text ... letters ...". The reading markers are "чита", "букв",
# "read" and "letter", so that boilerplate matched "readable"/"letters" and
# shadowed the publication's real semantics. Run #500 is the production case: a
# brief whose action was "The preschool child points to the book" with props
# "book, picture" was classified `reading_prep` instead of
# `books_vocab_phrases_stories`.

_RUN_500_TITLE = "\u0427\u0442\u043e \u043e\u0431\u044b\u0447\u043d\u043e \u0443\u043c\u0435\u0435\u0442 \u0442\u0440\u0435\u0445\u043b\u0435\u0442\u043d\u0438\u0439 \u0440\u0435\u0431\u0435\u043d\u043e\u043a"
_RUN_500_ACTION = "The preschool child points to the book"
_RUN_500_PROPS = ("book", "picture")


class ObjectSceneCompiledContextContaminationTest(unittest.TestCase):
    """The object category must read publication semantics, not style boilerplate."""

    def _category(self, title, action, props, rubric="age_norms"):
        return _object_scene_category(
            title, rubric, context_hint=_compiled_context(action, props, rubric)
        )

    def test_compiled_style_tail_is_excluded_from_classification(self):
        """The helper cuts the hint at the style marker and nowhere else."""

        compiled = _compiled_context(_RUN_500_ACTION, _RUN_500_PROPS)
        self.assertIn(OBJECT_SCENE_CONTEXT_STYLE_MARKER, compiled)
        # The tail really does carry the words the reading markers look for.
        self.assertIn("readable text", compiled)
        self.assertIn("letters", compiled)

        semantics = _object_scene_context_semantics(compiled)
        self.assertTrue(compiled.startswith(semantics))
        self.assertNotIn(OBJECT_SCENE_CONTEXT_STYLE_MARKER, semantics)
        self.assertNotIn("readable", semantics)
        self.assertNotIn("letters", semantics)
        # Genuine semantics before the marker survive untouched.
        self.assertIn(_RUN_500_ACTION, semantics)
        # A hint without the marker is passed through unchanged.
        self.assertEqual(
            _object_scene_context_semantics("the child looks at letter cards"),
            "the child looks at letter cards",
        )
        self.assertEqual(_object_scene_context_semantics(""), "")

    def test_run_500_book_picture_brief_is_books_vocab_not_reading_prep(self):
        """Regression 1: the exact run #500 semantic case."""

        category = self._category(_RUN_500_TITLE, _RUN_500_ACTION, _RUN_500_PROPS)

        self.assertEqual(category, "books_vocab_phrases_stories")
        self.assertNotEqual(category, "reading_prep")

    def test_neutral_compiled_brief_stays_default(self):
        """Regression 2: `readable`/`letters` alone must not mean reading_prep."""

        category = self._category(
            "\u0421\u043f\u043e\u043a\u043e\u0439\u043d\u043e\u0435 \u0443\u0442\u0440\u043e \u0434\u043e\u043c\u0430",
            "The child sits calmly at the low table",
            (),
        )

        self.assertEqual(category, "default")
        self.assertNotEqual(category, "reading_prep")

    def test_game_compiled_brief_stays_games_everyday_communication(self):
        """Regression 3: genuine ball/game semantics win over the boilerplate."""

        category = self._category(
            "\u0412\u0435\u0441\u0435\u043b\u0430\u044f \u0438\u0433\u0440\u0430 \u0441 \u043c\u044f\u0447\u043e\u043c",
            "The parent and child roll a ball to each other",
            ("ball", "basket"),
        )

        self.assertEqual(category, "games_everyday_communication")
        self.assertNotEqual(category, "reading_prep")

    def test_genuine_reading_semantics_still_classify_as_reading_prep(self):
        """Regression 4: real reading content before the marker is preserved."""

        category = self._category(
            "\u0413\u043e\u0442\u043e\u0432\u0438\u043c \u0440\u0435\u0431\u0435\u043d\u043a\u0430 \u043a \u0447\u0442\u0435\u043d\u0438\u044e",
            "The parent and child look at letter cards and learn to read",
            ("letter cards",),
        )

        self.assertEqual(category, "reading_prep")

    def test_higher_priority_categories_are_preserved(self):
        """Regression 5: precedence above reading_prep is untouched."""

        cases = (
            (
                "articulation_speech",
                "\u041a\u0430\u043a \u043f\u043e\u0441\u0442\u0430\u0432\u0438\u0442\u044c \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u044f\u0446\u0438\u044e \u0437\u0432\u0443\u043a\u0430 \u0421",
                "The child watches tongue position in a mirror",
                ("mirror",),
            ),
            (
                "bilingual_languages",
                "\u0420\u0435\u0431\u0435\u043d\u043e\u043a \u0440\u0430\u0441\u0442\u0435\u0442 \u0432 \u0434\u0432\u0443\u044f\u0437\u044b\u0447\u043d\u043e\u0439 \u0441\u0435\u043c\u044c\u0435",
                "The parent speaks the home language with the child",
                ("globe",),
            ),
            (
                "hearing_sounds_music",
                "\u0420\u0435\u0430\u043a\u0446\u0438\u044f \u043c\u0430\u043b\u044b\u0448\u0430 \u043d\u0430 \u043a\u043e\u043b\u043e\u043a\u043e\u043b\u044c\u0447\u0438\u043a",
                "The parent rings a small bell",
                ("bell", "drum"),
            ),
            (
                "household_routines",
                "\u0420\u0430\u0437\u0433\u043e\u0432\u043e\u0440\u044b \u0432\u043e \u0432\u0440\u0435\u043c\u044f \u0431\u044b\u0442\u043e\u0432\u044b\u0445 \u0434\u0435\u043b",
                "The parent folds laundry with the child",
                ("laundry basket",),
            ),
        )

        for expected, title, action, props in cases:
            with self.subTest(category=expected):
                self.assertEqual(self._category(title, action, props), expected)

    def test_run_500_like_ladder_uses_books_vocab_then_text_safe(self):
        """Regression 6: the full offline ladder for the production brief."""

        qa_results = iter(
            [
                _human_fail("action_mismatch"),
                _human_fail("object_contains_text"),
                _object_text_fail(),
                _object_text_fail(),
            ]
        )
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"image-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ), patch(
            "src.services.visual_pipeline.build_fallback_cover_buffer",
            return_value=BytesIO(b"text-card"),
        ):
            buffer, meta = build_post_visual(
                title=_RUN_500_TITLE,
                day_key="2026-09-25",
                image_prompt=_compiled_context(_RUN_500_ACTION, _RUN_500_PROPS),
                rubric_id="age_norms",
                audience="parents",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        human_prompts, object_prompts = prompts[:2], prompts[2:]
        categories = ObjectContainsTextSafeRetryTest._marker_categories(object_prompts)

        # Human stage untouched.
        self.assertEqual(len(human_prompts), 2)
        self.assertEqual(meta["visual_qa_attempts"], "2")
        # First object category is the clean semantic one ...
        self.assertEqual(categories[0], "books_vocab_phrases_stories")
        # ... and the text rejection switches only the last object attempt.
        self.assertEqual(
            categories, ["books_vocab_phrases_stories", OBJECT_TEXT_SAFE_RETRY_CATEGORY]
        )
        self.assertEqual(meta["object_scene_category"], OBJECT_TEXT_SAFE_RETRY_CATEGORY)
        # Object budget and terminal fallback unchanged.
        self.assertEqual(len(object_prompts), 2)
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["final_reason"], "object_fallback_rejected")
        self.assertEqual(buffer.getvalue(), b"text-card")

    def test_pr_78_text_safe_retry_still_fires_for_genuine_reading(self):
        """Regression 7: PR #78 behavior is preserved where it belongs."""

        qa_results = iter(
            [
                _human_fail("action_mismatch"),
                _human_fail("object_contains_text"),
                _object_text_fail(),
                _object_pass(),
            ]
        )
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"image-{len(prompts)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            _buffer, meta = build_post_visual(
                title="\u0413\u043e\u0442\u043e\u0432\u0438\u043c \u0440\u0435\u0431\u0435\u043d\u043a\u0430 \u043a \u0447\u0442\u0435\u043d\u0438\u044e",
                day_key="2026-09-25",
                image_prompt=_compiled_context(
                    "The parent and child look at letter cards and learn to read",
                    ("letter cards",),
                ),
                rubric_id="age_norms",
                audience="parents",
                visual_qa_fn=lambda *_a, **_k: next(qa_results),
            )

        object_prompts = prompts[2:]
        categories = ObjectContainsTextSafeRetryTest._marker_categories(object_prompts)

        self.assertEqual(categories[0], "reading_prep")
        self.assertEqual(categories, ["reading_prep", OBJECT_TEXT_SAFE_RETRY_CATEGORY])
        self.assertEqual(meta["object_scene_category"], OBJECT_TEXT_SAFE_RETRY_CATEGORY)
        self.assertEqual(len(object_prompts), 2)
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["visual_qa_attempts"], "2")


class Run501BooksTextSafeRetryTest(unittest.TestCase):
    """The Monday book/picture brief retains its category and bounded budget."""

    _TITLE = "Покажите картинку в книге и попросите назвать"
    _PROMPT = _compile_visual_prompt(
        VisualBrief(
            rubric_id="tip_of_day",
            role_rule="Exactly one adult parent and exactly one toddler, visibly different in age and height, no other people.",
            age_descriptor="toddler",
            setting="simple uncluttered home play area",
            action='Open the book to a page and point at the picture while asking “What is this?”',
            props=("book", "picture"),
        )
    )

    def _run(self, first_object_qa, second_object_qa):
        qa_results = iter([
            _human_fail("missing_required_child"),
            _human_fail("wrong_character_roles"),
            first_object_qa,
            second_object_qa,
        ])
        prompts = []
        qa_calls = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(f"image-{len(prompts)}".encode()), {"attempts_used": "1"}

        def qa(*args, **kwargs):
            qa_calls.append((args, kwargs))
            return next(qa_results)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ), patch(
            "src.services.visual_pipeline.build_fallback_cover_buffer",
            return_value=BytesIO(b"text-card"),
        ):
            buffer, meta = build_post_visual(
                title=self._TITLE,
                day_key="2026-09-28",
                image_prompt=self._PROMPT,
                rubric_id="tip_of_day",
                audience="parents",
                visual_qa_fn=qa,
            )
        return buffer, meta, prompts, qa_calls

    def test_text_rejection_switches_only_final_attempt_and_passes_qa(self):
        self.assertEqual(
            _object_scene_category(self._TITLE, "tip_of_day", context_hint=self._PROMPT),
            "books_vocab_phrases_stories",
        )
        buffer, meta, prompts, qa_calls = self._run(_object_text_fail(), _object_pass())
        object_prompts = prompts[2:]
        self.assertEqual(len(prompts[:2]), 2)
        self.assertEqual(len(object_prompts), 2)
        self.assertEqual(
            ObjectContainsTextSafeRetryTest._marker_categories(object_prompts),
            ["books_vocab_phrases_stories", OBJECT_TEXT_SAFE_RETRY_CATEGORY],
        )
        self.assertEqual(len(qa_calls), 4)
        first_match = OBJECT_SCENE_MARKER_RE.search(object_prompts[0])
        self.assertEqual(
            _prepare_pollinations_prompt(object_prompts[0]),
            build_object_provider_prompt("books_vocab_phrases_stories", first_match.group(2)),
        )
        retry = _prepare_pollinations_prompt(object_prompts[1]).lower()
        scanned = retry.replace("subtle watercolor paper texture", " ").replace(
            OBJECT_PROVIDER_NEGATIVES.lower(), " "
        )
        self.assertIn(OBJECT_SCENE_CATEGORIES[OBJECT_TEXT_SAFE_RETRY_CATEGORY].lower(), retry)
        for token in OBJECT_TEXT_SAFE_COMPOSITION_BANNED_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token, scanned)
        self.assertEqual(buffer.getvalue(), b"image-4")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["object_scene_category"], OBJECT_TEXT_SAFE_RETRY_CATEGORY)
        self.assertEqual(meta["object_qa_status"], "pass")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["object_qa_attempts"], "2")
        self.assertEqual(meta["human_qa_retry_reason"], "wrong_character_roles")

    def test_repeat_text_rejection_uses_text_card_after_two_objects(self):
        buffer, meta, prompts, qa_calls = self._run(_object_text_fail(), _object_text_fail())
        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(qa_calls), 4)
        self.assertEqual(
            ObjectContainsTextSafeRetryTest._marker_categories(prompts[2:]),
            ["books_vocab_phrases_stories", OBJECT_TEXT_SAFE_RETRY_CATEGORY],
        )
        self.assertEqual(buffer.getvalue(), b"text-card")
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["object_qa_reason"], OBJECT_TEXT_FAILURE_REASON)
        self.assertEqual(meta["final_reason"], "object_fallback_rejected")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["object_qa_attempts"], "2")

    def test_other_book_qa_failure_keeps_book_category(self):
        self.assertEqual(_text_safe_object_retry_category("books_vocab_phrases_stories", ""), "")
        _buffer, meta, prompts, qa_calls = self._run(_object_non_text_fail(), _object_pass())
        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(qa_calls), 4)
        self.assertEqual(
            ObjectContainsTextSafeRetryTest._marker_categories(prompts[2:]),
            ["books_vocab_phrases_stories"] * 2,
        )
        self.assertEqual(meta["object_scene_category"], "books_vocab_phrases_stories")
        self.assertEqual(meta["mode"], "ai_object_fallback")



if __name__ == "__main__":
    unittest.main()

from io import BytesIO
from contextlib import redirect_stdout
import os
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import requests

from src.services.visual_pipeline import (
    _object_scene_category,
    _visual_qa_key_candidates,
    build_object_only_visual_prompt,
    build_post_visual,
    evaluate_visual_quality,
)


def _qa_response(status_code, payload=None):
    response = Mock(status_code=status_code)
    response.json.return_value = payload or {
        "candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1}'}]}}]
    }
    return response


class VisualFallbackPolicyTest(unittest.TestCase):
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

    def test_two_403_responses_use_object_fallback(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[_qa_response(403), _qa_response(403)],
        ) as request, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {}), (BytesIO(b"object"), {})],
        ):
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
            )

        self.assertEqual(buffer.getvalue(), b"object")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["human_qa_key_source"], "general")
        self.assertEqual(meta["human_qa_key_attempts"], "2")
        self.assertEqual(meta["fallback_trigger"], "qa_unavailable_for_required_rubric")
        self.assertEqual(request.call_count, 2)

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

    def test_timeout_can_use_next_key_once(self):
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET", "GEMINI_API_KEY": "GENERAL_SECRET"},
            clear=True,
        ), patch(
            "src.services.visual_pipeline.requests.post",
            side_effect=[requests.Timeout(), _qa_response(200)],
        ) as request:
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["human_qa_key_source"], "general")
        self.assertEqual(result["human_qa_key_fallback_trigger"], "timeout")
        self.assertEqual(request.call_count, 2)

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
        hard = _qa_response(200, {"candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ghosted_figure", "people_count": 2, "adult_count": 1, "child_count": 1}'}]}}]})
        with patch.dict(os.environ, {"GEMINI_VISUAL_QA_API_KEY": "VISUAL_SECRET"}, clear=True), patch(
            "src.services.visual_pipeline.requests.post", return_value=hard
        ):
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "ghosted_figure")

    def test_object_prompt_is_people_free_and_does_not_include_raw_title(self):
        prompt = build_object_only_visual_prompt(
            "Русская игра с мячом и ребёнком", "play_and_speak", "raw Russian prompt"
        )
        self.assertIn("Object-only educational still life", prompt)
        self.assertIn("No people", prompt)
        self.assertIn("No text", prompt)
        self.assertIn("16:9 landscape", prompt)
        self.assertNotIn("Русская игра", prompt)
        self.assertNotIn("raw Russian prompt", prompt)

    def test_skipped_human_qa_uses_object_fallback_without_object_qa(self):
        qa_calls = []

        def qa(*args, **kwargs):
            qa_calls.append(kwargs)
            return {"status": "skipped", "pass": True, "reason": "qa_http_429"}

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
        self.assertEqual(len(qa_calls), 1)
        self.assertEqual(download.call_count, 2)

    def test_two_human_failures_then_object_failure_use_text_fallback(self):
        qa_results = iter(
            [
                {"status": "fail", "pass": False, "reason": "ghosted_figure"},
                {"status": "fail", "pass": False, "reason": "action_mismatch"},
            ]
        )
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {}), (BytesIO(b"retry"), {}), RuntimeError("object failed")],
        ):
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )
        self.assertNotIn(buffer.getvalue(), {b"human", b"retry"})
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["human_qa_first_reason"], "ghosted_figure")
        self.assertEqual(meta["human_qa_retry_reason"], "action_mismatch")

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
        )
        for title, rubric_id, expected in cases:
            with self.subTest(title=title, rubric_id=rubric_id):
                self.assertEqual(_object_scene_category(title, rubric_id), expected)

    def test_legacy_bilingual_rubric_does_not_override_neutral_title(self):
        self.assertNotEqual(
            _object_scene_category("Речь в разных ситуациях", "bilingual_corner"),
            "bilingual_languages",
        )

    def test_lone_language_word_does_not_select_bilingual_category(self):
        self.assertNotEqual(_object_scene_category("Положение языка", "tip_of_day"), "bilingual_languages")


if __name__ == "__main__":
    unittest.main()

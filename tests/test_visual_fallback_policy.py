from io import BytesIO
from contextlib import redirect_stdout
import os
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import requests

from src.services.visual_pipeline import (
    DEFAULT_GEMINI_VISUAL_QA_MODEL,
    VISUAL_STYLE_TAIL,
    _enforce_object_visual_qa,
    _object_scene_category,
    _visual_qa_key_candidates,
    build_object_only_visual_prompt,
    build_post_visual,
    build_visual_role_rule,
    evaluate_visual_quality,
)


def _qa_response(status_code, payload=None):
    response = Mock(status_code=status_code)
    response.json.return_value = payload or {
        "candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1, "ppe_detected": false}'}]}}]
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


if __name__ == "__main__":
    unittest.main()

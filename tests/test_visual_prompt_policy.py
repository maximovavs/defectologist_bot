from io import BytesIO
import os
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.services.llm_generator import _validate_image_prompt, build_image_prompt_prompt
from src.services.visual_pipeline import (
    POLLINATIONS_GEN_HEIGHT,
    POLLINATIONS_GEN_WIDTH,
    _enhance_image_prompt,
    _safe_visual_qa,
    _visual_qa_is_required,
    build_post_visual,
    build_visual_retry_prompt,
    evaluate_visual_quality,
    _normalize_pollinations_image,
)


class VisualPromptPolicyTest(unittest.TestCase):
    def test_enhanced_prompt_contains_anti_distortion_rules(self):
        prompt = _enhance_image_prompt("parent and child reading together")
        lower = prompt.lower()

        self.assertIn("horizontal cover composition suitable for telegram", lower)
        self.assertIn("natural human proportions", lower)
        self.assertIn("no stretched faces", lower)
        self.assertIn("no widened bodies", lower)
        self.assertIn("no widened torsos", lower)
        self.assertIn("no elongated arms or enlarged hands", lower)
        self.assertIn("normal camera perspective", lower)
        self.assertIn("avoid wide-angle lens distortion", lower)
        self.assertIn("avoid panoramic distortion", lower)
        self.assertIn("one clear focal group", lower)
        self.assertIn("do not place people edge-to-edge across the frame", lower)
        self.assertNotIn("full-bleed 16:9 landscape", lower)
        self.assertNotIn("no blurred side panels", lower)
        self.assertIn("no headphones", lower)
        self.assertIn("no holiday imagery", lower)
        self.assertIn("warm editorial illustration", lower)
        self.assertIn("soft beige cream and warm pastel palette", lower)
        self.assertIn("no deformed hands", lower)
        self.assertIn("no extra or missing limbs", lower)
        self.assertIn("no anime", lower)
        self.assertIn("no 3d toy style", lower)

    def test_method_prompt_allows_only_present_props(self):
        prompt = build_image_prompt_prompt(
            title="Игра с карточками",
            body_text="Специалист показывает карточки и просит ребенка назвать картинку.",
            audience="pros",
            rubric_id="method_piggybank",
        )

        allowed_props = prompt.split("Allowed props:", 1)[1].split("Title:", 1)[0]
        self.assertIn("picture cards", allowed_props)
        self.assertNotIn("mirror", allowed_props)
        self.assertNotIn("headphones", allowed_props)

    def test_default_generation_dimensions_are_landscape(self):
        self.assertEqual(POLLINATIONS_GEN_WIDTH, 1280)
        self.assertEqual(POLLINATIONS_GEN_HEIGHT, 720)

    def test_normalized_image_is_landscape(self):
        source = Image.new("RGB", (900, 900), "red")
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))

    def test_normalized_portrait_image_preserves_foreground_aspect_ratio(self):
        source = Image.new("RGB", (100, 200), "white")
        for y in range(200):
            source.putpixel((0, y), (0, 0, 0))
            source.putpixel((99, y), (0, 0, 0))
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))
            row = [
                x
                for x in range(image.width)
                if all(channel < 40 for channel in image.getpixel((x, image.height // 2))[:3])
            ]

        self.assertGreater(len(row), 0)
        self.assertLess(max(row) - min(row), 390)

    def test_image_prompt_uses_safe_cover_language(self):
        prompt = build_image_prompt_prompt(
            title="Игра с карточками",
            body_text="Специалист показывает карточки и просит ребенка назвать картинку.",
            audience="pros",
            rubric_id="method_piggybank",
        ).lower()

        self.assertIn("horizontal cover composition suitable for telegram", prompt)
        self.assertIn("natural human proportions", prompt)
        self.assertIn("no stretched faces", prompt)
        self.assertIn("no widened bodies", prompt)
        self.assertIn("no widened torsos", prompt)
        self.assertIn("avoid wide-angle lens distortion", prompt)
        self.assertIn("avoid panoramic distortion", prompt)
        self.assertIn("breathing room around the main figures", prompt)
        self.assertIn("one clear focal group", prompt)
        self.assertIn("do not place people edge-to-edge across the frame", prompt)
        self.assertIn("exactly 1 adult specialist and 1 child", prompt)
        self.assertIn("hard maximum 2 visible people", prompt)
        self.assertIn("no classroom group", prompt)
        self.assertIn("never add siblings", prompt)
        self.assertNotIn("native full-bleed 16:9 landscape composition", prompt)
        self.assertNotIn("no blurred side panels", prompt)

    def test_normalized_near_16_9_image_uses_full_frame(self):
        source = Image.new("RGB", (1600, 900), "white")
        for x in range(1600):
            for y in range(8):
                source.putpixel((x, y), (0, 0, 0))
                source.putpixel((x, 899 - y), (0, 0, 0))
        for y in range(900):
            for x in range(8):
                source.putpixel((x, y), (0, 0, 0))
                source.putpixel((1599 - x, y), (0, 0, 0))
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))
            center_y = image.height // 2
            row = [
                x
                for x in range(image.width)
                if all(channel < 40 for channel in image.getpixel((x, center_y))[:3])
            ]
            center_x = image.width // 2
            column = [
                y
                for y in range(image.height)
                if all(channel < 40 for channel in image.getpixel((center_x, y))[:3])
            ]

        self.assertGreater(len(row), 0)
        self.assertGreater(len(column), 0)
        self.assertLessEqual(min(row), 10)
        self.assertGreaterEqual(max(row), 1269)
        self.assertLessEqual(min(column), 10)
        self.assertGreaterEqual(max(column), 709)

    def test_rubric_people_limits_are_explicit(self):
        parent_prompt = build_image_prompt_prompt(
            title="A home speech game",
            body_text="A parent and child name picture cards.",
            audience="parents",
            rubric_id="tip_of_day",
        ).lower()
        age_prompt = build_image_prompt_prompt(
            title="A developmental milestone",
            body_text="A child points to a familiar object.",
            audience="parents",
            rubric_id="age_norms",
        ).lower()

        self.assertIn("exactly 1 adult and 1 child", parent_prompt)
        self.assertIn("hard maximum 2 visible people", parent_prompt)
        self.assertIn("one child only", age_prompt)
        self.assertIn("no extra people", age_prompt)

    def test_visual_retry_prompt_is_stricter(self):
        base = _enhance_image_prompt("an adult and child practicing a speech game")
        retry = build_visual_retry_prompt(base, rubric_id="tip_of_day", audience="parents")

        self.assertGreater(len(retry), len(base))
        self.assertIn("exactly one adult parent and exactly one toddler or young child", retry.lower())
        self.assertIn("hard maximum two visible people", retry.lower())
        self.assertIn("no crowd", retry.lower())
        self.assertIn("no duplicate or ghosted figures", retry.lower())

    def test_method_retry_prompt_forbids_partial_and_background_people(self):
        retry = build_visual_retry_prompt(
            "one specialist demonstrates an articulation exercise with a child",
            rubric_id="method_piggybank",
            audience="pros",
        ).lower()

        self.assertIn("exactly one adult speech specialist", retry)
        self.assertIn("exactly one child", retry)
        self.assertIn("no other faces, heads, reflections, silhouettes", retry)
        self.assertIn("one activity only", retry)
        self.assertIn("no third person", retry)
        self.assertIn("no reading scene unless reading is explicitly required", retry)

    def test_people_limit_overrides_gemini_pass(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 3, "adult_count": 2, "child_count": 1}'}]}}]
        }
        with patch("src.services.visual_pipeline.requests.post", return_value=response):
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="method_piggybank",
                gemini_api_key="test-key",
                expected_prompt="one specialist and one child perform an articulation exercise",
            )

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "too_many_people")
        self.assertEqual(result["people_count"], 3)

    def test_people_limit_two_passes_and_unknown_is_fail_open(self):
        passed = _safe_visual_qa(
            lambda *_args, **_kwargs: {"status": "pass", "pass": True, "reason": "ok", "people_count": 2},
            BytesIO(b"image"),
            rubric_id="method_piggybank",
            audience="pros",
        )
        unknown = _safe_visual_qa(
            lambda *_args, **_kwargs: {"status": "pass", "pass": True, "reason": "ok", "people_count": "unknown"},
            BytesIO(b"image"),
            rubric_id="method_piggybank",
            audience="pros",
        )

        self.assertTrue(passed["pass"])
        self.assertEqual(passed["people_count"], 2)
        self.assertTrue(unknown["pass"])
        self.assertEqual(unknown["people_count"], "unknown")

    def test_visual_qa_hard_reasons_override_pass_and_normalize(self):
        cases = (
            ("ghosted_figure", "ghosted_figure"),
            ("action_mismatch", "action_mismatch"),
            ("duplicate-figure", "duplicate_figure"),
            ("widened_torso", "widened_torso"),
            ("horizontal_stretch", "horizontal_stretch"),
        )

        for reason, normalized_reason in cases:
            with self.subTest(reason=reason):
                result = _safe_visual_qa(
                    lambda *_args, reason=reason, **_kwargs: {
                        "status": "pass",
                        "pass": True,
                        "reason": reason,
                        "people_count": 2,
                        "adult_count": 1,
                        "child_count": 1,
                    },
                    BytesIO(b"image"),
                    rubric_id="method_piggybank",
                    audience="pros",
                )

                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], normalized_reason)

    def test_parent_rubric_adult_only_scene_fails_missing_required_child(self):
        result = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": 2,
                "child_count": 0,
            },
            BytesIO(b"image"),
            rubric_id="tip_of_day",
            audience="parents",
        )

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "missing_required_child")
        self.assertEqual(result["adult_count"], 2)
        self.assertEqual(result["child_count"], 0)

    def test_parent_rubric_one_adult_one_child_passes(self):
        result = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
            },
            BytesIO(b"image"),
            rubric_id="tip_of_day",
            audience="parents",
        )

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertEqual(result["reason"], "ok")

    def test_visual_qa_technical_skipped_result_remains_fail_open(self):
        result = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "skipped",
                "pass": True,
                "reason": "qa timeout",
                "people_count": "unknown",
            },
            BytesIO(b"image"),
            rubric_id="method_piggybank",
            audience="pros",
        )

        self.assertEqual(result["status"], "skipped")
        self.assertTrue(result["pass"])
        self.assertEqual(result["reason"], "qa_timeout")

    def test_visual_qa_prompt_contains_counting_and_expected_action_rules(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1}'}]}}]
        }
        with patch("src.services.visual_pipeline.requests.post", return_value=response) as post:
            evaluate_visual_quality(
                BytesIO(b"image"),
                rubric_id="method_piggybank",
                audience="pros",
                gemini_api_key="test-key",
                expected_prompt="one specialist and one child perform an articulation exercise",
            )

        qa_text = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"].lower()
        self.assertIn("count every visible human face, head, torso, reflection", qa_text)
        self.assertIn("adult_count", qa_text)
        self.assertIn("child_count", qa_text)
        self.assertIn("count adults, children, and all visible people separately", qa_text)
        self.assertIn("exactly 1 adult parent and exactly 1 toddler or young child", qa_text)
        self.assertIn("exactly 1 adult specialist and exactly 1 child", qa_text)
        self.assertIn("missing_required_child", qa_text)
        self.assertIn("do not ignore small background figures", qa_text)
        self.assertIn("expected image prompt/action", qa_text)
        self.assertIn("articulation exercise", qa_text)
        self.assertIn("action_mismatch", qa_text)

    def test_visual_qa_prefers_separate_key_over_shared_key(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"pass": true, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1}'}]}}]
        }
        with patch.dict(
            os.environ,
            {"GEMINI_VISUAL_QA_API_KEY": "visual-key", "GEMINI_API_KEY": "shared-key"},
        ), patch("src.services.visual_pipeline.requests.post", return_value=response) as post:
            result = evaluate_visual_quality(BytesIO(b"image"), rubric_id="tip_of_day")

        self.assertTrue(result["pass"])
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "visual-key")

    def test_visual_qa_pass_does_not_retry(self):
        fake_buffer = BytesIO(b"first")
        qa_calls = []

        def qa(buffer, **kwargs):
            qa_calls.append(buffer)
            return {"status": "pass", "pass": True, "reason": "ok", "people_count": "2", "adult_count": 1, "child_count": 1}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(fake_buffer, {"attempts_used": "1", "final_reason": "ok"}),
        ) as download:
            _, meta = build_post_visual(
                title="Speech game",
                day_key="MO",
                image_prompt="an adult and child practicing a speech game",
                visual_qa_fn=qa,
                rubric_id="tip_of_day",
            )

        self.assertEqual(download.call_count, 1)
        self.assertEqual(len(qa_calls), 1)
        self.assertEqual(meta["mode"], "ai")
        self.assertEqual(meta["visual_qa_attempts"], "1")

    def test_method_piggybank_qa_http_429_uses_fallback_without_retry(self):
        first = BytesIO(b"first")

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(first, {"attempts_used": "1", "final_reason": "ok"}),
        ) as download:
            buffer, meta = build_post_visual(
                title="Method card",
                day_key="SA",
                image_prompt="speech specialist and child play a drum rhythm balance game",
                visual_qa_fn=lambda *_args, **_kwargs: {
                    "status": "skipped",
                    "pass": True,
                    "reason": "qa_http_429",
                    "people_count": "unknown",
                },
                rubric_id="method_piggybank",
            )

        self.assertEqual(download.call_count, 1)
        self.assertNotEqual(buffer.getvalue(), b"first")
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(meta["fallback_reason"], "qa_unavailable_for_required_rubric")
        self.assertEqual(meta["visual_qa_required"], "True")
        self.assertEqual(meta["visual_qa"], "skipped")
        self.assertEqual(meta["visual_qa_status"], "skipped")
        self.assertEqual(meta["visual_qa_reason"], "qa_http_429")
        self.assertEqual(meta["visual_qa_attempts"], "1")

    def test_method_piggybank_missing_visual_qa_key_uses_fallback(self):
        with patch.dict(os.environ, {"GEMINI_VISUAL_QA_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False), patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
        ) as download:
            buffer, meta = build_post_visual(
                title="Method card",
                day_key="SA",
                image_prompt="speech specialist and child play a drum rhythm balance game",
                rubric_id="method_piggybank",
            )

        self.assertEqual(download.call_count, 1)
        self.assertNotEqual(buffer.getvalue(), b"first")
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(meta["visual_qa_reason"], "gemini_key_missing")
        self.assertEqual(meta["visual_qa_attempts"], "1")

    def test_method_piggybank_invalid_qa_response_uses_fallback(self):
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
        ) as download:
            _, meta = build_post_visual(
                title="Method card",
                day_key="SA",
                image_prompt="speech specialist and child play a drum rhythm balance game",
                visual_qa_fn=lambda *_args, **_kwargs: {
                    "status": "skipped",
                    "pass": True,
                    "reason": "invalid_qa_response",
                    "people_count": "unknown",
                },
                rubric_id="method_piggybank",
            )

        self.assertEqual(download.call_count, 1)
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(meta["visual_qa_reason"], "invalid_qa_response")
        self.assertEqual(meta["fallback_reason"], "qa_unavailable_for_required_rubric")

    def test_method_piggybank_visual_qa_pass_uses_ai_image(self):
        first = BytesIO(b"first")

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(first, {"attempts_used": "1", "final_reason": "ok"}),
        ) as download:
            buffer, meta = build_post_visual(
                title="Method card",
                day_key="SA",
                image_prompt="speech specialist and child play a drum rhythm balance game",
                visual_qa_fn=lambda *_args, **_kwargs: {
                    "status": "pass",
                    "pass": True,
                    "reason": "ok",
                    "people_count": 2,
                    "adult_count": 1,
                    "child_count": 1,
                },
                rubric_id="method_piggybank",
            )

        self.assertEqual(download.call_count, 1)
        self.assertEqual(buffer.getvalue(), b"first")
        self.assertEqual(meta["mode"], "ai")
        self.assertEqual(meta["visual_qa_required"], "True")
        self.assertEqual(meta["visual_qa_status"], "pass")

    def test_visual_qa_failure_retries_once_then_accepts(self):
        first = BytesIO(b"first")
        second = BytesIO(b"second")
        qa_results = iter([
            {"status": "pass", "pass": True, "reason": "ghosted_figure", "people_count": 2, "adult_count": 1, "child_count": 1},
            {"status": "pass", "pass": True, "reason": "ok", "people_count": "2", "adult_count": 1, "child_count": 1},
        ])

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (first, {"attempts_used": "1", "final_reason": "ok"}),
                (second, {"attempts_used": "1", "final_reason": "ok"}),
            ],
        ) as download:
            _, meta = build_post_visual(
                title="Speech game",
                day_key="MO",
                image_prompt="an adult and child practicing a speech game",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
                rubric_id="tip_of_day",
            )

        self.assertEqual(download.call_count, 2)
        retry_prompt = download.call_args_list[1].kwargs["prompt"].lower()
        self.assertIn("hard maximum two visible people", retry_prompt)
        self.assertIn("exactly one adult parent and exactly one toddler or young child", retry_prompt)
        self.assertIn("no second adult", retry_prompt)
        self.assertIn("natural body width", retry_prompt)
        self.assertIn("no horizontal stretching", retry_prompt)
        self.assertIn("show the exact activity from the post", retry_prompt)
        self.assertEqual(meta["mode"], "ai")
        self.assertEqual(meta["visual_retry_used"], "True")
        self.assertEqual(meta["visual_qa_attempts"], "2")

    def test_method_piggybank_visual_qa_fail_then_retry_pass_uses_retry_image(self):
        qa_results = iter([
            {"status": "fail", "pass": False, "reason": "action_mismatch", "people_count": 2, "adult_count": 1, "child_count": 1},
            {"status": "pass", "pass": True, "reason": "ok", "people_count": 2, "adult_count": 1, "child_count": 1},
        ])

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
                (BytesIO(b"second"), {"attempts_used": "1", "final_reason": "ok"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Method card",
                day_key="SA",
                image_prompt="speech specialist and child play a drum rhythm balance game",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
                rubric_id="method_piggybank",
            )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(buffer.getvalue(), b"second")
        self.assertEqual(meta["mode"], "ai")
        self.assertEqual(meta["visual_retry_used"], "True")
        self.assertEqual(meta["visual_qa_status"], "pass")
        self.assertEqual(meta["visual_qa_attempts"], "2")

    def test_parent_rubric_skipped_visual_qa_uses_fallback_without_retry(self):
        first = BytesIO(b"first")

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(first, {"attempts_used": "1", "final_reason": "ok"}),
        ) as download:
            buffer, meta = build_post_visual(
                title="Speech game",
                day_key="MO",
                image_prompt="an adult and child practicing a speech game",
                visual_qa_fn=lambda *_args, **_kwargs: {
                    "status": "skipped",
                    "pass": True,
                    "reason": "qa_http_429",
                    "people_count": "unknown",
                },
                rubric_id="tip_of_day",
            )

        self.assertEqual(download.call_count, 1)
        self.assertNotEqual(buffer.getvalue(), b"first")
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(meta["fallback_reason"], "qa_unavailable_for_required_rubric")
        self.assertEqual(meta["visual_qa_required"], "True")
        self.assertEqual(meta["visual_qa_status"], "skipped")

    def test_visual_qa_required_rubrics_env_parses_multiple_ids(self):
        with patch.dict(os.environ, {"VISUAL_QA_REQUIRED_RUBRICS": "method_piggybank, tip_of_day age_norms"}):
            self.assertTrue(_visual_qa_is_required("method_piggybank"))
            self.assertTrue(_visual_qa_is_required("tip_of_day"))
            self.assertTrue(_visual_qa_is_required("age_norms"))
            self.assertFalse(_visual_qa_is_required("play_and_speak"))

    def test_visual_qa_failure_after_retry_uses_fallback(self):
        qa_results = iter([
            {"status": "pass", "pass": True, "reason": "action_mismatch", "people_count": 2, "adult_count": 1, "child_count": 1},
            {"status": "pass", "pass": True, "reason": "duplicate-figure", "people_count": 2, "adult_count": 1, "child_count": 1},
        ])

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
                (BytesIO(b"second"), {"attempts_used": "1", "final_reason": "ok"}),
            ],
        ) as download:
            _, meta = build_post_visual(
                title="Speech game",
                day_key="MO",
                image_prompt="an adult and child practicing a speech game",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
                rubric_id="tip_of_day",
            )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(meta["visual_qa"], "fail")
        self.assertEqual(meta["visual_qa_attempts"], "2")
        self.assertIn("duplicate_figure", meta["reason"])

    def test_rejects_santa_and_headphones_for_plain_speech_post(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of Santa wearing headphones with a child",
            body_text="Родитель и ребёнок повторяют короткую фразу во время игры с мячом.",
            rubric_id="tip_of_day",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "visual_prompt_topic_mismatch")

    def test_allows_headphones_for_explicit_listening_task(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of a child wearing headphones during a listening game",
            body_text="Ребёнок слушает аудио в наушниках и выбирает картинку с нужным звуком.",
            rubric_id="method_piggybank",
        )

        self.assertTrue(ok, reason)

    def test_allows_letter_cards_for_reading_post(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of a parent using letter cards with a child",
            body_text="Родитель показывает буквы и читает короткие слова вместе с ребёнком.",
            rubric_id="tip_of_day",
        )

        self.assertTrue(ok, reason)

    def test_rejects_random_floating_letters_for_dialogue_post(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of a parent and child with random floating letters",
            body_text="Родитель задаёт вопрос, ребёнок отвечает короткой фразой во время игры.",
            rubric_id="tip_of_day",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "visual_prompt_topic_mismatch")


if __name__ == "__main__":
    unittest.main()

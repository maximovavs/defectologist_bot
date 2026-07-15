import asyncio
from io import BytesIO
import importlib
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from src.services.llm_generator import (
    _compile_image_prompt_from_payload,
    _deterministic_visual_prompt,
    _extract_visual_age_descriptor,
    _mentioned_visual_props,
    _validate_image_prompt,
    build_image_prompt_prompt,
)
from src.services.visual_pipeline import (
    POLLINATIONS_GEN_HEIGHT,
    POLLINATIONS_GEN_WIDTH,
    VISUAL_STYLE_TAIL,
    _build_visual_qa_expected_brief,
    _compile_visual_prompt,
    _enhance_image_prompt,
    _parse_compiled_visual_prompt,
    _safe_visual_qa,
    _visual_qa_is_required,
    VisualBrief,
    build_visual_role_rule,
    build_post_visual,
    build_visual_retry_prompt,
    evaluate_visual_quality,
    _normalize_pollinations_image,
)


class VisualPromptPolicyTest(unittest.TestCase):
    def test_enhanced_prompt_adds_one_compact_style_tail(self):
        prompt = _enhance_image_prompt("parent and child reading together")
        lower = prompt.lower()

        self.assertIn("natural human proportions", lower)
        self.assertIn("simple naturally posed hands away from the camera", lower)
        self.assertIn("wide-angle distortion", lower)
        self.assertIn("stretched anatomy", lower)
        self.assertEqual(lower.count("warm soft editorial illustration"), 1)

    def test_enhance_image_prompt_is_idempotent(self):
        prompt = "parent and child reading together"
        once = _enhance_image_prompt(prompt)

        self.assertEqual(_enhance_image_prompt(once), once)

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

    def test_parent_18_30_months_compiles_parent_and_toddler_roles(self):
        body = (
            "Возраст: 18–30 мес.\n"
            "Как играть:\n"
            "Родитель показывает вежливую просьбу, ребёнок указывает на игрушку рядом с чашкой воды."
        )
        payload = {
            "action": "the parent models a polite request while the toddler points to a toy beside a cup of water",
            "setting": "simple home play area",
            "props": ["toy", "cup of water"],
        }

        prompt, brief, reason = _compile_image_prompt_from_payload(
            payload,
            body_text=body,
            audience="parents",
            rubric_id="tip_of_day",
        )

        self.assertEqual(reason, "ok")
        self.assertIsNotNone(brief)
        self.assertEqual(_extract_visual_age_descriptor(body), "2-year-old toddler")
        self.assertTrue(
            prompt.startswith(
                "Exactly one adult parent and exactly one 2-year-old toddler, "
                "visibly different in age and height, no other people."
            )
        )

    def test_parent_polite_request_compiles_exact_action_and_body_props(self):
        body = (
            "Возраст: 18–30 мес.\n"
            "Как играть: предложите ребёнку вежливо попросить игрушку рядом с чашкой воды."
        )
        action = "the parent models a polite request while the toddler points to a toy beside a cup of water"
        prompt, brief, reason = _compile_image_prompt_from_payload(
            {
                "action": action,
                "setting": "simple home play area",
                "props": ["toy", "cup of water", "mirror"],
            },
            body_text=body,
            audience="parents",
            rubric_id="play_and_speak",
        )

        self.assertEqual(reason, "ok")
        self.assertEqual(brief.action, action)
        self.assertEqual(brief.props, ("toy", "cup", "water"))
        self.assertIn(f"Action: {action}; allowed props: toy, cup, water.", prompt)
        self.assertNotIn("mirror", prompt.lower())

    def test_parent_compiled_prompt_has_no_conflicting_character_groups(self):
        body = "Возраст: 4 года. Как играть: родитель и ребёнок называют игрушку."
        prompt = _deterministic_visual_prompt(body, "parents", "tip_of_day").lower()

        for phrase in ("two adults", "two women", "family group", "classroom group", "siblings", "background people"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, prompt)

    def test_method_drum_and_metronome_compile_exact_roles_and_props(self):
        body = (
            "Цель: ритм и равновесие.\n"
            "Материалы: барабан и метроном.\n"
            "Как провести: специалист отбивает ритм на барабане, ребёнок повторяет."
        )
        prompt, brief, reason = _compile_image_prompt_from_payload(
            {
                "action": "the speech specialist taps a drum in time with a metronome while the child copies the rhythm",
                "setting": "busy classroom",
                "props": ["drum", "metronome", "picture cards"],
            },
            body_text=body,
            audience="pros",
            rubric_id="method_piggybank",
        )

        self.assertEqual(reason, "ok")
        self.assertTrue(prompt.startswith("Exactly one adult speech specialist and exactly one clearly younger child"))
        self.assertEqual(brief.props, ("drum", "metronome"))
        self.assertIn("Simple uncluttered speech therapy room", prompt)
        self.assertNotIn("classroom", prompt.lower())

    def test_unmentioned_and_risky_props_never_enter_compiled_prompt(self):
        body = "Как играть: родитель катит мяч ребёнку. Материал: мяч."
        prompt, brief, reason = _compile_image_prompt_from_payload(
            {
                "action": "the parent rolls a ball while the child repeats one target word",
                "setting": "simple home play area",
                "props": ["ball", "mirror", "spatula", "oral probe"],
            },
            body_text=body,
            audience="parents",
            rubric_id="tip_of_day",
        )

        self.assertEqual(reason, "ok")
        self.assertEqual(brief.props, ("ball",))
        self.assertNotIn("mirror", prompt.lower())
        self.assertNotIn("spatula", prompt.lower())
        self.assertNotIn("probe", prompt.lower())

    def test_source_url_and_hashtags_do_not_create_visual_props(self):
        body = (
            "Как играть: ребёнок выполняет движение без предметов.\n"
            "Источник: https://example.com/book-and-ball\n"
            "#игра_с_мячом #книга"
        )

        self.assertEqual(_mentioned_visual_props(body), [])

    def test_compiled_prompt_is_concise_and_style_is_not_duplicated(self):
        prompt = _deterministic_visual_prompt(
            "Возраст: 5 лет. Как играть: родитель показывает карточку, ребёнок называет картинку.",
            "parents",
            "question_week",
        )

        self.assertLessEqual(len(prompt), 900)
        self.assertEqual(prompt.count(VISUAL_STYLE_TAIL), 1)
        self.assertEqual(prompt.lower().count("normal 50mm perspective"), 1)
        self.assertEqual(prompt.lower().count("allowed props:"), 1)

    def test_invalid_json_gets_one_brief_repair(self):
        body = "Возраст: 4 года. Как играть: родитель катит мяч ребёнку."
        repaired_json = json.dumps(
            {
                "action": "the parent rolls a ball while the child names the movement",
                "setting": "simple home play area",
                "props": ["ball"],
            }
        )
        groq = AsyncMock(side_effect=["not json", repaired_json])
        llm_generator = importlib.import_module("src.services.llm_generator")

        with patch.object(llm_generator, "groq_chat", groq):
            prompt, ok, note = asyncio.run(
                llm_generator.generate_image_prompt_async(
                    title="Игра с мячом",
                    body_text=body,
                    audience="parents",
                    provider="groq",
                    groq_key="test-key",
                    gemini_key="",
                    rubric_id="tip_of_day",
                )
            )

        self.assertTrue(ok)
        self.assertEqual(groq.await_count, 2)
        self.assertEqual(note, "ok:groq_retry")
        self.assertIn("Action: the parent rolls a ball", prompt)

    def test_invalid_json_after_repair_uses_deterministic_brief(self):
        body = "Возраст: 4 года. Как играть: родитель катит мяч ребёнку."
        groq = AsyncMock(side_effect=["not json", "still not json"])
        llm_generator = importlib.import_module("src.services.llm_generator")

        with patch.object(llm_generator, "groq_chat", groq):
            prompt, ok, note = asyncio.run(
                llm_generator.generate_image_prompt_async(
                    title="Игра с мячом",
                    body_text=body,
                    audience="parents",
                    provider="groq",
                    groq_key="test-key",
                    gemini_key="",
                    rubric_id="tip_of_day",
                )
            )

        self.assertTrue(ok)
        self.assertEqual(groq.await_count, 2)
        self.assertIn("deterministic_fallback", note)
        self.assertIn("Action: the parent rolls a ball", prompt)

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

    def test_image_prompt_requests_json_action_without_style_or_camera(self):
        prompt = build_image_prompt_prompt(
            title="Игра с карточками",
            body_text="Специалист показывает карточки и просит ребенка назвать картинку.",
            audience="pros",
            rubric_id="method_piggybank",
        ).lower()

        self.assertIn("return json only", prompt)
        self.assertIn('"action"', prompt)
        self.assertIn('"setting"', prompt)
        self.assertIn('"props"', prompt)
        self.assertIn("exactly one adult speech specialist", prompt)
        self.assertIn("do not choose the number, roles, ages, art style, camera", prompt)
        self.assertNotIn("warm soft editorial illustration", prompt)
        self.assertNotIn("wide-angle distortion", prompt)

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

        self.assertIn("exactly one adult parent and exactly one young child", parent_prompt)
        self.assertIn("no other people", parent_prompt)
        self.assertIn("exactly one young child, no adults and no other people", age_prompt)

    def test_visual_retry_prompt_is_reason_aware(self):
        base = _compile_visual_prompt(
            VisualBrief(
                rubric_id="tip_of_day",
                role_rule=build_visual_role_rule("tip_of_day", "young child"),
                age_descriptor="young child",
                setting="simple uncluttered home play area",
                action="the parent models one target word while the child points to a toy",
                props=("toy",),
            )
        )
        retry = build_visual_retry_prompt(
            base,
            rubric_id="tip_of_day",
            audience="parents",
            qa_reason="missing_required_child",
        )

        self.assertLessEqual(len(retry), 900)
        self.assertTrue(retry.startswith("Exactly one adult parent"))
        self.assertIn("one unmistakably young child", retry.lower())
        self.assertEqual(retry.lower().count("warm soft editorial illustration"), 1)

    def test_method_retry_prompt_targets_extra_people(self):
        retry = build_visual_retry_prompt(
            "one specialist demonstrates an articulation exercise with a child",
            rubric_id="method_piggybank",
            audience="pros",
            qa_reason="partial_human_figure",
        ).lower()

        self.assertIn("exactly one adult speech specialist", retry)
        self.assertIn("exactly one clearly younger child", retry)
        self.assertIn("empty setting without reflections, portraits, silhouettes", retry)
        self.assertNotIn("background people", retry)

    def test_action_mismatch_retry_repeats_exact_expected_action(self):
        action = "the parent models a polite request while the toddler points to a toy"
        base = _compile_visual_prompt(
            VisualBrief(
                rubric_id="tip_of_day",
                role_rule=build_visual_role_rule("tip_of_day", "2-year-old toddler"),
                age_descriptor="2-year-old toddler",
                setting="simple uncluttered home play area",
                action="the parent and child sit near a toy",
                props=("toy",),
            )
        )

        retry = build_visual_retry_prompt(
            base,
            rubric_id="tip_of_day",
            qa_reason="action_mismatch",
            expected_action=action,
        )

        self.assertIn(f"Action: {action};", retry)
        self.assertLess(retry.index(f"Action: {action}"), retry.index("Warm soft editorial illustration"))
        self.assertLessEqual(len(retry), 900)

    def test_horizontal_stretch_retry_uses_normal_50mm_and_body_width(self):
        base = _deterministic_visual_prompt(
            "Возраст: 4 года. Как играть: родитель катит мяч ребёнку.",
            "parents",
            "tip_of_day",
        )

        retry = build_visual_retry_prompt(base, rubric_id="tip_of_day", qa_reason="horizontal_stretch").lower()

        self.assertIn("normal 50mm perspective", retry)
        self.assertIn("normal body width", retry)
        self.assertIn("non-panoramic composition", retry)

    def test_character_counts_unknown_retry_requires_separate_unobstructed_figures(self):
        base = _deterministic_visual_prompt(
            "Возраст: 4 года. Как играть: родитель катит мяч ребёнку.",
            "parents",
            "tip_of_day",
        )

        retry = build_visual_retry_prompt(base, rubric_id="tip_of_day", qa_reason="character_counts_unknown").lower()

        self.assertIn("both unobstructed figures separately", retry)
        self.assertIn("visible heads and upper bodies", retry)
        self.assertIn("without overlap", retry)

    def test_parent_role_rule_is_identical_in_compiled_prompt_and_qa_brief(self):
        prompt = _deterministic_visual_prompt(
            "Возраст: 18–30 мес. Как играть: родитель показывает игрушку ребёнку.",
            "parents",
            "tip_of_day",
        )
        brief = _parse_compiled_visual_prompt(prompt, rubric_id="tip_of_day")
        expected = _build_visual_qa_expected_brief(prompt, "tip_of_day")

        self.assertIsNotNone(brief)
        self.assertIn(f"Expected roles: {brief.role_rule}", expected)

    def test_build_post_visual_passes_short_expected_roles_action_and_props_to_qa(self):
        prompt = _deterministic_visual_prompt(
            "Возраст: 4 года. Как играть: родитель катит мяч ребёнку.",
            "parents",
            "tip_of_day",
        )
        expected_values = []

        def qa(*_args, **kwargs):
            expected_values.append(kwargs["expected_prompt"])
            return {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
            }

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
        ):
            _, meta = build_post_visual(
                title="Ball game",
                day_key="MO",
                image_prompt=prompt,
                visual_qa_fn=qa,
                rubric_id="tip_of_day",
            )

        self.assertEqual(len(expected_values), 1)
        expected = expected_values[0]
        self.assertIn("Expected roles:", expected)
        self.assertIn("Expected action:", expected)
        self.assertIn("Allowed props: ball", expected)
        self.assertNotIn("Warm soft editorial illustration", expected)
        self.assertLess(len(expected), 450)
        self.assertEqual(meta["visual_brief_props"], "ball")
        self.assertEqual(meta["compiled_prompt_len"], str(len(prompt)))

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
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
            },
            BytesIO(b"image"),
            rubric_id="method_piggybank",
            audience="pros",
        )
        unknown = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": "unknown",
                "adult_count": 1,
                "child_count": 1,
            },
            BytesIO(b"image"),
            rubric_id="method_piggybank",
            audience="pros",
        )

        self.assertTrue(passed["pass"])
        self.assertEqual(passed["people_count"], 2)
        self.assertTrue(unknown["pass"])
        self.assertEqual(unknown["people_count"], "unknown")

    def test_parent_rubric_unknown_character_counts_fail(self):
        result = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": "unknown",
                "child_count": "unknown",
            },
            BytesIO(b"image"),
            rubric_id="tip_of_day",
            audience="parents",
        )

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "character_counts_unknown")

    def test_parent_rubric_missing_character_counts_fail(self):
        result = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
            },
            BytesIO(b"image"),
            rubric_id="play_and_speak",
            audience="parents",
        )

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "character_counts_unknown")

    def test_method_rubric_unknown_character_counts_fail(self):
        result = _safe_visual_qa(
            lambda *_args, **_kwargs: {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": "unknown",
                "child_count": "unknown",
            },
            BytesIO(b"image"),
            rubric_id="method_piggybank",
            audience="pros",
        )

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "character_counts_unknown")

    def test_all_role_sensitive_rubrics_reject_non_numeric_character_counts(self):
        rubrics = (
            "method_piggybank",
            "tip_of_day",
            "play_and_speak",
            "question_week",
            "myth_fact",
            "bilingual_corner",
            "bilingual_parents",
            "age_norms",
        )

        for rubric_id in rubrics:
            with self.subTest(rubric_id=rubric_id):
                result = _safe_visual_qa(
                    lambda *_args, **_kwargs: {
                        "status": "pass",
                        "pass": True,
                        "reason": "ok",
                        "people_count": 2,
                        "adult_count": "one",
                        "child_count": 1,
                    },
                    BytesIO(b"image"),
                    rubric_id=rubric_id,
                    audience="pros" if rubric_id == "method_piggybank" else "parents",
                )

                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], "character_counts_unknown")

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
        self.assertIn("exactly one adult parent and exactly one clearly younger child", qa_text)
        self.assertIn("exactly one adult speech specialist and exactly one clearly younger child", qa_text)
        self.assertIn("missing_required_child", qa_text)
        self.assertIn("do not ignore small background figures", qa_text)
        self.assertIn("expected visual brief", qa_text)
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
        self.assertIn("exactly one adult parent", retry_prompt)
        self.assertIn("no other people", retry_prompt)
        self.assertIn("empty setting without reflections, portraits, silhouettes", retry_prompt)
        self.assertIn("normal 50mm perspective", retry_prompt)
        self.assertLessEqual(len(retry_prompt), 900)
        self.assertEqual(meta["mode"], "ai")
        self.assertEqual(meta["visual_retry_used"], "True")
        self.assertEqual(meta["visual_qa_attempts"], "2")
        self.assertEqual(meta["visual_retry_target_reason"], "ghosted_figure")

    def test_unknown_character_counts_retry_once_then_accepts(self):
        qa_results = iter([
            {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": "unknown",
                "child_count": "unknown",
            },
            {
                "status": "pass",
                "pass": True,
                "reason": "ok",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
            },
        ])

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
                (BytesIO(b"second"), {"attempts_used": "1", "final_reason": "ok"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Speech game",
                day_key="MO",
                image_prompt="an adult and child practicing a speech game",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
                rubric_id="tip_of_day",
            )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(buffer.getvalue(), b"second")
        self.assertEqual(meta["mode"], "ai")
        self.assertEqual(meta["visual_retry_used"], "True")
        self.assertEqual(meta["visual_qa_attempts"], "2")

    def test_unknown_character_counts_after_retry_uses_fallback(self):
        unknown_counts = {
            "status": "pass",
            "pass": True,
            "reason": "ok",
            "people_count": 2,
            "adult_count": "unknown",
            "child_count": "unknown",
        }

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"first"), {"attempts_used": "1", "final_reason": "ok"}),
                (BytesIO(b"second"), {"attempts_used": "1", "final_reason": "ok"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Speech game",
                day_key="MO",
                image_prompt="an adult and child practicing a speech game",
                visual_qa_fn=lambda *_args, **_kwargs: dict(unknown_counts),
                rubric_id="tip_of_day",
            )

        self.assertEqual(download.call_count, 2)
        self.assertNotIn(buffer.getvalue(), {b"first", b"second"})
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(meta["visual_qa"], "fail")
        self.assertEqual(meta["visual_qa_reason"], "character_counts_unknown")
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

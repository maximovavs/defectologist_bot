from io import BytesIO
import unittest
from unittest.mock import patch

from src.services.visual_pipeline import (
    VISUAL_QA_HARD_REASONS,
    VisualBrief,
    _compile_visual_prompt,
    _enforce_object_visual_qa,
    _normalize_child_only_visual_action,
    _parse_compiled_visual_prompt,
    _validate_compiled_visual_prompt,
    build_post_visual,
    build_visual_role_rule,
)


def _object_qa(**overrides):
    """A safe object QA verdict: zero people, no PPE, no text, channel style."""
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


class ObjectStyleVisualQaTest(unittest.TestCase):
    def test_photorealistic_object_image_is_rejected(self):
        result = _enforce_object_visual_qa(_object_qa(illustration_style_match=False))

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "object_style_mismatch")

    def test_watercolor_object_image_passes(self):
        result = _enforce_object_visual_qa(_object_qa())

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])
        self.assertEqual(result["reason"], "ok")
        self.assertTrue(result["illustration_style_match"])

    def test_unknown_style_fails_closed(self):
        missing = _object_qa()
        missing.pop("illustration_style_match")

        for label, payload in (
            ("absent", missing),
            ("unknown", _object_qa(illustration_style_match="unknown")),
            ("garbage", _object_qa(illustration_style_match="maybe")),
        ):
            with self.subTest(style=label):
                result = _enforce_object_visual_qa(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], "object_style_unknown")

    def test_style_reasons_are_hard_failures(self):
        self.assertIn("object_style_mismatch", VISUAL_QA_HARD_REASONS)
        self.assertIn("object_style_unknown", VISUAL_QA_HARD_REASONS)

    def test_existing_object_safety_checks_are_retained(self):
        cases = (
            ("object_contains_person", _object_qa(people_count=1, adult_count=1)),
            ("object_contains_text", _object_qa(text_detected=True)),
            ("unexpected_ppe", _object_qa(ppe_detected=True)),
            ("object_counts_unknown", _object_qa(people_count="unknown")),
            ("object_text_unknown", _object_qa(text_detected="unknown")),
        )
        for expected_reason, payload in cases:
            with self.subTest(reason=expected_reason):
                result = _enforce_object_visual_qa(payload)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["pass"])
                self.assertEqual(result["reason"], expected_reason)

    def test_two_object_style_failures_end_in_text_fallback(self):
        qa_results = iter([
            # human attempt rejected
            {
                "status": "fail",
                "pass": False,
                "reason": "action_mismatch",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
                "ppe_detected": False,
            },
            # human retry rejected
            {
                "status": "fail",
                "pass": False,
                "reason": "duplicate_figure",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
                "ppe_detected": False,
            },
            _object_qa(illustration_style_match=False),
            _object_qa(illustration_style_match=False),
        ])
        downloads = []

        def download(*, prompt, token):
            downloads.append(prompt)
            return BytesIO(f"image-{len(downloads)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            buffer, meta = build_post_visual(
                title="Книги и новые слова",
                day_key="2026-08-09",
                image_prompt="an adult and child looking at a picture book together",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )

        # human -> human retry -> object #1 -> object #2 -> text fallback
        self.assertEqual(len(downloads), 4)
        self.assertNotIn(buffer.getvalue(), {b"image-1", b"image-2", b"image-3", b"image-4"})
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["text_fallback_used"], "True")
        self.assertEqual(meta["final_reason"], "object_fallback_rejected")
        self.assertEqual(meta["object_generation_attempts"], "2")
        self.assertEqual(meta["object_qa_reason"], "object_style_mismatch")
        self.assertEqual(meta["object_qa_style_match"], "False")

    def test_styled_object_image_still_publishes(self):
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
                "reason": "duplicate_figure",
                "people_count": 2,
                "adult_count": 1,
                "child_count": 1,
                "ppe_detected": False,
            },
            _object_qa(),
        ])
        downloads = []

        def download(*, prompt, token):
            downloads.append(prompt)
            return BytesIO(f"image-{len(downloads)}".encode()), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            buffer, meta = build_post_visual(
                title="Книги и новые слова",
                day_key="2026-08-09",
                image_prompt="an adult and child looking at a picture book together",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )

        self.assertEqual(len(downloads), 3)
        self.assertEqual(buffer.getvalue(), b"image-3")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["object_generation_attempts"], "1")
        self.assertEqual(meta["object_qa_style_match"], "True")


class AgeNormsChildOnlyActionTest(unittest.TestCase):
    ROLE_RULE = build_visual_role_rule("age_norms", age_descriptor="2-year-old toddler")

    ADULT_DIRECTED_MARKERS = (
        "show the child",
        "ask the child",
        "give the child",
        "read to the child",
        "hold up",
        "wait for the child",
        "the child to",
        "parent",
        "adult",
        "mother",
        "father",
        "caregiver",
    )

    def _normalized(self, action, props=()):
        return _normalize_child_only_visual_action(
            action,
            role_rule=self.ROLE_RULE,
            age_descriptor="2-year-old toddler",
            props=props,
        )

    def test_role_rule_is_child_only(self):
        self.assertEqual(
            self.ROLE_RULE,
            "Exactly one 2-year-old toddler, no adults and no other people.",
        )

    def test_production_action_becomes_observable_child_action(self):
        action = self._normalized(
            'Point to a picture in a book while saying "mom" and wait for the child to name it or add a word',
            props=("picture", "book"),
        )

        self.assertTrue(action.lower().startswith("the 2-year-old toddler "))
        self.assertIn("points to", action.lower())
        self.assertNotIn('"mom"', action)
        for marker in self.ADULT_DIRECTED_MARKERS:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, action.lower())

    def test_representative_adult_directed_variants_are_rewritten(self):
        cases = (
            ("Show the child a ball and ask them to name it", ("ball",), "looks at"),
            ("Give the child a cup and wait for a word", ("cup",), "holds"),
            ("Read a picture book and ask the child to point", ("picture",), "points to"),
            ("Hold up a toy and wait for the child to say its name", ("toy",), "holds"),
        )
        for action, props, expected_verb in cases:
            with self.subTest(action=action):
                rewritten = self._normalized(action, props=props)
                self.assertTrue(rewritten.lower().startswith("the 2-year-old toddler "))
                self.assertIn(expected_verb, rewritten.lower())
                for marker in self.ADULT_DIRECTED_MARKERS:
                    self.assertNotIn(marker, rewritten.lower())

    def test_speech_expectation_never_asks_for_readable_text(self):
        rewritten = self._normalized(
            'Show the child a ball and ask them to say "ball"',
            props=("ball",),
        ).lower()

        self.assertIn("saying a simple word or short two-word phrase", rewritten)
        for phrase in ("written text", "letters", "caption", "label", '"'):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, rewritten)

    def test_child_subject_action_is_left_untouched(self):
        for action in (
            "The 2-year-old toddler stacks three blocks",
            "The toddler holds a cup and says a word",
        ):
            with self.subTest(action=action):
                self.assertEqual(self._normalized(action, props=("blocks", "cup")), action)

    def test_adult_allowed_role_rule_is_not_rewritten(self):
        adult_role = build_visual_role_rule(
            "age_norms",
            age_descriptor="2-year-old toddler",
            adult_required=True,
        )
        action = "Show the child a ball and ask them to name it"

        self.assertEqual(
            _normalize_child_only_visual_action(
                action,
                role_rule=adult_role,
                age_descriptor="2-year-old toddler",
                props=("ball",),
            ),
            action,
        )

    def test_compiled_child_only_prompt_does_not_contradict_role_rule(self):
        brief = VisualBrief(
            rubric_id="age_norms",
            role_rule=self.ROLE_RULE,
            age_descriptor="2-year-old toddler",
            setting="simple home play area",
            action=(
                'Point to a picture in a book while saying "mom" '
                "and wait for the child to name it or add a word"
            ),
            props=("picture", "book"),
        )
        prompt = _compile_visual_prompt(brief)

        ok, reason = _validate_compiled_visual_prompt(prompt, "age_norms")
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "ok")

        compiled = _parse_compiled_visual_prompt(prompt, rubric_id="age_norms")
        self.assertIsNotNone(compiled)
        self.assertTrue(compiled.action.lower().startswith("the 2-year-old toddler "))
        for marker in self.ADULT_DIRECTED_MARKERS:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, compiled.action.lower())


class VisualShotSelectionTest(unittest.TestCase):
    @staticmethod
    def _compile(role_rule: str, rubric_id: str) -> str:
        return _compile_visual_prompt(
            VisualBrief(
                rubric_id=rubric_id,
                role_rule=role_rule,
                age_descriptor="2-year-old toddler",
                setting="simple home play area",
                action="The toddler looks at a picture",
                props=("picture",),
            )
        )

    def test_age_norms_child_only_uses_medium_shot(self):
        prompt = self._compile(
            "Exactly one 2-year-old toddler, no adults and no other people.",
            "age_norms",
        )

        self.assertIn("eye-level medium shot,", prompt)
        self.assertNotIn("medium two-shot", prompt)

    def test_age_norms_with_required_adult_uses_medium_two_shot(self):
        role_rule = build_visual_role_rule(
            "age_norms",
            age_descriptor="2-year-old toddler",
            adult_required=True,
        )

        self.assertIn("eye-level medium two-shot,", self._compile(role_rule, "age_norms"))

    def test_parent_rubric_uses_medium_two_shot(self):
        role_rule = build_visual_role_rule(
            "tip_of_day",
            age_descriptor="2-year-old toddler",
        )

        self.assertIn("eye-level medium two-shot,", self._compile(role_rule, "tip_of_day"))

    def test_method_piggybank_uses_medium_two_shot(self):
        role_rule = build_visual_role_rule("method_piggybank")

        self.assertIn("eye-level medium two-shot,", self._compile(role_rule, "method_piggybank"))


if __name__ == "__main__":
    unittest.main()

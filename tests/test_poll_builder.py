import re
import unittest

from src.services.poll_builder import POLL_TEMPLATES, PollSpec, build_poll_spec


CONTENT_RUBRICS = {
    "tip_of_day",
    "play_and_speak",
    "myth_fact",
    "bilingual_corner",
    "question_week",
    "method_piggybank",
    "age_norms",
}


class PollBuilderTest(unittest.TestCase):
    def test_all_content_rubrics_return_valid_poll_specs(self):
        self.assertEqual(set(POLL_TEMPLATES), CONTENT_RUBRICS)
        for rubric_id in CONTENT_RUBRICS:
            with self.subTest(rubric_id=rubric_id):
                poll = build_poll_spec(
                    rubric_id,
                    "Опубликованный контентный пост",
                    f"https://example.org/{rubric_id}",
                    "2026-07-16",
                )
                self.assertIsInstance(poll, PollSpec)
                self.assertGreaterEqual(len(POLL_TEMPLATES[rubric_id]), 3)
                self.assertLessEqual(len(POLL_TEMPLATES[rubric_id]), 5)

    def test_non_content_and_unknown_rubrics_return_none(self):
        for rubric_id in ("quality_dashboard", "diagnostic_message", "semantic_alert", "unknown"):
            with self.subTest(rubric_id=rubric_id):
                self.assertIsNone(
                    build_poll_spec(rubric_id, "text", "https://example.org", "2026-07-16")
                )

    def test_same_inputs_select_the_same_poll(self):
        args = (
            "tip_of_day",
            "Один и тот же пост",
            "https://example.org/advice",
            "2026-07-16",
        )
        self.assertEqual(build_poll_spec(*args), build_poll_spec(*args))

    def test_dates_and_urls_can_select_different_variants(self):
        selected = {
            build_poll_spec(
                "play_and_speak",
                "Пост",
                f"https://example.org/game/{index}",
                f"2026-07-{10 + index:02d}",
            )
            for index in range(7)
        }
        self.assertGreater(len(selected), 1)

    def test_every_template_meets_length_and_uniqueness_limits(self):
        for rubric_id, variants in POLL_TEMPLATES.items():
            for poll in variants:
                with self.subTest(rubric_id=rubric_id, question=poll.question):
                    self.assertGreaterEqual(len(poll.question), 1)
                    self.assertLessEqual(len(poll.question), 180)
                    self.assertIn(len(poll.options), (3, 4))
                    self.assertEqual(len(poll.options), len({option.casefold() for option in poll.options}))
                    for option in poll.options:
                        self.assertGreaterEqual(len(option), 1)
                        self.assertLessEqual(len(option), 80)

    def test_templates_have_no_markup_links_hashtags_or_placeholders(self):
        unsafe = re.compile(
            r"https?://|www\.|[#*`<>]|\{\{|\}\}|\[[^\]]*\]|\bplaceholder\b",
            re.IGNORECASE,
        )
        for rubric_id, variants in POLL_TEMPLATES.items():
            for poll in variants:
                for text in (poll.question, *poll.options):
                    with self.subTest(rubric_id=rubric_id, text=text):
                        self.assertIsNone(unsafe.search(text))


if __name__ == "__main__":
    unittest.main()

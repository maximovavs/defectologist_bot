import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.publisher import run_publisher as publisher
from src.services.engagement_builder import (
    ENGAGEMENT_POLICY,
    EngagementSpec,
    append_engagement_footer,
    build_engagement_spec,
)
from src.services.poll_builder import PollSpec


CONTENT_RUBRICS = set(ENGAGEMENT_POLICY)


class EngagementBuilderTest(unittest.TestCase):
    def build(self, rubric_id, url="https://example.org/post", date_key="2026-07-16", mode="auto"):
        return build_engagement_spec(rubric_id, "Проверенный текст поста.", url, date_key, mode)

    def test_auto_polls_are_reserved_for_two_rubrics(self):
        self.assertEqual(self.build("myth_fact").kind, "poll")
        self.assertEqual(self.build("method_piggybank").kind, "poll")
        for rubric_id in CONTENT_RUBRICS - {"myth_fact", "method_piggybank"}:
            with self.subTest(rubric_id=rubric_id):
                self.assertNotEqual(self.build(rubric_id).kind, "poll")

    def test_auto_matches_the_declared_policy_modes(self):
        for rubric_id, allowed in ENGAGEMENT_POLICY.items():
            for index in range(12):
                spec = self.build(rubric_id, url=f"https://example.org/{index}")
                expected_kind = "poll" if "poll" in allowed else "footer" if spec.mode != "none" else "none"
                self.assertEqual(spec.kind, expected_kind)
                self.assertIn(spec.mode, allowed)

    def test_polls_only_off_unknown_and_determinism(self):
        for rubric_id in CONTENT_RUBRICS:
            with self.subTest(rubric_id=rubric_id):
                self.assertEqual(self.build(rubric_id, mode="polls_only").kind, "poll")
        self.assertEqual(self.build("tip_of_day", mode="off"), EngagementSpec(kind="none", mode="off"))
        self.assertEqual(self.build("technical_rubric"), EngagementSpec(kind="none", mode="none"))
        args = ("question_week", "https://example.org/stable", "2026-07-16")
        self.assertEqual(self.build(args[0], url=args[1], date_key=args[2]), self.build(args[0], url=args[1], date_key=args[2]))
        self.assertNotIn("hash(", inspect.getsource(__import__("src.services.engagement_builder", fromlist=["x"])))

    def test_footer_validation_and_template_limits(self):
        for rubric_id in CONTENT_RUBRICS - {"myth_fact", "method_piggybank"}:
            for index in range(10):
                spec = self.build(rubric_id, url=f"https://example.org/footer/{index}")
                if spec.kind == "footer":
                    self.assertLessEqual(len(spec.footer_text), 140)
                    self.assertNotRegex(spec.footer_text, r"https?://|www\.|[#*`<>]|\{\{|\}\}|\n")
        with self.assertRaises(ValueError):
            EngagementSpec(kind="footer", mode="comment", footer_text="<b>Оцените ребёнка</b>")
        with self.assertRaises(ValueError):
            EngagementSpec(kind="poll", mode="poll", footer_text="CTA", poll=PollSpec("Вопрос?", ("Да", "Нет", "Позже")))

    def test_footer_is_before_source_link_and_hashtags(self):
        plain = "Заголовок\n\nОсновной текст.\n\nИсточник: example.org\n🔗 https://example.org/post\n\n#играем_и_говорим #речь"
        footer = "💬 В какой ситуации вам удобнее попробовать эту игру?"
        result = append_engagement_footer(plain, footer, 1000)
        self.assertEqual(result.count(footer), 1)
        self.assertLess(result.index(footer), result.index("Источник:"))
        self.assertLess(result.index(footer), result.index("🔗"))
        self.assertLess(result.index(footer), result.index("#играем"))
        self.assertIn("Основной текст.", result)

    def test_footer_overflow_returns_original_without_truncation(self):
        plain = "Основной текст.\n\nИсточник: example.org\n🔗 https://example.org/post"
        footer = "💬 В какой ситуации вам удобнее попробовать эту игру?"
        self.assertEqual(append_engagement_footer(plain, footer, len(plain)), plain)


class EngagementPublisherIntegrationTest(unittest.TestCase):
    def test_payloads_and_send_rules(self):
        footer = EngagementSpec(kind="footer", mode="comment", footer_text="💬 Расскажите о вашем опыте.")
        poll = PollSpec("Какой формат удобнее?", ("Первый", "Второй", "Третий"))
        none = EngagementSpec(kind="none", mode="none")
        self.assertEqual(publisher._engagement_json_payload(footer)["footer_text"], footer.footer_text)
        self.assertEqual(publisher._engagement_json_payload(none), {"kind": "none", "mode": "none"})
        with patch.object(publisher, "send_post_poll") as sender:
            publisher._handle_post_engagement(spec=none, rubric_id="tip_of_day", canonical_url="", post_message_id=101)
            publisher._handle_post_engagement(spec=footer, rubric_id="tip_of_day", canonical_url="", post_message_id=101)
        sender.assert_not_called()
        with patch.object(publisher, "send_post_poll", return_value=303) as sender:
            result = publisher._handle_post_engagement(
                spec=EngagementSpec(kind="poll", mode="poll", poll=poll),
                rubric_id="myth_fact",
                canonical_url="https://example.org/post",
                chat_id="chat",
                post_message_id=101,
            )
        self.assertEqual(result, 303)
        sender.assert_called_once_with("chat", poll, 101)

    def test_dry_run_engagement_json_never_sends_poll(self):
        poll = PollSpec("Какой формат удобнее?", ("Первый", "Второй", "Третий"))
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            spec = EngagementSpec(kind="poll", mode="poll", poll=poll)
            with patch.object(publisher, "send_post_poll") as sender:
                publisher._write_dry_run_engagement(output_dir, "01_test_myth_fact", spec)
                publisher._handle_post_engagement(
                    spec=spec,
                    rubric_id="myth_fact",
                    canonical_url="https://example.org/post",
                    dry_run=True,
                    dry_run_dir=output_dir,
                    dry_run_stem="01_test_myth_fact",
                )
            sender.assert_not_called()
            payload = json.loads((output_dir / "01_test_myth_fact.engagement.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "poll")
            self.assertEqual(payload["question"], poll.question)

    def test_publisher_keeps_plain_for_dedup_history_and_image_but_renders_display(self):
        source = inspect.getsource(publisher.amain)
        self.assertIn("body_hash = sha1(norm_space(plain))", source)
        self.assertIn("body_text=plain", source)
        self.assertIn("body_text=plain", source)
        self.assertIn("render_plain_to_telegram_html(display_plain)", source)
        self.assertIn("body_text=plain", inspect.getsource(publisher.generate_image_prompt_async) if False else source)
        self.assertLess(source.index("store.record_publication("), source.index("_handle_post_engagement(", source.index("store.record_publication(")))


if __name__ == "__main__":
    unittest.main()

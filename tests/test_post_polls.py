from contextlib import redirect_stdout
from io import BytesIO, StringIO
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.publisher import run_publisher as publisher
from src.services.poll_builder import PollSpec


POLL = PollSpec(
    question="Попробуете этот приём сегодня?",
    options=("Да, обязательно", "Уже используем", "Сохраню на потом", "Пока не подходит"),
)


def _telegram_payload(message_id: int):
    return {"ok": True, "result": {"message_id": message_id}}


class TelegramMessageIdTest(unittest.TestCase):
    def test_extract_message_id_rejects_success_without_valid_id(self):
        for payload in ({"ok": True, "result": {}}, {"ok": True}, {"result": {"message_id": True}}):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(RuntimeError, "result.message_id"):
                    publisher._extract_telegram_message_id(payload)

    def test_send_message_returns_message_id(self):
        with patch.object(publisher, "tg_request", return_value=_telegram_payload(101)) as request:
            message_id = publisher.send_message("chat", "<b>Текст</b>")

        self.assertEqual(message_id, 101)
        self.assertEqual(request.call_args.args[0], "sendMessage")

    def test_send_message_html_fallback_returns_fallback_message_id(self):
        with patch.object(
            publisher,
            "tg_request",
            side_effect=[RuntimeError("Bad Request: can't parse entities"), _telegram_payload(111)],
        ) as request:
            message_id = publisher.send_message("chat", "<b>Текст</b>")

        self.assertEqual(message_id, 111)
        self.assertEqual(request.call_count, 2)
        self.assertNotIn("parse_mode", request.call_args_list[1].kwargs["data"])

    def test_send_post_with_visual_returns_photo_id_for_caption(self):
        photo = BytesIO(b"image")
        with (
            patch.object(publisher, "TG_CAPTION_MAX_BYTES", 950),
            patch.object(publisher, "tg_request", return_value=_telegram_payload(202)) as request,
        ):
            message_id = publisher.send_post_with_visual("chat", photo, "Короткий пост", "Короткий пост")

        self.assertEqual(message_id, 202)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], "sendPhoto")

    def test_send_post_with_visual_returns_text_id_when_sent_separately(self):
        photo = BytesIO(b"image")
        with (
            patch.object(publisher, "TG_CAPTION_MAX_BYTES", 1),
            patch.object(
                publisher,
                "tg_request",
                side_effect=[_telegram_payload(303), _telegram_payload(404)],
            ) as request,
        ):
            message_id = publisher.send_post_with_visual("chat", photo, "Длинный пост", "Длинный пост")

        self.assertEqual(message_id, 404)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0], "sendPhoto")
        self.assertEqual(request.call_args_list[1].args[0], "sendMessage")

    def test_send_post_with_visual_html_fallback_returns_text_message_id(self):
        photo = BytesIO(b"image")
        with patch.object(
            publisher,
            "tg_request",
            side_effect=[
                RuntimeError("Bad Request: can't parse entities"),
                _telegram_payload(505),
                _telegram_payload(606),
            ],
        ) as request:
            message_id = publisher.send_post_with_visual("chat", photo, "Пост", "<b>Пост</b>")

        self.assertEqual(message_id, 505)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[1].args[0], "sendPhoto")
        self.assertNotIn("parse_mode", request.call_args_list[1].kwargs["data"])


class SendPostPollTest(unittest.TestCase):
    def test_send_post_poll_uses_regular_anonymous_single_answer_reply(self):
        with (
            patch.object(publisher, "POLL_OPEN_PERIOD_SECONDS", 86400),
            patch.object(publisher, "POLL_DISABLE_NOTIFICATION", True),
            patch.object(publisher, "tg_request", return_value=_telegram_payload(505)) as request,
        ):
            message_id = publisher.send_post_poll("chat", POLL, 202)

        self.assertEqual(message_id, 505)
        self.assertEqual(request.call_args.args[0], "sendPoll")
        data = request.call_args.kwargs["data"]
        self.assertEqual(
            json.loads(data["options"]),
            [{"text": option} for option in POLL.options],
        )
        self.assertEqual(
            json.loads(data["reply_parameters"]),
            {"message_id": 202, "allow_sending_without_reply": True},
        )
        self.assertEqual(data["is_anonymous"], "true")
        self.assertEqual(data["type"], "regular")
        self.assertEqual(data["allows_multiple_answers"], "false")
        self.assertEqual(data["allows_revoting"], "true")
        self.assertEqual(data["disable_notification"], "true")
        self.assertEqual(data["open_period"], "86400")
        self.assertNotIn("correct_option_ids", data)


class PostPublishPollHandlingTest(unittest.TestCase):
    def test_poll_failure_does_not_become_telegram_send_failed(self):
        output = StringIO()
        with (
            patch.object(publisher, "send_post_poll", side_effect=RuntimeError("poll unavailable")),
            redirect_stdout(output),
        ):
            result = publisher._handle_post_poll(
                rubric_id="tip_of_day",
                plain_post="Пост",
                canonical_url="https://example.org/post",
                date_key="2026-07-16",
                chat_id="chat",
                post_message_id=101,
                enabled=True,
            )

        self.assertIsNone(result)
        self.assertIn(
            "[POLL][WARN] poll_send_failed rubric=tip_of_day url=https://example.org/post",
            output.getvalue(),
        )
        self.assertNotIn("telegram_send_failed", output.getvalue())

    def test_record_publication_and_posted_count_happen_before_poll(self):
        source = inspect.getsource(publisher.amain)
        record_at = source.index("store.record_publication(")
        posted_at = source.index("posted += 1", record_at)
        poll_at = source.index("_handle_post_engagement(", posted_at)

        self.assertLess(record_at, posted_at)
        self.assertLess(posted_at, poll_at)

    def test_poll_failure_does_not_undo_existing_record(self):
        store = Mock()
        store.record_publication(canonical_url="https://example.org/post")

        with patch.object(publisher, "send_post_poll", side_effect=RuntimeError("poll unavailable")):
            publisher._handle_post_poll(
                rubric_id="tip_of_day",
                plain_post="Пост",
                canonical_url="https://example.org/post",
                date_key="2026-07-16",
                chat_id="chat",
                post_message_id=101,
                enabled=True,
            )

        store.record_publication.assert_called_once_with(canonical_url="https://example.org/post")

    def test_disabled_flag_does_not_build_or_send_poll(self):
        with (
            patch.object(publisher, "build_poll_spec") as builder,
            patch.object(publisher, "send_post_poll") as sender,
        ):
            result = publisher._handle_post_poll(
                rubric_id="tip_of_day",
                plain_post="Пост",
                canonical_url="https://example.org/post",
                date_key="2026-07-16",
                chat_id="chat",
                post_message_id=101,
                enabled=False,
            )

        self.assertIsNone(result)
        builder.assert_not_called()
        sender.assert_not_called()

    def test_dry_run_writes_poll_json_without_telegram(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch.object(publisher, "send_post_poll") as sender:
                path = publisher._handle_post_poll(
                    rubric_id="tip_of_day",
                    plain_post="Пост",
                    canonical_url="https://example.org/post",
                    date_key="2026-07-16",
                    dry_run=True,
                    dry_run_dir=output_dir,
                    dry_run_stem="01_parents_tip_of_day",
                    enabled=True,
                )

            sender.assert_not_called()
            self.assertEqual(path, output_dir / "01_parents_tip_of_day.poll.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["question"], POLL.question)
            self.assertGreaterEqual(len(payload["options"]), 3)
            self.assertTrue(payload["is_anonymous"])
            self.assertEqual(payload["type"], "regular")
            self.assertEqual(payload["open_period"], 86400)

    def test_quality_dashboard_logs_no_template_and_does_not_send(self):
        output = StringIO()
        with patch.object(publisher, "send_post_poll") as sender, redirect_stdout(output):
            result = publisher._handle_post_poll(
                rubric_id="quality_dashboard",
                plain_post="Диагностика",
                canonical_url="",
                date_key="2026-07-16",
                chat_id="chat",
                post_message_id=101,
                enabled=True,
            )

        self.assertIsNone(result)
        sender.assert_not_called()
        self.assertIn("[POLL][SKIP] rubric=quality_dashboard reason=no_template", output.getvalue())


class PollWorkflowTest(unittest.TestCase):
    def test_workflow_uses_auto_engagement_for_schedule_and_manual_by_default(self):
        workflow = (publisher.ROOT / ".github" / "workflows" / "post.yml").read_text(encoding="utf-8")

        self.assertIn("engagement_mode:", workflow)
        self.assertIn('default: "auto"', workflow)
        self.assertIn("POST_ENGAGEMENT_MODE:", workflow)
        self.assertIn("github.event.inputs.engagement_mode", workflow)
        self.assertNotIn("post_poll:", workflow)
        self.assertIn('POLL_OPEN_PERIOD_SECONDS: "86400"', workflow)
        self.assertIn('POLL_DISABLE_NOTIFICATION: "1"', workflow)


if __name__ == "__main__":
    unittest.main()

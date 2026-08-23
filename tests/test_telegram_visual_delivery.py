from io import BytesIO
import unittest
from unittest.mock import patch

from src.publisher import run_publisher as publisher


def _payload(message_id):
    return {"ok": True, "result": {"message_id": message_id}}


class TelegramVisualDeliveryTest(unittest.TestCase):
    def test_utf16_units_count_surrogate_pair_as_two_units(self):
        self.assertEqual(publisher._telegram_utf16_units("A\U0001f431\nB"), 5)

    def test_multiline_caption_uses_utf16_limit(self):
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 20), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 20
        ), patch.object(publisher, "tg_request", return_value=_payload(1)) as request:
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "line one\n\nline two", "line one\n\nline two"
            )
        self.assertEqual(message_id, 1)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], "sendPhoto")

    def test_parse_error_retries_plain_caption_without_parse_mode(self):
        with patch.object(
            publisher,
            "tg_request",
            side_effect=[RuntimeError("Bad Request: can't parse entities"), _payload(2)],
        ) as request:
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "line one\n\nline two", "<b>line one</b>\n\nline two"
            )
        self.assertEqual(message_id, 2)
        self.assertEqual(request.call_count, 2)
        self.assertNotIn("parse_mode", request.call_args_list[1].kwargs["data"])
        self.assertEqual(request.call_args_list[1].kwargs["data"]["caption"], "line one\n\nline two")

    def test_long_split_success_keeps_photo_then_text_without_delete(self):
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[_payload(101), _payload(202)],
        ) as request:
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
            )

        self.assertEqual(message_id, 202)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["sendPhoto", "sendMessage"])
        self.assertEqual(request.call_args_list[0].kwargs["data"]["caption"], "")
        self.assertNotIn("deleteMessage", [call.args[0] for call in request.call_args_list])

    def test_long_split_text_failure_rolls_back_exact_photo_message(self):
        send_error = RuntimeError("telegram_api_error:503:text unavailable")
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[_payload(303), send_error, {"ok": True, "result": True}],
        ) as request:
            with self.assertRaises(RuntimeError) as raised:
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        self.assertIs(raised.exception, send_error)
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            ["sendPhoto", "sendMessage", "deleteMessage"],
        )
        self.assertEqual(
            request.call_args_list[2].kwargs["data"],
            {"chat_id": "chat", "message_id": 303},
        )

    def test_long_split_html_and_plain_text_failure_rolls_back_photo(self):
        parse_error = RuntimeError("Bad Request: can't parse entities")
        plain_error = RuntimeError("telegram_api_error:503:plain unavailable")
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[
                _payload(404),
                parse_error,
                plain_error,
                {"ok": True, "result": True},
            ],
        ) as request:
            with self.assertRaises(RuntimeError) as raised:
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        self.assertIs(raised.exception, plain_error)
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            ["sendPhoto", "sendMessage", "sendMessage", "deleteMessage"],
        )
        self.assertIn("parse_mode", request.call_args_list[1].kwargs["data"])
        self.assertNotIn("parse_mode", request.call_args_list[2].kwargs["data"])
        self.assertEqual(
            request.call_args_list[3].kwargs["data"],
            {"chat_id": "chat", "message_id": 404},
        )

    def test_long_split_rollback_failure_surfaces_explicit_diagnostic(self):
        send_error = RuntimeError("telegram_api_error:503:text unavailable")
        rollback_error = RuntimeError("telegram_api_error:500:delete unavailable")
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[_payload(505), send_error, rollback_error],
        ) as request:
            with self.assertRaisesRegex(
                RuntimeError,
                r"telegram_split_delivery_rollback_failed:send_error_type=RuntimeError:rollback_error_type=RuntimeError",
            ) as raised:
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        self.assertIs(raised.exception.__cause__, send_error)
        self.assertEqual(
            [call.args[0] for call in request.call_args_list],
            ["sendPhoto", "sendMessage", "deleteMessage"],
        )
        self.assertEqual(
            request.call_args_list[2].kwargs["data"],
            {"chat_id": "chat", "message_id": 505},
        )

    def test_short_caption_success_never_uses_rollback(self):
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 100), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 100
        ), patch.object(publisher, "tg_request", return_value=_payload(606)) as request:
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "short", "<b>short</b>"
            )

        self.assertEqual(message_id, 606)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["sendPhoto"])
        self.assertNotIn("deleteMessage", [call.args[0] for call in request.call_args_list])


if __name__ == "__main__":
    unittest.main()

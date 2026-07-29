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


if __name__ == "__main__":
    unittest.main()

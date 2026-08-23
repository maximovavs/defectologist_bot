from io import BytesIO
import inspect
import unittest
from unittest.mock import Mock, patch

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

    def test_short_caption_transport_timeout_fails_closed_without_retry_or_split(self):
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=publisher.requests.Timeout("network timeout"),
        ) as post:
            with self.assertRaisesRegex(
                publisher.TelegramDeliveryOutcomeAmbiguous,
                "telegram_delivery_outcome_ambiguous:transport_error_type=Timeout",
            ):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                )

        self.assertEqual(post.call_count, 1)

    def test_parse_error_then_plain_timeout_does_not_enter_split(self):
        ambiguous = publisher.TelegramDeliveryOutcomeAmbiguous(
            "telegram_delivery_outcome_ambiguous:transport_error_type=Timeout"
        )
        with patch.object(
            publisher,
            "tg_request",
            side_effect=[RuntimeError("Bad Request: can't parse entities"), ambiguous],
        ) as request:
            with self.assertRaises(publisher.TelegramDeliveryOutcomeAmbiguous) as raised:
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                )

        self.assertIs(raised.exception, ambiguous)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["sendPhoto", "sendPhoto"])
        self.assertNotIn("deleteMessage", [call.args[0] for call in request.call_args_list])

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

    def test_long_split_text_timeout_fails_closed_without_rollback(self):
        ambiguous = publisher.TelegramDeliveryOutcomeAmbiguous(
            "telegram_delivery_outcome_ambiguous:transport_error_type=Timeout"
        )
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[_payload(303), ambiguous],
        ) as request:
            with self.assertRaises(publisher.TelegramDeliveryOutcomeAmbiguous) as raised:
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        self.assertIs(raised.exception, ambiguous)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["sendPhoto", "sendMessage"])
        self.assertNotIn("deleteMessage", [call.args[0] for call in request.call_args_list])

    def test_long_split_text_failure_rolls_back_exact_photo_message(self):
        send_error = RuntimeError("telegram_api_error:400:text rejected")
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[_payload(404), send_error, {"ok": True, "result": True}],
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
            {"chat_id": "chat", "message_id": 404},
        )

    def test_long_split_html_and_plain_text_failure_rolls_back_photo(self):
        parse_error = RuntimeError("Bad Request: can't parse entities")
        plain_error = RuntimeError("telegram_api_error:400:plain rejected")
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[
                _payload(505),
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
            {"chat_id": "chat", "message_id": 505},
        )

    def test_long_split_rollback_failure_is_ambiguous_and_stops(self):
        send_error = RuntimeError("telegram_api_error:400:text rejected")
        rollback_error = RuntimeError("telegram_api_error:400:delete rejected")
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(
            publisher,
            "tg_request",
            side_effect=[_payload(606), send_error, rollback_error],
        ) as request:
            with self.assertRaisesRegex(
                publisher.TelegramDeliveryOutcomeAmbiguous,
                r"telegram_delivery_outcome_ambiguous:telegram_split_delivery_rollback_failed:"
                r"send_error_type=RuntimeError:rollback_error_type=RuntimeError",
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
            {"chat_id": "chat", "message_id": 606},
        )

    def test_success_without_valid_message_id_is_ambiguous(self):
        with patch.object(publisher, "tg_request", return_value={"ok": True, "result": {}}) as request:
            with self.assertRaisesRegex(
                publisher.TelegramDeliveryOutcomeAmbiguous,
                "telegram_delivery_outcome_ambiguous:missing_result_message_id",
            ):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                )

        self.assertEqual([call.args[0] for call in request.call_args_list], ["sendPhoto"])

    def test_server_5xx_is_ambiguous_without_response_body_in_error(self):
        response = Mock()
        response.status_code = 503
        response.ok = False
        response.json.return_value = {"ok": False, "description": "sensitive upstream detail"}
        response.text = "sensitive body"
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests, "post", return_value=response
        ):
            with self.assertRaises(publisher.TelegramDeliveryOutcomeAmbiguous) as raised:
                publisher.tg_request("sendPhoto", data={"chat_id": "chat"})

        self.assertEqual(
            str(raised.exception),
            "telegram_delivery_outcome_ambiguous:http_status=503",
        )
        self.assertNotIn("sensitive", str(raised.exception))
        self.assertNotIn("test-token", str(raised.exception))
        self.assertNotIn("chat", str(raised.exception))

    def test_deterministic_parse_reject_is_not_ambiguous(self):
        error = RuntimeError("Bad Request: can't parse entities")
        self.assertTrue(publisher._is_probably_parse_mode_error(error))
        self.assertNotIsInstance(error, publisher.TelegramDeliveryOutcomeAmbiguous)

    def test_publisher_level_ambiguous_primary_send_is_fail_closed_before_record_and_posted(self):
        source = inspect.getsource(publisher.amain)
        send_at = source.index("post_message_id = send_post_with_visual(")
        ambiguous_at = source.index("except TelegramDeliveryOutcomeAmbiguous as e:", send_at)
        generic_at = source.index("except Exception as e:", ambiguous_at)
        record_at = source.index("store.record_publication(", generic_at)
        posted_at = source.index("posted += 1", record_at)
        zero_alert_at = source.index("if posted == 0 and not DRY_RUN:", posted_at)
        ambiguous_clause = source[ambiguous_at:generic_at]

        self.assertIn("telegram_delivery_outcome_ambiguous", ambiguous_clause)
        self.assertIn("raise", ambiguous_clause)
        self.assertNotIn("continue", ambiguous_clause)
        self.assertLess(ambiguous_at, record_at)
        self.assertLess(record_at, posted_at)
        self.assertLess(posted_at, zero_alert_at)
        self.assertEqual(source.count("post_message_id = send_post_with_visual("), 1)

    def test_short_caption_success_never_uses_rollback(self):
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 100), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 100
        ), patch.object(publisher, "tg_request", return_value=_payload(707)) as request:
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "short", "<b>short</b>"
            )

        self.assertEqual(message_id, 707)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["sendPhoto"])
        self.assertNotIn("deleteMessage", [call.args[0] for call in request.call_args_list])


if __name__ == "__main__":
    unittest.main()

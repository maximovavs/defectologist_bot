from contextlib import redirect_stdout
from io import BytesIO, StringIO
import inspect
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.publisher import run_publisher as publisher
from src.services import publication_store as store_module
from src.services.publication_store import (
    PublicationDeliveryStateBlocked,
    PublicationStore,
)


def _telegram_response(message_id: int):
    response = Mock()
    response.status_code = 200
    response.ok = True
    response.json.return_value = {"ok": True, "result": {"message_id": message_id}}
    response.text = ""
    return response


def _telegram_result(payload):
    response = Mock()
    response.status_code = 200
    response.ok = True
    response.json.return_value = payload
    response.text = ""
    return response


def _telegram_reject(description: str, status_code: int = 400):
    response = Mock()
    response.status_code = status_code
    response.ok = False
    response.json.return_value = {"ok": False, "description": description}
    response.text = description
    return response


class DurableDeliveryStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.state_dir = Path(self.tmp.name) / ".state"
        self.state_dir.mkdir()
        self.env = patch.dict(
            os.environ,
            {
                "DRY_RUN": "0",
                "PRODUCTION_STATE_RESTORED": "",
            },
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)

    def _test_store(self) -> PublicationStore:
        store = PublicationStore(self.state_dir / "publication_history_test.sqlite3")
        self.addCleanup(store.deactivate_publisher_delivery_hooks)
        return store

    def _record(self, store: PublicationStore, canonical_url: str = "https://example.org/post") -> None:
        with patch.object(store_module, "text_batch_to_embeddings", return_value=[[], []]):
            store.record_publication(
                canonical_url=canonical_url,
                body_hash="b" * 64,
                body_text="body",
                evidence_hash="e" * 64,
                evidence_text="evidence",
                posted_at="2026-08-24T00:00:00+00:00",
                audience="parents",
                rubric_id="tip_of_day",
                rubric_title="Tip",
                source_domain="example.org",
            )

    def _attempt(self, store: PublicationStore):
        attempts = store.delivery_attempts()
        self.assertEqual(len(attempts), 1)
        return attempts[0]

    def test_ambiguous_primary_timeout_persists_cross_run_quarantine(self):
        store = self._test_store()
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=publisher.requests.Timeout("network timeout"),
        ) as post:
            with self.assertRaises(publisher.TelegramDeliveryOutcomeAmbiguous):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                )

        self.assertEqual(post.call_count, 1)
        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "ambiguous")
        self.assertEqual(attempt["primary_message_ids"], [])

        store.deactivate_publisher_delivery_hooks()
        with self.assertRaisesRegex(
            PublicationDeliveryStateBlocked,
            "unresolved_delivery_quarantine",
        ):
            PublicationStore(store.db_path)

    def test_server_5xx_persists_cross_run_quarantine(self):
        store = self._test_store()
        response = _telegram_reject("upstream unavailable", status_code=503)
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests, "post", return_value=response
        ):
            with self.assertRaises(publisher.TelegramDeliveryOutcomeAmbiguous):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                )

        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "ambiguous")
        self.assertEqual(attempt["primary_message_ids"], [])

    def test_deterministic_reject_does_not_leave_permanent_quarantine(self):
        store = self._test_store()
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 1), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 1
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            return_value=_telegram_reject("Bad Request: chat not found"),
        ):
            with self.assertRaisesRegex(RuntimeError, "chat not found"):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        self.assertFalse(store.has_unresolved_delivery_attempts())
        store.deactivate_publisher_delivery_hooks()
        reopened = PublicationStore(store.db_path)
        reopened.deactivate_publisher_delivery_hooks()
        self.assertFalse(reopened.has_unresolved_delivery_attempts())

    def test_short_caption_success_records_and_clears_attempt_atomically(self):
        store = self._test_store()
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 100), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 100
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            return_value=_telegram_response(707),
        ):
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "short", "<b>short</b>"
            )

        self.assertEqual(message_id, 707)
        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "confirmed")
        self.assertEqual(attempt["primary_message_ids"], [707])

        self._record(store)
        self.assertFalse(store.has_unresolved_delivery_attempts())
        self.assertTrue(store.has_url("https://example.org/post"))

    def test_html_parse_reject_then_plain_success_preserves_delivery_contract(self):
        store = self._test_store()
        responses = [
            _telegram_reject("Bad Request: can't parse entities"),
            _telegram_response(808),
        ]
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 100), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 100
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=responses,
        ) as post:
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "short", "<b>short</b>"
            )

        self.assertEqual(message_id, 808)
        self.assertEqual(post.call_count, 2)
        self.assertNotIn("parse_mode", post.call_args_list[1].kwargs["data"])
        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "confirmed")
        self.assertEqual(attempt["primary_message_ids"], [808])
        self._record(store)
        self.assertFalse(store.has_unresolved_delivery_attempts())

    def test_long_split_success_tracks_both_primary_message_ids(self):
        store = self._test_store()
        responses = [_telegram_response(101), _telegram_response(202)]
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=responses,
        ):
            message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
            )

        self.assertEqual(message_id, 202)
        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "confirmed")
        self.assertEqual(attempt["primary_message_ids"], [101, 202])
        self._record(store)
        self.assertFalse(store.has_unresolved_delivery_attempts())

    def test_long_split_ambiguous_text_keeps_known_photo_receipt(self):
        store = self._test_store()
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=[
                _telegram_response(303),
                publisher.requests.Timeout("text timeout"),
            ],
        ):
            with self.assertRaises(publisher.TelegramDeliveryOutcomeAmbiguous):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "ambiguous")
        self.assertEqual(attempt["primary_message_ids"], [303])

    def test_deterministic_split_rollback_clears_delivery_attempt(self):
        store = self._test_store()
        delete_ok = _telegram_result({"ok": True, "result": True})
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=[
                _telegram_response(404),
                _telegram_reject("text rejected"),
                delete_ok,
            ],
        ) as post:
            with self.assertRaisesRegex(RuntimeError, "text rejected"):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        self.assertEqual(post.call_count, 3)
        self.assertFalse(store.has_unresolved_delivery_attempts())

    def test_split_rollback_failure_remains_ambiguous_with_photo_receipt(self):
        store = self._test_store()
        with patch.object(publisher, "TG_CAPTION_MAX_UTF16_UNITS", 5), patch.object(
            publisher, "TG_CAPTION_MAX_BYTES", 5
        ), patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            side_effect=[
                _telegram_response(606),
                _telegram_reject("text rejected"),
                _telegram_reject("delete rejected"),
            ],
        ):
            with self.assertRaisesRegex(
                publisher.TelegramDeliveryOutcomeAmbiguous,
                "telegram_split_delivery_rollback_failed",
            ):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "long plain text", "<b>long plain text</b>"
                )

        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "ambiguous")
        self.assertEqual(attempt["primary_message_ids"], [606])

    def test_success_without_message_id_is_ambiguous_and_quarantined(self):
        store = self._test_store()
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            return_value=_telegram_result({"ok": True, "result": {}}),
        ):
            with self.assertRaisesRegex(
                publisher.TelegramDeliveryOutcomeAmbiguous,
                "missing_result_message_id",
            ):
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                )

        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "ambiguous")
        self.assertEqual(attempt["primary_message_ids"], [])

    def test_confirmed_send_then_record_failure_keeps_confirmed_quarantine(self):
        store = self._test_store()
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            return_value=_telegram_response(909),
        ):
            self.assertEqual(
                publisher.send_post_with_visual(
                    "chat", BytesIO(b"image"), "short", "<b>short</b>"
                ),
                909,
            )

        self.assertEqual(self._attempt(store)["state"], "confirmed")
        with patch.object(store_module, "text_batch_to_embeddings", return_value=[[], []]), patch.object(
            store,
            "_connect",
            side_effect=sqlite3.OperationalError("disk unavailable"),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "disk unavailable"):
                store.record_publication(
                    canonical_url="https://example.org/post",
                    body_hash="b" * 64,
                    body_text="body",
                    evidence_hash="e" * 64,
                    evidence_text="evidence",
                    posted_at="2026-08-24T00:00:00+00:00",
                    audience="parents",
                    rubric_id="tip_of_day",
                    rubric_title="Tip",
                    source_domain="example.org",
                )

        attempt = self._attempt(store)
        self.assertEqual(attempt["state"], "confirmed")
        self.assertEqual(attempt["primary_message_ids"], [909])

        source = inspect.getsource(publisher.amain)
        record_at = source.index("store.record_publication(")
        seen_at = source.index("seen_urls_this_run.add(canon)", record_at)
        posted_at = source.index("posted += 1", seen_at)
        self.assertNotIn("except ", source[record_at:seen_at])
        self.assertLess(record_at, seen_at)
        self.assertLess(seen_at, posted_at)

    def test_poll_failure_does_not_reopen_or_damage_primary_delivery_state(self):
        store = self._test_store()
        with patch.object(publisher, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(
            publisher.requests,
            "post",
            return_value=_telegram_response(1001),
        ):
            post_message_id = publisher.send_post_with_visual(
                "chat", BytesIO(b"image"), "short", "<b>short</b>"
            )
        self._record(store)
        self.assertFalse(store.has_unresolved_delivery_attempts())

        poll = publisher.PollSpec(
            question="Попробуете?",
            options=("Да", "Нет", "Позже"),
        )
        spec = publisher.EngagementSpec(kind="poll", mode="auto", poll=poll)
        output = StringIO()
        with patch.object(
            publisher,
            "send_post_poll",
            side_effect=RuntimeError("poll unavailable"),
        ), redirect_stdout(output):
            result = publisher._handle_post_engagement(
                spec=spec,
                rubric_id="tip_of_day",
                canonical_url="https://example.org/post",
                chat_id="chat",
                post_message_id=post_message_id,
            )

        self.assertIsNone(result)
        self.assertIn("[POLL][WARN] poll_send_failed", output.getvalue())
        self.assertFalse(store.has_unresolved_delivery_attempts())

    def test_production_missing_or_unrestored_history_fails_before_db_creation(self):
        prod_path = self.state_dir / "publication_history.sqlite3"
        with patch.dict(
            os.environ,
            {"DRY_RUN": "0", "PRODUCTION_STATE_RESTORED": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(
                PublicationDeliveryStateBlocked,
                "production_state_not_restored",
            ):
                PublicationStore(prod_path)
        self.assertFalse(prod_path.exists())

        with patch.dict(
            os.environ,
            {"DRY_RUN": "0", "PRODUCTION_STATE_RESTORED": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(
                PublicationDeliveryStateBlocked,
                "production_history_missing",
            ):
                PublicationStore(prod_path)
        self.assertFalse(prod_path.exists())

    def test_restored_production_history_is_allowed(self):
        prod_path = self.state_dir / "publication_history.sqlite3"
        sqlite3.connect(prod_path).close()
        with patch.dict(
            os.environ,
            {"DRY_RUN": "0", "PRODUCTION_STATE_RESTORED": "1"},
            clear=False,
        ):
            store = PublicationStore(prod_path)
        self.addCleanup(store.deactivate_publisher_delivery_hooks)
        self.assertTrue(prod_path.exists())
        self.assertFalse(store.has_unresolved_delivery_attempts())

    def test_test_state_and_dry_run_do_not_require_production_restore(self):
        test_store = self._test_store()
        self.assertTrue(test_store.db_path.exists())

        dry_prod = self.state_dir / "dry" / ".state" / "publication_history.sqlite3"
        dry_prod.parent.mkdir(parents=True)
        with patch.dict(
            os.environ,
            {"DRY_RUN": "1", "PRODUCTION_STATE_RESTORED": ""},
            clear=False,
        ):
            store = PublicationStore(dry_prod)
        self.assertTrue(dry_prod.exists())
        self.assertFalse(store.has_unresolved_delivery_attempts())


class ProductionWorkflowDeliveryStateTest(unittest.TestCase):
    def test_prod_cache_restore_is_verified_before_publisher(self):
        workflow = (publisher.ROOT / ".github" / "workflows" / "post.yml").read_text(
            encoding="utf-8"
        )

        restore_at = workflow.index("- name: Restore .state cache")
        verify_at = workflow.index("- name: Verify production state restore")
        publisher_at = workflow.index("- name: Run Publisher")

        self.assertLess(restore_at, verify_at)
        self.assertLess(verify_at, publisher_at)
        self.assertIn("id: restore-state", workflow)
        self.assertIn("steps.restore-state.outputs.cache-matched-key", workflow)
        self.assertIn(".state/publication_history.sqlite3", workflow)
        self.assertIn('PRODUCTION_STATE_RESTORED=1', workflow)
        self.assertIn("if: ${{ env.DRY_RUN != '1' && env.STATE_SCOPE == 'prod' }}", workflow)


if __name__ == "__main__":
    unittest.main()

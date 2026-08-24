from contextlib import redirect_stdout
from io import BytesIO, StringIO
import inspect
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scripts import resolve_production_state_predecessor as continuity
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


def _workflow_run(
    run_id: int,
    run_number: int,
    *,
    channel: str = "prod",
    conclusion: str = "success",
    status: str = "completed",
    run_attempt: int = 1,
    branch: str = "main",
    event: str = "schedule",
    title: str = "",
):
    return {
        "id": run_id,
        "run_number": run_number,
        "run_attempt": run_attempt,
        "head_branch": branch,
        "status": status,
        "conclusion": conclusion,
        "display_title": title
        or f"Logoped Bot • {event} • channel={channel} • provider=auto",
        "event": event,
    }


class ProductionStatePredecessorTest(unittest.TestCase):
    def _resolve(self, runs, *, current_attempt=1, jobs_loader=None):
        return continuity.resolve_predecessor(
            runs,
            current_run_id=100,
            current_run_number=100,
            current_run_attempt=current_attempt,
            ref_name="main",
            jobs_loader=jobs_loader or (lambda _run_id: {"jobs": []}),
        )

    def test_exact_predecessor_is_selected(self):
        predecessor = self._resolve([_workflow_run(99, 99), _workflow_run(98, 98)])
        self.assertEqual((predecessor.run_id, predecessor.run_number), (99, 99))

    def test_test_channel_run_is_not_prod_predecessor(self):
        predecessor = self._resolve(
            [_workflow_run(99, 99, channel="test"), _workflow_run(98, 98)]
        )
        self.assertEqual(predecessor.run_id, 98)

    def test_current_run_is_excluded_before_incomplete_current_metadata_matters(self):
        predecessor = self._resolve(
            [{"id": 100, "run_number": 100}, _workflow_run(99, 99)]
        )
        self.assertEqual(predecessor.run_id, 99)

    def test_newer_run_is_never_accepted_as_predecessor(self):
        predecessor = self._resolve(
            [{"id": 101, "run_number": 101}, _workflow_run(99, 99)]
        )
        self.assertEqual(predecessor.run_id, 99)

    def test_missing_predecessor_fails_closed(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "production_predecessor_missing",
        ):
            self._resolve([_workflow_run(99, 99, channel="test")])

    def test_ambiguous_lineage_metadata_fails_closed(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "ambiguous_run_metadata:channel",
        ):
            self._resolve([_workflow_run(99, 99, title="Logoped Bot without channel")])

    def test_current_production_rerun_attempt_fails_closed(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "production_rerun_not_safe",
        ):
            self._resolve([_workflow_run(99, 99)], current_attempt=2)

    def test_predecessor_rerun_attempt_fails_closed(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "production_predecessor_rerun_not_safe",
        ):
            self._resolve([_workflow_run(99, 99, run_attempt=2)])

    def test_cancelled_before_start_is_skipped_only_with_job_proof(self):
        jobs = {
            "jobs": [
                {
                    "name": "post",
                    "status": "completed",
                    "conclusion": "cancelled",
                    "started_at": None,
                }
            ]
        }
        predecessor = self._resolve(
            [_workflow_run(99, 99, conclusion="cancelled"), _workflow_run(98, 98)],
            jobs_loader=lambda run_id: jobs if run_id == 99 else {"jobs": []},
        )
        self.assertEqual(predecessor.run_id, 98)

    def test_cancelled_started_run_is_not_skipped(self):
        jobs = {
            "jobs": [
                {
                    "name": "post",
                    "status": "completed",
                    "conclusion": "cancelled",
                    "started_at": "2026-08-24T00:00:00Z",
                }
            ]
        }
        predecessor = self._resolve(
            [_workflow_run(99, 99, conclusion="cancelled"), _workflow_run(98, 98)],
            jobs_loader=lambda _run_id: jobs,
        )
        self.assertEqual(predecessor.run_id, 99)

    def test_cancelled_run_without_exact_post_job_proof_fails_closed(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "ambiguous_post_job_identity",
        ):
            self._resolve(
                [_workflow_run(99, 99, conclusion="cancelled"), _workflow_run(98, 98)],
                jobs_loader=lambda _run_id: {"jobs": []},
            )

    def test_failed_started_run_cannot_be_silently_skipped(self):
        jobs_loader = Mock(side_effect=AssertionError("jobs should not be queried"))
        predecessor = self._resolve(
            [_workflow_run(99, 99, conclusion="failure"), _workflow_run(98, 98)],
            jobs_loader=jobs_loader,
        )
        self.assertEqual(predecessor.run_id, 99)
        jobs_loader.assert_not_called()

    def test_timed_out_started_run_cannot_be_silently_skipped(self):
        jobs_loader = Mock(side_effect=AssertionError("jobs should not be queried"))
        predecessor = self._resolve(
            [_workflow_run(99, 99, conclusion="timed_out"), _workflow_run(98, 98)],
            jobs_loader=jobs_loader,
        )
        self.assertEqual(predecessor.run_id, 99)
        jobs_loader.assert_not_called()

    def test_duplicate_prod_run_number_is_ambiguous(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "ambiguous_production_predecessor_order",
        ):
            self._resolve([_workflow_run(99, 99), _workflow_run(97, 99)])

    def test_expected_cache_key_uses_exact_predecessor_run_id(self):
        self.assertEqual(
            continuity.build_expected_cache_key(
                cache_version="v12",
                ref_name="main",
                predecessor_run_id=99,
            ),
            "logoped-state-v12-prod-main-99",
        )

    def test_pure_resolver_tests_do_not_make_network_calls(self):
        with patch.object(continuity.urllib.request, "urlopen") as urlopen:
            predecessor = self._resolve([_workflow_run(99, 99)])
        self.assertEqual(predecessor.run_id, 99)
        urlopen.assert_not_called()


class ProductionWorkflowDeliveryStateTest(unittest.TestCase):
    def setUp(self):
        self.workflow = (
            publisher.ROOT / ".github" / "workflows" / "post.yml"
        ).read_text(encoding="utf-8")

    def test_predecessor_resolution_and_exact_prod_restore_precede_publisher(self):
        resolver_at = self.workflow.index("- name: Resolve production state predecessor")
        restore_at = self.workflow.index("- name: Restore production .state cache")
        verify_at = self.workflow.index("- name: Verify production state continuity")
        publisher_at = self.workflow.index("- name: Run Publisher")
        self.assertLess(resolver_at, restore_at)
        self.assertLess(restore_at, verify_at)
        self.assertLess(verify_at, publisher_at)
        self.assertIn("actions: read", self.workflow)

    def test_production_restore_has_no_prefix_fallback_and_fails_on_exact_miss(self):
        start = self.workflow.index("- name: Restore production .state cache")
        end = self.workflow.index("- name: Restore test .state cache", start)
        block = self.workflow[start:end]
        self.assertIn("steps.state-predecessor.outputs.expected_cache_key", block)
        self.assertIn("fail-on-cache-miss: true", block)
        self.assertNotIn("restore-keys:", block)

    def test_production_verification_checks_hit_key_and_history_before_markers(self):
        start = self.workflow.index("- name: Verify production state continuity")
        end = self.workflow.index("- name: Set up Python", start)
        block = self.workflow[start:end]
        self.assertIn('if [ "$CACHE_HIT" != "true" ]', block)
        self.assertIn('if [ "$RESTORED_CACHE_KEY" != "$EXPECTED_CACHE_KEY" ]', block)
        self.assertIn(".state/publication_history.sqlite3", block)
        restored_at = block.index("PRODUCTION_STATE_RESTORED=1")
        continuity_at = block.index("PROD_STATE_CONTINUITY_OK=1")
        output_at = block.index("continuity_ok=true")
        self.assertGreater(restored_at, block.index("production_history_missing"))
        self.assertGreater(continuity_at, restored_at)
        self.assertGreater(output_at, continuity_at)

    def test_production_save_requires_verified_continuity_and_keeps_always_semantics(self):
        start = self.workflow.index("- name: Save production .state cache")
        end = self.workflow.index("- name: Save test .state cache", start)
        block = self.workflow[start:end]
        self.assertIn("always()", block)
        self.assertIn("env.STATE_SCOPE == 'prod'", block)
        self.assertIn("env.PROD_STATE_CONTINUITY_OK == '1'", block)
        self.assertIn("steps.verify-prod-state.outputs.continuity_ok == 'true'", block)
        self.assertIn("github.run_id", block)

    def test_prod_dry_run_does_not_bypass_continuity(self):
        resolver_start = self.workflow.index("- name: Resolve production state predecessor")
        verify_end = self.workflow.index("- name: Set up Python", resolver_start)
        prod_gate = self.workflow[resolver_start:verify_end]
        self.assertIn("env.STATE_SCOPE == 'prod'", prod_gate)
        self.assertNotIn("DRY_RUN !=", prod_gate)

    def test_test_state_uses_separate_restore_and_can_keep_prefix_fallback(self):
        start = self.workflow.index("- name: Restore test .state cache")
        end = self.workflow.index("- name: Verify production state continuity", start)
        block = self.workflow[start:end]
        self.assertIn("env.STATE_SCOPE != 'prod'", block)
        self.assertIn("restore-keys:", block)

    def test_publisher_policy_ci_compiles_and_tracks_resolver_without_runtime_secrets(self):
        policy = (
            publisher.ROOT / ".github" / "workflows" / "publisher_policy_pr_checks.yml"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            policy.count("scripts/resolve_production_state_predecessor.py"),
            2,
        )
        self.assertIn('TELEGRAM_BOT_TOKEN: ""', policy)
        self.assertIn('GEMINI_API_KEY: ""', policy)
        self.assertIn('GROQ_API_KEY: ""', policy)
        self.assertIn('POLLINATIONS_TOKEN: ""', policy)


if __name__ == "__main__":
    unittest.main()

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import inspect
import json
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
    head_sha: str = "test-head-sha",
    event: str = "schedule",
    title: str = "",
):
    return {
        "id": run_id,
        "run_number": run_number,
        "run_attempt": run_attempt,
        "head_branch": branch,
        "head_sha": head_sha,
        "status": status,
        "conclusion": conclusion,
        "display_title": title
        or f"Logoped Bot • {event} • channel={channel} • provider=auto",
        "event": event,
    }


def _post_job_payload(
    *,
    run_conclusion: str = "failure",
    publisher_status: str = "completed",
    publisher_conclusion: str = "skipped",
    started_at: object = "2026-08-24T09:10:08Z",
    steps: object = None,
):
    if steps is None:
        steps = [
            {
                "name": "Run Publisher",
                "status": publisher_status,
                "conclusion": publisher_conclusion,
            }
        ]
    return {
        "jobs": [
            {
                "name": "post",
                "status": "completed",
                "conclusion": run_conclusion,
                "started_at": started_at,
                "steps": steps,
            }
        ]
    }


class ProductionStatePredecessorTest(unittest.TestCase):
    def _resolve(
        self,
        runs,
        *,
        current_attempt=1,
        jobs_loader=None,
        current_run_id=100,
        current_run_number=100,
    ):
        return continuity.resolve_predecessor(
            runs,
            current_run_id=current_run_id,
            current_run_number=current_run_number,
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

    def test_incident_446_pre_publisher_failure_skips_to_445(self):
        jobs_446 = _post_job_payload(
            run_conclusion="failure",
            steps=[
                {
                    "name": "Set up job",
                    "status": "completed",
                    "conclusion": "success",
                },
                {
                    "name": "Run actions/checkout@v6",
                    "status": "completed",
                    "conclusion": "success",
                },
                {
                    "name": "Resolve production state predecessor",
                    "status": "completed",
                    "conclusion": "failure",
                },
                {
                    "name": "Restore production .state cache",
                    "status": "completed",
                    "conclusion": "skipped",
                },
                {
                    "name": "Run Publisher",
                    "status": "completed",
                    "conclusion": "skipped",
                },
                {
                    "name": "Save production .state cache",
                    "status": "completed",
                    "conclusion": "skipped",
                },
            ],
        )
        runs = [
            _workflow_run(4440, 444, title="Legacy run before channel marker"),
            _workflow_run(4460, 446, conclusion="failure"),
            _workflow_run(4450, 445),
        ]
        with patch.object(continuity.urllib.request, "urlopen") as urlopen:
            predecessor = self._resolve(
                runs,
                current_run_id=4470,
                current_run_number=447,
                jobs_loader=lambda run_id: jobs_446 if run_id == 4460 else {"jobs": []},
            )
        self.assertEqual((predecessor.run_id, predecessor.run_number), (4450, 445))
        urlopen.assert_not_called()

    def test_incident_470_failure_skips_through_471_to_last_valid_state(self):
        incident_head_sha = "a1bf73dd1e647d1a2436a7f6896f55edc1a8412f"
        runs = [
            _workflow_run(
                34390642457,
                471,
                conclusion="failure",
                event="workflow_dispatch",
                head_sha=incident_head_sha,
            ),
            _workflow_run(
                34389481621,
                470,
                conclusion="failure",
                event="workflow_dispatch",
                head_sha=incident_head_sha,
            ),
            _workflow_run(
                34354544933,
                469,
                head_sha="045ac50da80ac5a73bba5bfe11142809c1accc43",
            ),
        ]
        jobs = {
            34390642457: _post_job_payload(
                run_conclusion="failure",
                publisher_conclusion="skipped",
            ),
            34389481621: _post_job_payload(
                run_conclusion="failure",
                publisher_conclusion="failure",
            ),
        }

        predecessor = self._resolve(
            runs,
            current_run_id=34392000000,
            current_run_number=472,
            jobs_loader=lambda run_id: jobs[run_id],
        )

        self.assertEqual(
            (predecessor.run_id, predecessor.run_number),
            (34354544933, 469),
        )
        self.assertEqual(
            continuity.build_expected_cache_key(
                cache_version="v12",
                ref_name="main",
                predecessor_run_id=predecessor.run_id,
            ),
            "logoped-state-v12-prod-main-34354544933",
        )

    def test_incident_exception_requires_every_immutable_metadata_field(self):
        exact = _workflow_run(
            34389481621,
            470,
            conclusion="failure",
            event="workflow_dispatch",
            head_sha="a1bf73dd1e647d1a2436a7f6896f55edc1a8412f",
        )
        mismatches = {
            "run_id": {**exact, "id": 34389481622},
            "run_number": {**exact, "run_number": 469},
            "event": {**exact, "event": "schedule"},
            "head_branch": {**exact, "head_branch": "incident"},
            "head_sha": {**exact, "head_sha": "0" * 40},
            "conclusion": {**exact, "conclusion": "cancelled"},
        }

        for field, run in mismatches.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                continuity.StateContinuityError,
                "proven_pre_mutation_incident_metadata_mismatch",
            ):
                self._resolve(
                    [run, _workflow_run(34354544933, 468)],
                    current_run_id=34392000000,
                    current_run_number=472,
                    jobs_loader=lambda _run_id: _post_job_payload(
                        run_conclusion="failure",
                        publisher_conclusion="failure",
                    ),
                )

    def test_incident_metadata_does_not_skip_nonfailure_publisher_outcome(self):
        incident = _workflow_run(
            34389481621,
            470,
            conclusion="failure",
            event="workflow_dispatch",
            head_sha="a1bf73dd1e647d1a2436a7f6896f55edc1a8412f",
        )
        predecessor = self._resolve(
            [incident, _workflow_run(34354544933, 469)],
            current_run_id=34392000000,
            current_run_number=472,
            jobs_loader=lambda _run_id: _post_job_payload(
                run_conclusion="failure",
                publisher_conclusion="success",
            ),
        )
        self.assertEqual((predecessor.run_id, predecessor.run_number), (34389481621, 470))

    def test_newest_to_oldest_selection_is_independent_of_input_order(self):
        jobs_99 = _post_job_payload(run_conclusion="failure")
        predecessor = self._resolve(
            [
                _workflow_run(97, 97),
                _workflow_run(99, 99, conclusion="failure"),
                _workflow_run(98, 98),
            ],
            jobs_loader=lambda run_id: jobs_99 if run_id == 99 else {"jobs": []},
        )
        self.assertEqual(predecessor.run_id, 98)

    def test_legacy_metadata_older_than_selected_predecessor_is_not_validated(self):
        predecessor = self._resolve(
            [
                _workflow_run(97, 97, title="Legacy run before channel marker"),
                _workflow_run(99, 99),
            ]
        )
        self.assertEqual(predecessor.run_id, 99)

    def test_legacy_metadata_between_current_and_predecessor_fails_closed(self):
        with self.assertRaisesRegex(
            continuity.StateContinuityError,
            "ambiguous_run_metadata:channel",
        ):
            self._resolve(
                [
                    _workflow_run(98, 98),
                    _workflow_run(99, 99, title="Legacy run before channel marker"),
                ]
            )

    def test_pre_publisher_terminal_runs_are_skipped_only_with_step_proof(self):
        for conclusion in ("failure", "cancelled", "timed_out", "skipped"):
            with self.subTest(conclusion=conclusion):
                jobs = _post_job_payload(run_conclusion=conclusion)
                predecessor = self._resolve(
                    [
                        _workflow_run(99, 99, conclusion=conclusion),
                        _workflow_run(98, 98),
                    ],
                    jobs_loader=lambda run_id, payload=jobs: (
                        payload if run_id == 99 else {"jobs": []}
                    ),
                )
                self.assertEqual(predecessor.run_id, 98)

    def test_mutation_capable_publisher_outcomes_are_never_skipped(self):
        for publisher_conclusion in ("success", "failure", "cancelled", "timed_out"):
            with self.subTest(publisher_conclusion=publisher_conclusion):
                jobs = _post_job_payload(
                    run_conclusion="failure",
                    publisher_conclusion=publisher_conclusion,
                )
                predecessor = self._resolve(
                    [
                        _workflow_run(99, 99, conclusion="failure"),
                        _workflow_run(98, 98),
                    ],
                    jobs_loader=lambda run_id, payload=jobs: (
                        payload if run_id == 99 else {"jobs": []}
                    ),
                )
                self.assertEqual(predecessor.run_id, 99)

    def test_missing_or_ambiguous_job_step_metadata_fails_closed(self):
        payloads = {
            "missing_post_job": {"jobs": []},
            "missing_steps": {
                "jobs": [
                    {
                        "name": "post",
                        "status": "completed",
                        "conclusion": "failure",
                        "started_at": "2026-08-24T09:10:08Z",
                    }
                ]
            },
            "missing_publisher_step": _post_job_payload(run_conclusion="failure", steps=[]),
            "duplicate_publisher_step": _post_job_payload(
                run_conclusion="failure",
                steps=[
                    {
                        "name": "Run Publisher",
                        "status": "completed",
                        "conclusion": "skipped",
                    },
                    {
                        "name": "Run Publisher",
                        "status": "completed",
                        "conclusion": "skipped",
                    },
                ],
            ),
            "incomplete_publisher_step": _post_job_payload(
                run_conclusion="failure",
                steps=[
                    {
                        "name": "Run Publisher",
                        "status": "",
                        "conclusion": "",
                    }
                ],
            ),
        }
        for case, jobs in payloads.items():
            with self.subTest(case=case), self.assertRaises(continuity.StateContinuityError):
                self._resolve(
                    [
                        _workflow_run(99, 99, conclusion="failure"),
                        _workflow_run(98, 98),
                    ],
                    jobs_loader=lambda run_id, payload=jobs: (
                        payload if run_id == 99 else {"jobs": []}
                    ),
                )

    def test_cancelled_started_run_with_publisher_cancelled_is_not_skipped(self):
        jobs = _post_job_payload(
            run_conclusion="cancelled",
            publisher_conclusion="cancelled",
        )
        predecessor = self._resolve(
            [_workflow_run(99, 99, conclusion="cancelled"), _workflow_run(98, 98)],
            jobs_loader=lambda run_id: jobs if run_id == 99 else {"jobs": []},
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



# Run #514 (37486543863) requested the cache of #342 (27962120533) although
# #513 (37347011941) was the successful prod predecessor. Root cause: PARTIAL.
# These tests pin the stderr diagnostics added to make a recurrence
# distinguishable, and that they change no decision.
_DIAG_TOKEN = "ghs_DIAGNOSTIC_TOKEN_MUST_NOT_LEAK"
_RUN_513 = _workflow_run(37347011941, 513)
_RUN_343 = _workflow_run(28023253494, 343)
_RUN_342 = _workflow_run(27962120533, 342, event="workflow_dispatch")
_RUN_514 = {"id": 37486543863, "run_number": 514, "run_attempt": 1, "status": "in_progress"}


class _FakeApiResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, *_args):
        body, self._body = self._body, b""
        return body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class ProductionStatePredecessorDiagnosticsTest(unittest.TestCase):
    ENV = {
        "GITHUB_RUN_ID": "37486543863",
        "GITHUB_RUN_NUMBER": "514",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REF_NAME": "main",
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_REPOSITORY": "maximovavs/defectologist_bot",
        "GITHUB_TOKEN": _DIAG_TOKEN,
        "STATE_CACHE_VERSION": "v12",
    }

    def _resolve(self, runs, *, jobs_loader=None, current_run_id=100, current_run_number=100,
                 current_attempt=1):
        stderr = StringIO()
        stdout = StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            predecessor = continuity.resolve_predecessor(
                runs,
                current_run_id=current_run_id,
                current_run_number=current_run_number,
                current_run_attempt=current_attempt,
                ref_name="main",
                jobs_loader=jobs_loader or (lambda _run_id: {"jobs": []}),
            )
        self.assertEqual(stdout.getvalue(), "")
        return predecessor, stderr.getvalue()

    def _main(self, pages, jobs=None, env=None):
        requests = []

        def urlopen(request, timeout=None):
            requests.append(request)
            url = request.full_url
            if "/jobs?" in url:
                run_id = int(url.split("/actions/runs/", 1)[1].split("/", 1)[0])
                return _FakeApiResponse((jobs or {}).get(run_id, {"jobs": []}))
            page = int(url.rsplit("page=", 1)[1])
            return _FakeApiResponse({"workflow_runs": pages[page - 1]})

        stdout = StringIO()
        stderr = StringIO()
        with patch.dict(os.environ, env or self.ENV, clear=True), patch.object(
            continuity.urllib.request, "urlopen", side_effect=urlopen
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = continuity.main()
        return code, stdout.getvalue(), stderr.getvalue(), requests

    def _lines(self, stderr, tag):
        prefix = f"[STATE_CONTINUITY][{tag}] "
        return [line[len(prefix):] for line in stderr.splitlines() if line.startswith(prefix)]

    # --- 1 / 2 / 3 / 7 / 10 / 16 / 18: main() on a #513-shaped history -----

    def test_main_selects_513_with_unchanged_stdout_and_diagnostics_on_stderr(self):
        code, stdout, stderr, requests = self._main([[_RUN_514, _RUN_513, _RUN_343, _RUN_342]])

        self.assertEqual(code, 0)
        # stdout is the $GITHUB_OUTPUT contract: exactly these two lines.
        self.assertEqual(
            stdout,
            "predecessor_run_id=37347011941\n"
            "expected_cache_key=logoped-state-v12-prod-main-37347011941\n",
        )
        self.assertNotIn("STATE_CONTINUITY", stdout)
        self.assertEqual(
            self._lines(stderr, "CURRENT"),
            ["run_id=37486543863 run_number=514 run_attempt=1 ref=main"],
        )
        self.assertEqual(
            self._lines(stderr, "SELECTED"),
            ["run_id=37347011941 run_number=513 conclusion=success"],
        )
        # Only the history page was read: a success is selected without jobs.
        self.assertEqual(len(requests), 1)

    def test_token_and_authorization_never_reach_stdout_or_stderr(self):
        code, stdout, stderr, requests = self._main([[_RUN_514, _RUN_513]])
        self.assertEqual(code, 0)
        self.assertEqual(requests[0].get_header("Authorization"), f"Bearer {_DIAG_TOKEN}")
        for stream in (stdout, stderr):
            self.assertNotIn(_DIAG_TOKEN, stream)
            self.assertNotIn("Authorization", stream)
            self.assertNotIn("Bearer", stream)
            self.assertNotIn("https://", stream)

    def test_blocked_main_keeps_stdout_empty_and_reports_on_stderr(self):
        env = dict(self.ENV, GITHUB_RUN_ATTEMPT="2")
        code, stdout, stderr, requests = self._main([[_RUN_514, _RUN_513]], env=env)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("production_state_continuity_blocked:production_rerun_not_safe", stderr)
        # The rerun identity is visible before resolution refuses it.
        self.assertEqual(
            self._lines(stderr, "CURRENT"),
            ["run_id=37486543863 run_number=514 run_attempt=2 ref=main"],
        )
        self.assertNotIn(_DIAG_TOKEN, stderr)

    # --- 4 / 17: page diagnostics and exact pagination -----------------------

    def test_each_page_logs_count_and_run_number_range_with_exact_requests(self):
        full_page = [_workflow_run(10_000 + n, 400 - n) for n in range(100)]
        full_page[0] = _RUN_513
        last_page = [_workflow_run(20_000 + n, 300 - n) for n in range(5)]
        code, stdout, stderr, requests = self._main([full_page, last_page])

        self.assertEqual(code, 0)
        self.assertEqual(
            [request.full_url for request in requests],
            [
                "https://api.github.com/repos/maximovavs/defectologist_bot/actions/workflows/"
                "post.yml/runs?branch=main&per_page=100&page=1",
                "https://api.github.com/repos/maximovavs/defectologist_bot/actions/workflows/"
                "post.yml/runs?branch=main&per_page=100&page=2",
            ],
        )
        self.assertEqual(
            self._lines(stderr, "PAGE"),
            [
                "page=1 count=100 min_run_number=301 max_run_number=513 unparsed_run_number=0",
                "page=2 count=5 min_run_number=296 max_run_number=300 unparsed_run_number=0",
            ],
        )

    def test_page_limit_and_truncation_failure_are_unchanged(self):
        full_page = [_workflow_run(n + 1, n + 1) for n in range(100)]
        code, stdout, stderr, requests = self._main([full_page] * 20)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(len(requests), 20)
        self.assertTrue(requests[-1].full_url.endswith("&per_page=100&page=20"))
        self.assertIn("production_state_continuity_blocked:github_actions_history_truncated", stderr)
        self.assertEqual(len(self._lines(stderr, "PAGE")), 20)

    def test_malformed_page_run_numbers_are_reported_not_normalised(self):
        page = [_RUN_514, {"id": 1, "run_number": "not-a-number"}, _RUN_513]
        code, stdout, stderr, _requests = self._main([page])
        # Resolution still fails closed exactly as before on the malformed run.
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("production_state_continuity_blocked:ambiguous_run_metadata:run_number", stderr)
        self.assertEqual(
            self._lines(stderr, "PAGE"),
            ["page=1 count=3 min_run_number=513 max_run_number=514 unparsed_run_number=1"],
        )

    def test_page_diagnostic_never_raises_on_pathological_run_number(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            continuity._emit_page_diagnostic(
                1, [{"run_number": float("inf")}, {"run_number": True}, "not-a-run", _RUN_513]
            )
        self.assertEqual(
            self._lines(stderr.getvalue(), "PAGE"),
            ["page=1 count=4 min_run_number=513 max_run_number=513 unparsed_run_number=3"],
        )

    def test_missing_recent_history_is_visible_in_page_and_candidate_diagnostics(self):
        # A #514-like recurrence: #513..#344 absent from the history the
        # runtime received. Resolution selects #343 exactly as before; the
        # diagnostics now show that the newest prior run seen was #343.
        code, stdout, stderr, _requests = self._main([[_RUN_514, _RUN_343, _RUN_342]])
        self.assertEqual(code, 0)
        self.assertIn("predecessor_run_id=28023253494\n", stdout)
        self.assertEqual(
            self._lines(stderr, "PAGE"),
            ["page=1 count=3 min_run_number=342 max_run_number=514 unparsed_run_number=0"],
        )
        candidates = self._lines(stderr, "CANDIDATE")
        self.assertTrue(candidates[0].startswith("rank=1 run_number=343 run_id=28023253494 "))

    # --- 5 / 6: bounded candidate summary, unchanged selection ---------------

    def test_candidate_summary_is_bounded_and_newest_first(self):
        runs = [_workflow_run(1000 + n, n) for n in range(1, 31)]
        predecessor, stderr = self._resolve(runs, current_run_id=5000, current_run_number=31)
        self.assertEqual((predecessor.run_id, predecessor.run_number), (1030, 30))
        self.assertEqual(
            self._lines(stderr, "CANDIDATES"),
            ["history_count=30 prior_count=30 shown=10 limit=10"],
        )
        candidates = self._lines(stderr, "CANDIDATE")
        self.assertEqual(len(candidates), continuity.DIAGNOSTIC_CANDIDATE_LIMIT)
        self.assertEqual(
            [line.split()[1] for line in candidates],
            [f"run_number={n}" for n in range(30, 20, -1)],
        )
        self.assertIn(
            "event=schedule head_branch=main run_attempt=1 status=completed "
            "conclusion=success lineage=prod",
            candidates[0],
        )

    def test_diagnostics_do_not_change_newest_to_oldest_selection(self):
        runs = [_RUN_342, _RUN_513, _RUN_343]
        for order in (runs, list(reversed(runs))):
            with self.subTest(order=[r["run_number"] for r in order]):
                predecessor, _stderr = self._resolve(
                    order, current_run_id=37486543863, current_run_number=514
                )
                self.assertEqual(
                    (predecessor.run_id, predecessor.run_number), (37347011941, 513)
                )

    def test_candidate_diagnostics_never_raise_on_irrelevant_metadata(self):
        legacy = {"id": 50, "run_number": 50, "display_title": None, "event": None}
        predecessor, stderr = self._resolve(
            [_workflow_run(99, 99), legacy], current_run_id=100, current_run_number=100
        )
        self.assertEqual(predecessor.run_id, 99)
        self.assertIn("run_number=50 run_id=50 event=none", stderr)
        self.assertIn("lineage=unknown", stderr)

    # --- 8 / 9: safe-skip result and its diagnostics ---------------------------

    def test_safe_skip_returns_the_same_predecessor_and_logs_the_skip(self):
        jobs = _post_job_payload(run_conclusion="failure")
        predecessor, stderr = self._resolve(
            [_workflow_run(99, 99, conclusion="failure"), _workflow_run(98, 98)],
            jobs_loader=lambda run_id: jobs if run_id == 99 else {"jobs": []},
        )
        self.assertEqual((predecessor.run_id, predecessor.run_number), (98, 98))
        self.assertEqual(
            self._lines(stderr, "SKIP"),
            ["run_id=99 run_number=99 conclusion=failure reason=publisher_not_executed"],
        )
        self.assertEqual(
            self._lines(stderr, "SELECTED"), ["run_id=98 run_number=98 conclusion=success"]
        )

    def test_proven_incident_skip_logs_its_distinct_reason(self):
        incident = continuity.PROVEN_PRE_MUTATION_INCIDENT
        runs = [
            _workflow_run(
                incident.run_id,
                incident.run_number,
                conclusion="failure",
                event=incident.event,
                head_sha=incident.head_sha,
            ),
            _workflow_run(34000000000, 469),
        ]
        jobs = _post_job_payload(run_conclusion="failure", publisher_conclusion="failure")
        predecessor, stderr = self._resolve(
            runs,
            current_run_id=34400000000,
            current_run_number=471,
            jobs_loader=lambda run_id: jobs if run_id == incident.run_id else {"jobs": []},
        )
        self.assertEqual(predecessor.run_number, 469)
        self.assertEqual(
            self._lines(stderr, "SKIP"),
            [
                f"run_id={incident.run_id} run_number={incident.run_number} "
                "conclusion=failure reason=proven_pre_mutation_incident"
            ],
        )

    # --- 11 - 15: fail-closed behaviour is unchanged ---------------------------

    def test_fail_closed_paths_are_unchanged(self):
        cases = (
            ("ambiguous_run_metadata:channel",
             [_workflow_run(99, 99, title="Logoped Bot without channel")], 1),
            ("ambiguous_run_metadata:run_number", [{"id": 99, "run_number": "x"}], 1),
            ("production_rerun_not_safe", [_workflow_run(99, 99)], 2),
            ("production_predecessor_rerun_not_safe", [_workflow_run(99, 99, run_attempt=2)], 1),
            ("ambiguous_production_predecessor_order",
             [_workflow_run(99, 99), _workflow_run(97, 99)], 1),
        )
        for expected, runs, attempt in cases:
            with self.subTest(expected=expected):
                with patch.object(continuity.urllib.request, "urlopen") as urlopen:
                    with self.assertRaisesRegex(continuity.StateContinuityError, expected):
                        self._resolve(runs, current_attempt=attempt)
                urlopen.assert_not_called()

    def test_expected_cache_key_is_byte_identical(self):
        self.assertEqual(
            continuity.build_expected_cache_key(
                cache_version="v12", ref_name="main", predecessor_run_id=37347011941
            ),
            "logoped-state-v12-prod-main-37347011941",
        )

    def test_diagnostic_values_are_single_line_and_bounded(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            continuity._emit_diagnostic("CURRENT", ref="main\n::set-output name=x::y " + "z" * 200)
        line = stderr.getvalue()
        self.assertEqual(line.count("\n"), 1)
        self.assertTrue(line.startswith("[STATE_CONTINUITY][CURRENT] ref=main_::set-output"))
        self.assertLessEqual(len(line.split("ref=", 1)[1].strip()), 64)


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

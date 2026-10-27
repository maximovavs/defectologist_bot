"""SAFE03 production-seam compatibility checks; deliberately no live calls.

Blocking observations are EXPECTED and tested: they are not 'ready to publish'.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from src.services import safe03_r6n_warning_acceptance_offline as n
from src.services import safe03_r6g_optin_offline as g
from src.services import safe03_r6h_trust_offline as t
from src.services.safe03_r6h_delivery_offline import OfflineLedger
from src.services import safe03_r6d_offline_acceptance as frozen

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / 'src/publisher/run_publisher.py'
VISUAL = ROOT / 'src/services/visual_pipeline.py'
STORE = ROOT / 'src/services/publication_store.py'
CANONICAL = ROOT / '.github/workflows/publisher_policy_pr_checks.yml'
WORKFLOW = ROOT / '.github/workflows/safe03_offline_pr_checks.yml'


def imported_safe03(path):
    tree = ast.parse(path.read_text('utf-8'))
    return [node for node in ast.walk(tree)
            if ((isinstance(node, ast.ImportFrom) and 'safe03' in (node.module or ''))
                or (isinstance(node, ast.Import) and any('safe03' in alias.name for alias in node.names)))]


class SeamReadiness(unittest.TestCase):
    def test_public_route_blocks_new_and_preserves_old_identity(self):
        old = {'message': 'ordinary legacy publication'}
        result = g.route(old)
        self.assertEqual(result.route, 'LEGACY_UNTOUCHED')
        self.assertIs(result.payload, old)
        self.assertEqual(g.route({'safe03_mode':'catalog','proposal':{}}).route, 'NO_POST')
        self.assertEqual(n.public_candidate().accepted_for, 'NO_POST')
        self.assertEqual(t.public_candidate().status, 'NO_POST')

    def test_current_production_lacks_safe03_wiring(self):
        for p in (PUBLISHER, VISUAL, STORE):
            self.assertEqual(imported_safe03(p), [], p)
        text = (ROOT / '.github/workflows/post.yml').read_text('utf-8')
        self.assertNotIn('SAFE03_ENABLED', text)
        self.assertNotIn('safe03_r6', text)

    def test_no_autoregistration_of_oral_attestations_or_source_roots(self):
        self.assertEqual(dict(n.PRODUCTION_APPROVALS), {})
        self.assertEqual(dict(n.PRODUCTION_SOURCE_RECEIPTS), {})
        self.assertEqual(dict(t.PRODUCTION_REVIEWER_KEYS), {})
        self.assertEqual(dict(t.PRODUCTION_SOURCE_RECORDS), {})
        catalog = n.validated_offline_entries()
        self.assertEqual(len(catalog), 5)
        for e in catalog.values():
            self.assertEqual(g._utf16_units(e.text), len(e.text.encode('utf-16-le')) // 2)
            self.assertLessEqual(g._utf16_units(e.text), 1000)
        for k in n.NEW_WARNING_DIGESTS:
            e = catalog[k]
            self.assertEqual(frozen.digest(e), n.NEW_WARNING_DIGESTS[k][1])

    def test_exact_sender_requires_new_seam_not_generic_telegram_split(self):
        p = ast.parse(PUBLISHER.read_text('utf-8'))
        fn = next(x for x in ast.walk(p) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and x.name == 'send_post_with_visual')
        source = ast.get_source_segment(PUBLISHER.read_text('utf-8'), fn)
        self.assertIn('sendPhoto', source)
        self.assertIn('send_message(chat_id, html_full_post)', source)
        self.assertIn('parse_mode', source)
        self.assertIn('TELEGRAM_PARSE_MODE', source)
        self.assertIn('caption_too_long', source)
        self.assertIn('plain_fallback', source)
        # A future SAFE03 sender must reject if the single exact caption fails,
        # not switch to a multi-message send or HTML fallback.

    def test_exact_bodyhash_parity_and_global_dedup_blocker(self):
        from src.publisher.run_publisher import norm_space, sha1, _telegram_utf16_units
        text = PUBLISHER.read_text('utf-8')
        self.assertIn('if store.has_body_hash(body_hash):', text)
        self.assertIn('dup_body_hash_db', text)
        for entry in n.validated_offline_entries().values():
            value = hashlib.sha1(re.sub(r'\s+', ' ', entry.text.strip()).encode()).hexdigest()
            self.assertEqual(value, sha1(norm_space(entry.text)))
            self.assertEqual(_telegram_utf16_units(entry.text), g._utf16_units(entry.text))

    def test_visual_fallback_requires_separate_static_text_route(self):
        text = VISUAL.read_text('utf-8')
        self.assertIn('def _question_week_sound_object_scene(', text)
        self.assertIn('question_week_sound_object_ungrounded', text)
        self.assertNotIn('safe03_r6n_warning_acceptance', text)
        from src.services import visual_pipeline
        # No LLM call or generated image needed to prove fail-closed routing.
        self.assertEqual(visual_pipeline._question_week_sound_object_scene(
            'Ребёнок слышит звуки вокруг', 'question_week', ''), ('', True))
        self.assertEqual(visual_pipeline._question_week_sound_object_scene(
            'Ребёнок не слушается', 'question_week', ''), ('', True))

    def test_source_grants_synthetic_only_cannot_open_public_route(self):
        catalog = n.validated_offline_entries()
        for cid in ('warning_button_batteries', 'warning_small_magnets'):
            with self.subTest(cid=cid):
                e = catalog[cid]
                with patch.object(frozen, 'CATALOG', catalog):
                    bogus = {'safe03_mode':'catalog', 'proposal':frozen.proposal(e, 30)}
                    self.assertEqual(g.route(bogus).route, 'NO_POST')
                    self.assertEqual(t.public_candidate(catalog_id=cid).status, 'NO_POST')

    def test_offline_ledger_preserves_unknown_quarantine_and_idempotency(self):
        from src.services.safe03_r6g_optin_offline import StoreSnapshot, EvidenceGrant
        catalog = n.validated_offline_entries()
        e = catalog['warning_small_magnets']
        with patch.object(frozen, 'CATALOG', catalog):
            approvals = {e.catalog_id:frozen.digest(e)}
            grants = {e.catalog_id:EvidenceGrant(e.catalog_id, frozen.digest(e),
                       'offline_fixture:magnet', 'a'*64, 'reviewed_guidance')}
            snapshot = StoreSnapshot(frozenset(), frozenset(), trusted_snapshot=True)
            plan = g._offline_fixture_preview({'safe03_mode':'catalog','proposal':frozen.proposal(e, 24)},
                     approvals=approvals, grants=grants, history=snapshot)
            self.assertEqual(plan.status, 'OFFLINE_SIMULATED_ALLOW')
            with tempfile.TemporaryDirectory() as td:
                ledger = OfflineLedger(Path(td)/'test-only.sqlite3')
                kwargs = {'plan':plan, 'send_window':'2026Q4','external_exact_hash_seen':False,
                          'external_history_known':True,'external_unresolved_delivery':False,
                          'now_utc':datetime(2026,10,26,18,tzinfo=timezone.utc)}
                reserved = ledger.reserve(**kwargs)
                self.assertTrue(reserved.allowed, reserved.reason)
                self.assertFalse(ledger.reserve(**kwargs).allowed)
                self.assertTrue(ledger.mark_inflight(reserved.key))
                self.assertTrue(ledger.mark_unknown(reserved.key))
                self.assertFalse(ledger.reserve(**{**kwargs, 'send_window':'2027Q1'}).allowed)
                self.assertFalse(ledger.mark_confirmed(reserved.key, message_id=19))

    def test_ci_contract_no_dispatch_no_secrets_and_correct_path_filters(self):
        src = WORKFLOW.read_text('utf-8')
        self.assertIn('pull_request:', src)
        self.assertIn('contents: read', src)
        self.assertIn('python-version: "3.11"', src)
        self.assertNotIn('workflow_dispatch:', src)
        self.assertNotIn('pull_request_target:', src)
        self.assertNotIn('secrets.', src)
        for name in ('tests/test_safe03_*.py','src/services/safe03_*.py',
                     'src/services/publication_store.py','src/services/visual_pipeline.py',
                     'src/publisher/run_publisher.py', 'SAFE03_*.json'):
            self.assertIn(name, src)
        self.assertIn('safe03_r6p_ci_entrypoint.py', src)

    def test_existing_canonical_workflow_does_not_claim_safe03_ci(self):
        text = CANONICAL.read_text('utf-8')
        self.assertNotIn('tests.test_safe03_', text)
        self.assertNotIn('tests/test_safe03_', text)


if __name__ == '__main__':
    unittest.main()

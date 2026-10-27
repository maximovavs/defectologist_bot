"""R6-V: exercise the REAL PublicationStore installed hook with fake transport.

No real Telegram/provider calls and no mutation of production databases. The
fixture gate is statically OFF unless patched in an individual offline test.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest import TestCase
from unittest.mock import patch
import hashlib
import os
import sqlite3
import sys
import tempfile

from src.services import publication_store as store_mod
from src.services.publication_store import PublicationStore
from src.services.safe03_r6s_joint_reservation_offline import normal_fixture
from src.services.safe03_r6u_unified_lifecycle_offline import OfflineLifecycle
from src.services import safe03_r6v_delivery_handoff_offline as _r6v_loaded

CHAT = '123456789'
CAPTION = 'Ребёнок играет с мячом'
NOW = datetime(2026,10,27,tzinfo=timezone.utc)

class MockTelegramAmbiguous(RuntimeError):
    pass


class R6VHookHandoffTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'test_history.sqlite3'
        with patch.dict(os.environ, {'DRY_RUN':'1'}):
            self.store = PublicationStore(self.path)
        self.lifecycle = OfflineLifecycle(self.path)
        self.proposal = normal_fixture(caption=CAPTION,canonical_url='https://example.org/r6v',evidence_hash='b'*64)
        self.assertIsNotNone(self.proposal)
        self.calls=[]
        self.transport_mode='receipt'
        self.fake = ModuleType('src.publisher.run_publisher')
        self.fake.TelegramDeliveryOutcomeAmbiguous = MockTelegramAmbiguous

        def mocked_tg(method, data, files=None):
            self.calls.append(method)
            if self.transport_mode == 'ambiguous':
                raise MockTelegramAmbiguous('socket_lost_after_remote_send')
            if self.transport_mode == 'reject':
                raise ValueError('deterministic_rejection')
            if self.transport_mode == 'missing_receipt':
                return {'ok': True,'result': {}}
            if self.transport_mode == 'two_receipts':
                self.fake.tg_request('sendMessage', {'chat_id':CHAT}) if method == 'sendPhoto' else None
            return {'ok':True,'result': {'message_id':1001 if method=='sendPhoto' else 1002}}

        def mocked_sender(chat_id, photo_buffer, plain_post, html_full_post):
            self.fake.tg_request('sendPhoto', {'chat_id':chat_id, 'caption':plain_post})
            return 1

        self.fake.tg_request = mocked_tg
        self.fake.send_post_with_visual = mocked_sender
        self.fake_orig=(mocked_tg,mocked_sender)
        self.override = patch.dict(sys.modules, {'src.publisher.run_publisher':self.fake})
        self.override.start()
        self.addCleanup(self.override.stop)
        self.old_global = {name: getattr(store_mod,name) for name in (
            '_DELIVERY_HOOKS_INSTALLED','_ACTIVE_DELIVERY_STORE','_ACTIVE_DELIVERY_ATTEMPT_KEY',
            '_DELIVERY_ORIGINAL_TG_REQUEST','_DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL')}
        store_mod._DELIVERY_HOOKS_INSTALLED=False
        store_mod._ACTIVE_DELIVERY_STORE=None
        store_mod._ACTIVE_DELIVERY_ATTEMPT_KEY=''
        store_mod._DELIVERY_ORIGINAL_TG_REQUEST=None
        store_mod._DELIVERY_ORIGINAL_SEND_POST_WITH_VISUAL=None
        self.addCleanup(self.restore_globals)

    def restore_globals(self):
        for name,value in self.old_global.items():setattr(store_mod,name,value)

    def reserve(self):
        x=self.lifecycle.reserve(self.proposal,display_plain=CAPTION,chat_id=CHAT,now=NOW)
        self.assertEqual(x.status,'OFFLINE_RESERVED_ONLY',x)
        self.store._r6v_test_lifecycle=self.lifecycle
        self.store._r6v_test_attempt_key=x.key
        return x.key

    @contextmanager
    def installed_hook(self, offline_gate=True):
        with patch.object(store_mod, '_R6V_TEST_ONLY_HANDOFF_ENABLED', offline_gate):
            self.store._install_publisher_delivery_hooks()
            try:
                yield
            finally:
                self.store.deactivate_publisher_delivery_hooks()
                self.fake.tg_request,self.fake.send_post_with_visual=self.fake_orig
                store_mod._DELIVERY_HOOKS_INSTALLED=False

    def send(self, chat=CHAT, caption=CAPTION):
        return self.fake.send_post_with_visual(chat,None,caption,caption)

    def test_offline_gate_is_hard_false_even_with_env_var(self):
        self.assertIs(store_mod._R6V_TEST_ONLY_HANDOFF_ENABLED,False)
        with patch.dict(os.environ, {'SAFE03_ENABLED':'1','SAFE03_R6V_ENABLED':'1'}):
            self.assertFalse(store_mod._R6V_TEST_ONLY_HANDOFF_ENABLED)

    def test_legacy_hook_unchanged_without_test_gate(self):
        with self.installed_hook(False):
            self.assertEqual(self.send(),1)
            rows=self.store.delivery_attempts()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['state'],'confirmed')
            self.assertEqual(rows[0]['primary_message_ids'],[1001])
            self.assertEqual(self.calls,['sendPhoto'])
            self.assertEqual(self.store._confirmed_delivery_attempt_key,rows[0]['attempt_key'])

    def test_handoff_consumes_one_existing_attempt_and_atomic_mock_ack(self):
        key=self.reserve()
        with self.installed_hook():
            with patch.object(self.store,'begin_delivery_attempt',side_effect=AssertionError('DOUBLE_BEGIN')):
                self.assertEqual(self.send(),1)
        self.assertEqual(self.calls,['sendPhoto'])
        self.assertEqual(self.store.delivery_attempts(),[])
        self.assertTrue(self.store.has_body_hash(self.proposal.body_hash))
        with sqlite3.connect(self.path) as c:
            row=c.execute('SELECT state,receipt_message_id FROM safe03_r6u_fixture_claims').fetchone()
            self.assertEqual(row,('mock_ack',1001))
            self.assertEqual(c.execute('SELECT count(*) FROM publications').fetchone()[0],1)
        self.assertEqual(self.lifecycle.reserve(self.proposal,display_plain=CAPTION,chat_id=CHAT,now=NOW).status,'NO_POST')

    def test_handoff_rejects_modified_caption_before_fake_transport(self):
        self.reserve()
        with self.installed_hook():
            with self.assertRaisesRegex(MockTelegramAmbiguous,'delivery_state_begin_failed'):
                self.send(caption=CAPTION+' #tag')
        self.assertEqual(self.calls,[])
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'pending')

    def test_handoff_rejects_wrong_chat_before_fake_transport(self):
        self.reserve()
        with self.installed_hook():
            with self.assertRaises(MockTelegramAmbiguous):self.send(chat='9')
        self.assertEqual(self.calls,[])
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'pending')

    def test_handoff_missing_key_refuses_send(self):
        self.reserve();self.store._r6v_test_attempt_key=''
        with self.installed_hook():
            with self.assertRaises(ValueError):self.send()
        self.assertEqual(self.calls,[])

    def test_unknown_response_stays_quarantined_without_retry(self):
        self.reserve();self.transport_mode='ambiguous'
        with self.installed_hook():
            with self.assertRaisesRegex(MockTelegramAmbiguous,'socket_lost'):
                self.send()
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'ambiguous')
        self.assertEqual(self.lifecycle.reserve(self.proposal,display_plain=CAPTION,chat_id=CHAT,now=NOW).status,'NO_POST')
        self.assertEqual(self.calls,['sendPhoto'])

    def test_deterministic_rejection_is_conservatively_quarantined(self):
        self.reserve();self.transport_mode='reject'
        with self.installed_hook():
            with self.assertRaisesRegex(MockTelegramAmbiguous,'fixture_unclassified_send_error'):
                self.send()
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'ambiguous')

    def test_missing_receipt_blocks_atomic_completion(self):
        self.reserve();self.transport_mode='missing_receipt'
        with self.installed_hook():
            with self.assertRaisesRegex(MockTelegramAmbiguous,'delivery_state_confirm_failed'):
                self.send()
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'in_flight')
        self.assertFalse(self.store.has_body_hash(self.proposal.body_hash))

    def test_multiple_receipts_blocks_atomic_completion(self):
        self.reserve();self.transport_mode='two_receipts'
        with self.installed_hook():
            with self.assertRaisesRegex(MockTelegramAmbiguous,'delivery_state_confirm_failed'):
                self.send()
        self.assertEqual(self.store.delivery_attempts()[0]['primary_message_ids'],[1002,1001])
        self.assertFalse(self.store.has_body_hash(self.proposal.body_hash))

    def test_database_error_cannot_ack_or_clear_quarantine(self):
        self.reserve()
        with sqlite3.connect(self.path) as conn:
            conn.execute('CREATE TRIGGER stop_pub BEFORE INSERT ON publications BEGIN SELECT RAISE(ABORT,"force_rollback"); END')
        with self.installed_hook():
            with self.assertRaisesRegex(MockTelegramAmbiguous,'delivery_state_confirm_failed'):
                self.send()
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'in_flight')
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute('SELECT count(*) FROM publications').fetchone()[0],0)

    def test_double_send_with_reused_key_does_not_dispatch(self):
        self.reserve()
        with self.installed_hook():
            self.assertEqual(self.send(),1)
            with self.assertRaises(ValueError):self.send()
        self.assertEqual(self.calls,['sendPhoto'])

    def test_pending_after_crash_disallows_new_reservation(self):
        self.reserve()
        other=OfflineLifecycle(self.path)
        self.assertEqual(other.reserve(self.proposal,display_plain=CAPTION,chat_id=CHAT,now=NOW).status,'NO_POST')
        self.assertEqual(self.store.delivery_attempts()[0]['state'],'pending')

    def test_500_forged_handoffs_cannot_reach_transport(self):
        key=self.reserve()
        with self.installed_hook():
            for index in range(250):
                self.store._r6v_test_attempt_key = f'{index:064x}'
                if self.store._r6v_test_attempt_key == key:
                    continue
                with self.assertRaises(MockTelegramAmbiguous):
                    self.send()
            self.store._r6v_test_attempt_key = key
            for index in range(250):
                with self.assertRaises(MockTelegramAmbiguous):
                    self.send(caption=CAPTION + ' ' + str(index))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.store.delivery_attempts()[0]['state'], 'pending')

    def test_untrusted_lifecycle_rejected_before_send(self):
        self.reserve()
        self.store._r6v_test_lifecycle = object()
        with self.installed_hook():
            with self.assertRaises(ValueError):
                self.send()
        self.assertEqual(self.calls, [])

    def test_public_paths_still_no_post(self):
        from src.services.safe03_r6u_unified_lifecycle_offline import public_safe03, public_normal
        self.assertEqual(public_safe03(),'NO_POST')
        self.assertEqual(public_normal(),'LEGACY_UNCHANGED')

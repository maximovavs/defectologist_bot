"""R6-V: test-only handoff exercising PublicationStore's installed hook.

Hard OFF except an explicit monkeypatch in a disposable test. No transport.
Never a production authorization or a source trust root.
"""
from pathlib import Path
import hashlib
from src.services.safe03_r6u_unified_lifecycle_offline import OfflineLifecycle


def preflight(store, chat_id: str, exact_caption: str):
    lifecycle = getattr(store, '_r6v_test_lifecycle', None)
    key = getattr(store, '_r6v_test_attempt_key', None)
    if (not isinstance(store.db_path, Path) or store.db_path.name != 'test_history.sqlite3'
            or store.db_path.is_symlink() or
            any(parent.is_symlink() for parent in store.db_path.parents)
            or type(lifecycle) is not OfflineLifecycle or lifecycle.path != store.db_path
            or type(key) is not str or len(key) != 64
            or not all(ch in '0123456789abcdef' for ch in key)
            or type(chat_id) is not str or type(exact_caption) is not str):
        raise ValueError('r6v_offline_fixture_only')
    # Claim checks chat, exact UTF-8 caption, in-flight transition and HMAC.
    return key, lifecycle


def claim(lifecycle: OfflineLifecycle, key: str, chat_id: str, exact_caption: str):
    result = lifecycle.mock_hook_begin(key, chat_id=chat_id, display_plain=exact_caption)
    if result.status != 'OFFLINE_TRANSITION_ONLY' or result.reason != 'in_flight':
        raise ValueError('r6v_claim_not_accepted:' + result.reason)


def quarantine(lifecycle: OfflineLifecycle, key: str):
    result = lifecycle.mock_unknown(key)
    if result.status != 'OFFLINE_TRANSITION_ONLY':
        raise ValueError('r6v_quarantine_failure:' + result.reason)


def reconcile_mock_ack(store, lifecycle: OfflineLifecycle, key: str):
    """Use the *fake* receipt persisted by actual tg_request receipt wrapper.

    No network call or claim of real Telegram delivery. Publication INSERT and
    deletion of the one attempt are in one transaction in mock_ack.
    """
    attempts = store.delivery_attempts()
    if len(attempts) != 1 or attempts[0]['attempt_key'] != key:
        raise ValueError('r6v_missing_single_attempt')
    ids = attempts[0]['primary_message_ids']
    if len(ids) != 1 or type(ids[0]) is not int or ids[0] <= 0:
        raise ValueError('r6v_mock_receipt_missing_or_multiple')
    result = lifecycle.mock_ack(key, ids[0])
    if result.status != 'OFFLINE_TRANSITION_ONLY' or result.reason != 'mock_ack':
        raise ValueError('r6v_atomic_ack_not_accepted:' + result.reason)

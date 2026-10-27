"""SAFE03 R6-G OFFLINE-ONLY opt-in routing contract. Not wired into publisher.

This module is deliberately disconnected. Public route() never grants a catalog
publication. A separate private preview fixture models *future* integration,
not a production authorisation. Never persist or publish its preview plans.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import re
import secrets
from typing import Mapping

from src.services import safe03_r6d_offline_acceptance as catalog

_CAPTION_MAX_UTF16 = 1000  # matches current publisher default; explicit lower bound
_TOKEN = secrets.token_bytes(32)  # local process only, not a portable signature
_CARD_TITLES = {
    'warning_button_batteries': 'КНОПОЧНЫЕ БАТАРЕЙКИ',
    'warning_small_magnets': 'МЕЛКИЕ МАГНИТЫ',
    'picture_naming_24_48': 'ИГРА С КАРТИНКОЙ',
    'soft_ball_roll_36_60': 'КАТАЕМ МЯЧ',
    'clap_rhythm_24_48': 'ПОВТОРЯЕМ РИТМ',
}


@dataclass(frozen=True)
class EvidenceGrant:
    """Synthetic offline stand-in; no production-grounded/source reviewed grants exist."""
    catalog_id: str
    entry_digest: str
    source_ref: str
    evidence_digest: str
    source_category: str  # reviewed_guidance | expert_record


@dataclass(frozen=True)
class StoreSnapshot:
    """Offline stand-in; production must obtain fresh state at delivery time."""
    posted_keys: frozenset[str]
    posted_body_hashes: frozenset[str]
    unresolved_delivery: bool = False
    trusted_snapshot: bool = False  # real production path has no trusted snapshot here


@dataclass(frozen=True)
class Plan:
    status: str
    reason: str
    lane: str = 'none'
    catalog_id: str = ''
    entry_digest: str = ''
    full_text: str = ''
    card_title: str = ''
    card_type: str = ''
    delivery_mode: str = ''
    caption_parse_mode: str = ''
    idempotency_key: str = ''
    content_hash: str = ''  # exact current publisher body_hash: SHA1 normalized text
    source_ref: str = ''
    evidence_digest: str = ''
    age_months: int = -1
    signature: str = ''


@dataclass(frozen=True)
class Routed:
    route: str
    payload: object
    plan: Plan | None = None


def route(request: object) -> Routed:
    """Safe public behavior. Legacy is exact identity, catalog stays NO_POST.

    Caller must select a trusted routing mode at the application boundary;
    never infer the route from an LLM response. This standalone function is
    NOT imported by production. Unknown routes fail closed.
    """
    if type(request) is not dict:
        return Routed('NO_POST', request, Plan('NO_POST', 'invalid_route_request'))
    if 'safe03_mode' not in request:
        return Routed('LEGACY_UNTOUCHED', request)  # same object, no mutation
    if request.get('safe03_mode') == 'catalog':
        return Routed('NO_POST', request, Plan('NO_POST', 'trusted_approvals_unavailable'))
    return Routed('NO_POST', request, Plan('NO_POST', 'unsupported_route'))


def _utf16_units(value: str) -> int:
    return len(value.encode('utf-16-le')) // 2


def _key(entry: catalog.Entry) -> str:
    """Stable one-shot identity; no silent dedup bypass or automatic republishing."""
    return f'safe03:{entry.catalog_id}:v{entry.version}:{catalog.digest(entry)}'


def _signature(plan: Plan) -> str:
    data = dict(plan.__dict__)
    data.pop('signature', None)
    data_raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hmac.new(_TOKEN, data_raw, hashlib.sha256).hexdigest()


def _deny(reason: str) -> Plan:
    return Plan('NO_POST', reason)


def _offline_fixture_preview(
    incoming: object, *, approvals: Mapping[str, str], grants: Mapping[str, EvidenceGrant],
    history: StoreSnapshot, caption_limit: int = _CAPTION_MAX_UTF16,
) -> Plan:
    """Only for OFFLINE simulation with synthetic approvals and source proofs.

    No public code path passes these arguments. This does NOT authorise sending.
    """
    if type(incoming) is not dict or frozenset(incoming) != {'safe03_mode', 'proposal'}:
        return _deny('invalid_catalog_envelope')
    if incoming['safe03_mode'] != 'catalog':
        return _deny('wrong_route')
    if type(caption_limit) is not int or caption_limit <= 0 or caption_limit > 1000:
        return _deny('invalid_caption_budget')
    if type(history) is not StoreSnapshot or not history.trusted_snapshot:
        return _deny('no_trusted_delivery_snapshot')
    if history.unresolved_delivery:
        return _deny('unresolved_delivery')
    raw = incoming['proposal']
    rendered = catalog._test_fixture_only_render(raw, approvals)
    if rendered.decision != 'ALLOW':
        return _deny('catalog_denied:' + rendered.reason)
    entry = catalog.CATALOG[rendered.catalog_id]
    # Ensure no prepended title, source, hashtags, footer, or repaired content.
    verdict = catalog._test_fixture_only_validate(rendered, entry.text, rendered.lane, approvals)
    if verdict.decision != 'ALLOW':
        return _deny('render_final_invalid')
    grant = grants.get(entry.catalog_id)
    if type(grant) is not EvidenceGrant:
        return _deny('missing_source_provenance')
    if (grant.catalog_id != entry.catalog_id or grant.entry_digest != catalog.digest(entry)
        or grant.source_category not in ('reviewed_guidance','expert_record')
        or not grant.source_ref or not grant.evidence_digest or len(grant.evidence_digest) != 64
        or any(c not in '0123456789abcdef' for c in grant.evidence_digest)):
        return _deny('invalid_source_binding')
    # The image is a precomposed text-card ONLY, never an object/LLM scene.
    # No split messages: multi-request atomicity is not established.
    if _utf16_units(entry.text) > caption_limit:
        return _deny('exact_caption_too_long')
    key = _key(entry)
    # Match run_publisher.py: sha1(norm_space(plain)), NOT sha256(raw text).
    norm_body = re.sub(r'\s+', ' ', entry.text.strip())
    body_hash = hashlib.sha1(norm_body.encode('utf-8')).hexdigest()
    if (key in history.posted_keys or body_hash in history.posted_body_hashes):
        return _deny('duplicate_without_policy_override')
    lane = 'safety_warning' if entry.kind == 'warning' else 'exercise'
    plan = Plan(
        'OFFLINE_SIMULATED_ALLOW', 'synthetic_evidence_and_approval_only',
        lane=lane, catalog_id=entry.catalog_id, entry_digest=catalog.digest(entry),
        full_text=entry.text, card_title=_CARD_TITLES[entry.catalog_id],
        card_type='static_text_card', delivery_mode='single_photo_exact_caption',
        caption_parse_mode='none',
        idempotency_key=key, content_hash=body_hash, source_ref=grant.source_ref,
        evidence_digest=grant.evidence_digest, age_months=rendered.age_months,
    )
    return replace(plan, signature=_signature(plan))


def _offline_fixture_final_check(
    plan: object, *, final_caption: object, visual_type: object, delivery_mode: object,
    approvals: Mapping[str, str], grants: Mapping[str, EvidenceGrant],
    history: StoreSnapshot, caption_limit: int = _CAPTION_MAX_UTF16,
) -> bool:
    """All checks must be re-run immediately before any future send path.

    HMAC covers the local offline plan, not a durable cross-process token.
    """
    if type(plan) is not Plan or plan.status != 'OFFLINE_SIMULATED_ALLOW':
        return False
    if type(final_caption) is not str or type(visual_type) is not str or type(delivery_mode) is not str:
        return False
    if not hmac.compare_digest(plan.signature, _signature(plan)):
        return False
    if final_caption.encode('utf-8') != plan.full_text.encode('utf-8'):
        return False
    if (visual_type != 'static_text_card' or delivery_mode != 'single_photo_exact_caption'
        or plan.caption_parse_mode != 'none'):
        return False
    if type(plan.age_months) is not int or plan.catalog_id not in catalog.CATALOG:
        return False
    entry = catalog.CATALOG[plan.catalog_id]
    rebuilt = _offline_fixture_preview(
        {'safe03_mode': 'catalog', 'proposal': catalog.proposal(entry, plan.age_months)},
        approvals=approvals, grants=grants, history=history, caption_limit=caption_limit,
    )
    return rebuilt == plan  # includes all source, dedup, metadata, visual and content fields

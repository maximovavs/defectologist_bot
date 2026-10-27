"""SAFE03 R6-H offline-only provenance, immutable caption and source-link contract.

DO NOT wire to production or regard user-relayed attestations as cryptographic
signatures. The production trust roots remain deliberately EMPTY. Synthetic
fixture capabilities only exercise a possible downstream trust decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import re
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

from src.services import safe03_r6d_offline_acceptance as catalog
from src.services.safe03_r6g_optin_offline import (
    EvidenceGrant, Plan, StoreSnapshot, _offline_fixture_final_check,
    _offline_fixture_preview, _utf16_units,
)

# No trusted source/reviewer has been independently enrolled.
PRODUCTION_REVIEWER_KEYS: Mapping[str, str] = MappingProxyType({})
PRODUCTION_SOURCE_RECORDS: Mapping[str, str] = MappingProxyType({})
HEX = re.compile(r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class ReviewerReceipt:
    catalog_id: str
    whole_entry_digest: str
    reviewer_key_id: str
    reviewer_role: str
    receipt_sha256: str
    # provenance string is informational ONLY; never grants trust.
    reported_origin: str = 'USER_RELAYED'


@dataclass(frozen=True)
class SourceReceipt:
    catalog_id: str
    whole_entry_digest: str
    source_ref: str
    evidence_digest: str
    vetted_host: str
    source_category: str  # reviewed_guidance / expert_record
    reviewed_version: str
    source_review_sha256: str


@dataclass(frozen=True)
class VerifiedRoots:
    """Injected only in TESTS; a real root loader/signature verifier is absent.

    Pinning a checksum is not proof a specialist authored an attestation.
    An integration would need authenticated signing and source verification,
    not trust arbitrary user- or provider-provided dictionary entries.
    """
    signed_review_receipts: Mapping[str, str]  # key_id -> verified receipt sha256
    audited_source_receipts: Mapping[str, str]  # catalog_id -> source_review_sha256
    permitted_hosts: frozenset[str]


@dataclass(frozen=True)
class BoundPreview:
    status: str
    reason: str
    plan: Plan | None = None
    caption: str = ''
    source_keyboard: tuple[tuple[str, str], ...] = ()


def _sha(v: object) -> bool:
    return type(v) is str and bool(HEX.fullmatch(v))


def _source_https(url: object, permitted_hosts: frozenset[str]) -> bool:
    if type(url) is not str or len(url) > 2048 or any(ord(c) < 32 or ord(c) == 127 for c in url):
        return False
    try:
        p = urlsplit(url)
        host = p.hostname
        if p.scheme != 'https' or not host or p.username or p.password or p.fragment or p.port not in (None, 443):
            return False
        if host != host.lower() or host not in permitted_hosts or not p.path.startswith('/'):
            return False
        ipaddress.ip_address(host)
        return False  # no IP-literal source endpoint
    except ValueError:
        # a hostname is expected to raise ValueError for IP parsing
        return bool(host and re.fullmatch(r'[a-z0-9.-]+', host)
                    and '..' not in host and not host.startswith('.') and not host.endswith('.'))


def reviewer_role_for(entry: catalog.Entry) -> frozenset[str]:
    if entry.catalog_id == 'clap_rhythm_24_48':
        return frozenset({'audiologist'})
    if entry.kind == 'warning':
        return frozenset({'pediatrician_child_safety'})
    return frozenset({'speech_defectologist'})


def _offline_provenance_gate(
    entry: catalog.Entry, reviewer: ReviewerReceipt, source: SourceReceipt,
    roots: VerifiedRoots,
) -> str:
    """No signatures are verified here: roots MUST originate upstream, never from a post.

    This function is deliberately private and fixture-only. It checks the
    complete binding that a future trust verifier must supply.
    """
    if type(entry) is not catalog.Entry or type(reviewer) is not ReviewerReceipt or type(source) is not SourceReceipt:
        return 'invalid_record_types'
    if type(roots) is not VerifiedRoots:
        return 'no_verified_roots'
    dig = catalog.digest(entry)
    if (reviewer.catalog_id != entry.catalog_id or reviewer.whole_entry_digest != dig
        or reviewer.reviewer_role not in reviewer_role_for(entry)
        or not _sha(reviewer.receipt_sha256)
        or roots.signed_review_receipts.get(reviewer.reviewer_key_id) != reviewer.receipt_sha256):
        return 'reviewer_not_independently_authenticated'
    if (source.catalog_id != entry.catalog_id or source.whole_entry_digest != dig
        or not _sha(source.evidence_digest) or not _sha(source.source_review_sha256)
        or not source.reviewed_version or source.source_category not in ('expert_record', 'reviewed_guidance')
        or roots.audited_source_receipts.get(entry.catalog_id) != source.source_review_sha256
        or source.vetted_host not in roots.permitted_hosts
        or not _source_https(source.source_ref, roots.permitted_hosts)
        or urlsplit(source.source_ref).hostname != source.vetted_host):
        return 'source_not_independently_grounded'
    return 'OFFLINE_VERIFIED_FIXTURE_ONLY'


def preview_from_synthetic_verified_roots(
    *, catalog_id: str, age_months: int, reviewer: ReviewerReceipt, source: SourceReceipt,
    roots: VerifiedRoots, history: StoreSnapshot,
) -> BoundPreview:
    """ONLY for offline proof of contract; never a production send authorization."""
    if type(catalog_id) is not str or catalog_id not in catalog.CATALOG or type(age_months) is not int:
        return BoundPreview('NO_POST', 'invalid_catalog_or_age')
    entry = catalog.CATALOG[catalog_id]
    reason = _offline_provenance_gate(entry, reviewer, source, roots)
    if reason != 'OFFLINE_VERIFIED_FIXTURE_ONLY':
        return BoundPreview('NO_POST', reason)
    approvals = {catalog_id: catalog.digest(entry)}  # MOCK only after both mock receipts validated
    grants = {catalog_id: EvidenceGrant(catalog_id, catalog.digest(entry), source.source_ref,
                                        source.evidence_digest, source.source_category)}
    plan = _offline_fixture_preview(
        {'safe03_mode':'catalog','proposal':catalog.proposal(entry,age_months)},
        approvals=approvals, grants=grants, history=history,
    )
    if plan.status != 'OFFLINE_SIMULATED_ALLOW':
        return BoundPreview('NO_POST', plan.reason)
    # Attribution outside immutable caption: future Telegram inline URL button.
    # This does not prove the source is clinically sufficient or live/reachable.
    keyboard = (('Источник', source.source_ref),)
    if not _offline_fixture_final_check(
        plan, final_caption=entry.text, visual_type='static_text_card',
        delivery_mode='single_photo_exact_caption', approvals=approvals,
        grants=grants, history=history,
    ) or _utf16_units(entry.text) > 1000:
        return BoundPreview('NO_POST', 'final_integrity_invalid')
    return BoundPreview('OFFLINE_SIMULATED_ALLOW', 'mock_roots_only', plan,
                        entry.text, keyboard)


def public_candidate(*_args: object, **_kwargs: object) -> BoundPreview:
    """A separate real registrar/signature check and authorization are required."""
    return BoundPreview('NO_POST', 'no_authenticated_reviewer_or_source_roots')

"""SAFE03 R6-N local-only candidate, never a production publisher route.

This is NOT an independently certified medical opinion, source receipt, or
publish permission. The user's report of a telephone approval is editorial
provenance only. Existing R6-H strict roots and R6-K owner model are unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from src.services import safe03_r6d_offline_acceptance as frozen

RECORD_FIELDS = ('catalog_id','version','kind','age_min_months','age_max_months',
                 'object_id','action','actor','recipient','constraints','text')
PRODUCTION_APPROVALS: Mapping[str, str] = MappingProxyType({})
PRODUCTION_SOURCE_RECEIPTS: Mapping[str, str] = MappingProxyType({})
FROZEN_CATALOG_SHA256 = '0557614596b05b17b3a44b94421e0970b665daaca5858b8db8a865da7b4d1a9d'
NEW_WARNING_DIGESTS = MappingProxyType({
    'warning_button_batteries': ('502488755f3279c9af0612b1710092d0c0d8192a5d0eb1953a48a87d97cb684c',
                                 'adaa80ebd6cd4707615e190fad1f75876eb2f0bcba740bf6f5fd1b2ae4e71f29'),
    'warning_small_magnets': ('81516094e7e0ff5184c522505a2a3a96460ee3bef23d6fa1b0d45abc660ea341',
                              '120a28f67e7f97faadb453d9d5d08c233e8fef5ae1de2d111b8ae5a72399b275'),
})

@dataclass(frozen=True)
class Candidate:
    accepted_for: str
    reason: str
    entry: frozen.Entry | None = None
    reported_approval: str = ''


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load() -> tuple[dict, dict]:
    root = Path(__file__).resolve().parents[2]
    catalog_file = root / 'SAFE03_R6N_CATALOG_EDITORIAL_ONLY.json'
    frozen_file = root / 'SAFE03_R6M_FROZEN_SOURCE_UNMODIFIED.json'
    raw_frozen = frozen_file.read_bytes()
    if _sha(raw_frozen) != FROZEN_CATALOG_SHA256:
        raise ValueError('frozen_source_drift')
    spec = json.loads(catalog_file.read_bytes())
    legacy = json.loads(raw_frozen)
    if (spec.get('mode') != 'OFFLINE_ONLY_NO_POST'
        or spec.get('production_access') != 'NO_POST'
        or spec.get('trusted_approvals') != {}
        or spec.get('trusted_sources') != {}
        or spec.get('frozen_catalog_sha256') != FROZEN_CATALOG_SHA256):
        raise ValueError('unsafe_candidate_status')
    return spec, legacy


def validated_offline_entries() -> Mapping[str, frozen.Entry]:
    spec, legacy = _load()
    # Recheck status at the decision boundary, even if the loader is replaced.
    if (type(spec) is not dict
        or spec.get('stage') != 'SAFE03_R6N'
        or spec.get('mode') != 'OFFLINE_ONLY_NO_POST'
        or spec.get('production_access') != 'NO_POST'
        or spec.get('legacy_route') != 'UNCHANGED'
        or spec.get('trusted_approvals') != {}
        or spec.get('trusted_sources') != {}
        or spec.get('frozen_catalog_sha256') != FROZEN_CATALOG_SHA256):
        raise ValueError('unsafe_catalog_status_or_trust_roots')
    baseline = {e['catalog_id']:e for e in legacy['entries']}
    incoming = {e['catalog_id']:e for e in spec['entries']}
    if len(incoming) != 5 or set(incoming) != set(baseline):
        raise ValueError('catalog_scope_changed')
    if len(spec['editorial_receipts']) != 2:
        raise ValueError('editorial_receipt_scope_changed')
    approved_by_user = {}
    for r in spec['editorial_receipts']:
        key = r['catalog_id']
        if key in approved_by_user or key not in NEW_WARNING_DIGESTS:
            raise ValueError('unexpected_editorial_receipt')
        if (r['version'] != incoming[key]['version']
            or r['evidence_grade'] != 'USER_ATTESTED_CLINICAL_OK'
            or r['report_date'] != '2026-10-26'
            or r['reported_role'] != 'врач-педиатр, специалист по детской безопасности'
            or r['independent_identity_verified'] is not False
            or r['independent_signature_verified'] is not False
            or (r['text_sha256'],r['entry_sha256']) != NEW_WARNING_DIGESTS[key]):
            raise ValueError('approval_origin_or_digest_mismatch')
        approved_by_user[key]=r
    if set(approved_by_user) != set(NEW_WARNING_DIGESTS):
        raise ValueError('missing_editorial_attestation')
    out = {}
    for key, proposed in incoming.items():
        if set(proposed) != set(RECORD_FIELDS):
            raise ValueError('unexpected_entry_fields')
        old = baseline[key]
        if key not in NEW_WARNING_DIGESTS:
            if any(old[k] != proposed[k] for k in RECORD_FIELDS):
                raise ValueError('old_exercise_drift')
        else:
            if proposed['version'] != old['version']+1:
                raise ValueError('wrong_warning_version')
            if any(old[k]!=proposed[k] for k in RECORD_FIELDS if k not in ('version','text')):
                raise ValueError('warning_metadata_changed')
            actual_sha = _sha(proposed['text'].encode('utf-8'))
            full = _sha(json.dumps(proposed,ensure_ascii=False,sort_keys=True,
                                   separators=(',',':')).encode('utf-8'))
            if (actual_sha,full)!=NEW_WARNING_DIGESTS[key]:
                raise ValueError('warning_text_or_entry_tampered')
        entry = frozen.Entry(**{**proposed,'constraints':tuple(proposed['constraints'])})
        if entry.kind=='warning' and (entry.age_max_months is not None
                                      or entry.age_min_months != 0
                                      or 'no_handling_exercise' not in entry.constraints):
            raise ValueError('warning_lane_invalid')
        if entry.kind=='exercise' and entry.catalog_id=='clap_rhythm_24_48' and 'Это упражнение не является проверкой слуха' not in entry.text:
            raise ValueError('hearing_guard_lost')
        out[key]=entry
    return MappingProxyType(out)


def public_candidate(*_args: object,**_kwargs: object) -> Candidate:
    """Only a separate authorized stage can provide a real approval root and sender."""
    return Candidate('NO_POST','no_independent_approval_or_source_root')


def offline_editorial_statement(catalog_id: str) -> Candidate:
    """Classification, never a release decision. No external calls."""
    try:
        entries = validated_offline_entries()
    except (OSError,ValueError,KeyError,TypeError,UnicodeError,json.JSONDecodeError):
        return Candidate('NO_POST','source_or_receipt_invalid')
    if type(catalog_id) is not str or catalog_id not in entries:
        return Candidate('NO_POST','unknown_entry')
    e=entries[catalog_id]
    return Candidate('OFFLINE_EDITORIAL_ONLY','requires_release_review',e,
                     'USER_ATTESTED_CLINICAL_OK' if e.kind=='warning' else 'USER_RELAYED_2026-10-26')

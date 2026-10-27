"""SAFE03 R6-Q: pure, offline contract for source review and single-photo delivery.

NO Telegram API, NO real trust root, NO production credential, NO send methods.
All fixture decisions are non-authoritative and NEVER publishable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from io import BytesIO
import secrets
from PIL import Image
import hmac
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Mapping

from src.services import safe03_r6d_offline_acceptance as frozen
from src.services import safe03_r6h_trust_offline as trust
from src.services import safe03_r6n_warning_acceptance_offline as current
from src.services.safe03_r6g_optin_offline import _utf16_units

# Clinical claims are *identifiers* for an evidence-mapping requirement;
# they are not proof that any source actually establishes the proposition.
REQUIRED_CLAIMS: Mapping[str, frozenset[str]] = MappingProxyType({
    'picture_naming_24_48': frozenset({'gesture_communication','accessible_picture_play'}),
    'soft_ball_roll_36_60': frozenset({'voluntary_turn_taking','motor_accommodation'}),
    'clap_rhythm_24_48': frozenset({'comfortable_sound_level','not_hearing_screen'}),
    'warning_button_batteries': frozenset({'urgent_evaluation','honey_eligibility','hearing_aid_battery_exception'}),
    'warning_small_magnets': frozenset({'urgent_evaluation','radiography_required'}),
})

# Source/reviewer registries must be independently enrolled in a separate stage.
PRODUCTION_TRUST_ROOTS: Mapping[str, str] = MappingProxyType({})
PRODUCTION_SEND_CAPABILITIES: Mapping[str, str] = MappingProxyType({})
_LOCAL_FIXTURE_KEY = secrets.token_bytes(32)  # nonportable offline tamper-evidence only


@dataclass(frozen=True)
class ClaimMap:
    catalog_id: str
    whole_entry_sha256: str
    evidence_sha256: str
    claim_ids: frozenset[str]
    clinical_review_receipt_sha256: str


@dataclass(frozen=True)
class SourceFinding:
    status: str
    reason: str
    catalog_id: str = ''
    whole_entry_sha256: str = ''
    source_url: str = ''
    source_receipt_sha256: str = ''
    local_fixture_seal: str = ''


@dataclass(frozen=True)
class PhotoIntent:
    status: str
    reason: str
    catalog_id: str = ''
    whole_entry_sha256: str = ''
    caption: str = ''
    photo_sha256: str = ''
    method: str = ''
    caption_parse_mode: str = ''
    source_button_url: str = ''
    attempts: int = 0
    local_fixture_seal: str = ''


@dataclass(frozen=True)
class StoreInspection:
    known: bool
    reason: str
    existing_body_hash: bool = False
    unresolved_delivery: bool = True
    last_posted_utc: datetime | None = None


def public_source(*_args: object, **_kwargs: object) -> SourceFinding:
    return SourceFinding('NO_POST', 'trusted_source_and_reviewer_roots_absent')


def public_sender(*_args: object, **_kwargs: object) -> PhotoIntent:
    return PhotoIntent('NO_POST', 'no_authenticated_sender_or_source')


def _seal(*parts: str) -> str:
    return hmac.new(_LOCAL_FIXTURE_KEY, '\0'.join(parts).encode('utf-8'), hashlib.sha256).hexdigest()


def _hex64(value: object) -> bool:
    return type(value) is str and bool(re.fullmatch('[0-9a-f]{64}', value))


def _offline_source_fixture(entry: object, *, reviewer: object, source: object,
                            roots: object, claims: object) -> SourceFinding:
    """Verify fixture-shaped data; NO cryptographic identity proof is performed."""
    if (type(entry) is not frozen.Entry or type(claims) is not ClaimMap
        or entry.catalog_id not in REQUIRED_CLAIMS):
        return SourceFinding('NO_POST', 'invalid_entry_or_claim_map')
    try:
        expected = current.validated_offline_entries()[entry.catalog_id]
    except (ValueError, OSError, KeyError, TypeError):
        return SourceFinding('NO_POST', 'local_catalog_invalid')
    digest = frozen.digest(entry)
    if (entry != expected or type(reviewer) is not trust.ReviewerReceipt
        or type(source) is not trust.SourceReceipt
        or type(roots) is not trust.VerifiedRoots
        or claims.catalog_id != entry.catalog_id
        or claims.whole_entry_sha256 != digest
        or claims.evidence_sha256 != source.evidence_digest
        or not _hex64(claims.clinical_review_receipt_sha256)
        or claims.clinical_review_receipt_sha256 != source.source_review_sha256
        or claims.claim_ids != REQUIRED_CLAIMS[entry.catalog_id]):
        return SourceFinding('NO_POST', 'unbound_or_incomplete_claim_mapping')
    gate = trust._offline_provenance_gate(entry, reviewer, source, roots)
    if gate != 'OFFLINE_VERIFIED_FIXTURE_ONLY':
        return SourceFinding('NO_POST', gate)
    return SourceFinding('OFFLINE_FIXTURE_ONLY', 'external_authentication_not_implemented',
                         entry.catalog_id, digest, source.source_ref, source.source_review_sha256,
                         _seal(entry.catalog_id,digest,source.source_ref,source.source_review_sha256))


def _offline_photo_fixture(entry: object, finding: object, png_bytes: object,
                           card_digest: object) -> PhotoIntent:
    """Pure preparation only. No chat ID, transport or Telegram credentials."""
    if (type(entry) is not frozen.Entry or type(finding) is not SourceFinding
        or type(png_bytes) is not bytes or type(card_digest) is not str):
        return PhotoIntent('NO_POST', 'invalid_types')
    if (finding.status != 'OFFLINE_FIXTURE_ONLY'
        or finding.catalog_id != entry.catalog_id
        or not hmac.compare_digest(finding.whole_entry_sha256, frozen.digest(entry))
        or not hmac.compare_digest(finding.local_fixture_seal,
              _seal(finding.catalog_id,finding.whole_entry_sha256,
                    finding.source_url,finding.source_receipt_sha256))):
        return PhotoIntent('NO_POST', 'source_not_fixture_verified_for_entry')
    if entry != current.validated_offline_entries().get(entry.catalog_id):
        return PhotoIntent('NO_POST', 'entry_drift')
    if (_utf16_units(entry.text) > 1000 or not entry.text
        or len(png_bytes) > 4 * 1024 * 1024
        or not _hex64(card_digest)
        or not hmac.compare_digest(hashlib.sha256(png_bytes).hexdigest(),card_digest)):
        return PhotoIntent('NO_POST', 'caption_or_card_integrity_invalid')
    try:
        with Image.open(BytesIO(png_bytes)) as im:
            if im.format != 'PNG' or im.width < 512 or im.height < 512:
                return PhotoIntent('NO_POST','static_card_dimensions_invalid')
            im.verify()
    except (OSError,ValueError,SyntaxError):
        return PhotoIntent('NO_POST','static_card_png_invalid')
    # A valid PNG header/hash alone does NOT establish visual QA or static title.
    return PhotoIntent('OFFLINE_FIXTURE_ONLY', 'static_visual_qa_and_delivery_unverified',
                       entry.catalog_id, frozen.digest(entry), entry.text,
                       card_digest, 'sendPhoto', 'omitted', finding.source_url, 1,
                       _seal(entry.catalog_id, frozen.digest(entry),entry.text,card_digest,finding.source_url))


def inspect_fixture_publication_store(db_path: Path, caption: str) -> StoreInspection:
    """Read a TEST snapshot using actual PublicationStore schema; never mutate DB.

    Deliberately refuses a production history filename; this is not a live
    source of truth for a send (external, mutable state may change after read).
    """
    if not isinstance(db_path, Path) or db_path.name == 'publication_history.sqlite3':
        return StoreInspection(False, 'production_store_not_permitted')
    if type(caption) is not str or not caption or not db_path.is_file():
        return StoreInspection(False, 'missing_fixture_store_or_caption')
    body_hash = hashlib.sha1(re.sub(r'\s+', ' ', caption.strip()).encode('utf-8')).hexdigest()
    try:
        # query-only URI: accidental writes are rejected by SQLite.
        con = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
        try:
            con.execute('PRAGMA query_only=ON')
            pub = con.execute('SELECT posted_at FROM publications WHERE body_hash=? ORDER BY posted_at DESC LIMIT 1',
                              (body_hash,)).fetchone()
            unresolved = con.execute('SELECT 1 FROM delivery_attempts LIMIT 1').fetchone() is not None
        finally:
            con.close()
        last = datetime.fromisoformat(pub[0]) if pub and pub[0] else None
        if pub and (last is None or last.tzinfo is None):
            return StoreInspection(False, 'untrusted_history_timestamp')
        return StoreInspection(True, 'fixture_snapshot_not_atomic', bool(pub), unresolved, last)
    except (sqlite3.Error, ValueError, TypeError, OSError):
        return StoreInspection(False, 'store_schema_or_timestamp_untrusted')


def simulate_one_attempt(intent: object, store: object, *, outcome: str) -> str:
    """No I/O. Models blocking; accepted is NOT a delivery receipt or publish proof."""
    if type(intent) is not PhotoIntent or intent.status != 'OFFLINE_FIXTURE_ONLY':
        return 'NO_POST'
    if type(store) is not StoreInspection or not store.known or store.unresolved_delivery:
        return 'NO_POST'
    if store.existing_body_hash:  # no global dedup bypass, even for warnings
        return 'NO_POST'
    if (not hmac.compare_digest(intent.local_fixture_seal,
            _seal(intent.catalog_id,intent.whole_entry_sha256,intent.caption,
                  intent.photo_sha256,intent.source_button_url))
        or intent.method != 'sendPhoto' or intent.caption_parse_mode != 'omitted'
        or intent.attempts != 1 or _utf16_units(intent.caption) > 1000):
        return 'NO_POST'
    if outcome == 'timeout':
        return 'UNKNOWN_QUARANTINE_NO_AUTORETRY'
    if outcome == 'mock_rejected':
        return 'DETERMINISTIC_REJECT_NO_FALLBACK'
    if outcome == 'mock_accepted':
        return 'SIMULATED_ONLY_NOT_DELIVERY_PROOF'
    return 'NO_POST'

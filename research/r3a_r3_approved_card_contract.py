"""R3-A R3 OFFLINE ONLY: exact expert-approved whole-card variants.

No production imports, no source approvals, no Telegram calls. An LLM is allowed
only to return a card_id, never a clinical string. Every delivered text byte is
reconstructed from a signed, independently approved, versioned card. Test keys
and test source bytes do NOT amount to real reviewer or clinical authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from datetime import datetime, timezone, timedelta
import unicodedata
from typing import Any
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Exact reference copied from merged PR #109; git blob cf88303ca2f7b2bfe08391867e7acc2e486cb4b2.
# Reuse independently security-reviewed point validator instead of making a new one.
from r3a_r2_final_post_contract import (
    _ed25519_prime_order_public_key, _full_refutation_sentence, Span
)

SCHEMA = "logopedia-r3a-r3-whole-card-v2"
NO_POST = "NO_POST"
_MAX_CHARS = 3500  # strict self-contained sendMessage, no photo caption/split
_MAX_VALIDITY = timedelta(days=90)  # Offline conservative bound; not clinical approval.
# Historical inventory: mandatory independent denial even when caller omits it.
# Never treat this list as an exhaustive publication source policy.
_HISTORICAL_NO_POST = frozenset({
    "healthychildren_bilingual_myths",
    "nationwide_bilingual_speech_delay",
    "asha_speech_sound_multilingual_influence",
    "nlg_nhs_tongue_tie_speech_myth",
    "asha_newborn_hearing_screening",
    "nationwide_school_hearing_screening",
    "nationwide_newborn_hearing_screening",
    "healthychildren_one_year_talking",
    "nationwide_aac_myths",
    "healthychildren_crawling_reading_myth",
    "healthychildren_learning_disorders_dyslexia",
})
_ALLOWED_SECTIONS = ("card_id", "myth", "fact", "qualifier", "action")
_PACK_FIELDS = frozenset(("schema", "pack_id", "revision", "reviewer_id", "valid_from_utc",
    "valid_until_utc", "source_id", "canonical_url", "raw_html_sha256", "evidence_sha256",
    "evidence_refutation", "source_qualifiers", "topic_id", "audience", "age_scope", "channel_id", "rubric_id", "reviewer_role", "cards"))


@dataclass(frozen=True)
class SignedPack:
    payload: bytes
    signature: bytes


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    canonical_url: str
    raw_html: bytes
    evidence_text: str
    topic_id: str
    audience: str
    age_scope: str


@dataclass(frozen=True)
class OfflineAuthority:
    # SIMULATION ONLY. Never substitute these caller-supplied fields for a trusted
    # production trust root. A real root requires owner-controlled key provisioning.
    reviewer_public_keys: dict[str, bytes]
    reviewer_roles: dict[str, frozenset[str]]
    approved_channel_id: str
    canonical_source_urls: dict[str, str]
    revoked_reviewer_ids: frozenset[str]
    revoked_pack_ids: frozenset[str]
    revoked_source_ids: frozenset[str]
    historical_source_ids: frozenset[str]
    # A separate offline registry must name the *current* pack for each
    # (source, topic, channel, rubric). Values: (revision, pack_id, payload_sha256).
    # This is caller-controlled in simulation, NOT a trustworthy production root.
    active_packs: dict[tuple[str, str, str, str], tuple[int, str, str]]
    now_utc: datetime


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str
    send_plan: Mapping[str, Any] | None = None


def _stop(reason: str) -> Verdict:
    return Verdict(False, NO_POST + ":" + reason)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_payload(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _utc(s: str) -> datetime:
    if type(s) is not str or not s.endswith("Z"):
        raise ValueError("non_utc_time")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.utcoffset() is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("non_utc_time")
    return dt


# Unicode Default_Ignorable_Code_Point frozen range table (17 intervals).
# Unlike categories C/Z alone, this also catches Khmer inherent-vowel controls,
# Mongolian variation selectors and supplementary-plane variation selectors.
# Conservative research policy: reject ANY such character, even in a signed pack.
# Keep the table versioned and re-audit on Unicode database updates.
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C),
    (0x115F, 0x1160), (0x17B4, 0x17B5), (0x180B, 0x180F),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F),
    (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)


def _is_default_ignorable(codepoint: int) -> bool:
    return any(low <= codepoint <= high for low, high in _DEFAULT_IGNORABLE_RANGES)


def _valid_rendered_string(s: object) -> bool:
    # No implicit HTML parsing, malformed entities or visible text divergence
    # if a future caller accidentally selects the existing HTML delivery path.
    return _valid_string(s) and not any(ch in s for ch in "<>&")


def _valid_string(s: object) -> bool:
    # Strict closed display/review vocabulary without a new CI dependency.
    # The canonical signed bytes must have a visible, stable representation.
    return (type(s) is str and bool(s) and s == s.strip()
            and unicodedata.normalize("NFC", s) == s
            and all(
                (ch == " " or unicodedata.category(ch)[0] not in "CZ")
                and not _is_default_ignorable(ord(ch))
                and ord(ch) != 0x2800  # BRAILLE PATTERN BLANK
                and unicodedata.bidirectional(ch) not in ("R", "AL", "AN")
                for ch in s
            ))


def _valid_hex(s: object) -> bool:
    return type(s) is str and len(s) == 64 and all(ch in "0123456789abcdef" for ch in s)


def _valid_https_url(s: object) -> bool:
    if not _valid_rendered_string(s):
        return False
    try:
        parts = urlsplit(s)
        return parts.scheme == "https" and bool(parts.netloc) and parts.username is None and parts.password is None and not any(ch.isspace() for ch in s)
    except ValueError:
        return False


def _valid_anchor(anchor: object) -> bool:
    return (type(anchor) is dict and set(anchor)=={"start","text"}
            and type(anchor["start"]) is int and anchor["start"] >= 0
            and _valid_string(anchor["text"]))


def _verify_record(signed: SignedPack, auth: OfflineAuthority) -> dict[str, Any] | None:
    if (type(signed) is not SignedPack or type(signed.payload) is not bytes
        or type(signed.signature) is not bytes or len(signed.signature) != 64
        or not signed.payload or len(signed.payload) > 64000):
        return None
    try:
        rec = json.loads(signed.payload.decode("utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=lambda v: (_ for _ in ()).throw(ValueError("non_json_number")))
        if type(rec) is not dict or set(rec) != _PACK_FIELDS or canonical_payload(rec) != signed.payload:
            return None
        for field in _PACK_FIELDS - {"cards", "revision", "evidence_refutation", "source_qualifiers"}:
            if not _valid_string(rec[field]):
                return None
        if rec["schema"] != SCHEMA or type(rec["revision"]) is not int or rec["revision"] < 1:
            return None
        if not _valid_hex(rec["raw_html_sha256"]) or not _valid_hex(rec["evidence_sha256"]):
            return None
        if not _valid_https_url(rec["canonical_url"]) or not _valid_rendered_string(rec["age_scope"]):
            return None
        try:
            start, end = _utc(rec["valid_from_utc"]), _utc(rec["valid_until_utc"])
            if not start < end or end - start > _MAX_VALIDITY:
                return None
        except ValueError:
            return None
        anchor = rec["evidence_refutation"]
        qualifiers = rec["source_qualifiers"]
        if not _valid_anchor(anchor) or type(qualifiers) is not list or not qualifiers or len(qualifiers) > 10:
            return None
        if not all(_valid_anchor(item) for item in qualifiers):
            return None
        if type(rec["cards"]) is not list or not 1 <= len(rec["cards"]) <= 12:
            return None
        ids = set()
        for card in rec["cards"]:
            if type(card) is not dict or set(card) != set(_ALLOWED_SECTIONS):
                return None
            if not all(_valid_rendered_string(card[key]) for key in _ALLOWED_SECTIONS):
                return None
            if card["card_id"] in ids:
                return None
            ids.add(card["card_id"])
            # No unreviewed markup, injection of extra lines, or placeholder slots.
            if any("\n" in card[key] or "{{" in card[key] or "}}" in card[key]
                   for key in _ALLOWED_SECTIONS):
                return None
        reviewer = rec["reviewer_id"]
        if (rec["reviewer_role"] != "clinical_reviewer" or
                rec["reviewer_role"] not in auth.reviewer_roles.get(reviewer, frozenset())):
            return None
        if reviewer in auth.revoked_reviewer_ids or rec["pack_id"] in auth.revoked_pack_ids:
            return None
        key = auth.reviewer_public_keys.get(reviewer)
        if not _ed25519_prime_order_public_key(key):
            return None
        Ed25519PublicKey.from_public_bytes(key).verify(signed.signature, signed.payload)
        return rec
    except (InvalidSignature, ValueError, UnicodeError, TypeError, KeyError, OverflowError):
        return None


def _validate_authority(auth: object) -> bool:
    if type(auth) is not OfflineAuthority or type(auth.now_utc) is not datetime:
        return False
    if auth.now_utc.tzinfo is None or auth.now_utc.utcoffset() != timezone.utc.utcoffset(None):
        return False
    if (not _valid_string(auth.approved_channel_id) or
            any(type(x) is not dict for x in (auth.reviewer_public_keys,
                                              auth.reviewer_roles, auth.canonical_source_urls,
                                              auth.active_packs))):
        return False
    if any(type(x) is not frozenset for x in (auth.revoked_reviewer_ids, auth.revoked_pack_ids,
                                            auth.revoked_source_ids, auth.historical_source_ids)):
        return False
    if any(type(k) is not str or type(v) is not frozenset or
           any(type(role) is not str for role in v)
           for k,v in auth.reviewer_roles.items()):
        return False
    if any(type(k) is not str or type(v) is not bytes for k,v in auth.reviewer_public_keys.items()):
        return False
    if any(type(k) is not str or type(v) is not str for k,v in auth.canonical_source_urls.items()):
        return False
    if any(type(v) is not str for s in (auth.revoked_reviewer_ids, auth.revoked_pack_ids,
                                      auth.revoked_source_ids, auth.historical_source_ids) for v in s):
        return False
    # Reject missing, inherited, malformed or weakened registry entries.
    for scope, identity in auth.active_packs.items():
        if (type(scope) is not tuple or len(scope) != 4
                or not all(_valid_string(v) for v in scope)
                or type(identity) is not tuple or len(identity) != 3
                or type(identity[0]) is not int or identity[0] < 1
                or not _valid_string(identity[1]) or not _valid_hex(identity[2])):
            return False
    return True


def _render(pack: dict[str, Any], card: dict[str, str]) -> str:
    return (f"🔴 Миф: {card['myth']}\n\n"
            f"🟢 Факт: {card['fact']}\n\n"
            f"⚠️ Важно: {card['qualifier']}\n\n"
            f"👶 Возраст: {pack['age_scope']}\n\n"
            f"🧩 Что попробовать сегодня: {card['action']}\n\n"
            f"Источник: {pack['canonical_url']}\n\n"
            f"#мифы")


def _allowed_send_plan(record: dict[str, Any], card: dict[str,str]) -> dict[str,Any] | None:
    text = _render(record, card)
    # Pure plain text: no parse_mode, HTML, Markdown, photo captions, splits or
    # extra delivery text. Human-visible text is byte-exact when sent as plain.
    if len(text.encode("utf-16-le")) // 2 > _MAX_CHARS:
        return None
    return {"method":"sendMessage", "text":text, "parse_mode":None,
            "disable_web_page_preview":True}


def evaluate_last_boundary(*, signed_pack: SignedPack, source: SourceSnapshot,
                            authority: OfflineAuthority, card_id: str,
                            proposed_transport: dict[str,Any] | None = None,
                            simulation_only: bool = False) -> Verdict:
    """Offline dry-run of the *last* transport boundary, never a Telegram send.

    Untrusted model picks a card ID only. All actual words come from a signed
    expert-approved whole-card variant. Any changed transport text fails closed.
    Future publisher wiring must put this check after finalization and before the
    actual Telegram request, while retaining every existing validator and dedup.
    """
    if simulation_only is not True:
        return _stop("not_production_wired")
    if not _validate_authority(authority):
        return _stop("independent_trust_root_missing")
    if type(card_id) is not str or not card_id:
        return _stop("invalid_selection")
    if type(source) is not SourceSnapshot:
        return _stop("untrusted_source_snapshot")
    if (any(type(getattr(source,k)) is not str for k in ("source_id","canonical_url",
            "evidence_text", "topic_id", "audience", "age_scope"))
            or type(source.raw_html) is not bytes or not source.raw_html):
        return _stop("untrusted_source_snapshot")
    if source.source_id in _HISTORICAL_NO_POST or source.source_id in authority.historical_source_ids:
        return _stop("historical_11_no_post")
    if source.source_id in authority.revoked_source_ids:
        return _stop("source_revoked")
    if authority.canonical_source_urls.get(source.source_id) != source.canonical_url:
        return _stop("source_url_not_authorized")
    pack = _verify_record(signed_pack, authority)
    if pack is None:
        return _stop("signed_approval_invalid")
    scope = (pack["source_id"], pack["topic_id"], pack["channel_id"], pack["rubric_id"])
    current = authority.active_packs.get(scope)
    # Exact-current digest prevents rollback under a fresh pack_id or a
    # different payload at the same revision. Independent registry required.
    if current != (pack["revision"], pack["pack_id"], _sha256(signed_pack.payload)):
        return _stop("pack_not_current_revision")
    try:
        if not (_utc(pack["valid_from_utc"]) <= authority.now_utc <= _utc(pack["valid_until_utc"])):
            return _stop("approval_expired_or_not_yet_valid")
    except ValueError:
        return _stop("invalid_approval_window")
    if pack["channel_id"] != authority.approved_channel_id or pack["rubric_id"] != "myth_fact":
        return _stop("channel_or_rubric_mismatch")
    for field in ("source_id","canonical_url","topic_id","audience","age_scope"):
        if pack[field] != getattr(source,field):
            return _stop("source_scope_mismatch")
    if pack["raw_html_sha256"] != _sha256(source.raw_html):
        return _stop("source_html_digest_changed")
    try:
        evidence_digest = _sha256(source.evidence_text.encode("utf-8"))
    except UnicodeError:
        return _stop("invalid_evidence_encoding")
    if pack["evidence_sha256"] != evidence_digest:
        return _stop("source_evidence_digest_changed")
    anchor = pack["evidence_refutation"]
    if not _full_refutation_sentence(source.evidence_text, Span(anchor["start"],anchor["text"])):
        return _stop("source_refutation_missing_or_truncated")
    for item in pack["source_qualifiers"]:
        if not _full_refutation_sentence(source.evidence_text, Span(item["start"], item["text"])):
            return _stop("source_qualifier_missing_or_truncated")
    card = next((c for c in pack["cards"] if c["card_id"] == card_id), None)
    if card is None:
        return _stop("unapproved_variant")
    plan = _allowed_send_plan(pack, card)
    if plan is None:
        return _stop("oversize_plain_transport")
    if proposed_transport is not None:
        if (type(proposed_transport) is not dict or set(proposed_transport) != set(plan)
            or any(type(proposed_transport[k]) is not type(plan[k])
                   or proposed_transport[k] != plan[k] for k in plan)):
            return _stop("last_boundary_transport_changed")
    return Verdict(True, "SIMULATION_ONLY_APPROVED_WHOLE_CARD", MappingProxyType(plan))

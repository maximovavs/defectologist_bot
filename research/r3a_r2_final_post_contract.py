"""R3-A R2: isolated, fail-closed *final-post* admission experiment.

No production registration or source approvals. This module is deliberately NOT
imported by the publisher. The only positive evaluation path is explicit
``simulation_only=True`` with a separately supplied trusted registry and an
Ed25519 verification key. A signature verifies the reviewed bytes; it cannot
independently establish medical accuracy or clinical reviewer authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps, loads
from hmac import compare_digest
from typing import Mapping, Sequence
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

REJECTION_REASON = "myth_claim_not_grounded"  # Existing publisher validation reason.
PERMIT_SCHEMA = "logopedia-r3a-r2-final-post-v2"
_MYTH_LINE = re.compile(r"^🔴\s*Миф\s*[:：]\s*(.+\S)\s*$", re.MULTILINE)
_AGE_LINE = re.compile(r"^👶\s*Возраст\s*[:：]\s*(.+\S)\s*$", re.MULTILINE)
_ACTION_HEADER = re.compile(r"^🧩\s*Что попробовать сегодня\s*[:：]", re.MULTILINE)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


@dataclass(frozen=True)
class Span:
    """Exact character position in the *reviewed extracted evidence*."""

    start: int
    text: str

    def matches(self, evidence: str) -> bool:
        return (isinstance(self.start, int) and not isinstance(self.start, bool)
                and self.start >= 0 and bool(self.text)
                and evidence[self.start:self.start + len(self.text)] == self.text)


@dataclass(frozen=True)
class SourceSnapshot:
    """Data supplied by a trusted fetch/extraction stage, NOT the generated post.

    Raw bytes and exact extraction are both required. This module does not fetch
    sources or implement HTML extraction, and cannot prove that a caller supplied
    authentic bytes. The signed review must refer to the same two digests.
    """

    source_id: str
    canonical_url: str
    raw_response_bytes: bytes
    evidence_text: str
    topic_id: str
    population: str
    age_scope: str


@dataclass(frozen=True)
class ReviewPermit:
    """All semantically material admission fields are signed, including final post."""

    permit_id: str
    reviewer_id: str
    source_id: str
    canonical_url: str
    raw_html_sha256: str
    evidence_sha256: str
    topic_id: str
    population: str
    age_scope: str
    myth_line: str
    fact_text: str
    refutation: Span
    mandatory_source_qualifiers: tuple[Span, ...]
    mandatory_post_qualifiers: tuple[str, ...]
    approved_final_post_sha256: str
    schema: str = PERMIT_SCHEMA

    def signed_bytes(self) -> bytes:
        # Do NOT include any unsigned fields outside this canonical representation.
        record = {
            "schema": self.schema,
            "permit_id": self.permit_id,
            "reviewer_id": self.reviewer_id,
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "raw_html_sha256": self.raw_html_sha256,
            "evidence_sha256": self.evidence_sha256,
            "topic_id": self.topic_id,
            "population": self.population,
            "age_scope": self.age_scope,
            "myth_line": self.myth_line,
            "fact_text": self.fact_text,
            "refutation": {"start": self.refutation.start, "text": self.refutation.text},
            "mandatory_source_qualifiers": [
                {"start": span.start, "text": span.text}
                for span in self.mandatory_source_qualifiers
            ],
            "mandatory_post_qualifiers": list(self.mandatory_post_qualifiers),
            "approved_final_post_sha256": self.approved_final_post_sha256,
        }
        return dumps(record, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class SignedReview:
    permit: ReviewPermit
    signature: bytes


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    detail: str


def _deny(detail: str) -> Decision:
    return Decision(False, REJECTION_REASON, detail)


def _full_refutation_sentence(evidence: str, span: Span) -> bool:
    """Conservative sentence-boundary heuristic, NOT semantic entailment.

    A shortened phrase like 'There is no scientific evidence' is never a full
    sentence. This rejects genuine multi-clause text unless separately reviewed.
    """
    if not span.matches(evidence) or span.text != span.text.strip():
        return False
    before = evidence[:span.start].rstrip()
    if before and before[-1] not in ".!?":
        return False
    return span.text[-1] in ".!?"


def _extract_fact_section(final_post: str, myth_match: re.Match[str]) -> str:
    tail = final_post[myth_match.end():]
    marker = _ACTION_HEADER.search(tail)
    return tail[:marker.start()] if marker else tail


# Reject arbitrary subclasses and overridden magic methods *before* signature,
# lookup, hash or text comparisons. Frozen dataclasses do not enforce field types.
_SIGNED_STRING_FIELDS = (
    "schema", "permit_id", "reviewer_id", "source_id", "canonical_url",
    "raw_html_sha256", "evidence_sha256", "topic_id", "population",
    "age_scope", "myth_line", "fact_text", "approved_final_post_sha256",
)
_SOURCE_STRING_FIELDS = (
    "source_id", "canonical_url", "evidence_text", "topic_id",
    "population", "age_scope",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", flags=re.ASCII)


# Ed25519 verifier backends do not consistently reject low-order public keys.
# Before asking the backend to verify a signature, require a canonical point in
# the *prime-order* Ed25519 subgroup. The subgroup test rejects all small-order
# and mixed-order keys, including noncanonical representations. Offline only.
_ED25519_P = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493
_ED25519_D = (-121665 * pow(121666, _ED25519_P - 2, _ED25519_P)) % _ED25519_P
_ED25519_SQRT_MINUS_ONE = pow(2, (_ED25519_P - 1) // 4, _ED25519_P)


def _edwards_add(a: tuple[int, int, int, int],
                 b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Complete extended-coordinate twisted-Edwards addition (a=-1)."""
    p = _ED25519_P
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    aa = ((y1 - x1) * (y2 - x2)) % p
    bb = ((y1 + x1) * (y2 + x2)) % p
    cc = (2 * _ED25519_D * t1 * t2) % p
    dd = (2 * z1 * z2) % p
    e, f, g, h = (bb - aa) % p, (dd - cc) % p, (dd + cc) % p, (bb + aa) % p
    return (e * f % p, g * h % p, f * g % p, e * h % p)


def _ed25519_prime_order_public_key(key: object) -> bool:
    """Reject malformed, noncanonical, small-order and mixed-order keys.

    Encoding is the RFC 8032 compressed Edwards point. Main-subgroup
    membership is checked by [L]P == identity, with P != identity.
    This is a defensive input validation preceding the library verifier,
    not a replacement for cryptographic signature verification.
    """
    if type(key) is not bytes or len(key) != 32:
        return False
    encoded = int.from_bytes(key, "little")
    sign = encoded >> 255
    y = encoded & ((1 << 255) - 1)
    p = _ED25519_P
    if y >= p:
        return False  # No reduction of noncanonical encodings.
    yy = y * y % p
    denominator = (_ED25519_D * yy + 1) % p
    if denominator == 0:
        return False
    xx = ((yy - 1) * pow(denominator, p - 2, p)) % p
    x = pow(xx, (p + 3) // 8, p)
    if x * x % p != xx:
        x = x * _ED25519_SQRT_MINUS_ONE % p
    if x * x % p != xx or (x == 0 and sign):
        return False
    if x % 2 != sign:
        x = (-x) % p
    if x == 0 and y == 1:
        return False  # Identity has order one.
    point = (x, y, 1, x * y % p)
    acc = (0, 1, 1, 0)
    scalar = _ED25519_L
    while scalar:
        if scalar & 1:
            acc = _edwards_add(acc, point)
        point = _edwards_add(point, point)
        scalar >>= 1
    ax, ay, az, at = acc
    return ax == 0 and (ay - az) % p == 0 and at == 0


def _strict_span(span: object) -> bool:
    return (type(span) is Span and type(span.start) is int
            and type(span.text) is str and span.start >= 0 and bool(span.text))


def _verified_payload(signed: object) -> tuple[bytes, dict] | None:
    """Serialize only exact trusted schema types; never call virtual signed_bytes().

    The returned dictionary contains ordinary JSON primitive values from the
    exact signed byte string. All subsequent permit comparisons use this dict,
    rather than caller-controlled dataclass attributes or subclass equality.
    Signature verification is performed by the caller over these same bytes.
    """
    if type(signed) is not SignedReview or type(signed.signature) is not bytes:
        return None
    if len(signed.signature) != 64:
        return None
    permit = signed.permit
    if type(permit) is not ReviewPermit:
        return None
    if any(type(getattr(permit, key)) is not str for key in _SIGNED_STRING_FIELDS):
        return None
    if (not permit.permit_id or not permit.reviewer_id or not permit.source_id
            or not permit.canonical_url or not permit.topic_id
            or not permit.population or not permit.age_scope
            or not permit.myth_line or not permit.fact_text):
        return None
    if any(not _SHA256_RE.fullmatch(getattr(permit, field)) for field in (
        "raw_html_sha256", "evidence_sha256", "approved_final_post_sha256"
    )):
        return None
    if not _strict_span(permit.refutation):
        return None
    if (type(permit.mandatory_source_qualifiers) is not tuple
            or any(not _strict_span(span) for span in permit.mandatory_source_qualifiers)):
        return None
    if (type(permit.mandatory_post_qualifiers) is not tuple
            or any(type(q) is not str or not q for q in permit.mandatory_post_qualifiers)):
        return None
    try:
        canonical_bytes = ReviewPermit.signed_bytes(permit)
        record = loads(canonical_bytes.decode("utf-8"))
    except (ValueError, TypeError, UnicodeError, OverflowError):
        return None
    # Safety invariant: built-in JSON materialization, never a user object.
    if type(record) is not dict:
        return None
    return canonical_bytes, record


def assess_finalized_post(
    source: SourceSnapshot,
    final_post: str,
    *,
    signed_reviews: Mapping[str, Sequence[SignedReview]] | None = None,
    trusted_reviewer_keys: Mapping[str, bytes] | None = None,
    trusted_source_urls: Mapping[str, str] | None = None,
    simulation_only: bool = False,
) -> Decision:
    """Only an *offline* test path; no production registry or activation.

    A cryptographic signature is proof of byte integrity, not proof of clinical
    correctness, reviewer authority, or freshness of credentials. In a future
    production design the trust root and revocation must be independent.
    """
    if simulation_only is not True:
        return _deny("not_wired_to_a_trusted_production_authority")
    if (type(signed_reviews) is not dict or type(trusted_reviewer_keys) is not dict
            or type(trusted_source_urls) is not dict
            or not signed_reviews or not trusted_reviewer_keys
            or not trusted_source_urls):
        return _deny("no_independent_admission_registry")
    if type(final_post) is not str or not final_post:
        return _deny("empty_final_post")
    if (any(type(key) is not str for key in signed_reviews)
            or any(type(key) is not str for key in trusted_reviewer_keys)
            or any(type(key) is not str for key in trusted_source_urls)):
        return _deny("invalid_registry_keys")
    if type(source) is not SourceSnapshot:
        return _deny("untrusted_source_input")
    if (any(type(getattr(source, key)) is not str for key in _SOURCE_STRING_FIELDS)
            or type(source.raw_response_bytes) is not bytes
            or not source.raw_response_bytes or not source.evidence_text
            or not source.source_id or not source.canonical_url or not source.topic_id
            or not source.population or not source.age_scope):
        return _deny("missing_source_provenance")
    registered_url = trusted_source_urls.get(source.source_id)
    if type(registered_url) is not str or registered_url != source.canonical_url:
        return _deny("source_url_not_registered")

    try:
        post_hash = digest(final_post.encode("utf-8"))
        evidence_hash = digest(source.evidence_text.encode("utf-8"))
    except UnicodeError:
        return _deny("invalid_text_encoding")
    entries = signed_reviews.get(source.source_id, ())
    if type(entries) is not tuple:
        return _deny("invalid_review_registry_shape")
    matched: list[tuple[SignedReview, bytes, dict]] = []
    for signed in entries:
        payload_and_record = _verified_payload(signed)
        if payload_and_record is None:
            return _deny("invalid_review_object")
        raw, record = payload_and_record
        if compare_digest(record["approved_final_post_sha256"], post_hash):
            matched.append((signed, raw, record))
    if len(matched) != 1:
        return _deny("no_unique_permitted_final_post")
    reviewed, signed_bytes, permit = matched[0]
    if permit["schema"] != PERMIT_SCHEMA:
        return _deny("stale_or_unknown_contract")

    trusted_key_bytes = trusted_reviewer_keys.get(permit["reviewer_id"])
    if not _ed25519_prime_order_public_key(trusted_key_bytes):
        return _deny("reviewer_not_trusted")
    try:
        Ed25519PublicKey.from_public_bytes(trusted_key_bytes).verify(
            reviewed.signature, signed_bytes)
    except (InvalidSignature, ValueError, TypeError):
        return _deny("review_signature_invalid")

    for left, right in (
        (permit["source_id"], source.source_id),
        (permit["canonical_url"], source.canonical_url),
        (permit["topic_id"], source.topic_id),
        (permit["population"], source.population),
        (permit["age_scope"], source.age_scope),
        (permit["raw_html_sha256"], digest(source.raw_response_bytes)),
        (permit["evidence_sha256"], evidence_hash),
    ):
        if not left or left != right:
            return _deny("reviewed_provenance_mismatch")
    ref = permit["refutation"]
    if not _full_refutation_sentence(source.evidence_text, Span(ref["start"], ref["text"])):
        return _deny("missing_complete_refutation")
    qualifiers = permit["mandatory_source_qualifiers"]
    if (not qualifiers or any(not _full_refutation_sentence(
            source.evidence_text, Span(item["start"], item["text"]))
            for item in qualifiers)):
        return _deny("source_qualifier_missing_or_incomplete")

    myth_matches = list(_MYTH_LINE.finditer(final_post))
    if len(myth_matches) != 1 or myth_matches[0].group(1).strip() != permit["myth_line"]:
        return _deny("myth_line_changed")
    ages = list(_AGE_LINE.finditer(final_post))
    if len(ages) != 1 or ages[0].group(1).strip() != permit["age_scope"]:
        return _deny("post_age_scope_changed")
    fact_area = _extract_fact_section(final_post, myth_matches[0])
    if not permit["fact_text"] or fact_area.count(permit["fact_text"]) != 1:
        return _deny("reviewed_fact_missing_or_ambiguous")
    if (not permit["mandatory_post_qualifiers"] or any(
            not q or q not in permit["fact_text"]
            for q in permit["mandatory_post_qualifiers"])):
        return _deny("post_qualifier_not_in_fact")
    return Decision(True, "ok", "simulated_review_only")

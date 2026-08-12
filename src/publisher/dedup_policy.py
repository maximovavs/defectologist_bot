"""
Deduplication policy helpers for publisher-level checks.

This module intentionally stays small and dependency-light:
- no Telegram logic
- no DB/store access
- no LLM logic
- no publisher orchestration

Goal:
Keep rubric-specific dedup thresholds and bypass rules out of the large
run_publisher.py file, so future dedup changes can be made safely in a small
full-file replacement.
"""

from __future__ import annotations

import os
import re


def _env_float(name: str, default: str) -> float:
    """Read a float env var safely, falling back to default on invalid values."""
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: str) -> int:
    """Read an int env var safely, falling back to default on invalid values."""
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


SEMANTIC_THRESHOLD = _env_float("SEMANTIC_THRESHOLD", "0.95")

# Source-level (evidence) semantic threshold. Source texts are long and share a lot
# of professional vocabulary, so the source layer gets its own threshold instead of
# borrowing the global one.
SEMANTIC_THRESHOLD_SOURCE = _env_float("SEMANTIC_THRESHOLD_SOURCE", "0.93")

# Editorial-core threshold for the cross-rubric freshness check. The editorial core
# is short and stripped of boilerplate, so near-identical advice scores high and a
# lower threshold than the full-body one is appropriate.
SEMANTIC_THRESHOLD_POST = _env_float("SEMANTIC_THRESHOLD_POST", "0.86")

# Freshness cooldowns, in days.
SOURCE_COOLDOWN_DAYS = _env_int("SOURCE_COOLDOWN_DAYS", "28")
EDITORIAL_CORE_COOLDOWN_DAYS = _env_int("EDITORIAL_CORE_COOLDOWN_DAYS", "28")

# How many recently used source domains feed the soft diversity preference.
RECENT_SOURCE_DOMAIN_WINDOW = _env_int("RECENT_SOURCE_DOMAIN_WINDOW", "3")

SEMANTIC_THRESHOLD_POST_AGE_NORMS = _env_float(
    "SEMANTIC_THRESHOLD_POST_AGE_NORMS",
    "0.985",
)

SEMANTIC_THRESHOLD_POST_PLAY_AND_SPEAK = _env_float(
    "SEMANTIC_THRESHOLD_POST_PLAY_AND_SPEAK",
    "0.94",
)

SEMANTIC_THRESHOLD_POST_QUESTION_WEEK = _env_float(
    "SEMANTIC_THRESHOLD_POST_QUESTION_WEEK",
    "0.94",
)

SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER = _env_float(
    "SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER",
    "0.92",
)

SEMANTIC_THRESHOLD_POST_TIP_OF_DAY = _env_float(
    "SEMANTIC_THRESHOLD_POST_TIP_OF_DAY",
    "0.94",
)

SEMANTIC_THRESHOLD_POST_MYTH_FACT = _env_float(
    "SEMANTIC_THRESHOLD_POST_MYTH_FACT",
    "0.94",
)

SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK = _env_float(
    "SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK",
    "0.985",
)


def normalize_rubric_id(rubric_id: str | None) -> str:
    """Normalize rubric id for stable policy comparisons."""
    return (rubric_id or "").strip().lower()


def should_allow_evergreen_source_reuse(rubric_id: str | None) -> bool:
    """
    Return True for rubrics that may reuse the same trusted source/evidence
    because they are evergreen recurring formats.

    This bypasses only source URL / evidence hash blocking.
    Final post protections must remain active:
    - dup_body_hash_db
    - dup_semantic_post
    - validation
    """
    return normalize_rubric_id(rubric_id) in {
        "question_week",
        "tip_of_day",
        "play_and_speak",
        "myth_fact",
        "age_norms",
        "bilingual_corner",
        "method_piggybank",
    }


EVERGREEN_SOURCE_REUSE_REASONS = frozenset({"dup_url_db", "dup_evidence_hash_db"})


def should_bypass_duplicate_reason(rubric_id: str | None, reason: str | None) -> bool:
    """Limit evergreen reuse to persisted source URL/evidence duplicates."""
    return (
        should_allow_evergreen_source_reuse(rubric_id)
        and (reason or "").strip() in EVERGREEN_SOURCE_REUSE_REASONS
    )


def semantic_post_threshold_for_rubric(rubric_id: str | None) -> float:
    """
    Return semantic body/post dedup threshold for the current rubric.

    age_norms naturally produces similar milestone-style texts every Sunday.
    For this rubric we keep exact/hash dedup active, but make semantic post
    dedup stricter so only near-identical final posts are blocked.

    question_week also naturally repeats broad parent-question topics
    (bilingualism, language delay, communication milestones). Its final Q&A
    posts can be useful even when semantically close to previous evidence-based
    posts, so it uses a slightly stricter-than-global post threshold.

    Other rubrics keep the global SEMANTIC_THRESHOLD.
    """
    normalized = normalize_rubric_id(rubric_id)

    if normalized == "myth_fact":
        return SEMANTIC_THRESHOLD_POST_MYTH_FACT

    if normalized == "age_norms":
        return SEMANTIC_THRESHOLD_POST_AGE_NORMS

    if normalized == "play_and_speak":
        return SEMANTIC_THRESHOLD_POST_PLAY_AND_SPEAK

    if normalized == "question_week":
        return SEMANTIC_THRESHOLD_POST_QUESTION_WEEK

    if normalized == "tip_of_day":
        return SEMANTIC_THRESHOLD_POST_TIP_OF_DAY

    if normalized == "bilingual_corner":
        return SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER

    if normalized == "method_piggybank":
        return SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK

    return SEMANTIC_THRESHOLD


def should_bypass_source_semantic_dedup(rubric_id: str | None) -> bool:
    """
    Return True when source-level semantic dedup should not block candidates.

    Some rubrics naturally reuse the same broad evidence topics while still
    producing different final posts.

    We only bypass source-level semantic dedup. Other dedup policies separately
    decide whether trusted evergreen source URL/evidence may be reused.
    Safer final checks remain active:
    - dup_body_hash_db
    - dup_semantic_post

    Rationale:
    - method_piggybank: professional method articles genuinely share the same
      terminology while still producing different practical techniques, and the
      rubric keeps a very strict full-body threshold (0.985) behind this bypass.

    Every other rubric lost this bypass: a blanket source-semantic bypass let the
    same evidence topic return again and again, which is exactly the editorial
    staleness this policy exists to prevent. Those rubrics are still protected by
    their own full-body thresholds and by the editorial-core freshness check.
    """
    return normalize_rubric_id(rubric_id) == "method_piggybank"


# ---------------------------------------------------------------------------
# Editorial core extraction (deterministic, no LLM)
# ---------------------------------------------------------------------------

_LINK_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
_URL_INLINE_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HASHTAG_ONLY_LINE_RE = re.compile(r"^\s*(?:#[^\s#]+\s*)+$")
_HASHTAG_TOKEN_RE = re.compile(r"(?<!\w)#[A-Za-zА-Яа-яЁё0-9_]+")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_BOLD_MARKS_RE = re.compile(r"[*_`]{1,3}")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-–—•·▪️*]|\d{1,2}[.)])\s+")

# Service headings that structure the post but carry no editorial meaning. Kept
# deliberately small: only headings that are pure labels.
#
# Two groups, both label-only:
# - footer/apparatus labels ("Источник", "Теги", ...);
# - rubric labels ("Совет дня", "Приём из копилки", ...). The same advice has to
#   compare equal no matter which rubric wrapper it arrived in, so a line that is
#   nothing but the rubric's own name is noise for the editorial core. Headings
#   that carry real content ("Как поддержать речь в 2 года") are absent from this
#   list and are kept.
_SERVICE_HEADING_WORDS = frozenset(
    {
        "источник",
        "источники",
        "ссылка",
        "ссылки",
        "материал",
        "материалы",
        "литература",
        "читать далее",
        "подробнее",
        "дисклеймер",
        "важно",
        "примечание",
        "теги",
        # Rubric labels.
        "совет",
        "совет дня",
        "приём",
        "прием",
        "приём дня",
        "прием дня",
        "приём из копилки",
        "прием из копилки",
        "копилка методов",
        "вопрос недели",
        "миф",
        "факт",
        "миф и факт",
        "уголок билингва",
        "играем и говорим",
        "нормы возраста",
    }
)


# Wrapper labels used by the live Telegram templates. They structure a post but
# carry no editorial meaning, so the same advice has to compare equal whichever
# wrapper delivered it — parent or pro, one rubric or another.
#
# Only the label itself is removed. Content that shares the line with its label
# ("💡 Что это дает: ребёнок выбирает картинку") is kept.
#
# The emoji is part of the signature and is REQUIRED. A bare word before a colon
# is ordinary Russian prose, not structure: "Это важный факт: ребёнок уже
# понимает просьбу" and "Наша цель: помочь ребёнку начать фразу" must survive
# intact, so nothing here is matched without its template emoji.
_WRAPPER_LABEL_WORDS = (
    # Parent templates.
    "возраст",
    "что попробовать сегодня",
    "что попробовать",
    "пример",
    "что это дает",
    "что это даёт",
    "как играть",
    "миф",
    # Pro templates.
    "аудитория",
    "цель",
    "материалы",
    "как провести",
    "на что смотреть",
    "вариант усложнения",
)

# Footer/apparatus labels. These may appear without an emoji, but only where the
# structure proves they are apparatus: the label has to introduce links or
# hashtags, nothing else. "Наш источник: мама описывает действия ребёнка" is an
# ordinary sentence and must survive whole.
_APPARATUS_LABEL_WORDS = (
    "источник",
    "источники",
    "ссылка",
    "ссылки",
    "теги",
)

# What an apparatus label is allowed to introduce: links and/or hashtags.
_APPARATUS_PAYLOAD_TOKEN = r"(?:(?:https?://|www\.)\S+|#[^\s#]+)"

# An emoji prefix such as "👶", "✅" or the ZWJ sequence "👩‍⚕️". Letters, digits
# and punctuation are excluded, so neither a real word nor the full stop closing
# the previous sentence is ever swallowed as decoration.
_LABEL_PREFIX_PUNCTUATION = ".,;:!?…()[]{}«»\"'`*_#/\\|-–—+=<>@%&~^$"
_LABEL_EMOJI_PREFIX = (
    r"[^\w\s" + re.escape(_LABEL_PREFIX_PUNCTUATION) + r"]{1,3}"
    r"(?:‍[^\w\s" + re.escape(_LABEL_PREFIX_PUNCTUATION) + r"]{1,3})*[️‍]*"
)
def _label_alternation(words) -> str:
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


_WRAPPER_ALTERNATION = _label_alternation(_WRAPPER_LABEL_WORDS)
_APPARATUS_ALTERNATION = _label_alternation(_APPARATUS_LABEL_WORDS)

# Structural template signature: emoji + label + colon. The emoji is mandatory.
_WRAPPER_LABEL_PATTERN = rf"{_LABEL_EMOJI_PREFIX}\s*(?:{_WRAPPER_ALTERNATION})\s*:"
# Apparatus footer: emoji optional, but the label set is limited to real apparatus.
_APPARATUS_LABEL_PATTERN = rf"(?:{_LABEL_EMOJI_PREFIX}\s*)?(?:{_APPARATUS_ALTERNATION})\s*:"

# A bare apparatus label only starts a new segment when links or hashtags follow
# it; otherwise the sentence continues and must not be cut apart.
_APPARATUS_SPLIT_PATTERN = rf"{_APPARATUS_LABEL_PATTERN}(?=\s*{_APPARATUS_PAYLOAD_TOKEN})"

_LABEL_INLINE_RE = re.compile(
    rf"(?:{_WRAPPER_LABEL_PATTERN}|{_APPARATUS_SPLIT_PATTERN})",
    re.IGNORECASE,
)
_LABEL_PREFIX_RE = re.compile(rf"^{_WRAPPER_LABEL_PATTERN}\s*", re.IGNORECASE)

# A whole line that is an apparatus label followed by nothing but links and/or
# hashtags. Anything else after the label is editorial text and is kept.
_APPARATUS_LINE_RE = re.compile(
    rf"^{_APPARATUS_LABEL_PATTERN}\s*(?:{_APPARATUS_PAYLOAD_TOKEN}[\s,;]*)*$",
    re.IGNORECASE,
)


def _resegment_stored_body(text: str) -> str:
    """
    Restore line structure that `normalize_publication_text` collapsed away.

    `body_norm` is stored as a single line, so line-anchored cleanup (source
    lines, label-only headings, bullet prefixes) would silently do nothing on
    stored rows while working fine on a freshly generated post. Re-inserting a
    break before every wrapper label makes both paths behave identically.
    """
    value = str(text or "").replace("\r\n", "\n")
    return _LABEL_INLINE_RE.sub(lambda match: "\n" + match.group(0), value)


def _is_service_heading(line: str) -> bool:
    """
    True for short label-only lines such as 'Источник:', '## Приём' or '**Совет дня**'.

    Markdown heading and bullet marks are stripped first, so the same label is
    recognised whichever markup a rubric happens to wrap it in.
    """
    cleaned = _MARKDOWN_HEADING_RE.sub("", line)
    cleaned = _BULLET_PREFIX_RE.sub("", cleaned)
    cleaned = _BOLD_MARKS_RE.sub("", cleaned).strip().rstrip(":").strip().lower()
    if not cleaned or len(cleaned) > 32:
        return False
    return cleaned in _SERVICE_HEADING_WORDS


def extract_editorial_core(post_text: str) -> str:
    """
    Deterministically reduce a finished post to its editorial core.

    Removed: source lines, bare link lines, inline URLs, hashtags, markdown
    heading/emphasis marks and pure service headings.

    Kept: the practical action, the scenario it happens in and the observable
    child reaction — i.e. everything that makes two posts editorially the same
    or genuinely different. No LLM and no network access are involved.
    """
    lines_out: list[str] = []
    for raw_line in _resegment_stored_body(post_text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _APPARATUS_LINE_RE.match(line):
            continue
        if _LINK_LINE_RE.match(line):
            continue
        if _HASHTAG_ONLY_LINE_RE.match(line):
            continue
        if _is_service_heading(line):
            continue

        line = _MARKDOWN_HEADING_RE.sub("", line)
        line = _URL_INLINE_RE.sub(" ", line)
        line = _HASHTAG_TOKEN_RE.sub(" ", line)
        line = _BOLD_MARKS_RE.sub("", line)
        line = _BULLET_PREFIX_RE.sub("", line)
        # Drop the wrapper label, keep whatever it introduced on the same line.
        line = _LABEL_PREFIX_RE.sub("", line)
        line = " ".join(line.split()).strip()
        # Drop separator-only leftovers.
        if not line or not re.search(r"[A-Za-zА-Яа-яЁё0-9]", line):
            continue
        lines_out.append(line)

    return " ".join(" ".join(lines_out).split()).strip()


def semantic_editorial_core_threshold() -> float:
    """Threshold for the cross-rubric editorial-core freshness check."""
    return SEMANTIC_THRESHOLD_POST


# ---------------------------------------------------------------------------
# Soft source diversity / authority preferences
# ---------------------------------------------------------------------------


def normalize_domain(domain: str | None) -> str:
    """Normalize a source domain for stable comparisons."""
    value = (domain or "").strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/", 1)[0]
    return value[4:] if value.startswith("www.") else value


def is_recent_source_domain(domain: str | None, recent_domains) -> bool:
    """True when the domain is among the recently used ones."""
    normalized = normalize_domain(domain)
    if not normalized:
        return False
    return normalized in {normalize_domain(d) for d in (recent_domains or []) if d}


def is_scientific_domain(domain: str | None, scientific_domains) -> bool:
    """True when the domain is (or is a subdomain of) a configured authority."""
    normalized = normalize_domain(domain)
    if not normalized:
        return False
    for raw in scientific_domains or []:
        candidate = normalize_domain(raw)
        if not candidate:
            continue
        if normalized == candidate or normalized.endswith("." + candidate):
            return True
    return False


def source_diversity_sort_key(
    domain: str | None,
    *,
    recent_domains=(),
    scientific_domains=(),
    prefer_scientific: bool = False,
) -> tuple[int, int]:
    """
    Soft preference key: lower sorts first.

    Fresh domains outrank recently used ones, and for authority-driven rubrics a
    configured scientific domain outranks a non-scientific one. Both are strictly
    preferences — a recent or non-scientific domain still remains a usable
    fallback, it simply loses the tie.
    """
    recent_rank = 1 if is_recent_source_domain(domain, recent_domains) else 0
    if prefer_scientific:
        science_rank = 0 if is_scientific_domain(domain, scientific_domains) else 1
    else:
        science_rank = 0
    return (recent_rank, science_rank)


def should_prefer_scientific_sources(rubric_id: str | None) -> bool:
    """age_norms makes milestone claims, so authoritative sources sort first."""
    return normalize_rubric_id(rubric_id) == "age_norms"

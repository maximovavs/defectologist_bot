from __future__ import annotations
"""
Publisher (cron/GitHub Actions) v4.3.3-safe

Основа: production v4.3.1-safe
Минимальные безопасные улучшения:
1) Разделение test/prod history DB.
2) Soft skip / hard skip: лимит рубрики тратится только на hard skips.
3) Relevance guard для тематических хештегов.
4) Мягкий topic guard для age_norms.
5) Более умный выбор H1 для обложки.
6) Диагностический alert при Posted: 0.
7) Новый мягкий post-fit guard для Monday / tip_of_day.
8) Диагностика HTML fallback для Telegram send/caption.
9) Для method_piggybank source-level semantic dedup не блокирует кандидата:
  финальный body-level dedup остаётся активным.
"""

import asyncio
from io import BytesIO
import hashlib
import html as _html
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlparse
from src.publisher.dedup_policy import (
    EDITORIAL_CORE_COOLDOWN_DAYS,
    RECENT_SOURCE_DOMAIN_WINDOW,
    SEMANTIC_THRESHOLD_SOURCE,
    SOURCE_COOLDOWN_DAYS,
    extract_editorial_core,
    is_scientific_domain,
    semantic_editorial_core_threshold,
    semantic_post_threshold_for_rubric,
    should_bypass_duplicate_reason,
    should_bypass_source_semantic_dedup,
    should_prefer_scientific_sources,
    source_diversity_sort_key,
)

import feedparser
import requests
import urllib3
import yaml
from bs4 import BeautifulSoup
from dateutil import tz

from src.services.llm_generator import (
    _validate_question_week_output,
    gemini_text_provider_status,
    generate_image_prompt_async,
    generate_post_plain_from_evidence_async,
    validate_myth_fact_evidence_for_generation,
    validate_pro_evidence_for_generation,
)
from src.services.publication_store import PublicationStore
from src.services.poll_builder import PollSpec, build_poll_spec
from src.services.engagement_builder import (
    EngagementSpec,
    append_engagement_footer,
    build_engagement_spec,
)
from src.services.topic_policy import (
    RUBRIC_TOPIC_ROTATION,
    TOPIC_HASHTAGS,
    TOPICS,
    detect_evidence_topics,
    rank_candidates_for_topic,
    select_topic_plan,
    topic_matches_text,
)
from src.services.visual_pipeline import build_post_visual


ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/4.3.2-safe (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "").strip().lower()
POLLINATIONS_TOKEN = os.getenv("POLLINATIONS_TOKEN", "").strip()

DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
POST_POLLS_ENABLED = os.getenv("POST_POLLS_ENABLED", "0").strip().lower() in ("1", "true", "yes")
_POST_ENGAGEMENT_MODE_RAW = os.getenv("POST_ENGAGEMENT_MODE", "").strip().lower()
if _POST_ENGAGEMENT_MODE_RAW in {"auto", "polls_only", "off"}:
    POST_ENGAGEMENT_MODE = _POST_ENGAGEMENT_MODE_RAW
elif "POST_POLLS_ENABLED" in os.environ:
    POST_ENGAGEMENT_MODE = "polls_only" if POST_POLLS_ENABLED else "off"
else:
    POST_ENGAGEMENT_MODE = "auto"
POST_TOPIC_ID = os.getenv("POST_TOPIC_ID", "auto").strip().lower() or "auto"
POLL_OPEN_PERIOD_SECONDS = int(os.getenv("POLL_OPEN_PERIOD_SECONDS", "86400"))
POLL_DISABLE_NOTIFICATION = os.getenv("POLL_DISABLE_NOTIFICATION", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()

PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_VISUAL_QA_API_KEY = os.getenv("GEMINI_VISUAL_QA_API_KEY", "").strip()


def _normalize_selected_rubric(raw: str) -> str:
    value = (raw or "").strip()
    if not value or value.lower() == "auto":
        return ""
    if "|" in value:
        value = value.split("|", 1)[0].strip()
    return value.lower()


def _parse_csv_env(raw: str) -> List[str]:
    out: List[str] = []
    for part in (raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        if item not in out:
            out.append(item)
    return out


AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()
RUBRIC_ID = _normalize_selected_rubric(os.getenv("RUBRIC_ID", ""))
INCLUDE_SOURCES = _parse_csv_env(os.getenv("INCLUDE_SOURCES", ""))
EXCLUDE_SOURCES = _parse_csv_env(os.getenv("EXCLUDE_SOURCES", ""))
RESET_TEST_DB = os.getenv("RESET_TEST_DB", "").strip().lower() in ("1", "true", "yes")

POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1000"))
if "TG_CAPTION_MAX_UTF16_UNITS" in os.environ:
    TG_CAPTION_MAX_UTF16_UNITS = int(os.getenv("TG_CAPTION_MAX_UTF16_UNITS", "1000"))
elif "TG_CAPTION_MAX_BYTES" in os.environ:
    TG_CAPTION_MAX_UTF16_UNITS = int(os.getenv("TG_CAPTION_MAX_BYTES", "1000"))
    print("[WARN] TG_CAPTION_MAX_BYTES is deprecated; treating value as UTF-16 caption units", flush=True)
else:
    TG_CAPTION_MAX_UTF16_UNITS = 1000
# Kept only for import compatibility; caption decisions use UTF-16 units.
TG_CAPTION_MAX_BYTES = TG_CAPTION_MAX_UTF16_UNITS
IMAGE_PROMPT_TIMEOUT_SECONDS = int(os.getenv("IMAGE_PROMPT_TIMEOUT_SECONDS", "60"))

SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.95"))
RECENT_ALERT_HOURS = int(os.getenv("RECENT_ALERT_HOURS", "36"))

MAX_RUN_SECONDS = int(os.getenv("MAX_RUN_SECONDS", "1500"))
MAX_CANDIDATES_PER_RUBRIC = int(os.getenv("MAX_CANDIDATES_PER_RUBRIC", "25"))
MAX_SKIPS_PER_RUBRIC = int(os.getenv("MAX_SKIPS_PER_RUBRIC", "12"))
MAX_LLM_SECONDS_PER_CANDIDATE = int(os.getenv("MAX_LLM_SECONDS_PER_CANDIDATE", "180"))

INSECURE_TLS_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("INSECURE_TLS_DOMAINS", "") or "").split(",")
    if d.strip()
]

if os.getenv("SUPPRESS_INSECURE_TLS_WARNINGS", "1").strip().lower() in ("1", "true", "yes"):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SECTION_HEADERS = {
    "Введение",
    "Методы",
    "Главные выводы",
    "Практическое применение",
    "Источник",
}

GAME_HEADING_RE = re.compile(r"^🎲\s*Как играть\s*:?\s*$", re.IGNORECASE)
TRY_TODAY_HEADING_RE = re.compile(r"^🧩\s*Что попробовать сегодня\s*:?\s*$", re.IGNORECASE)
BILINGUAL_HEADING_RE = re.compile(r"^🌍\s*Что помогает в двуязычной семье\s*:?\s*$", re.IGNORECASE)
HOME_HEADING_RE = re.compile(r"^🏠\s*Что можно попробовать дома\s*:?\s*$", re.IGNORECASE)
EXAMPLE_HEADING_RE = re.compile(r"^👄\s*Пример\s*:?\s*$", re.IGNORECASE)
BENEFIT_HEADING_RE = re.compile(r"^💡\s*Что это да[её]т\s*:?\s*$", re.IGNORECASE)

AGE_LINE_RE = re.compile(r"^👶\s*Возраст\s*:\s*.+\S$", re.IGNORECASE)
AUDIENCE_LINE_RE = re.compile(r"^👩‍⚕️\s*Аудитория\s*:\s*.+\S$", re.IGNORECASE)
SOURCE_LINE_RE = re.compile(r"^Источник:\s*\S.+$", re.IGNORECASE)
BENEFIT_LINE_RE = re.compile(r"^💡\s*Что это да[её]т\s*:\s*.+\S$", re.IGNORECASE)
MYTH_LINE_RE = re.compile(r"^🔴\s*Миф\s*:\s*.+\S$", re.IGNORECASE)
QUESTION_LINE_RE = re.compile(r"^❓\s*Вопрос недели\s*:\s*.+\S$", re.IGNORECASE)
ORIENTIRS_LINE_RE = re.compile(r"^Ориентиры:\s*.+\S$", re.IGNORECASE)

HASHTAG_TOKEN_RE = re.compile(r"(?<!\w)#([A-Za-zА-Яа-яЁё0-9_]+)")

RUBRIC_TAGS_BY_DAY = {
    "MO": "#совет_логопеда",
    "TU": "#играем_и_говорим",
    "WE": "#миф_факт",
    "TH": "#речь_в_разных_ситуациях",
    "FR": "#вопрос_недели",
    "SA": "#методическая_копилка",
    "SU": "#возрастная_норма",
}

MYTH_FACT_CANONICAL_SOURCE_IDS = frozenset({
    "healthychildren_bilingual_myths",
    "asha_speech_sound_multilingual_influence",
    "asha_newborn_hearing_screening",
    "healthychildren_one_year_talking",
    "healthychildren_crawling_reading_myth",
})

POLICY_OWNERSHIP_REASONS = frozenset({
    "myth_claim_not_grounded",
    "parent_age_not_grounded",
    "parent_modality_not_grounded",
    "parent_diagnostic_role_violation",
    "parent_false_hearing_inference",
    "parent_risky_oral_manipulation",
    "exercise_coherence_violation",
    "parent_professional_role_violation",
})

PUBLISHER_DIAGNOSTIC_STAGES = (
    "url_cooldown",
    "evidence",
    "source_authority",
    "pre_llm",
    "llm_validation",
)

LEGACY_VALIDATION_NOTE_ALIASES = {
    "groq_failed_after_modality_repair:": "parent_modality_not_grounded",
    "groq_failed_after_diagnostic_repair:": "parent_diagnostic_role_violation",
}

QUOTA_SKIP_REASONS = (
    "gemini_quota_exhausted_cached",
    "gemini_quota_exhausted",
)

SOFT_SKIP_REASONS = {
    "bad_candidate_url",
    "skip_non_html_asset",
    "dup_url_same_run",
    "dup_url_db",
    "dup_url_recent",
    "no_evidence_short",
    "dup_evidence_same_run",
    "dup_evidence_hash_db",
    "dup_evidence_hash_recent",
    "dup_semantic_source",
    "dup_editorial_core_recent",
    "dup_body_same_run",
    "dup_body_hash_db",
    "dup_semantic_post",
    "rubric_topic_mismatch_source",
    "rubric_topic_mismatch_post",
    "source_authority_required",
    "myth_evidence_missing_refutation_anchor",
    "myth_missing_claim",
    "myth_topic_mismatch",
    "myth_unsupported_sensitive_claim",
    "myth_unsupported_numeric_detail",
    "myth_unsupported_phoneme_detail",
    "tip_of_day_post_too_generic",
    "unsupported_mechanism_claim",
    "pro_unsupported_concrete_detail",
    "pro_unsupported_numeric_detail",
    "no_data_in_source",
    "gemini_quota_exhausted",
    "gemini_quota_exhausted_cached",
    "empty",
    "too_short",
    "template_leak",
    "pro_insufficient_evidence",
    "pro_empty",
    "pro_title_too_long",
    "pro_markdown_or_template_leak",
    "pro_generic_benefit",
    "pro_missing_goal",
    "pro_missing_materials",
    "pro_missing_steps",
    "pro_missing_observation_criterion",
    "pro_unsupported_observation_claim",
    "pro_too_abstract",
    "pro_old_academic_structure",
    "pro_risky_manual_technique",
    "pro_missing_method_card_heading",
    "bilingual_topic_mismatch",
    "bilingual_missing_family_action",
    "bilingual_false_causality",
    "bilingual_unsupported_mechanism",
    "thematic_topic_mismatch",
    "thematic_missing_home_action",
    "thematic_unsupported_mechanism",
    "thematic_missing_heading",
    "thematic_nonobservable_benefit",
    "parent_ambiguous_latin_phoneme",
    "parent_age_range_too_broad",
    "parent_age_action_mismatch",
    "parent_nonobservable_benefit",
    "parent_cross_language_sound_norm",
    "parent_too_many_numbered_steps",
    "missing_parent_safety_note",
    "blanket_reassurance",
    "misleading_politeness_framing",
    "visual_prompt_topic_mismatch",
    "unknown_source_id",
    "llm_invalid_output",
    "final_invalid_output",
    "no_candidates",
    "max_skips_per_rubric",
    *POLICY_OWNERSHIP_REASONS,
}
HARD_SKIP_REASONS = {
    "source_fetch_failed",
    "evidence_fetch_failed",
    "llm_timeout",
    "llm_failed",
    "visual_build_failed",
    "telegram_send_failed",
    "max_run_seconds",
}

VALIDATION_SKIP_REASONS = {
    "no_data_in_source",
    "myth_evidence_missing_refutation_anchor",
    "myth_missing_claim",
    "myth_topic_mismatch",
    "myth_unsupported_sensitive_claim",
    "myth_unsupported_numeric_detail",
    "myth_unsupported_phoneme_detail",
    "empty",
    "too_short",
    "template_leak",
    "missing_parent_safety_note",
    "blanket_reassurance",
    "misleading_politeness_framing",
    "bilingual_topic_mismatch",
    "bilingual_missing_family_action",
    "bilingual_false_causality",
    "bilingual_unsupported_mechanism",
    "thematic_topic_mismatch",
    "thematic_missing_home_action",
    "thematic_unsupported_mechanism",
    "thematic_missing_heading",
    "thematic_nonobservable_benefit",
    "parent_ambiguous_latin_phoneme",
    "parent_age_range_too_broad",
    "parent_age_action_mismatch",
    "parent_nonobservable_benefit",
    "parent_cross_language_sound_norm",
    "parent_too_many_numbered_steps",
    "pro_insufficient_evidence",
    "pro_empty",
    "pro_title_too_long",
    "pro_markdown_or_template_leak",
    "pro_generic_benefit",
    "pro_missing_goal",
    "pro_missing_materials",
    "pro_missing_steps",
    "pro_missing_observation_criterion",
    "pro_unsupported_observation_claim",
    "pro_too_abstract",
    "pro_old_academic_structure",
    "pro_risky_manual_technique",
    "pro_missing_method_card_heading",
    *POLICY_OWNERSHIP_REASONS,
}

VALIDATION_SKIP_PREFIXES = (
    "banned_phrase",
    "unsupported_mechanism_claim",
    "pro_unsupported_concrete_detail",
    "pro_unsupported_numeric_detail",
)

AGE_NORMS_BAD_MARKERS = [
    "дисграф",
    "дислекс",
    "заикан",
    "алал",
    "афаз",
    "дизартр",
    "ринолал",
    "коррекц",
    "нарушени",
    "дефект",
    "диагноз",
    "патолог",
]
AGE_NORMS_GOOD_MARKERS = [
    "возраст",
    "норма",
    "ориентир",
    "milestone",
    "development",
    "communication",
    "речевое развитие",
    "что умеет",
    "что обычно",
]
AGE_NORMS_SOURCE_GOOD_MARKERS = [
    "milestone",
    "developmental milestone",
    "communication milestone",
    "возрастн",
    "ориентир",
    "месяц",
    "months",
    "years old",
    "by age",
]
AGE_NORMS_SOURCE_BAD_MARKERS = [
    "speech delay",
    "language delay",
    "late language emergence",
    "speech disorder",
    "language disorder",
    "communication disorder",
    "hearing loss",
    "diagnos",
    "патолог",
    "диагноз",
    "задержк речи",
    "задержк язык",
    "нарушени речи",
]
DEVELOPMENTAL_RISK_MARKERS = [
    "speech delay",
    "language delay",
    "late language emergence",
    "regression",
    "lost skills",
    "loss of skills",
    "stopped talking",
    "stops talking",
    "does not understand speech",
    "doesn't understand speech",
    "hearing loss",
    "hearing screening",
    "speech disorder",
    "language disorder",
    "communication disorder",
    "diagnos",
    "задержк речи",
    "задержк язык",
    "регресс",
    "потерял навык",
    "потеря навык",
    "перестал говор",
    "не понимает речь",
    "снижение слух",
    "потеря слух",
    "проверка слух",
    "диагноз",
    "диагност",
    "нарушени речи",
]
TIP_OF_DAY_BAD_MARKERS = [
    "совет логопеда дня",
    "сегодня работаем над",
    "сегодня поговорим",
    "развитие речи",
    "общее недоразвитие речи",
    "онр",
    "дизартр",
    "алали",
    "дисграф",
    "дислекс",
    "диагноз",
    "коррекц",
]
TIP_OF_DAY_GOOD_MARKERS = [
    "👶 возраст:",
    "🧩 что попробовать сегодня",
    "👄 пример",
    "💡 что это дает",
]

_SKIP_EXT_RE = re.compile(
    r"\.(ppt|pptx|pdf|doc|docx|xls|xlsx|zip|rar|mp3|mp4)$",
    re.IGNORECASE,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def log_field(value: object, max_len: int = 220) -> str:
    text = norm_space(str(value or "")).replace('"', "'")
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_local_now(tzname: str) -> datetime:
    return datetime.now(tz=tz.gettz(tzname))


def iso_week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def weekday_key(dt: datetime) -> str:
    return ["MO", "TU", "WE", "TH", "FR", "SA", "SU"][dt.weekday()]


def is_due(rubric: Dict[str, Any], now: datetime) -> bool:
    cadence = (rubric.get("cadence") or "DAILY").upper()
    byweekday = rubric.get("byweekday") or []
    if byweekday and weekday_key(now) not in set(byweekday):
        return False
    return cadence in ("DAILY", "WEEKLY")


def safe_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _verify_for_url(url: str) -> bool:
    dom = safe_domain(url)
    if not dom:
        return True
    for bad in INSECURE_TLS_DOMAINS:
        if dom == bad or dom.endswith("." + bad):
            return False
    return True


def _escape(s: str) -> str:
    return _html.escape(s or "", quote=False)


def _start_recent_window(now: datetime) -> datetime:
    return now - timedelta(hours=RECENT_ALERT_HOURS)


def _resolve_publish_chat_id() -> str:
    return TELEGRAM_DRAFTS_CHAT_ID if TARGET_CHANNEL == "test" else TELEGRAM_CHAT_ID


def _resolve_state_scope() -> str:
    return "test" if TARGET_CHANNEL == "test" else "prod"


def _resolve_publication_db_path() -> Path:
    return STATE_DIR / (
        "publication_history_test.sqlite3"
        if _resolve_state_scope() == "test"
        else "publication_history.sqlite3"
    )


def _skip_kind(reason: str) -> str:
    return "hard" if reason in HARD_SKIP_REASONS else "soft"


def _canonical_validation_candidate(candidate: str) -> str:
    candidate = (candidate or "").strip()
    if not candidate:
        return ""
    if "=" in candidate:
        candidate = candidate.split("=", 1)[1].strip()
    if candidate.startswith("p2d_fail_closed:"):
        parts = candidate.split(":", 2)
        candidate = parts[1].strip() if len(parts) >= 2 else ""
    elif candidate.startswith("invalid_") and ":" in candidate:
        candidate = candidate.split(":", 1)[1].strip()
    if candidate in VALIDATION_SKIP_REASONS:
        return candidate
    if any(candidate.startswith(prefix + ":") for prefix in VALIDATION_SKIP_PREFIXES):
        return candidate
    return ""


def _extract_validation_skip_reason(llm_note: str) -> str:
    note = norm_space(llm_note)
    if not note:
        return ""

    candidates: List[str] = []
    candidates.extend(re.findall(r"p2d_fail_closed:[^|]+", note))
    candidates.extend(re.findall(r"invalid_(?:groq_retry|groq|gemini_retry|gemini):([^|]+)", note))
    candidates.extend(re.split(r"\s*\|\s*", note))

    for candidate in candidates:
        canonical = _canonical_validation_candidate(candidate)
        if canonical:
            return canonical

    for segment in re.split(r"\s*\|\s*", note):
        probe = segment.strip()
        if "=" in probe:
            probe = probe.split("=", 1)[1].strip()
        for prefix, canonical in LEGACY_VALIDATION_NOTE_ALIASES.items():
            if probe.startswith(prefix):
                return canonical
    return ""


def _resolve_llm_skip(llm_note: str) -> tuple[str, str]:
    validation_reason = _extract_validation_skip_reason(llm_note)
    if validation_reason:
        return validation_reason, "llm_validation"
    note = llm_note or ""
    for reason in QUOTA_SKIP_REASONS:
        if reason in note:
            return reason, ""
    return "llm_invalid_output", ""


def _extract_pro_validation_skip_reason(llm_note: str) -> str:
    """Backward-compatible alias for callers using the former pro-only helper."""
    return _extract_validation_skip_reason(llm_note)


def _record_skip(
    reason: str,
    url: str,
    soft_skip_reasons: Dict[str, int],
    hard_skip_reasons: Dict[str, int],
    samples: List[str],
    *,
    stage_skip_reasons: Optional[Dict[str, Dict[str, int]]] = None,
    stage: str = "",
) -> str:
    kind = _skip_kind(reason)
    target = hard_skip_reasons if kind == "hard" else soft_skip_reasons
    target[reason] = target.get(reason, 0) + 1
    if stage and stage in PUBLISHER_DIAGNOSTIC_STAGES and stage_skip_reasons is not None:
        bucket = stage_skip_reasons.setdefault(stage, {})
        bucket[reason] = bucket.get(reason, 0) + 1
    if len(samples) < 10:
        samples.append(f"[{kind}] {reason}: {url}")
    return kind


def _build_posted_zero_alert_plain(
    now: datetime,
    day: str,
    week_key: str,
    audience: str,
    provider: str,
    soft_skip_reasons: Dict[str, int],
    hard_skip_reasons: Dict[str, int],
    samples: List[str],
    state_scope: str,
    db_name: str,
    attempted_rubrics: List[str],
    topic_preference: str = "auto",
    stage_skip_reasons: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    soft_total = sum(soft_skip_reasons.values())
    hard_total = sum(hard_skip_reasons.values())
    soft_top = sorted(soft_skip_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
    hard_top = sorted(hard_skip_reasons.items(), key=lambda x: x[1], reverse=True)[:10]

    parts: List[str] = [
        "⚠️ Publisher diagnostic: пост не опубликован (Posted: 0)",
        f"Дата: {now.date()} | День: {day} | Неделя: {week_key}",
        f"AUDIENCE={audience} | PROVIDER={provider} | TARGET_CHANNEL={TARGET_CHANNEL}",
        f"STATE_SCOPE={state_scope} | History DB={db_name}",
        f"Rubrics attempted: {', '.join(attempted_rubrics) or '—'}",
        f"Topic preference: {topic_preference or 'auto'}",
        f"Soft skips: {soft_total} | Hard skips: {hard_total}",
    ]

    if hard_top:
        parts.extend(["", "Hard skip reasons:"])
        parts.extend([f"• {reason}: {count}" for reason, count in hard_top])

    if soft_top:
        parts.extend(["", "Soft skip reasons:"])
        parts.extend([f"• {reason}: {count}" for reason, count in soft_top])

    stage_lines: List[str] = []
    for stage in PUBLISHER_DIAGNOSTIC_STAGES:
        counts = (stage_skip_reasons or {}).get(stage, {})
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            stage_lines.append(f"• {stage} | {reason}: {count}")
    if stage_lines:
        parts.extend(["", "Skip stage attribution:"])
        parts.extend(stage_lines)

    visual_qa_status = (
        "separate_key"
        if GEMINI_VISUAL_QA_API_KEY
        else "shared_key"
        if GEMINI_API_KEY
        else "unavailable"
    )
    parts.extend(
        [
            "",
            "Provider availability:",
            f"• groq: {'available' if GROQ_API_KEY else 'unavailable'}",
            f"• gemini_text: {gemini_text_provider_status(GEMINI_API_KEY)}",
            f"• gemini_visual_qa: {visual_qa_status}",
        ]
    )

    if samples:
        parts.extend(["", "Examples:"])
        parts.extend(samples[:10])

    return "\n".join(parts)


def _production_slots_already_fulfilled(
    *,
    db_path: Path,
    attempted_rubrics: Sequence[str],
    now: datetime,
    state_scope: str,
    target_channel: str,
    dry_run: bool,
) -> bool:
    if dry_run or state_scope != "prod" or target_channel != "prod":
        return False
    if db_path.name != "publication_history.sqlite3" or db_path.parent.name != ".state":
        return False
    if now.tzinfo is None or now.utcoffset() is None or not db_path.is_file():
        return False

    required_rubrics = {
        (rubric_id or "").strip().lower()
        for rubric_id in attempted_rubrics
        if (rubric_id or "").strip()
    }
    if not required_rubrics:
        return False

    placeholders = ", ".join("?" for _ in required_rubrics)
    sql = (
        "SELECT rubric_id, posted_at FROM publications "
        f"WHERE lower(trim(rubric_id)) IN ({placeholders})"
    )
    try:
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute(sql, sorted(required_rubrics)).fetchall()
    except Exception:
        return False

    fulfilled_rubrics: set[str] = set()
    for rubric_id, posted_at in rows:
        rubric = (rubric_id or "").strip().lower()
        try:
            posted = datetime.fromisoformat((posted_at or "").strip())
            if posted.tzinfo is None or posted.utcoffset() is None:
                continue
            if posted.astimezone(now.tzinfo).date() == now.date():
                fulfilled_rubrics.add(rubric)
        except (TypeError, ValueError, OverflowError):
            continue

    return required_rubrics.issubset(fulfilled_rubrics)


def _send_posted_zero_alert_if_needed(
    *,
    db_path: Path,
    now: datetime,
    day: str,
    week_key: str,
    audience: str,
    provider: str,
    soft_skip_reasons: Dict[str, int],
    hard_skip_reasons: Dict[str, int],
    samples: List[str],
    state_scope: str,
    attempted_rubrics: List[str],
    topic_preference: str,
    stage_skip_reasons: Optional[Dict[str, Dict[str, int]]] = None,
) -> bool:
    if _production_slots_already_fulfilled(
        db_path=db_path,
        attempted_rubrics=attempted_rubrics,
        now=now,
        state_scope=state_scope,
        target_channel=TARGET_CHANNEL,
        dry_run=DRY_RUN,
    ):
        rubrics = ",".join(
            sorted({rubric.strip().lower() for rubric in attempted_rubrics if rubric.strip()})
        )
        print(
            "[INFO] posted_zero_alert_suppressed "
            "reason=production_slot_already_fulfilled "
            f"date={now.date()} rubrics={rubrics}",
            flush=True,
        )
        return False

    if not TELEGRAM_DRAFTS_CHAT_ID:
        print("[WARN] Posted:0 but TELEGRAM_DRAFTS_CHAT_ID not set; no alert sent.", flush=True)
        return False

    try:
        send_plain_message(
            TELEGRAM_DRAFTS_CHAT_ID,
            _build_posted_zero_alert_plain(
                now=now,
                day=day,
                week_key=week_key,
                audience=audience,
                provider=provider,
                soft_skip_reasons=soft_skip_reasons,
                hard_skip_reasons=hard_skip_reasons,
                samples=samples,
                state_scope=state_scope,
                db_name=db_path.name,
                attempted_rubrics=attempted_rubrics,
                topic_preference=topic_preference,
                stage_skip_reasons=stage_skip_reasons,
            ),
        )
    except Exception as e:
        print(f"[WARN] failed_to_send_posted_zero_alert err={e}", flush=True)
        return False
    return True


def _strip_html_tags_for_telegram(text: str) -> str:
    s = text or ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(
        r'<a\s+href="([^"]+)">(.+?)</a>',
        lambda m: f"{_html.unescape(m.group(2))} ({_html.unescape(m.group(1))})",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(
        r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre)>",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"<[^>]+>", "", s)
    return _html.unescape(s).strip()


def _is_probably_parse_mode_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "can't parse entities" in text
        or "unsupported start tag" in text
        or "bad request" in text
    )


def _safe_html_preview(text: str, limit: int = 400) -> str:
    preview = (text or "").replace("\n", "\\n")
    return (preview[:limit].rstrip() + "…") if len(preview) > limit else preview


def _log_telegram_html_fallback(context: str, html_text: str, exc: Exception) -> None:
    print(
        f"[WARN] telegram_html_fallback context={context} err={exc} html_preview={_safe_html_preview(html_text)}",
        flush=True,
    )


def _line_matches_structural(st: str) -> bool:
    return any(
        (
            AGE_LINE_RE.match(st),
            AUDIENCE_LINE_RE.match(st),
            SOURCE_LINE_RE.match(st),
            BENEFIT_LINE_RE.match(st),
            MYTH_LINE_RE.match(st),
            QUESTION_LINE_RE.match(st),
            ORIENTIRS_LINE_RE.match(st),
            GAME_HEADING_RE.match(st),
            TRY_TODAY_HEADING_RE.match(st),
            BILINGUAL_HEADING_RE.match(st),
            HOME_HEADING_RE.match(st),
            EXAMPLE_HEADING_RE.match(st),
            BENEFIT_HEADING_RE.match(st),
        )
    )


def _is_structural_heading(line: str) -> bool:
    st = (line or "").strip()
    return bool(st and (_line_matches_structural(st) or st in SECTION_HEADERS))


def _slugify_tag_body(text: str) -> str:
    s = (text or "").strip().lower().replace("ё", "е")
    s = re.sub(r"[-–—−]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zа-я_]+", "_", s, flags=re.IGNORECASE)
    return re.sub(r"_+", "_", s).strip("_")


def _extract_age_value(lines: List[str]) -> str:
    for line in lines:
        st = line.strip()
        if AGE_LINE_RE.match(st):
            return st.split(":", 1)[1].strip()
    return ""


AGE_TAG_ALIASES = {
    "дошкольный": "#для_дошкольников",
    "дошкольный возраст": "#для_дошкольников",
    "дошкольники": "#для_дошкольников",
    "для дошкольников": "#для_дошкольников",
    "младший дошкольный возраст": "#для_младших_дошкольников",
    "младшие дошкольники": "#для_младших_дошкольников",
    "старший дошкольный возраст": "#для_старших_дошкольников",
    "старшие дошкольники": "#для_старших_дошкольников",
    "школьный возраст": "#для_школьников",
    "школьники": "#для_школьников",
    "ранний возраст": "#для_детей_раннего_возраста",
}


def _normalize_age_alias_value(age_value: str) -> str:
    value = norm_space(age_value).strip().lower().replace("ё", "е")
    return re.sub(r"[\s.,;:!?…]+$", "", value).strip()


def _build_age_tag(age_value: str) -> str:
    alias = AGE_TAG_ALIASES.get(_normalize_age_alias_value(age_value))
    if alias:
        return alias

    value = _slugify_tag_body(age_value)
    if not value:
        return ""
    return f"#{value}" if value.startswith("для_детей_") else f"#для_детей_{value}"


def _extract_thematic_tags_and_clean_lines(lines: List[str]) -> tuple[List[str], List[str]]:
    tags: List[str] = []
    clean_lines: List[str] = []

    for line in lines:
        st = line.strip()
        if not st:
            clean_lines.append(line)
            continue

        if st.startswith("#"):
            for raw in HASHTAG_TOKEN_RE.findall(st):
                tag = f"#{raw.lower()}"
                if tag not in tags:
                    tags.append(tag)
            continue

        clean_lines.append(line)

    return tags[:2], clean_lines


def _normalize_tag_token(token: str) -> str:
    t = (token or "").strip().lower().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", "", t, flags=re.IGNORECASE)


def _body_supports_tag(tag: str, body_text: str) -> bool:
    if not tag or not body_text:
        return False
    tag_body = tag.lstrip("#").replace("_", " ").replace("-", " ")
    tokens = [_normalize_tag_token(x) for x in tag_body.split() if _normalize_tag_token(x)]
    body_norm = _normalize_tag_token(body_text)
    return sum(1 for tok in tokens if len(tok) >= 4 and tok in body_norm) >= 1


CONTROLLED_THEMATIC_TAGS: List[tuple[List[str], str]] = [
    (["фонемат", "различает звук"], "#фонематический_слух"),
    (["артикуляц", "положение языка", "положение губ"], "#артикуляция"),
    (["фраз", "два слова", "просьба"], "#фразовая_речь"),
    (["словар", "название предметов"], "#словарный_запас"),
    (["билингв", "двуязыч", "два языка"], "#билингвизм"),
    (["слог"], "#слоговая_структура"),
    (["связн", "рассказ", "пересказ"], "#связная_речь"),
    (["понимание речи", "выполняет просьбу"], "#понимание_речи"),
]


def _controlled_thematic_tag(
    body_text: str,
    day_key: str = "",
    rubric_id: str = "",
    topic_id: str = "",
) -> str:
    blob = (body_text or "").lower().replace("ё", "е")
    dk = (day_key or "").strip().upper()
    rubric = (rubric_id or "").strip().lower()
    effective_topic = (topic_id or "").strip().lower()
    if effective_topic in TOPIC_HASHTAGS and topic_matches_text(body_text, effective_topic):
        return TOPIC_HASHTAGS[effective_topic]
    bilingual_markers = ["билингв", "двуязыч", "два языка", "двух язык", "домашний язык"]
    if (rubric in {"bilingual_corner", "bilingual_parents"} or dk == "TH") and any(
        marker in blob for marker in bilingual_markers
    ):
        return "#билингвизм"
    for markers, tag in CONTROLLED_THEMATIC_TAGS:
        if any(marker in blob for marker in markers):
            return tag
    return ""


def _filter_relevant_thematic_tags(
    tags: List[str],
    body_text: str,
    day_key: str = "",
    rubric_id: str = "",
    topic_id: str = "",
) -> List[str]:
    controlled = _controlled_thematic_tag(
        body_text,
        day_key=day_key,
        rubric_id=rubric_id,
        topic_id=topic_id,
    )
    return [controlled] if controlled else []


def _extract_source_line(lines: List[str], fallback_domain: str) -> str:
    for line in lines:
        st = line.strip()
        if SOURCE_LINE_RE.match(st):
            return st
    return f"Источник: {fallback_domain}"


def _extract_link_line(lines: List[str], fallback_url: str) -> str:
    for line in lines:
        st = line.strip()
        if st.startswith("🔗 "):
            return st
    return f"🔗 {fallback_url}"


def _remove_footer_lines(lines: List[str]) -> List[str]:
    cleaned = [
        line
        for line in lines
        if not SOURCE_LINE_RE.match(line.strip()) and not line.strip().startswith("🔗 ")
    ]
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return cleaned


def _body_without_footer(plain_text: str) -> str:
    lines = (plain_text or "").replace("\r\n", "\n").split("\n")
    body_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if SOURCE_LINE_RE.match(stripped) or stripped.startswith("🔗 "):
            break
        if stripped.startswith("#"):
            break
        body_lines.append(line)

    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    return "\n".join(body_lines).strip()


def _looks_incomplete_final_body(body_text: str) -> bool:
    lines = [line.strip() for line in (body_text or "").replace("\r\n", "\n").split("\n") if line.strip()]
    if not lines:
        return True

    last = lines[-1].strip()
    if not last:
        return True

    if last.endswith(("...", "…", ",", ";", ":", "-", "—")):
        return True

    if last.count("«") != last.count("»"):
        return True

    if last.count("(") != last.count(")"):
        return True

    if re.search(r"[«\"“][^»\"”]{0,30}$", last):
        return True

    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", last)
    if words and len(last) < 20:
        return True

    if re.search(r"[A-Za-zА-Яа-яЁё]$", last) and not re.search(r"[.!?…»)]$", last):
        return True

    return False


def _trim_body_preserving_footer(body_text: str, footer_text: str, max_chars: int) -> str:
    body = (body_text or "").strip()
    footer = (footer_text or "").strip()

    if not body:
        return ""

    if not footer:
        if len(body) <= max_chars and not _looks_incomplete_final_body(body):
            return body
        return ""

    composed = f"{body}\n\n{footer}"
    if len(composed) <= max_chars and not _looks_incomplete_final_body(body):
        return body

    body_lines = body.split("\n")
    while body_lines:
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        candidate = "\n".join(body_lines).strip()
        if not candidate:
            return ""

        composed = f"{candidate}\n\n{footer}"
        if len(composed) <= max_chars and not _looks_incomplete_final_body(candidate):
            return candidate

        body_lines.pop()

    return ""


def finalize_plain_post_for_publication(
    plain_text: str,
    day_key: str,
    source_domain: str,
    source_url: str,
    max_chars: int,
    rubric_id: str = "",
    topic_id: str = "",
) -> str:
    raw_lines = (plain_text or "").replace("\r\n", "\n").split("\n")
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    thematic_tags, no_tag_lines = _extract_thematic_tags_and_clean_lines(raw_lines)
    source_line = _extract_source_line(no_tag_lines, source_domain)
    link_line = _extract_link_line(no_tag_lines, source_url)
    body_lines = _remove_footer_lines(no_tag_lines)

    age_value = _extract_age_value(body_lines)
    rubric_tag = RUBRIC_TAGS_BY_DAY.get((day_key or "").upper(), "")
    age_tag = _build_age_tag(age_value)

    body_text = "\n".join(body_lines).strip()
    thematic_tags = _filter_relevant_thematic_tags(
        thematic_tags,
        body_text,
        day_key=day_key,
        rubric_id=rubric_id,
        topic_id=topic_id,
    )

    final_tags: List[str] = []
    for tag in [rubric_tag, age_tag, *thematic_tags]:
        tag = (tag or "").strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        if tag not in final_tags:
            final_tags.append(tag)

    footer_parts = [source_line, link_line]
    if final_tags:
        footer_parts += ["", " ".join(final_tags)]
    footer_text = "\n".join(footer_parts).strip()

    trimmed_body = _trim_body_preserving_footer(body_text, footer_text, max_chars)
    if trimmed_body and _looks_incomplete_final_body(trimmed_body):
        body_lines = trimmed_body.split("\n")
        while body_lines:
            body_lines.pop()
            candidate = "\n".join(body_lines).strip()
            if candidate and not _looks_incomplete_final_body(candidate):
                trimmed_body = candidate
                break
        else:
            trimmed_body = ""

    return f"{trimmed_body}\n\n{footer_text}".strip() if trimmed_body else footer_text


def _normalize_title_probe(text: str) -> str:
    return norm_space(text).replace("ё", "е").lower()


def _strip_question_prefix(line: str) -> str:
    st = (line or "").strip()
    st = re.sub(r"^❓\s*", "", st)
    st = re.sub(r"^вопрос недели\s*:\s*", "", st, flags=re.IGNORECASE)
    st = re.sub(r"^вопрос\s*:\s*", "", st, flags=re.IGNORECASE)
    return st.strip()


def _first_sentence(text: str, max_len: int = 90) -> str:
    s = norm_space(text)
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", s)
    first = parts[0].strip() if parts else s
    if len(first) <= max_len:
        return first
    cut = first[:max_len].rstrip(" ,;:-")
    if " " in cut:
        cut = cut[:cut.rfind(" ")].rstrip(" ,;:-")
    return cut + "…"


def _extract_cover_title_from_plain_post(
    plain_text: str,
    fallback: str,
    rubric_title: str = "",
) -> str:
    lines = [x.strip() for x in (plain_text or "").splitlines() if x.strip()]
    rubric_probe = _normalize_title_probe(rubric_title)
    fallback_probe = _normalize_title_probe(fallback)
    narrative_candidates: List[str] = []

    for line in lines:
        st = line.strip()
        low = _normalize_title_probe(st)
        if (
            not st
            or st.startswith("#")
            or SOURCE_LINE_RE.match(st)
            or st.startswith("🔗 ")
            or AGE_LINE_RE.match(st)
            or AUDIENCE_LINE_RE.match(st)
            or _is_structural_heading(st)
        ):
            continue
        if rubric_probe and low.startswith(rubric_probe):
            continue
        if fallback_probe and low == fallback_probe:
            continue
        if QUESTION_LINE_RE.match(st) or low.startswith("вопрос недели"):
            q = _strip_question_prefix(st)
            if q:
                return q
        if len(st) <= 90 and not st.endswith(":"):
            return st
        narrative_candidates.append(st)

    for candidate in narrative_candidates:
        sentence = _first_sentence(candidate, max_len=90)
        if sentence:
            return sentence

    return fallback


def _contains_any_marker(text: str, markers: List[str]) -> bool:
    blob = (text or "").lower().replace("ё", "е")
    return any(m in blob for m in markers)


def _is_age_norms_content_fit(text: str) -> bool:
    blob = (text or "").lower().replace("ё", "е")
    return not _contains_any_marker(blob, AGE_NORMS_BAD_MARKERS) and _contains_any_marker(
        blob, AGE_NORMS_GOOD_MARKERS
    )


def _is_age_norms_source_fit(text: str) -> bool:
    blob = (text or "").lower().replace("ё", "е")
    return not _contains_any_marker(blob, AGE_NORMS_SOURCE_BAD_MARKERS) and _contains_any_marker(
        blob, AGE_NORMS_SOURCE_GOOD_MARKERS
    )


def _evidence_has_developmental_risk(text: str) -> bool:
    return _contains_any_marker(text, DEVELOPMENTAL_RISK_MARKERS)


def _requires_tier1_source(rubric_id: str, effective_topic_id: str, evidence: str) -> bool:
    rubric = (rubric_id or "").strip().lower()
    topic = (effective_topic_id or "").strip().lower()
    return (
        rubric == "age_norms"
        or topic == "hearing_and_speech"
        or _evidence_has_developmental_risk(evidence)
    )


def _is_tip_of_day_content_fit(text: str) -> bool:
    lines = [
        (x or "").strip()
        for x in (text or "").replace("\r\n", "\n").split("\n")
        if x.strip()
    ]
    if not lines:
        return False
    title = lines[0].lower().replace("ё", "е")
    if "совет логопеда дня" in title:
        return False
    blob = (text or "").lower().replace("ё", "е")
    return (not _contains_any_marker(blob, TIP_OF_DAY_BAD_MARKERS)) and _contains_any_marker(
        blob, TIP_OF_DAY_GOOD_MARKERS
    )


@dataclass
class Source:
    id: str
    name: str
    type: str
    url: Optional[str] = None
    urls: Optional[List[str]] = None
    parser: Optional[str] = None
    notes: str = ""


def load_sources() -> Dict[str, Source]:
    cfg = load_yaml(CFG_DIR / "sources.yml")
    return {s["id"]: Source(**s) for s in (cfg.get("sources", []) or [])}


def load_scientific_domains() -> List[str]:
    """Read the existing quality.scientific_domains list. config/sources.yml is not modified."""
    try:
        cfg = load_yaml(CFG_DIR / "sources.yml")
        quality = cfg.get("quality") or {}
        return [str(d).strip().lower() for d in (quality.get("scientific_domains") or []) if str(d).strip()]
    except Exception:
        return []


def _source_round_numbers(candidates: Sequence[Dict[str, str]]) -> List[int]:
    """
    Per-source occurrence index of every candidate — the same notion of a
    "source round" that `rank_candidates_for_topic` reconstructs downstream.

    With fewer than two distinct sources there is no round-robin interleaving to
    protect, so every candidate is reported as round 0 and the diversity
    preference stays free to order them.
    """
    source_ids = [
        str(candidate.get("source_id", "") or "").strip().lower() or "unknown"
        for candidate in candidates
    ]
    if len(set(source_ids)) < 2:
        return [0] * len(candidates)

    seen: Dict[str, int] = {}
    rounds: List[int] = []
    for source_id in source_ids:
        round_number = seen.get(source_id, 0)
        seen[source_id] = round_number + 1
        rounds.append(round_number)
    return rounds


def apply_source_diversity_preference(
    candidates: List[Dict[str, str]],
    *,
    recent_domains: Sequence[str],
    scientific_domains: Sequence[str],
    prefer_scientific: bool,
) -> List[Dict[str, str]]:
    """
    Soft, stable pre-ordering by source diversity — a tie-break, never a ranking.

    This runs *before* `rank_candidates_for_topic`, which stays the single
    authoritative topic scorer and has the final word: a stronger topic match
    always ends up above a weaker one, so a fresh domain can only win among
    candidates the canonical scorer rates equally.

    The source round is the primary key, so the publisher's source round-robin
    survives and the list is never regrouped by domain. Same-source candidates
    keep their relative order, which keeps the rounds themselves identical.
    Nothing is dropped: recent and non-scientific domains remain fallbacks.
    """
    if not candidates:
        return candidates
    if not recent_domains and not prefer_scientific:
        return candidates

    rounds = _source_round_numbers(candidates)

    def _key(item: tuple[int, Dict[str, str]]) -> tuple[int, int, int]:
        index, candidate = item
        link = (candidate.get("link") or "").strip()
        recent_rank, science_rank = source_diversity_sort_key(
            safe_domain(link),
            recent_domains=recent_domains,
            scientific_domains=scientific_domains,
            prefer_scientific=prefer_scientific,
        )
        return (rounds[index], recent_rank, science_rank)

    return [candidate for _index, candidate in sorted(enumerate(candidates), key=_key)]


def fetch_rss(url: str) -> List[Dict[str, str]]:
    d = feedparser.parse(url)
    return [
        {
            "title": norm_space(getattr(e, "title", "")),
            "link": getattr(e, "link", ""),
            "summary": norm_space(re.sub("<.*?>", "", getattr(e, "summary", ""))),
        }
        for e in d.entries[:50]
    ]


def fetch_static(urls: List[str]) -> List[Dict[str, str]]:
    return [{"title": "", "link": u, "summary": ""} for u in (urls or [])]


def _abs(base_url: str, href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(base_url, href)
    if href.startswith(("http://", "https://")):
        return href
    return urljoin(base_url, href)


def _collect_links(
    base_url: str,
    soup: BeautifulSoup,
    selector: str,
    href_re: Optional[str] = None,
) -> List[Dict[str, str]]:
    pat = re.compile(href_re) if href_re else None
    out: List[Dict[str, str]] = []
    for a in soup.select(selector):
        href = _abs(base_url, a.get("href", ""))
        if not href:
            continue
        if pat and not pat.search(href):
            continue
        title = norm_space(a.get_text(" ", strip=True))
        if not title or len(title) < 8:
            continue
        out.append({"title": title, "link": href, "summary": ""})

    seen: set[str] = set()
    uniq: List[Dict[str, str]] = []
    for it in out:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)
    return uniq


def parse_logorina_news(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "article a, div.news a, a", r"/news/[\w\-]+/?$")
    return [it for it in items if not re.search(r"/news/\d{4}-\d{2}/?$", it.get("link", ""))][
        :80
    ]


def parse_logomag_lib(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    return _collect_links(url, soup, "main a, div.content a, a", r"/lib/[^\"']+")[:80]


def parse_logoportal_articles(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    return _collect_links(url, soup, "main a, div#content a, article a, a", r"(statya-|/statya-)")[
        :80
    ]


def parse_logopedy_articles(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(
        url,
        soup,
        "div.content a, main a, a",
        r"logoped-article|logoped-literature|portal/[^#]+",
    )
    items.sort(key=lambda x: len(x["title"]), reverse=True)
    return items[:80]


def parse_logopediya_publ(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(
        url,
        soup,
        "div#dle-content a, div#dle-content h2 a, div#dle-content h3 a",
        r"/documents/[^\"']+|/publ/[^\"']+",
    )
    return [it for it in items if not re.search(r"/page/\d+/?$", it["link"])]


SITE_PARSERS = {
    "logorina_news": parse_logorina_news,
    "logomag_lib": parse_logomag_lib,
    "logoportal_articles": parse_logoportal_articles,
    "logopedy_articles": parse_logopedy_articles,
    "logopediya_publ": parse_logopediya_publ,
}


MOJIBAKE_MARKERS = (
    "Р°",
    "Рµ",
    "Рё",
    "Рѕ",
    "С‚",
    "СЏ",
    "Ð",
    "Ñ",
    "Ã",
    "Â",
    "â",
)


def _decode_candidate_score(text: str) -> tuple[int, int]:
    replacement_count = text.count("�")
    mojibake_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return (replacement_count * 10 + mojibake_count * 6, mojibake_count)


def _explicit_charset_from_headers(headers: object) -> str:
    content_type = ""
    if headers is not None:
        content_type = str(getattr(headers, "get", lambda *_args: "")("Content-Type", "") or "")
    match = re.search(r"(?:^|;)\s*charset\s*=\s*['\"]?([^;'\"]+)", content_type, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _decode_response_text(response: requests.Response) -> str:
    encodings: List[str] = []
    explicit_charset = _explicit_charset_from_headers(response.headers)
    if explicit_charset:
        encodings.append(explicit_charset)

    try:
        apparent_encoding = response.apparent_encoding
    except Exception:
        apparent_encoding = ""
    if apparent_encoding:
        encodings.append(apparent_encoding)

    # Do not use requests' implicit ISO-8859-1 choice when no charset is declared.
    encodings.extend(["utf-8", "windows-1251"])

    candidates: List[tuple[str, str]] = []
    seen: set[str] = set()
    for encoding in encodings:
        normalized = (encoding or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            candidates.append((normalized, response.content.decode(encoding)))
        except (LookupError, UnicodeDecodeError):
            continue

    if not candidates:
        return response.text

    explicit_candidate = next(
        (text for encoding, text in candidates if encoding == explicit_charset.strip().lower()),
        "",
    )
    if explicit_candidate and _decode_candidate_score(explicit_candidate) == (0, 0):
        return explicit_candidate

    return min(candidates, key=lambda item: _decode_candidate_score(item[1]))[1]


def fetch_html_site(url: str, parser_name: str) -> List[Dict[str, str]]:
    r = requests.get(url, headers=HEADERS, timeout=30, verify=_verify_for_url(url))
    r.raise_for_status()
    parser = SITE_PARSERS.get(parser_name)
    if not parser:
        raise ValueError(f"Unknown site parser: {parser_name}")
    items = parser(url, _decode_response_text(r))
    uniq: Dict[str, Dict[str, str]] = {}
    for it in items:
        uniq[it["link"]] = it
    return list(uniq.values())


def fetch_source(src: Source) -> List[Dict[str, str]]:
    if src.type == "rss":
        return fetch_rss(src.url or "")
    if src.type == "static":
        return fetch_static(src.urls or [])
    if src.type == "html_site":
        return fetch_html_site(src.url or "", src.parser or "")
    raise ValueError(f"Unsupported source type: {src.type}")


def order_candidates_for_rubric(
    rubric_id: str,
    candidates: List[Dict[str, str]],
    rng: random.Random,
) -> List[Dict[str, str]]:
    items = [dict(candidate) for candidate in candidates]
    if rubric_id not in {"method_piggybank", "bilingual_corner"}:
        rng.shuffle(items)
        return items

    grouped: Dict[str, List[Dict[str, str]]] = {}
    source_order: List[str] = []
    for candidate in items:
        source_id = (candidate.get("source_id") or "unknown").strip() or "unknown"
        if source_id not in grouped:
            grouped[source_id] = []
            source_order.append(source_id)
        grouped[source_id].append(candidate)

    for source_id in source_order:
        rng.shuffle(grouped[source_id])

    ordered: List[Dict[str, str]] = []
    while any(grouped[source_id] for source_id in source_order):
        for source_id in source_order:
            if grouped[source_id]:
                ordered.append(grouped[source_id].pop(0))

    return ordered


def get_canonical(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=25, verify=_verify_for_url(url))
        r.raise_for_status()
        soup = BeautifulSoup(_decode_response_text(r), "lxml")
        canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
        if canon and canon.get("href"):
            href = canon["href"].strip()
            if href.startswith("/"):
                href = urljoin(url, href)
            return href
        return url
    except Exception:
        return url


def extract_evidence_text(url: str, max_chars: int = 3600) -> str:
    r = requests.get(url, headers=HEADERS, timeout=35, verify=_verify_for_url(url))
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return ""

    soup = BeautifulSoup(_decode_response_text(r), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root = soup.select_one("div#dle-content") or soup.find("article") or soup.find("main")
    fallback_root = False
    if root is None:
        root = soup.body or soup
        fallback_root = True

    chunks: List[str] = []
    h1 = soup.find("h1")
    if h1:
        chunks.append(norm_space(h1.get_text(" ", strip=True)))

    elements = (
        h1.find_all_next(["h2", "h3", "p", "li"])
        if fallback_root and h1
        else root.select("h2, h3, p, li")
    )
    for el in elements:
        txt = norm_space(el.get_text(" ", strip=True))
        if len(txt) < 20:
            continue
        low = txt.lower()
        if any(
            bad in low
            for bad in ["cookie", "privacy", "политик", "подпис", "реклама", "скачать", "регистрация"]
        ):
            continue
        chunks.append(txt)
        if sum(len(x) for x in chunks) > max_chars * 1.35:
            break

    seen = set()
    uniq: List[str] = []
    for c in chunks:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    out = "\n".join(uniq).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0].strip()
    return out


def load_topic_source_ids() -> Dict[str, set[str]]:
    raw = load_yaml(CFG_DIR / "topics.yml").get("topics", {}) or {}
    return {
        str(topic_id).strip().lower(): {
            str(source_id).strip()
            for source_id in (topic_cfg.get("source_ids", []) or [])
            if str(source_id).strip()
        }
        for topic_id, topic_cfg in raw.items()
        if isinstance(topic_cfg, dict)
    }


def _resolve_effective_topic_id(
    rubric_id: str,
    source_id: str,
    preferred_topic_id: str,
    evidence: str,
    topic_source_ids: Dict[str, set[str]],
    detected_topic_ids: Optional[set[str]] = None,
) -> tuple[str, str]:
    rubric = (rubric_id or "").strip().lower()
    source = (source_id or "").strip()
    preferred = (preferred_topic_id or "").strip().lower()
    allowed_topic_ids = RUBRIC_TOPIC_ROTATION.get(rubric, ())

    if rubric == "myth_fact" and source in MYTH_FACT_CANONICAL_SOURCE_IDS:
        mapped_topic_ids = tuple(
            topic_id
            for topic_id in allowed_topic_ids
            if source in topic_source_ids.get(topic_id, set())
        )
        if len(mapped_topic_ids) != 1:
            return "", "myth_topic_mismatch"
        return mapped_topic_ids[0], ""

    if detected_topic_ids is None:
        detected_topic_ids = detect_evidence_topics(evidence)
    if preferred and preferred in detected_topic_ids:
        return preferred, ""
    return next(
        (topic_id for topic_id in allowed_topic_ids if topic_id in detected_topic_ids),
        "",
    ), ""


def render_plain_to_telegram_html(plain_text: str) -> str:
    lines = (plain_text or "").splitlines()
    if not lines:
        return ""

    def _link_anchor(url: str, prefix: str = "🔗 ") -> str:
        label = "Читать оригинальный материал"
        href = _html.escape(url, quote=True)
        return f'{prefix}<a href="{href}">{_escape(label)}</a>'

    out: List[str] = []
    for idx, raw in enumerate(lines):
        s = raw.rstrip("\n")
        st = s.strip()

        if idx == 0 and st:
            out.append(f"<b>{_escape(st)}</b>")
            continue

        if _is_structural_heading(st):
            out.append(f"<b>{_escape(st)}</b>")
            continue

        if st.startswith("🔗 "):
            url = st[2:].strip()
            out.append(_link_anchor(url, prefix="🔗 ") if url.startswith(("http://", "https://")) else _escape(st))
            continue

        if st.startswith("ℹ️ "):
            out.append(f"<i>{_escape(st)}</i>")
            continue

        out.append(_escape(s))

    return "\n".join(out).strip()


class TelegramDeliveryOutcomeAmbiguous(RuntimeError):
    """Telegram may have accepted a mutation, but the client cannot prove the outcome."""


def tg_request(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    try:
        r = requests.post(url, data=data, files=files, timeout=30)
    except requests.RequestException as exc:
        raise TelegramDeliveryOutcomeAmbiguous(
            "telegram_delivery_outcome_ambiguous:"
            f"transport_error_type={exc.__class__.__name__}"
        ) from exc

    try:
        payload = r.json()
    except Exception:
        payload = None

    if r.status_code >= 500:
        raise TelegramDeliveryOutcomeAmbiguous(
            f"telegram_delivery_outcome_ambiguous:http_status={r.status_code}"
        )

    if not r.ok:
        description = payload.get("description", "") if isinstance(payload, dict) else ""
        if not description:
            description = r.text or ""
        raise RuntimeError(f"telegram_api_error:{r.status_code}:{description}")

    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(f"telegram_api_error:{r.status_code}:{payload.get('description', '')}")

    return payload or {}


def _extract_telegram_message_id(payload: Dict[str, Any]) -> int:
    result = payload.get("result") if isinstance(payload, dict) else None
    message_id = result.get("message_id") if isinstance(result, dict) else None
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise TelegramDeliveryOutcomeAmbiguous(
            "telegram_delivery_outcome_ambiguous:missing_result_message_id"
        )
    return message_id


def send_message(chat_id: str, html_text: str) -> int:
    if not chat_id:
        raise RuntimeError("chat_id is missing")

    base_data: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": html_text,
        "disable_web_page_preview": "true",
    }

    if TELEGRAM_PARSE_MODE:
        try:
            data = dict(base_data)
            data["parse_mode"] = TELEGRAM_PARSE_MODE
            payload = tg_request("sendMessage", data=data)
            return _extract_telegram_message_id(payload)
        except Exception as e:
            if not _is_probably_parse_mode_error(e):
                raise
            _log_telegram_html_fallback("send_message", html_text, e)

    fallback_text = _strip_html_tags_for_telegram(html_text)
    payload = tg_request(
        "sendMessage",
        data={
            "chat_id": chat_id,
            "text": fallback_text,
            "disable_web_page_preview": "true",
        },
    )
    return _extract_telegram_message_id(payload)


def send_plain_message(chat_id: str, text: str) -> None:
    if not chat_id:
        raise RuntimeError("chat_id is missing")
    tg_request(
        "sendMessage",
        data={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        },
    )


def _photo_file_tuple(photo_buffer: BytesIO) -> tuple[str, bytes, str]:
    filename = getattr(photo_buffer, "name", "cover.png")
    mime_type = getattr(photo_buffer, "mime_type", "image/png")
    return (filename, photo_buffer.getvalue(), mime_type)


def _telegram_utf16_units(text: str) -> int:
    return len((text or "").encode("utf-16-le")) // 2


def _telegram_caption_limit_units() -> int:
    # Existing tests and integrations may still patch the deprecated name.
    if TG_CAPTION_MAX_BYTES != TG_CAPTION_MAX_UTF16_UNITS:
        return TG_CAPTION_MAX_BYTES
    return TG_CAPTION_MAX_UTF16_UNITS


def send_post_with_visual(chat_id: str, photo_buffer: BytesIO, plain_post: str, html_full_post: str) -> int:
    caption_units = _telegram_utf16_units(plain_post)
    caption_limit = _telegram_caption_limit_units()
    file_tuple = _photo_file_tuple(photo_buffer)

    if caption_units <= caption_limit:
        print(f"[TELEGRAM][CAPTION] mode=html units={caption_units}", flush=True)
        try:
            data: Dict[str, Any] = {
                "chat_id": chat_id,
                "caption": html_full_post,
            }
            if TELEGRAM_PARSE_MODE:
                data["parse_mode"] = TELEGRAM_PARSE_MODE
            payload = tg_request("sendPhoto", data=data, files={"photo": file_tuple})
        except Exception as e:
            if isinstance(e, TelegramDeliveryOutcomeAmbiguous):
                raise
            if _is_probably_parse_mode_error(e):
                print(f"[TELEGRAM][CAPTION] mode=plain_fallback units={caption_units}", flush=True)
                try:
                    payload = tg_request(
                        "sendPhoto",
                        data={"chat_id": chat_id, "caption": plain_post},
                        files={"photo": file_tuple},
                    )
                except Exception as plain_error:
                    if isinstance(plain_error, TelegramDeliveryOutcomeAmbiguous):
                        raise
                    print(
                        f"[TELEGRAM][SPLIT] reason=caption_send_failed units={caption_units} "
                        f"error_type={plain_error.__class__.__name__}",
                        flush=True,
                    )
                else:
                    return _extract_telegram_message_id(payload)
            else:
                print(
                    f"[TELEGRAM][SPLIT] reason=caption_send_failed units={caption_units} "
                    f"error_type={e.__class__.__name__}",
                    flush=True,
                )
        else:
            return _extract_telegram_message_id(payload)
    else:
        print(f"[TELEGRAM][SPLIT] reason=caption_too_long units={caption_units}", flush=True)

    photo_payload = tg_request(
        "sendPhoto",
        data={"chat_id": chat_id, "caption": ""},
        files={"photo": file_tuple},
    )
    photo_message_id = _extract_telegram_message_id(photo_payload)
    try:
        return send_message(chat_id, html_full_post)
    except Exception as send_error:
        if isinstance(send_error, TelegramDeliveryOutcomeAmbiguous):
            raise
        try:
            tg_request(
                "deleteMessage",
                data={"chat_id": chat_id, "message_id": photo_message_id},
            )
        except Exception as rollback_error:
            raise TelegramDeliveryOutcomeAmbiguous(
                "telegram_delivery_outcome_ambiguous:"
                "telegram_split_delivery_rollback_failed:"
                f"send_error_type={send_error.__class__.__name__}:"
                f"rollback_error_type={rollback_error.__class__.__name__}"
            ) from send_error
        print(
            f"[TELEGRAM][ROLLBACK] split_delivery_photo_deleted message_id={photo_message_id}",
            flush=True,
        )
        raise


def send_post_poll(
    chat_id: str,
    poll: PollSpec,
    reply_to_message_id: int,
) -> int:
    if not chat_id:
        raise RuntimeError("chat_id is missing")
    if isinstance(reply_to_message_id, bool) or not isinstance(reply_to_message_id, int) or reply_to_message_id <= 0:
        raise RuntimeError("reply_to_message_id must be a positive integer")

    data = {
        "chat_id": chat_id,
        "question": poll.question,
        "options": json.dumps(
            [{"text": option} for option in poll.options],
            ensure_ascii=False,
        ),
        "is_anonymous": "true",
        "type": "regular",
        "allows_multiple_answers": "false",
        "allows_revoting": "true",
        "open_period": str(POLL_OPEN_PERIOD_SECONDS),
        "disable_notification": "true" if POLL_DISABLE_NOTIFICATION else "false",
        "reply_parameters": json.dumps(
            {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }
        ),
    }
    payload = tg_request("sendPoll", data=data)
    return _extract_telegram_message_id(payload)


def _write_dry_run_poll(output_dir: Path, file_stem: str, poll: PollSpec) -> Path:
    payload = {
        "question": poll.question,
        "options": list(poll.options),
        "is_anonymous": True,
        "type": "regular",
        "open_period": POLL_OPEN_PERIOD_SECONDS,
    }
    path = output_dir / f"{file_stem}.poll.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _engagement_json_payload(spec: EngagementSpec) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"kind": spec.kind, "mode": spec.mode}
    if spec.kind == "footer":
        payload["footer_text"] = spec.footer_text
    elif spec.kind == "poll" and spec.poll is not None:
        payload["question"] = spec.poll.question
        payload["options"] = list(spec.poll.options)
    return payload


def _write_dry_run_engagement(output_dir: Path, file_stem: str, spec: EngagementSpec) -> Path:
    path = output_dir / f"{file_stem}.engagement.json"
    path.write_text(
        json.dumps(_engagement_json_payload(spec), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_dry_run_topic(
    output_dir: Path,
    file_stem: str,
    preferred_plan: Any,
    effective_topic_id: str,
    effective_topic_title: str,
) -> Path:
    path = output_dir / f"{file_stem}.topic.json"
    payload = {
        "preferred_topic_id": preferred_plan.preferred_topic_id,
        "preferred_topic_title": preferred_plan.preferred_topic_title,
        "effective_topic_id": effective_topic_id,
        "effective_topic_title": effective_topic_title,
        "override_used": preferred_plan.override_used,
        "fallback_used": not effective_topic_id or effective_topic_id != preferred_plan.preferred_topic_id,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


DRY_RUN_VISUAL_FIELDS = (
    "mode",
    "reason",
    "final_reason",
    "visual_source",
    "fallback_stage",
    "fallback_trigger",
    "fallback_reason",
    "visual_qa_required",
    "visual_qa_status",
    "visual_qa_reason",
    "visual_qa_attempts",
    "human_qa_first_status",
    "human_qa_first_reason",
    "human_qa_retry_status",
    "human_qa_retry_reason",
    "human_qa_key_source",
    "human_qa_key_attempts",
    "human_qa_key_fallback_used",
    "human_qa_key_fallback_trigger",
    "object_prompt_used",
    "object_scene_category",
    "object_generation_status",
    "text_fallback_used",
    "model",
    "gen_size",
    "output_size",
)


def _write_dry_run_visual(
    out_dir: Path,
    stem: str,
    visual_meta: dict[str, object],
) -> None:
    try:
        payload = {field: visual_meta.get(field) for field in DRY_RUN_VISUAL_FIELDS if field in visual_meta}
        (out_dir / f"{stem}.visual.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            f"[DRY_RUN][WARN] visual_metadata_write_failed stem={stem} error={exc.__class__.__name__}",
            flush=True,
        )


def _handle_post_engagement(
    *,
    spec: EngagementSpec,
    rubric_id: str,
    canonical_url: str,
    chat_id: str = "",
    post_message_id: Optional[int] = None,
    dry_run: bool = False,
    dry_run_dir: Optional[Path] = None,
    dry_run_stem: str = "",
) -> int | None:
    """Send only a selected poll, keeping optional engagement non-fatal."""
    if spec.kind != "poll" or spec.poll is None:
        return None
    if dry_run:
        if dry_run_dir is None or not dry_run_stem:
            raise RuntimeError("dry-run poll output path is missing")
        _write_dry_run_poll(dry_run_dir, dry_run_stem, spec.poll)
        return None
    try:
        poll_message_id = send_post_poll(chat_id, spec.poll, int(post_message_id or 0))
    except Exception as e:
        print(
            f"[POLL][WARN] poll_send_failed rubric={rubric_id} url={canonical_url} err={e}",
            flush=True,
        )
        return None
    print(
        f"[POLL][SENT] rubric={rubric_id} post_message_id={post_message_id} "
        f"poll_message_id={poll_message_id}",
        flush=True,
    )
    return poll_message_id


def _handle_post_poll(
    *,
    rubric_id: str,
    plain_post: str,
    canonical_url: str,
    date_key: str,
    chat_id: str = "",
    post_message_id: Optional[int] = None,
    dry_run: bool = False,
    dry_run_dir: Optional[Path] = None,
    dry_run_stem: str = "",
    enabled: Optional[bool] = None,
) -> int | Path | None:
    poll_enabled = POST_POLLS_ENABLED if enabled is None else enabled
    if not poll_enabled:
        return None

    try:
        poll = build_poll_spec(rubric_id, plain_post, canonical_url, date_key)
    except Exception as e:
        print(
            f"[POLL][WARN] poll_build_failed rubric={rubric_id} url={canonical_url} err={e}",
            flush=True,
        )
        return None
    if poll is None:
        print(f"[POLL][SKIP] rubric={rubric_id} reason=no_template", flush=True)
        return None

    if dry_run:
        if dry_run_dir is None or not dry_run_stem:
            raise RuntimeError("dry-run poll output path is missing")
        return _write_dry_run_poll(dry_run_dir, dry_run_stem, poll)

    try:
        poll_message_id = send_post_poll(chat_id, poll, int(post_message_id or 0))
    except Exception as e:
        print(
            f"[POLL][WARN] poll_send_failed rubric={rubric_id} url={canonical_url} err={e}",
            flush=True,
        )
        return None

    print(
        f"[POLL][SENT] rubric={rubric_id} post_message_id={post_message_id} "
        f"poll_message_id={poll_message_id}",
        flush=True,
    )
    return poll_message_id


def send_semantic_alert(
    chat_id: str,
    candidate_url: str,
    matched_url: str,
    score: float,
    audience: str,
    rubric_id: str,
    match_field: str,
) -> None:
    plain_text = (
        "⚠️ Semantic dedup alert\n"
        f"Материал отклонён: cosine similarity ≥ {SEMANTIC_THRESHOLD:.2f}\n"
        f"AUDIENCE={audience} | RUBRIC={rubric_id} | FIELD={match_field}\n\n"
        f"Новый кандидат: {candidate_url}\n"
        f"Похож на: {matched_url}\n"
        f"Cosine: {score:.3f}"
    )
    send_plain_message(chat_id, plain_text)


async def amain() -> None:
    rub_cfg = load_yaml(CFG_DIR / "rubrics.yml")
    channel_cfg = rub_cfg.get("channel", {}) or {}
    pub_cfg = rub_cfg.get("publishing", {}) or {}
    tzname = channel_cfg.get("timezone", "Asia/Nicosia")
    now = get_local_now(tzname)
    run_started_monotonic = time.monotonic()

    state_scope = _resolve_state_scope()
    db_path = _resolve_publication_db_path()
    selected_rubric_id = RUBRIC_ID

    if RESET_TEST_DB and state_scope == "test":
        try:
            if db_path.exists():
                db_path.unlink()
                print(f"[RESET_TEST_DB] removed {db_path}", flush=True)
            else:
                print(f"[RESET_TEST_DB] file not found: {db_path}", flush=True)
        except Exception as e:
            print(f"[RESET_TEST_DB][WARN] failed to remove {db_path}: {e}", flush=True)

    print(
        f"[START] Publisher started at {now.isoformat()} "
        f"target_channel={TARGET_CHANNEL} state_scope={state_scope} db={db_path.name} "
        f"rubric_id={selected_rubric_id or '(auto)'} topic_override={POST_TOPIC_ID} "
        f"reset_test_db={RESET_TEST_DB}",
        flush=True,
    )

    week_key = iso_week_key(now)
    day = weekday_key(now)
    max_posts = int(pub_cfg.get("max_posts_per_run", 1))
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    hashtags = channel_cfg.get("hashtags", []) or []
    sources = load_sources()
    registered_source_ids = set(sources)
    topic_source_ids = {
        topic_id: source_ids & registered_source_ids
        for topic_id, source_ids in load_topic_source_ids().items()
    }
    store = PublicationStore(db_path)
    recent_since_iso = _start_recent_window(now).isoformat()
    # Freshness cooldowns: a source may return, but only after the window passes.
    source_cooldown_since_iso = (now - timedelta(days=SOURCE_COOLDOWN_DAYS)).isoformat()
    editorial_core_since_iso = (now - timedelta(days=EDITORIAL_CORE_COOLDOWN_DAYS)).isoformat()
    recent_domains = store.recent_source_domains(RECENT_SOURCE_DOMAIN_WINDOW)
    scientific_domains = load_scientific_domains()

    audiences_cfg = rub_cfg.get("audiences", {}) or {}
    aud_list = (
        ["parents", "pros"]
        if AUDIENCE == "both"
        else ([AUDIENCE] if AUDIENCE in ("parents", "pros") else ["parents"])
    )

    posted = 0
    soft_skip_reasons: Dict[str, int] = {}
    hard_skip_reasons: Dict[str, int] = {}
    skip_stage_reasons: Dict[str, Dict[str, int]] = {}
    samples: List[str] = []
    attempted_rubrics: List[str] = []
    seen_urls_this_run: set[str] = set()
    seen_body_hashes_this_run: set[str] = set()
    seen_evidence_hashes_this_run: set[str] = set()

    def note(reason: str, url: str, stage: str = "") -> str:
        return _record_skip(
            reason,
            url,
            soft_skip_reasons,
            hard_skip_reasons,
            samples,
            stage_skip_reasons=skip_stage_reasons,
            stage=stage,
        )

    for aud in aud_list:
        if posted >= max_posts:
            break

        aud_cfg = audiences_cfg.get(aud, {}) or {}
        title_suffix = (aud_cfg.get("title_suffix", "") or "").strip()
        rubrics = aud_cfg.get("rubrics", []) or []

        for rubric in rubrics:
            if posted >= max_posts:
                break

            rf = (rubric.get("format") or "").strip().lower()
            if rf == "quality_dashboard":
                continue

            rubric_id = (rubric.get("id") or "").strip() or "unknown"
            rubric_title = rubric.get("title", "Рубрика") or "Рубрика"
            rubric_days = [str(x).strip().upper() for x in (rubric.get("byweekday") or []) if str(x).strip()]
            effective_day = rubric_days[0] if rubric_days else day

            if selected_rubric_id:
                if rubric_id.lower() != selected_rubric_id:
                    continue
            else:
                if not is_due(rubric, now):
                    continue

            try:
                topic_plan = select_topic_plan(rubric_id, week_key, POST_TOPIC_ID)
            except ValueError as e:
                print(
                    f"[TOPIC][WARN] invalid_override rubric={rubric_id} override={POST_TOPIC_ID} "
                    f"err={e}; using auto",
                    flush=True,
                )
                topic_plan = select_topic_plan(rubric_id, week_key, "auto")
            print(
                f"[TOPIC] rubric={rubric_id} week={week_key} "
                f"preferred={topic_plan.preferred_topic_id or '(none)'} "
                f"override={topic_plan.override_used}",
                flush=True,
            )

            if rubric_id not in attempted_rubrics:
                attempted_rubrics.append(rubric_id)

            rubric_skips = 0

            selected_sources = list(rubric.get("sources", []) or [])
            if INCLUDE_SOURCES:
                include_set = set(INCLUDE_SOURCES)
                selected_sources = [sid for sid in selected_sources if sid in include_set]
            if EXCLUDE_SOURCES:
                exclude_set = set(EXCLUDE_SOURCES)
                selected_sources = [sid for sid in selected_sources if sid not in exclude_set]

            print(
                f"[RUBRIC_SOURCES] rubric={rubric_id} selected_sources={selected_sources}",
                flush=True,
            )

            all_items: List[Dict[str, str]] = []
            for sid in selected_sources:
                src = sources.get(sid)
                if not src:
                    kind = note("unknown_source_id", sid)
                    print(f"[SKIP][{kind}] unknown_source_id id={sid}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    continue
                try:
                    for item in fetch_source(src):
                        candidate = dict(item)
                        candidate["source_id"] = sid
                        all_items.append(candidate)
                except Exception as e:
                    kind = note("source_fetch_failed", f"{sid}: {e}")
                    print(f"[SKIP][{kind}] source_fetch_failed source={sid} err={e}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1

            if not all_items:
                note("no_candidates", rubric_id)
                continue

            seed = int(hashlib.sha1(f"{now.date()}|{rubric_id}|{aud}".encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(seed)
            all_items = order_candidates_for_rubric(rubric_id, all_items, rng)
            # Diversity is only a stable pre-order/tie-break inside the source
            # rounds; the canonical topic scorer runs last and stays authoritative.
            all_items = apply_source_diversity_preference(
                all_items,
                recent_domains=recent_domains,
                scientific_domains=scientific_domains,
                prefer_scientific=should_prefer_scientific_sources(rubric_id),
            )
            all_items = rank_candidates_for_topic(
                all_items,
                topic_plan.preferred_topic_id,
                topic_source_ids.get(topic_plan.preferred_topic_id, set()),
            )

            print(
                f"[RUBRIC] rubric={rubric_id} audience={aud} candidates_total={len(all_items)} max_scan={MAX_CANDIDATES_PER_RUBRIC}",
                flush=True,
            )

            for cand in all_items[:MAX_CANDIDATES_PER_RUBRIC]:
                url = (cand.get("link") or "").strip()
                candidate_source_id = (cand.get("source_id") or "unknown").strip() or "unknown"

                elapsed = time.monotonic() - run_started_monotonic
                if elapsed > MAX_RUN_SECONDS:
                    kind = note("max_run_seconds", rubric_id)
                    print(f"[STOP][{kind}] max_run_seconds reached: {elapsed:.1f}s", flush=True)
                    break

                print(
                    f"[CANDIDATE] rubric={rubric_id} audience={aud} "
                    f"source={candidate_source_id} url={url}",
                    flush=True,
                )

                if not url.startswith(("http://", "https://")):
                    kind = note("bad_candidate_url", url or "(empty)")
                    print(f"[SKIP][{kind}] bad_candidate_url source={candidate_source_id} url={url}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if _SKIP_EXT_RE.search(url):
                    kind = note("skip_non_html_asset", url)
                    print(f"[SKIP][{kind}] skip_non_html_asset source={candidate_source_id} url={url}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                canon = get_canonical(url)
                if _SKIP_EXT_RE.search(canon):
                    kind = note("skip_non_html_asset", canon)
                    print(f"[SKIP][{kind}] skip_non_html_asset source={candidate_source_id} canon={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if canon in seen_urls_this_run:
                    kind = note("dup_url_same_run", canon)
                    print(f"[SKIP][{kind}] dup_url_same_run source={candidate_source_id} url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_url(canon):
                    url_within_cooldown = store.has_url_since(canon, source_cooldown_since_iso)
                    evergreen_url_reuse = should_bypass_duplicate_reason(rubric_id, "dup_url_db")
                    if url_within_cooldown:
                        kind = note("dup_url_recent", canon, stage="url_cooldown")
                        print(
                            f"[SKIP][{kind}] dup_url_recent source={candidate_source_id} url={canon} "
                            f"cooldown_days={SOURCE_COOLDOWN_DAYS}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue
                    if evergreen_url_reuse:
                        print(
                            f"[WARN] dup_url_db_ignored evergreen_reuse rubric={rubric_id} "
                            f"source={candidate_source_id} url={canon} "
                            f"cooldown_days={SOURCE_COOLDOWN_DAYS}",
                            flush=True,
                        )
                    else:
                        kind = note("dup_url_db", canon)
                        print(f"[SKIP][{kind}] dup_url_db source={candidate_source_id} url={canon}", flush=True)
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                try:
                    evidence = extract_evidence_text(canon, max_chars=3600)
                except Exception as e:
                    kind = note("evidence_fetch_failed", f"{canon} ({e})")
                    print(
                        f"[SKIP][{kind}] evidence_fetch_failed source={candidate_source_id} "
                        f"url={canon} err={e}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if len((evidence or "").strip()) < 260:
                    kind = note("no_evidence_short", canon, stage="evidence")
                    print(f"[SKIP][{kind}] no_evidence_short source={candidate_source_id} url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                preferred_topic_id = topic_plan.preferred_topic_id
                detected_topic_ids: Optional[set[str]] = None
                if not (
                    rubric_id.lower() == "myth_fact"
                    and candidate_source_id in MYTH_FACT_CANONICAL_SOURCE_IDS
                ):
                    detected_topic_ids = detect_evidence_topics(evidence)
                effective_topic_id, topic_routing_reason = _resolve_effective_topic_id(
                    rubric_id,
                    candidate_source_id,
                    preferred_topic_id,
                    evidence,
                    topic_source_ids,
                    detected_topic_ids,
                )
                effective_topic_title = TOPICS[effective_topic_id] if effective_topic_id else ""
                if topic_routing_reason:
                    kind = note(topic_routing_reason, canon, stage="pre_llm")
                    print(
                        f"[SKIP][{kind}] {topic_routing_reason} "
                        f"stage=pre_llm source={candidate_source_id} url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue
                if effective_topic_id:
                    route_label = "MATCH" if effective_topic_id == preferred_topic_id else "FALLBACK"
                    print(
                        f"[TOPIC][{route_label}] rubric={rubric_id} source={candidate_source_id} "
                        f"preferred={preferred_topic_id or '(none)'} effective={effective_topic_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"[TOPIC][UNDETECTED] rubric={rubric_id} source={candidate_source_id} "
                        f"preferred={preferred_topic_id or '(none)'}",
                        flush=True,
                    )

                if rubric_id == "age_norms" and not _is_age_norms_source_fit(evidence):
                    kind = note("rubric_topic_mismatch_source", canon)
                    print(
                        f"[SKIP][{kind}] rubric_topic_mismatch_source source={candidate_source_id} "
                        f"url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                candidate_domain = safe_domain(canon) or safe_domain(url)
                if _requires_tier1_source(rubric_id, effective_topic_id, evidence) and not is_scientific_domain(
                    candidate_domain,
                    scientific_domains,
                ):
                    # note("source_authority_required", canon)
                    kind = note("source_authority_required", canon, stage="source_authority")
                    print(
                        f"[SKIP][{kind}] source_authority_required source={candidate_source_id} "
                        f"rubric={rubric_id} topic={effective_topic_id or '(none)'} "
                        f"domain={candidate_domain or '(none)'} url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                evidence_hash = sha1(norm_space(evidence))
                if evidence_hash in seen_evidence_hashes_this_run:
                    kind = note("dup_evidence_same_run", canon)
                    print(f"[SKIP][{kind}] dup_evidence_same_run source={candidate_source_id} url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_evidence_hash(evidence_hash):
                    evidence_within_cooldown = store.has_evidence_hash_since(
                        evidence_hash, source_cooldown_since_iso
                    )
                    if evidence_within_cooldown:
                        kind = note("dup_evidence_hash_recent", canon)
                        print(
                            f"[SKIP][{kind}] dup_evidence_hash_recent source={candidate_source_id} "
                            f"url={canon} cooldown_days={SOURCE_COOLDOWN_DAYS}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue
                    if should_bypass_duplicate_reason(rubric_id, "dup_evidence_hash_db"):
                        print(
                            f"[WARN] dup_evidence_hash_db_ignored evergreen_reuse rubric={rubric_id} "
                            f"source={candidate_source_id} url={canon} "
                            f"cooldown_days={SOURCE_COOLDOWN_DAYS}",
                            flush=True,
                        )
                    else:
                        kind = note("dup_evidence_hash_db", canon)
                        print(
                            f"[SKIP][{kind}] dup_evidence_hash_db source={candidate_source_id} "
                            f"url={canon}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                sem_source_hit = store.find_semantic_duplicate(
                    evidence,
                    threshold=SEMANTIC_THRESHOLD_SOURCE,
                    since_iso=source_cooldown_since_iso,
                    limit=500,
                    compare="evidence",
                )
                if sem_source_hit:
                    if should_bypass_source_semantic_dedup(rubric_id):
                        print(
                            f"[WARN] semantic_source_match_ignored rubric={rubric_id} "
                            f"url={canon} matched={sem_source_hit.canonical_url} "
                            f"score={sem_source_hit.similarity:.3f}",
                            flush=True,
                        )
                    else:
                        kind = note("dup_semantic_source", canon)
                        print(
                            f"[SKIP][{kind}] dup_semantic_source source={candidate_source_id} url={canon} "
                            f"matched={sem_source_hit.canonical_url} "
                            f"score={sem_source_hit.similarity:.3f}",
                            flush=True,
                        )
                        if not DRY_RUN and TELEGRAM_DRAFTS_CHAT_ID:
                            recent_hit = store.find_semantic_duplicate(
                                evidence,
                                threshold=SEMANTIC_THRESHOLD,
                                since_iso=recent_since_iso,
                                limit=120,
                                compare="evidence",
                            )
                            if recent_hit:
                                try:
                                    send_semantic_alert(
                                        TELEGRAM_DRAFTS_CHAT_ID,
                                        canon,
                                        recent_hit.canonical_url,
                                        recent_hit.similarity,
                                        aud,
                                        rubric_id,
                                        recent_hit.match_field,
                                    )
                                except Exception as e:
                                    print(f"[WARN] failed_to_send_semantic_alert err={e}", flush=True)

                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                sd = candidate_domain or "источник"

                llm_rubric_format = rf
                if rubric_id.lower() == "bilingual_corner":
                    llm_rubric_format = (
                        "bilingual_parents"
                        if effective_topic_id == "bilingualism"
                        else "thematic_parents"
                    )

                myth_evidence_prevalidated = False
                if llm_rubric_format == "myth_fact":
                    myth_evidence_ok, myth_evidence_reason = validate_myth_fact_evidence_for_generation(
                        evidence,
                        effective_topic_id,
                    )
                    if not myth_evidence_ok:
                        kind = note(myth_evidence_reason, canon, stage="pre_llm")
                        print(
                            f"[SKIP][{kind}] {myth_evidence_reason} "
                            f"stage=pre_llm source={candidate_source_id} url={canon}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue
                    # Fail-closed evidence validation already passed for this
                    # candidate, so the prompt no longer needs the generic
                    # method-card "return НЕТ_ДАННЫХ" instructions. The myth
                    # evidence validator and every output validator stay active.
                    myth_evidence_prevalidated = True

                pro_evidence_prevalidated = False
                if rf == "pro_friendly":
                    pro_evidence_ok, pro_evidence_reason = validate_pro_evidence_for_generation(evidence)
                    if not pro_evidence_ok:
                        kind = note(pro_evidence_reason, canon, stage="pre_llm")
                        print(
                            f"[SKIP][{kind}] {pro_evidence_reason} "
                            f"stage=pre_llm source={candidate_source_id} url={canon}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue
                    pro_evidence_prevalidated = rubric_id == "method_piggybank"

                evidence_prevalidated = pro_evidence_prevalidated or myth_evidence_prevalidated

                try:
                    plain_raw, ok, llm_note = await asyncio.wait_for(
                        generate_post_plain_from_evidence_async(
                            rubric_title=rubric_title,
                            rubric_format=llm_rubric_format,
                            audience=aud,
                            title_suffix=title_suffix,
                            source_domain=sd,
                            source_url=canon,
                            evidence_text=evidence,
                            disclaimer=disclaimer,
                            hashtags=hashtags if aud != "pros" else [],
                            provider=PROVIDER,
                            groq_key=GROQ_API_KEY,
                            gemini_key=GEMINI_API_KEY,
                            max_chars=POST_MAX_CHARS,
                            day_key=effective_day,
                            evidence_prevalidated=evidence_prevalidated,
                            topic_id=effective_topic_id,
                            topic_title=effective_topic_title,
                        ),
                        timeout=MAX_LLM_SECONDS_PER_CANDIDATE,
                    )
                except asyncio.TimeoutError:
                    kind = note("llm_timeout", canon)
                    print(f"[SKIP][{kind}] llm_timeout source={candidate_source_id} url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue
                except Exception as e:
                    kind = note("llm_failed", f"{canon} ({e})")
                    print(
                        f"[SKIP][{kind}] llm_failed source={candidate_source_id} "
                        f"url={canon} err={e}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if not ok or not plain_raw:
                    skip_reason, skip_stage = _resolve_llm_skip(llm_note)
                    kind = note(skip_reason, canon, stage=skip_stage)
                    print(
                        f"[SKIP][{kind}] {llm_note} reason={skip_reason} "
                        f"source={candidate_source_id} url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                plain = finalize_plain_post_for_publication(
                    plain_text=plain_raw,
                    day_key=effective_day,
                    source_domain=sd,
                    source_url=canon,
                    max_chars=POST_MAX_CHARS,
                    rubric_id=rubric_id,
                    topic_id=effective_topic_id,
                )

                if not plain or _looks_incomplete_final_body(_body_without_footer(plain)):
                    kind = note("final_invalid_output", f"{canon} (final_body_incomplete)")
                    print(
                        f"[SKIP][{kind}] final_invalid_output "
                        f"reason=final_body_incomplete source={candidate_source_id} url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if rubric_id == "question_week":
                    final_ok, final_reason = _validate_question_week_output(plain)
                    if not final_ok:
                        kind = note("final_invalid_output", f"{canon} ({final_reason})")
                        print(
                            f"[SKIP][{kind}] final_invalid_output "
                            f"reason={final_reason} source={candidate_source_id} url={canon}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                if rubric_id == "age_norms" and not _is_age_norms_content_fit(plain):
                    kind = note("rubric_topic_mismatch_post", canon)
                    print(
                        f"[SKIP][{kind}] rubric_topic_mismatch_post source={candidate_source_id} "
                        f"url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if rubric_id == "tip_of_day" and not _is_tip_of_day_content_fit(plain):
                    kind = note("tip_of_day_post_too_generic", canon)
                    print(
                        f"[SKIP][{kind}] tip_of_day_post_too_generic source={candidate_source_id} "
                        f"url={canon}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                body_hash = sha1(norm_space(plain))
                if body_hash in seen_body_hashes_this_run:
                    kind = note("dup_body_same_run", canon)
                    print(f"[SKIP][{kind}] dup_body_same_run source={candidate_source_id} url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_body_hash(body_hash):
                    kind = note("dup_body_hash_db", canon)
                    print(f"[SKIP][{kind}] dup_body_hash_db source={candidate_source_id} url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                sem_body_threshold = semantic_post_threshold_for_rubric(rubric_id)
                sem_body_hit = store.find_semantic_duplicate(
                    plain,
                    threshold=sem_body_threshold,
                    since_iso=None,
                    limit=500,
                    compare="body",
                )
                if sem_body_hit:
                    kind = note("dup_semantic_post", canon)
                    print(
                        f"[SKIP][{kind}] dup_semantic_post source={candidate_source_id} url={canon} "
                        f"matched={sem_body_hit.canonical_url} "
                        f"score={sem_body_hit.similarity:.3f} "
                        f"threshold={sem_body_threshold:.3f}",
                        flush=True,
                    )
                    if not DRY_RUN and TELEGRAM_DRAFTS_CHAT_ID:
                        recent_post_hit = store.find_semantic_duplicate(
                            plain,
                            threshold=sem_body_threshold,
                            since_iso=recent_since_iso,
                            limit=120,
                            compare="body",
                        )
                        if recent_post_hit:
                            try:
                                send_semantic_alert(
                                    TELEGRAM_DRAFTS_CHAT_ID,
                                    canon,
                                    recent_post_hit.canonical_url,
                                    recent_post_hit.similarity,
                                    aud,
                                    rubric_id,
                                    recent_post_hit.match_field,
                                )
                            except Exception as e:
                                print(f"[WARN] failed_to_send_semantic_alert err={e}", flush=True)

                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                editorial_core = extract_editorial_core(plain)
                if editorial_core:
                    core_threshold = semantic_editorial_core_threshold()
                    core_hit = store.find_editorial_core_duplicate(
                        editorial_core,
                        threshold=core_threshold,
                        since_iso=editorial_core_since_iso,
                        limit=200,
                        core_extractor=extract_editorial_core,
                    )
                    if core_hit:
                        kind = note("dup_editorial_core_recent", canon)
                        print(
                            f"[SKIP][{kind}] dup_editorial_core_recent source={candidate_source_id} "
                            f"url={canon} matched={core_hit.canonical_url} "
                            f"matched_rubric={core_hit.rubric_id} "
                            f"score={core_hit.similarity:.3f} threshold={core_threshold:.3f} "
                            f"cooldown_days={EDITORIAL_CORE_COOLDOWN_DAYS}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                h1_title = _extract_cover_title_from_plain_post(
                    plain,
                    fallback=rubric_title,
                    rubric_title=rubric_title,
                )

                image_prompt = ""
                image_prompt_ok = False
                image_prompt_note = "skipped"
                try:
                    image_prompt, image_prompt_ok, image_prompt_note = await asyncio.wait_for(
                        generate_image_prompt_async(
                            title=h1_title,
                            body_text=plain,
                            audience=aud,
                            provider=PROVIDER,
                            groq_key=GROQ_API_KEY,
                            gemini_key=GEMINI_API_KEY,
                            rubric_id=rubric_id,
                        ),
                        timeout=IMAGE_PROMPT_TIMEOUT_SECONDS,
                    )
                    if not image_prompt_ok:
                        image_prompt = ""
                except asyncio.TimeoutError:
                    image_prompt = ""
                    image_prompt_note = "image_prompt_timeout"
                except Exception as e:
                    image_prompt = ""
                    image_prompt_note = f"image_prompt_failed:{e}"

                try:
                    visual_buffer, visual_meta = build_post_visual(
                        title=h1_title,
                        day_key=effective_day,
                        image_prompt=image_prompt,
                        pollinations_token=POLLINATIONS_TOKEN,
                        rubric_id=rubric_id,
                        audience=aud,
                        visual_qa_api_key=GEMINI_VISUAL_QA_API_KEY,
                    )
                except Exception as e:
                    kind = note("visual_build_failed", f"{canon} ({e})")
                    print(
                        f"[SKIP][{kind}] visual_build_failed source={candidate_source_id} "
                        f"url={canon} err={e}",
                        flush=True,
                    )
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                visual_title = visual_meta.get("visual_title") or visual_meta.get("title") or h1_title
                visual_prompt_len = visual_meta.get("prompt_len", len(image_prompt or ""))
                visual_has_prompt = bool(visual_meta.get("has_image_prompt", bool(image_prompt)))
                visual_has_token = bool(visual_meta.get("has_token", bool(POLLINATIONS_TOKEN)))
                print(
                    f"[VISUAL] rubric={rubric_id} "
                    f"mode={log_field(visual_meta.get('mode'))} "
                    f"reason={log_field(visual_meta.get('reason'))} "
                    f"fallback_reason={log_field(visual_meta.get('fallback_reason'))} "
                    f"fallback_stage={log_field(visual_meta.get('fallback_stage'))} "
                    f"fallback_trigger={log_field(visual_meta.get('fallback_trigger'))} "
                    f"visual_source={log_field(visual_meta.get('visual_source'))} "
                    f"visual_qa_required={log_field(visual_meta.get('visual_qa_required'))} "
                    f"visual_qa_status={log_field(visual_meta.get('visual_qa_status', visual_meta.get('visual_qa')))} "
                    f"visual_qa_reason={log_field(visual_meta.get('visual_qa_reason'))} "
                    f"human_qa_first_status={log_field(visual_meta.get('human_qa_first_status'))} "
                    f"human_qa_first_reason={log_field(visual_meta.get('human_qa_first_reason'))} "
                    f"human_qa_retry_status={log_field(visual_meta.get('human_qa_retry_status'))} "
                    f"human_qa_retry_reason={log_field(visual_meta.get('human_qa_retry_reason'))} "
                    f"human_qa_key_source={log_field(visual_meta.get('human_qa_key_source'))} "
                    f"human_qa_key_attempts={log_field(visual_meta.get('human_qa_key_attempts'))} "
                    f"human_qa_key_fallback_used={log_field(visual_meta.get('human_qa_key_fallback_used'))} "
                    f"human_qa_key_fallback_trigger={log_field(visual_meta.get('human_qa_key_fallback_trigger'))} "
                    f"object_prompt_used={log_field(visual_meta.get('object_prompt_used'))} "
                    f"object_scene_category={log_field(visual_meta.get('object_scene_category'))} "
                    f"object_generation_status={log_field(visual_meta.get('object_generation_status'))} "
                    f"image_prompt_note={log_field(image_prompt_note)} "
                    f"prompt_len={visual_prompt_len} "
                    f"has_image_prompt={visual_has_prompt} "
                    f"has_pollinations_token={visual_has_token} "
                    f"title=\"{log_field(visual_title, max_len=140)}\" "
                    f"image_prompt_ok={image_prompt_ok} "
                    f"exception_type={log_field(visual_meta.get('exception_type'))} "
                    f"model={log_field(visual_meta.get('model'))} "
                    f"gen_size={log_field(visual_meta.get('gen_size'))} "
                    f"output_size={log_field(visual_meta.get('output_size'))} "
                    f"timeout_seconds={visual_meta.get('timeout_seconds', '')}",
                    flush=True,
                )

                engagement_spec = EngagementSpec(kind="none", mode="none")
                display_plain = plain
                try:
                    engagement_spec = build_engagement_spec(
                        rubric_id=rubric_id,
                        plain_post=plain,
                        canonical_url=canon,
                        date_key=now.date().isoformat(),
                        policy_mode=POST_ENGAGEMENT_MODE,
                        topic_id=effective_topic_id,
                    )
                except Exception as e:
                    print(
                        f"[ENGAGEMENT][WARN] build_failed rubric={rubric_id} url={canon} err={e}",
                        flush=True,
                    )
                    engagement_spec = EngagementSpec(kind="none", mode="none")
                    display_plain = plain

                if engagement_spec.kind == "footer":
                    footer_failed = False
                    try:
                        candidate_display = append_engagement_footer(
                            plain,
                            engagement_spec.footer_text,
                            POST_MAX_CHARS,
                        )
                    except Exception as e:
                        footer_failed = True
                        print(
                            f"[ENGAGEMENT][WARN] footer_failed rubric={rubric_id} url={canon} err={e}",
                            flush=True,
                        )
                        candidate_display = plain
                    if footer_failed:
                        engagement_spec = EngagementSpec(kind="none", mode="none")
                    elif candidate_display == plain:
                        print(
                            f"[ENGAGEMENT][SKIP] rubric={rubric_id} mode={engagement_spec.mode} "
                            "reason=max_chars",
                            flush=True,
                        )
                        engagement_spec = EngagementSpec(kind="none", mode="none")
                    else:
                        display_plain = candidate_display

                if engagement_spec.kind == "poll":
                    print(
                        f"[ENGAGEMENT][POLL] rubric={rubric_id} mode={engagement_spec.mode}",
                        flush=True,
                    )
                elif engagement_spec.kind == "footer":
                    print(
                        f"[ENGAGEMENT][FOOTER] rubric={rubric_id} mode={engagement_spec.mode}",
                        flush=True,
                    )
                else:
                    print(
                        f"[ENGAGEMENT][NONE] rubric={rubric_id} mode={engagement_spec.mode}",
                        flush=True,
                    )

                html_full = render_plain_to_telegram_html(display_plain)
                post_message_id: Optional[int] = None
                target_chat_id = ""
                dry_run_out: Optional[Path] = None
                dry_run_stem = f"{posted+1:02d}_{aud}_{rubric_id}"

                if DRY_RUN:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    out = STATE_DIR / "dry_run" / ts
                    out.mkdir(parents=True, exist_ok=True)
                    dry_run_out = out
                    filename = getattr(visual_buffer, "name", f"{posted+1:02d}_{aud}_{rubric_id}.png")
                    ext = Path(filename).suffix or ".png"
                    (out / f"{dry_run_stem}{ext}").write_bytes(visual_buffer.getvalue())
                    (out / f"{dry_run_stem}.txt").write_text(display_plain, encoding="utf-8")
                    _write_dry_run_engagement(out, dry_run_stem, engagement_spec)
                    _write_dry_run_topic(
                        out,
                        dry_run_stem,
                        topic_plan,
                        effective_topic_id,
                        effective_topic_title,
                    )
                    _write_dry_run_visual(out, dry_run_stem, visual_meta)
                else:
                    target_chat_id = _resolve_publish_chat_id()
                    if not target_chat_id:
                        raise RuntimeError("Resolved target chat id is empty")
                    try:
                        post_message_id = send_post_with_visual(
                            target_chat_id,
                            visual_buffer,
                            display_plain,
                            html_full,
                        )
                    except TelegramDeliveryOutcomeAmbiguous as e:
                        print(
                            "[STOP] telegram_delivery_outcome_ambiguous "
                            f"error_class={e.__class__.__name__} "
                            f"rubric={rubric_id} source={candidate_source_id} url={canon}",
                            flush=True,
                        )
                        raise
                    except Exception as e:
                        kind = note("telegram_send_failed", f"{canon} ({e})")
                        print(
                            f"[SKIP][{kind}] telegram_send_failed source={candidate_source_id} "
                            f"url={canon} err={e}",
                            flush=True,
                        )
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                    store.record_publication(
                        canonical_url=canon,
                        body_hash=body_hash,
                        body_text=plain,
                        evidence_hash=evidence_hash,
                        evidence_text=evidence,
                        posted_at=now.isoformat(),
                        audience=aud,
                        rubric_id=rubric_id,
                        rubric_title=rubric_title,
                        source_domain=sd,
                    )

                seen_urls_this_run.add(canon)
                seen_body_hashes_this_run.add(body_hash)
                seen_evidence_hashes_this_run.add(evidence_hash)

                posted += 1
                print(
                    f"[POSTED] rubric={rubric_id} audience={aud} "
                    f"topic={effective_topic_id or '(none)'} url={canon}",
                    flush=True,
                )
                _handle_post_engagement(
                    spec=engagement_spec,
                    rubric_id=rubric_id,
                    canonical_url=canon,
                    chat_id=target_chat_id,
                    post_message_id=post_message_id,
                    dry_run=DRY_RUN,
                    dry_run_dir=dry_run_out,
                    dry_run_stem=dry_run_stem,
                )
                await asyncio.sleep(1.0)
                break

            if (time.monotonic() - run_started_monotonic) > MAX_RUN_SECONDS or posted >= max_posts:
                break

        if (time.monotonic() - run_started_monotonic) > MAX_RUN_SECONDS or posted >= max_posts:
            break

    if posted == 0 and not DRY_RUN:
        _send_posted_zero_alert_if_needed(
            db_path=db_path,
            now=now,
            day=day,
            week_key=week_key,
            audience=AUDIENCE,
            provider=PROVIDER,
            soft_skip_reasons=soft_skip_reasons,
            hard_skip_reasons=hard_skip_reasons,
            samples=samples,
            state_scope=state_scope,
            attempted_rubrics=attempted_rubrics,
            topic_preference=POST_TOPIC_ID,
            stage_skip_reasons=skip_stage_reasons,
        )

    print(
        f"Publisher done. Posted: {posted}. Week: {week_key}.{' [DRY_RUN]' if DRY_RUN else ''}",
        flush=True,
    )


def run() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    run()

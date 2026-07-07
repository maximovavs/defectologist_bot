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
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
from src.publisher.dedup_policy import (
    semantic_post_threshold_for_rubric,
    should_bypass_source_semantic_dedup,
    should_allow_evergreen_source_reuse,
)

import feedparser
import requests
import urllib3
import yaml
from bs4 import BeautifulSoup
from dateutil import tz

from src.services.llm_generator import (
    _validate_question_week_output,
    generate_image_prompt_async,
    generate_post_plain_from_evidence_async,
)
from src.services.publication_store import PublicationStore
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
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()

PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


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
TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))
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
BENEFIT_HEADING_RE = re.compile(r"^💡\s*Что это дает\s*:?\s*$", re.IGNORECASE)

AGE_LINE_RE = re.compile(r"^👶\s*Возраст\s*:\s*.+\S$", re.IGNORECASE)
AUDIENCE_LINE_RE = re.compile(r"^👩‍⚕️\s*Аудитория\s*:\s*.+\S$", re.IGNORECASE)
SOURCE_LINE_RE = re.compile(r"^Источник:\s*\S.+$", re.IGNORECASE)
BENEFIT_LINE_RE = re.compile(r"^💡\s*Что это дает\s*:\s*.+\S$", re.IGNORECASE)
MYTH_LINE_RE = re.compile(r"^🔴\s*Миф\s*:\s*.+\S$", re.IGNORECASE)
QUESTION_LINE_RE = re.compile(r"^❓\s*Вопрос недели\s*:\s*.+\S$", re.IGNORECASE)
ORIENTIRS_LINE_RE = re.compile(r"^Ориентиры:\s*.+\S$", re.IGNORECASE)

HASHTAG_TOKEN_RE = re.compile(r"(?<!\w)#([A-Za-zА-Яа-яЁё0-9_]+)")

RUBRIC_TAGS_BY_DAY = {
    "MO": "#совет_логопеда",
    "TU": "#играем_и_говорим",
    "WE": "#миф_факт",
    "TH": "#русский_за_границей",
    "FR": "#вопрос_недели",
    "SA": "#методическая_копилка",
    "SU": "#возрастная_норма",
}

SOFT_SKIP_REASONS = {
    "bad_candidate_url",
    "skip_non_html_asset",
    "dup_url_same_run",
    "dup_url_db",
    "no_evidence_short",
    "dup_evidence_same_run",
    "dup_evidence_hash_db",
    "dup_semantic_source",
    "dup_body_same_run",
    "dup_body_hash_db",
    "dup_semantic_post",
    "rubric_topic_mismatch_source",
    "rubric_topic_mismatch_post",
    "tip_of_day_post_too_generic",
    "unknown_source_id",
    "llm_invalid_output",
    "final_invalid_output",
    "no_candidates",
    "max_skips_per_rubric",
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
        f"Soft skips: {soft_total} | Hard skips: {hard_total}",
    ]

    if hard_top:
        parts.extend(["", "Hard skip reasons:"])
        parts.extend([f"• {reason}: {count}" for reason, count in hard_top])

    if soft_top:
        parts.extend(["", "Soft skip reasons:"])
        parts.extend([f"• {reason}: {count}" for reason, count in soft_top])

    if samples:
        parts.extend(["", "Examples:"])
        parts.extend(samples[:10])

    return "\n".join(parts)


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


def _build_age_tag(age_value: str) -> str:
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


def _filter_relevant_thematic_tags(tags: List[str], body_text: str) -> List[str]:
    return [tag for tag in tags if _body_supports_tag(tag, body_text)][:2]


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
    thematic_tags = _filter_relevant_thematic_tags(thematic_tags, body_text)

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


def fetch_html_site(url: str, parser_name: str) -> List[Dict[str, str]]:
    r = requests.get(url, headers=HEADERS, timeout=30, verify=_verify_for_url(url))
    r.raise_for_status()
    parser = SITE_PARSERS.get(parser_name)
    if not parser:
        raise ValueError(f"Unknown site parser: {parser_name}")
    items = parser(url, r.text)
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


def get_canonical(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=25, verify=_verify_for_url(url))
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
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

    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root = soup.select_one("div#dle-content") or soup.find("article") or soup.find("main") or soup.body or soup

    chunks: List[str] = []
    h1 = soup.find("h1")
    if h1:
        chunks.append(norm_space(h1.get_text(" ", strip=True)))

    for el in root.select("h2, h3, p, li"):
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


def tg_request(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, data=data, files=files, timeout=30)

    try:
        payload = r.json()
    except Exception:
        payload = None

    if not r.ok:
        description = payload.get("description", "") if isinstance(payload, dict) else ""
        if not description:
            description = r.text or ""
        raise RuntimeError(f"telegram_api_error:{r.status_code}:{description}")

    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(f"telegram_api_error:{r.status_code}:{payload.get('description', '')}")

    return payload or {}


def send_message(chat_id: str, html_text: str) -> None:
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
            tg_request("sendMessage", data=data)
            return
        except Exception as e:
            if not _is_probably_parse_mode_error(e):
                raise
            _log_telegram_html_fallback("send_message", html_text, e)

    fallback_text = _strip_html_tags_for_telegram(html_text)
    tg_request(
        "sendMessage",
        data={
            "chat_id": chat_id,
            "text": fallback_text,
            "disable_web_page_preview": "true",
        },
    )


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


def send_post_with_visual(chat_id: str, photo_buffer: BytesIO, plain_post: str, html_full_post: str) -> None:
    plain_bytes = len((plain_post or "").encode("utf-8"))
    file_tuple = _photo_file_tuple(photo_buffer)

    if plain_bytes <= TG_CAPTION_MAX_BYTES:
        try:
            data: Dict[str, Any] = {
                "chat_id": chat_id,
                "caption": html_full_post,
            }
            if TELEGRAM_PARSE_MODE:
                data["parse_mode"] = TELEGRAM_PARSE_MODE
            tg_request("sendPhoto", data=data, files={"photo": file_tuple})
            return
        except Exception as e:
            if _is_probably_parse_mode_error(e):
                _log_telegram_html_fallback("send_photo_caption", html_full_post, e)
            else:
                print(f"[WARN] send_photo_with_caption_failed err={e}", flush=True)

    tg_request("sendPhoto", data={"chat_id": chat_id, "caption": ""}, files={"photo": file_tuple})
    send_message(chat_id, html_full_post)


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
        f"rubric_id={selected_rubric_id or '(auto)'} reset_test_db={RESET_TEST_DB}",
        flush=True,
    )

    week_key = iso_week_key(now)
    day = weekday_key(now)
    max_posts = int(pub_cfg.get("max_posts_per_run", 1))
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    hashtags = channel_cfg.get("hashtags", []) or []
    sources = load_sources()
    store = PublicationStore(db_path)
    recent_since_iso = _start_recent_window(now).isoformat()

    audiences_cfg = rub_cfg.get("audiences", {}) or {}
    aud_list = (
        ["parents", "pros"]
        if AUDIENCE == "both"
        else ([AUDIENCE] if AUDIENCE in ("parents", "pros") else ["parents"])
    )

    posted = 0
    soft_skip_reasons: Dict[str, int] = {}
    hard_skip_reasons: Dict[str, int] = {}
    samples: List[str] = []
    attempted_rubrics: List[str] = []
    seen_urls_this_run: set[str] = set()
    seen_body_hashes_this_run: set[str] = set()
    seen_evidence_hashes_this_run: set[str] = set()

    def note(reason: str, url: str) -> str:
        kind = _skip_kind(reason)
        target = hard_skip_reasons if kind == "hard" else soft_skip_reasons
        target[reason] = target.get(reason, 0) + 1
        if len(samples) < 10:
            samples.append(f"[{kind}] {reason}: {url}")
        return kind

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
                    all_items.extend(fetch_source(src))
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
            rng.shuffle(all_items)

            print(
                f"[RUBRIC] rubric={rubric_id} audience={aud} candidates_total={len(all_items)} max_scan={MAX_CANDIDATES_PER_RUBRIC}",
                flush=True,
            )

            for cand in all_items[:MAX_CANDIDATES_PER_RUBRIC]:
                url = (cand.get("link") or "").strip()

                elapsed = time.monotonic() - run_started_monotonic
                if elapsed > MAX_RUN_SECONDS:
                    kind = note("max_run_seconds", rubric_id)
                    print(f"[STOP][{kind}] max_run_seconds reached: {elapsed:.1f}s", flush=True)
                    break

                print(f"[CANDIDATE] rubric={rubric_id} audience={aud} url={url}", flush=True)

                if not url.startswith(("http://", "https://")):
                    kind = note("bad_candidate_url", url or "(empty)")
                    print(f"[SKIP][{kind}] bad_candidate_url url={url}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if _SKIP_EXT_RE.search(url):
                    kind = note("skip_non_html_asset", url)
                    print(f"[SKIP][{kind}] skip_non_html_asset url={url}", flush=True)
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
                    print(f"[SKIP][{kind}] skip_non_html_asset canon={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if canon in seen_urls_this_run:
                    kind = note("dup_url_same_run", canon)
                    print(f"[SKIP][{kind}] dup_url_same_run url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_url(canon):
                    if should_allow_evergreen_source_reuse(rubric_id):
                        print(
                            f"[WARN] dup_url_db_ignored evergreen_reuse rubric={rubric_id} url={canon}",
                            flush=True,
                        )
                    else:
                        kind = note("dup_url_db", canon)
                        print(f"[SKIP][{kind}] dup_url_db url={canon}", flush=True)
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
                    print(f"[SKIP][{kind}] evidence_fetch_failed url={canon} err={e}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if len((evidence or "").strip()) < 260:
                    kind = note("no_evidence_short", canon)
                    print(f"[SKIP][{kind}] no_evidence_short url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if rubric_id == "age_norms" and not _is_age_norms_content_fit(evidence):
                    kind = note("rubric_topic_mismatch_source", canon)
                    print(f"[SKIP][{kind}] rubric_topic_mismatch_source url={canon}", flush=True)
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
                    print(f"[SKIP][{kind}] dup_evidence_same_run url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_evidence_hash(evidence_hash):
                    if should_allow_evergreen_source_reuse(rubric_id):
                        print(
                            f"[WARN] dup_evidence_hash_db_ignored evergreen_reuse rubric={rubric_id} url={canon}",
                            flush=True,
                        )
                    else:
                        kind = note("dup_evidence_hash_db", canon)
                        print(f"[SKIP][{kind}] dup_evidence_hash_db url={canon}", flush=True)
                        if kind == "hard":
                            rubric_skips += 1
                        if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                            note("max_skips_per_rubric", rubric_id)
                            print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                            break
                        continue

                sem_source_hit = store.find_semantic_duplicate(
                    evidence,
                    threshold=SEMANTIC_THRESHOLD,
                    since_iso=None,
                    limit=500,
                    compare="evidence",
                )
                if sem_source_hit:
                    # method_piggybank is a methodological/professional rubric.
                    # Source-level semantic dedup is too aggressive here:
                    # many different method articles use the same professional vocabulary
                    # and can score very high while still producing different practical posts.
                    #
                    # URL/evidence hash DB checks follow their own rubric policy.
                    # We still keep the final post checks:
                    # - dup_body_hash_db
                    # - dup_semantic_post
                    #
                    # So for this rubric we only warn and continue to LLM.
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
                            f"[SKIP][{kind}] dup_semantic_source url={canon} "
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

                sd = safe_domain(canon) or safe_domain(url) or "источник"

                try:
                    plain_raw, ok, llm_note = await asyncio.wait_for(
                        generate_post_plain_from_evidence_async(
                            rubric_title=rubric_title,
                            rubric_format=rf,
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
                        ),
                        timeout=MAX_LLM_SECONDS_PER_CANDIDATE,
                    )
                except asyncio.TimeoutError:
                    kind = note("llm_timeout", canon)
                    print(f"[SKIP][{kind}] llm_timeout url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue
                except Exception as e:
                    kind = note("llm_failed", f"{canon} ({e})")
                    print(f"[SKIP][{kind}] llm_failed url={canon} err={e}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if not ok or not plain_raw:
                    kind = note("llm_invalid_output", canon)
                    print(f"[SKIP][{kind}] {llm_note} url={canon}", flush=True)
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
                )

                if not plain or _looks_incomplete_final_body(_body_without_footer(plain)):
                    kind = note("final_invalid_output", f"{canon} (final_body_incomplete)")
                    print(
                        f"[SKIP][{kind}] final_invalid_output "
                        f"reason=final_body_incomplete url={canon}",
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
                            f"reason={final_reason} url={canon}",
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
                    print(f"[SKIP][{kind}] rubric_topic_mismatch_post url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if rubric_id == "tip_of_day" and not _is_tip_of_day_content_fit(plain):
                    kind = note("tip_of_day_post_too_generic", canon)
                    print(f"[SKIP][{kind}] tip_of_day_post_too_generic url={canon}", flush=True)
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
                    print(f"[SKIP][{kind}] dup_body_same_run url={canon}", flush=True)
                    if kind == "hard":
                        rubric_skips += 1
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_body_hash(body_hash):
                    kind = note("dup_body_hash_db", canon)
                    print(f"[SKIP][{kind}] dup_body_hash_db url={canon}", flush=True)
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
                        f"[SKIP][{kind}] dup_semantic_post url={canon} "
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
                    )
                except Exception as e:
                    kind = note("visual_build_failed", f"{canon} ({e})")
                    print(f"[SKIP][{kind}] visual_build_failed url={canon} err={e}", flush=True)
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

                html_full = render_plain_to_telegram_html(plain)

                if DRY_RUN:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    out = STATE_DIR / "dry_run" / ts
                    out.mkdir(parents=True, exist_ok=True)
                    filename = getattr(visual_buffer, "name", f"{posted+1:02d}_{aud}_{rubric_id}.png")
                    ext = Path(filename).suffix or ".png"
                    (out / f"{posted+1:02d}_{aud}_{rubric_id}{ext}").write_bytes(visual_buffer.getvalue())
                    (out / f"{posted+1:02d}_{aud}_{rubric_id}.txt").write_text(plain, encoding="utf-8")
                else:
                    target_chat_id = _resolve_publish_chat_id()
                    if not target_chat_id:
                        raise RuntimeError("Resolved target chat id is empty")
                    try:
                        send_post_with_visual(target_chat_id, visual_buffer, plain, html_full)
                    except Exception as e:
                        kind = note("telegram_send_failed", f"{canon} ({e})")
                        print(f"[SKIP][{kind}] telegram_send_failed url={canon} err={e}", flush=True)
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
                print(f"[POSTED] rubric={rubric_id} audience={aud} url={canon}", flush=True)
                await asyncio.sleep(1.0)
                break

            if (time.monotonic() - run_started_monotonic) > MAX_RUN_SECONDS or posted >= max_posts:
                break

        if (time.monotonic() - run_started_monotonic) > MAX_RUN_SECONDS or posted >= max_posts:
            break

    if posted == 0 and not DRY_RUN:
        if TELEGRAM_DRAFTS_CHAT_ID:
            try:
                send_plain_message(
                    TELEGRAM_DRAFTS_CHAT_ID,
                    _build_posted_zero_alert_plain(
                        now=now,
                        day=day,
                        week_key=week_key,
                        audience=AUDIENCE,
                        provider=PROVIDER,
                        soft_skip_reasons=soft_skip_reasons,
                        hard_skip_reasons=hard_skip_reasons,
                        samples=samples,
                        state_scope=state_scope,
                        db_name=db_path.name,
                        attempted_rubrics=attempted_rubrics,
                    ),
                )
            except Exception as e:
                print(f"[WARN] failed_to_send_posted_zero_alert err={e}", flush=True)
        else:
            print("[WARN] Posted:0 but TELEGRAM_DRAFTS_CHAT_ID not set; no alert sent.", flush=True)

    print(
        f"Publisher done. Posted: {posted}. Week: {week_key}.{' [DRY_RUN]' if DRY_RUN else ''}",
        flush=True,
    )


def run() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    run()

from __future__ import annotations
"""
Publisher (cron/GitHub Actions) v5.0.0

Что изменено:
1) LLM генерирует deep narrative summary вместо сухих тезисов.
2) Добавлена семантическая дедупликация:
   - exact URL
   - exact hash по evidence/post
   - cosine similarity по векторным представлениям evidence/post
3) Если новый материал семантически слишком похож на уже опубликованный,
   он пропускается. Если совпадение с недавним материалом >= порога — идёт alert в техчат.
4) Добавлены защитные лимиты:
   - глобальный лимит времени на весь run
   - лимит кандидатов на рубрику
   - лимит skip-ов на рубрику
   - таймаут на генерацию одного кандидата
5) Добавлено подробное логирование в stdout для GitHub Actions.
6) Tech alerts теперь отправляются безопасно для Telegram:
   - без <br>
   - с fallback на plain text, если HTML parse mode ломается
7) Telegram HTML renderer синхронизирован с новым narrative-форматом постов.
8) В каждый публикуемый пост внедрена система хештегов:
   - рубричный тег по дню недели
   - тег возраста из строки "👶 Возраст:"
   - 1–2 тематических тега, извлечённых из LLM-ответа
   Все хештеги ставятся строго в самом низу сообщения, под ссылкой.
9) Визуальный пайплайн теперь гибридный:
   - основной режим: Pollinations AI
   - fallback: шаблон дня недели + H1 поста через Pillow
10) При TARGET_CHANNEL=test публикация идёт в TELEGRAM_DRAFTS_CHAT_ID.
11) Введены soft_skip / hard_skip и диагностический alert для Posted:0.
12) Разделены state/test и state/prod для истории публикаций.
13) Введены отдельные semantic thresholds для source/evidence и post/body.
14) Добавлен prefilter мусорных documents/* URL для age_norms.
15) Добавлен rubric-guard для age_norms: whitelist/blacklist тем и reject при несоответствии рубрике.
16) Усилен age consistency для age_norms: узкий возрастной диапазон, базовая проверка milestone-age fit, stronger titles.
17) Добавлен site-specific extractor для logopedy.ru + boilerplate guard + защита от ложных semantic collisions score=1.000.
18) Для Monday/tip_of_day добавлена диверсификация candidate pool: round-robin по источникам, cap по домену и чистая test DB v10.
19) Для Monday/tip_of_day запрещён generic final H1: заголовок обязан отражать конкретный приём/действие, а не название рубрики.
"""

import asyncio
from collections import Counter, deque
from io import BytesIO
import json
import hashlib
import html as _html
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import feedparser
from aiogram.types import InlineKeyboardMarkup
import requests
import urllib3
import yaml
from bs4 import BeautifulSoup
from dateutil import tz

from src.services.llm_generator import (
    generate_image_prompt_async,
    generate_post_plain_from_evidence_async,
)
from src.services.publication_store import PublicationStore
from src.services.visual_pipeline import build_post_visual
from src.services.telegram_miniapp import build_mini_app_markup


# =========================
# Paths / env
# =========================

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/4.7.0 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip()
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "prod").strip().lower()
POLLINATIONS_TOKEN = os.getenv("POLLINATIONS_TOKEN", "").strip()

DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()

PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()
POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1000"))
TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))
IMAGE_PROMPT_TIMEOUT_SECONDS = int(os.getenv("IMAGE_PROMPT_TIMEOUT_SECONDS", "60"))
MINI_APP_URL = os.getenv("MINI_APP_URL", "").strip()

SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))
SEMANTIC_THRESHOLD_SOURCE = float(os.getenv("SEMANTIC_THRESHOLD_SOURCE", "0.93"))
SEMANTIC_THRESHOLD_POST = float(os.getenv("SEMANTIC_THRESHOLD_POST", "0.86"))
MAX_LLM_REGEN_ATTEMPTS = int(os.getenv("MAX_LLM_REGEN_ATTEMPTS", "3"))
RECENT_ALERT_HOURS = int(os.getenv("RECENT_ALERT_HOURS", "36"))
STATE_SCOPE = "test" if TARGET_CHANNEL == "test" else "prod"
TEST_DB_VERSION = os.getenv("TEST_DB_VERSION", "v10").strip() or "v10"
PUBLICATION_DB_NAME = f"publication_history_test_{TEST_DB_VERSION}.sqlite3" if STATE_SCOPE == "test" else "publication_history.sqlite3"
TIP_OF_DAY_DOMAIN_CAP = int(os.getenv("TIP_OF_DAY_DOMAIN_CAP", "3"))

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


# =========================
# Helpers
# =========================

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


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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


_AGE_NORMS_URL_NOISE_TOKENS = (
    "anketa",
    "anam",
    "karta",
    "obsl",
    "obsled",
    "obslyed",
    "diagnost",
    "programma",
    "otziv",
    "harakter",
    "haraktyer",
    "studenta-vuza",
    "logidoshkol",
)


def prefilter_candidate_url(rubric_id: str, url: str) -> Optional[str]:
    rid = (rubric_id or "").strip().lower()
    if rid != "age_norms":
        return None
    low = (url or "").lower()
    if "logopediya.com/documents/" not in low:
        return None
    if any(token in low for token in _AGE_NORMS_URL_NOISE_TOKENS):
        return "prefilter_noise_document"
    return None


def _resolve_publish_chat_id() -> str:
    if TARGET_CHANNEL == "test":
        return TELEGRAM_DRAFTS_CHAT_ID
    return TELEGRAM_CHAT_ID


def classify_skip_severity(reason: str) -> str:
    reason = (reason or "").strip().lower()
    if not reason:
        return "soft"

    soft_exact = {
        "bad_candidate_url",
        "skip_non_html_asset",
        "dup_url_same_run",
        "dup_url_db",
        "dup_evidence_same_run",
        "dup_evidence_hash_db",
        "dup_semantic_source",
        "dup_body_hash_after_regen",
        "dup_semantic_post_after_regen",
        "no_evidence_short",
        "no_candidates",
        "unknown_source_id",
        "prefilter_noise_document",
        "rubric_topic_mismatch_source",
        "rubric_topic_mismatch_post_after_regen",
        "age_consistency_missing_age_line",
        "age_consistency_unparsed_age_line",
        "age_consistency_range_too_wide",
        "age_consistency_milestone_mismatch",
        "evidence_boilerplate_extracted",
        "evidence_low_information",
        "extractor_collision_suspected",
        "tip_of_day_title_too_generic",
        "tip_of_day_first_action_missing",
        "tip_of_day_bilingual_too_general",
    }
    if reason in soft_exact:
        return "soft"

    soft_prefixes = ("dup_", "no_evidence_", "rubric_topic_mismatch")
    if reason.startswith(soft_prefixes):
        return "soft"

    hard_exact = {
        "source_fetch_failed",
        "evidence_fetch_failed",
        "llm_timeout",
        "llm_regen_exhausted",
        "max_run_seconds",
    }
    if reason in hard_exact:
        return "hard"

    hard_prefixes = (
        "invalid_",
        "llm_failed",
        "gemini_failed",
        "groq_failed",
        "image_prompt_failed",
        "image_prompt_timeout",
        "telegram_api_error",
        "banned_phrase:",
        "template_leak",
    )
    if reason.startswith(hard_prefixes):
        return "hard"

    return "hard"


def _build_posted_zero_alert_html(
    now: datetime,
    day: str,
    week_key: str,
    audience: str,
    provider: str,
    skip_reasons: Dict[str, int],
    samples: List[str],
) -> str:
    top = sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:12]

    parts: List[str] = [
        "⚠️ <b>Publisher: не удалось опубликовать пост (Posted: 0)</b>",
        f"Дата: {_escape(str(now.date()))} | День: {_escape(day)} | Неделя: {_escape(week_key)}",
        f"AUDIENCE={_escape(audience)} | PROVIDER={_escape(provider)} | TARGET_CHANNEL={_escape(TARGET_CHANNEL)}",
        "",
        "<b>Причины пропуска (топ):</b>",
    ]

    for reason, count in top:
        parts.append(f"• {_escape(reason)}: {_escape(str(count))}")

    if samples:
        parts.append("")
        parts.append("<b>Примеры:</b>")
        for sample in samples[:8]:
            parts.append(_escape(sample))

    return "\n".join(parts)


def _strip_html_tags_for_telegram(text: str) -> str:
    s = text or ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(
        r'<a\s+href=\"([^\"]+)\">(.+?)</a>',
        lambda m: f"{_html.unescape(m.group(2))} ({_html.unescape(m.group(1))})",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre)>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = _html.unescape(s)
    return s.strip()


def _is_probably_parse_mode_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "can't parse entities" in text or "unsupported start tag" in text or "bad request" in text


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
        )
    )


def _is_structural_heading(line: str) -> bool:
    st = (line or "").strip()
    if not st:
        return False
    if _line_matches_structural(st):
        return True
    if st in SECTION_HEADERS:
        return True
    return False


def _slugify_tag_body(text: str) -> str:
    s = (text or "").strip().lower().replace("ё", "е")
    s = re.sub(r"[-–—−]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zа-я_]+", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


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
    if value.startswith("для_детей_"):
        return f"#{value}"
    return f"#для_детей_{value}"


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


_TAG_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_TAG_RU_SUFFIXES = (
    "иями", "ями", "ами", "иях", "ого", "ему", "ому", "ыми", "ими", "ее", "ие", "ые",
    "ое", "ей", "ий", "ый", "ой", "ам", "ям", "ом", "ем", "ах", "ях", "ию", "ью", "ия", "ья",
    "ев", "ов", "ие", "ье", "а", "я", "ы", "и", "е", "о", "у",
)
_TAG_EN_SUFFIXES = ("ingly", "edly", "ing", "ed", "ly", "es", "s")


def _stem_tag_token(token: str) -> str:
    t = (token or "").strip().lower().replace("ё", "е")
    if len(t) <= 3:
        return t
    for suf in _TAG_RU_SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    for suf in _TAG_EN_SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _tag_match_tokens(text: str) -> set[str]:
    raw = _TAG_WORD_RE.findall((text or "").lower().replace("ё", "е"))
    out: set[str] = set()
    for token in raw:
        stem = _stem_tag_token(token)
        if len(stem) >= 3:
            out.add(stem)
    return out


def _is_thematic_tag_relevant(tag: str, body_text: str) -> bool:
    tag_body = (tag or "").strip().lstrip("#")
    if not tag_body:
        return False

    tag_tokens = [
        _stem_tag_token(tok)
        for tok in _TAG_WORD_RE.findall(tag_body.replace("_", " "))
        if len(_stem_tag_token(tok)) >= 3
    ]
    if not tag_tokens:
        return False

    body_tokens = _tag_match_tokens(body_text)
    if not body_tokens:
        return False

    matches = 0
    for tok in tag_tokens:
        if tok in body_tokens:
            matches += 1
            continue
        if any(tok in b or b in tok for b in body_tokens):
            matches += 1

    required = 1 if len(tag_tokens) == 1 else max(1, (len(tag_tokens) + 1) // 2)
    return matches >= required


def _filter_relevant_thematic_tags(tags: List[str], body_text: str) -> List[str]:
    filtered: List[str] = []
    for tag in tags:
        if _is_thematic_tag_relevant(tag, body_text):
            if tag not in filtered:
                filtered.append(tag)
    return filtered[:2]


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
    cleaned: List[str] = []
    for line in lines:
        st = line.strip()
        if SOURCE_LINE_RE.match(st):
            continue
        if st.startswith("🔗 "):
            continue
        cleaned.append(line)

    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return cleaned


def _trim_body_preserving_footer(body_text: str, footer_text: str, max_chars: int) -> str:
    body = (body_text or "").strip()
    footer = (footer_text or "").strip()

    if not footer:
        return body[:max_chars].rstrip()

    composed = f"{body}\n\n{footer}" if body else footer
    if len(composed) <= max_chars:
        return body

    allowance = max_chars - len(footer) - 2
    if allowance <= 0:
        return ""

    cut = body[:allowance]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()


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
        footer_parts.append("")
        footer_parts.append(" ".join(final_tags))
    footer_text = "\n".join(footer_parts).strip()

    trimmed_body = _trim_body_preserving_footer(body_text, footer_text, max_chars)
    if trimmed_body:
        return f"{trimmed_body}\n\n{footer_text}".strip()
    return footer_text


def _extract_h1_from_plain_post(plain_text: str, fallback: str) -> str:
    first = ""
    for line in (plain_text or "").splitlines():
        st = norm_space(line)
        if st:
            first = st
            break
    if not first:
        return norm_space(fallback) or "Логопедия и дефектология"
    if _is_structural_heading(first) or first.startswith('#'):
        return norm_space(fallback) or "Логопедия и дефектология"
    return first


AGE_NORMS_POSITIVE_MARKERS = (
    "возраст",
    "возрастн",
    "норма",
    "ориентир",
    "этап",
    "milestone",
    "milestones",
    "developmental",
    "development",
    "communication milestone",
    "speech development",
    "language development",
    "typical development",
    "по возрасту",
    "развитие речи",
    "развитие языка",
)

AGE_NORMS_NEGATIVE_MARKERS = (
    "дисграф",
    "дислекс",
    "заикан",
    "алали",
    "афази",
    "дизартр",
    "ринолал",
    "дислал",
    "нарушени",
    "коррекц",
    "диагноз",
    "патолог",
    "дефект",
    "therapy",
    "treatment",
    "intervention",
    "disorder",
    "impairment",
)

AGE_NORMS_RED_FLAG_PHRASES = (
    "трудности с письм",
    "замена букв",
    "пропуск слог",
    "ошибки письма",
    "коррекция письма",
    "нарушение письма",
    "reading disorder",
    "writing disorder",
)


def _topic_scan_normalize(text: str) -> str:
    return norm_space((text or "").lower().replace("ё", "е"))


def _contains_any_marker(text: str, markers: Tuple[str, ...]) -> bool:
    blob = _topic_scan_normalize(text)
    return any(marker in blob for marker in markers)


def _find_negative_marker(text: str, markers: Tuple[str, ...]) -> str:
    blob = _topic_scan_normalize(text)
    for marker in markers:
        if marker in blob:
            return marker
    return ""


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        st = norm_space(line)
        if st:
            return st
    return ""

TIP_OF_DAY_GENERIC_TITLE_MARKERS = (
    "совет логопеда дня",
    "совет логопеда",
    "логопедический совет",
    "работа с",
    "развитие ",
    "развитие_",
    "особенности",
    "формирование",
    "поддержка",
    "воспитание",
    "обучение",
    "языковые навыки",
    "речевые навыки",
    "билингваль",
    "двуязыч",
    "для родителей",
)

TIP_OF_DAY_ACTION_VERBS = (
    "попробуйте",
    "читайте",
    "играйте",
    "скажите",
    "говорите",
    "называйте",
    "попросите",
    "повторяйте",
    "покажите",
    "дайте",
    "выберите",
    "используйте",
    "добавьте",
    "опишите",
    "слушайте",
    "спойте",
    "чередуйте",
    "обсуждайте",
    "предложите",
    "прочитайте",
)

TIP_OF_DAY_BILINGUAL_MARKERS = (
    "билингв",
    "двуязыч",
    "два языка",
    "двух язы",
    "на каждом языке",
    "на двух языках",
    "язык среды",
    "родной язык",
    "второй язык",
)

TIP_OF_DAY_ACTION_SECTION_MARKERS = (
    "что попробовать сегодня",
    "что можно попробовать дома",
    "что сделать сегодня",
)

AGE_GENERIC_TITLE_MARKERS = (
    "возрастная норма",
    "норма речи",
    "развитие речи",
    "для родителей",
    "речевые нормы",
)

AGE_EARLY_COMPLEX_MARKERS = (
    "предложени",
    "рассказывает историю",
    "пересказывает",
    "длинные фразы",
    "сложные фразы",
)

AGE_LATE_SIMPLE_MARKERS = (
    "около 50 слов",
    "50 слов",
    "первые слова",
    "двухслов",
    "2-слов",
    "два слова",
)


def _extract_first_nonempty_line_index(lines: List[str]) -> int:
    for idx, line in enumerate(lines):
        if norm_space(line):
            return idx
    return -1


def _parse_age_bounds(age_value: str) -> Optional[Tuple[float, float]]:
    s = (age_value or "").strip().lower().replace("ё", "е")
    s = s.replace("–", "-").replace("—", "-").replace(",", ".")
    nums = []
    for part in re.findall(r"\d+(?:\.\d+)?", s):
        try:
            nums.append(float(part))
        except Exception:
            continue
    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (min(nums), max(nums))


def _age_range_too_wide(age_value: str) -> bool:
    bounds = _parse_age_bounds(age_value)
    if not bounds:
        return True
    lo, hi = bounds
    return (hi - lo) > 2.0


def _iter_content_sentences(text: str) -> List[str]:
    blob = norm_space(text)
    if not blob:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", blob)
    return [norm_space(p).strip(" .") for p in parts if norm_space(p)]


def _tip_of_day_title_too_generic(title: str) -> bool:
    blob = _topic_scan_normalize(title)
    if not blob:
        return True
    return any(marker in blob for marker in TIP_OF_DAY_GENERIC_TITLE_MARKERS)


def _tip_of_day_title_matches_rubric(title: str, rubric_title: str) -> bool:
    title_blob = _topic_scan_normalize(title)
    rubric_blob = _topic_scan_normalize(rubric_title)
    if not title_blob:
        return True
    if rubric_blob and title_blob == rubric_blob:
        return True
    if title_blob in {"совет логопеда дня", "совет логопеда", "логопедический совет"}:
        return True
    return False


def _cleanup_tip_of_day_title(action: str) -> str:
    s = norm_space(action)
    if not s:
        return ""
    replacements = [
        (r"^попробуйте\s+поиграть", "Поиграйте"),
        (r"^попробуйте\s+играть", "Играйте"),
        (r"^попробуйте\s+читать", "Читайте"),
        (r"^попробуйте\s+почитать", "Читайте"),
        (r"^попробуйте\s+называть", "Называйте"),
        (r"^попробуйте\s+повторять", "Повторяйте"),
        (r"^попробуйте\s+говорить", "Говорите"),
        (r"^попробуйте\s+сказать", "Скажите"),
        (r"^попробуйте\s+дать", "Дайте"),
        (r"^попробуйте\s+давать", "Давайте"),
        (r"^попробуйте\s+использовать", "Используйте"),
        (r"^попробуйте\s+слушать", "Слушайте"),
        (r"^попробуйте\s+спеть", "Спойте"),
        (r"^попробуйте\s+обсуждать", "Обсуждайте"),
        (r"^попробуйте\s+предложить", "Предложите"),
        (r"^попробуйте\s+выбрать", "Выберите"),
    ]
    for pattern, repl in replacements:
        s2 = re.sub(pattern, repl, s, flags=re.IGNORECASE)
        if s2 != s:
            s = s2
            break
    s = re.sub(r"^сегодня\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^что\s+(попробовать\s+сегодня|можно\s+попробовать\s+дома)\s*:?[ ]*", "", s, flags=re.IGNORECASE)
    s = re.split(r"[,;:](?=\s|$)", s, maxsplit=1)[0]
    s = re.split(r"чтобы", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = norm_space(s).strip(" .")
    if s:
        s = s[0].upper() + s[1:]
    return s


def _looks_tip_of_day_action_sentence(sentence: str) -> bool:
    blob = _topic_scan_normalize(sentence)
    if not blob:
        return False
    return any(re.search(rf"\b{re.escape(verb)}\b", blob) for verb in TIP_OF_DAY_ACTION_VERBS)


def _extract_tip_of_day_section_action(lines: List[str]) -> str:
    for idx, raw in enumerate(lines):
        st = norm_space(raw)
        if not st:
            continue
        low = _topic_scan_normalize(st)
        if any(marker in low for marker in TIP_OF_DAY_ACTION_SECTION_MARKERS):
            if ":" in st:
                after = norm_space(st.split(":", 1)[1])
                if after:
                    for sent in _iter_content_sentences(after):
                        if _looks_tip_of_day_action_sentence(sent):
                            return sent
            collected: List[str] = []
            for nxt in lines[idx + 1:]:
                s2 = norm_space(nxt)
                if not s2:
                    break
                if _is_structural_heading(s2) or SOURCE_LINE_RE.match(s2) or s2.startswith("🔗 ") or s2.startswith("#"):
                    break
                collected.append(s2)
                if len(" ".join(collected)) > 240:
                    break
            joined = " ".join(collected)
            for sent in _iter_content_sentences(joined):
                if _looks_tip_of_day_action_sentence(sent):
                    return sent
    return ""


def _extract_tip_of_day_first_substantive_sentence(lines: List[str]) -> str:
    collected: List[str] = []
    for idx, raw in enumerate(lines):
        st = norm_space(raw)
        if idx == 0:
            continue
        if not st:
            continue
        if AGE_LINE_RE.match(st) or AUDIENCE_LINE_RE.match(st):
            continue
        if SOURCE_LINE_RE.match(st) or st.startswith("🔗 ") or st.startswith("#"):
            break
        if _is_structural_heading(st):
            continue
        collected.append(st)
        if len(" ".join(collected)) > 260:
            break
    joined = " ".join(collected)
    sentences = _iter_content_sentences(joined)
    return sentences[0] if sentences else ""


def _build_tip_of_day_title_from_action(action_sentence: str, plain_text: str, rubric_title: str) -> str:
    action = norm_space(action_sentence)
    if not action:
        return rubric_title or "Совет логопеда дня"
    action = re.sub(r"^(сегодня\s+)?", "", action, flags=re.IGNORECASE)
    action = re.sub(r"^(что\s+попробовать\s+сегодня|что\s+можно\s+попробовать\s+дома)\s*:?\s*", "", action, flags=re.IGNORECASE)
    action = re.split(r"[,;:](?=\s|$)", action, maxsplit=1)[0]
    action = re.split(r"\bчтобы\b", action, maxsplit=1, flags=re.IGNORECASE)[0]
    action = norm_space(action).strip(" .")
    if not action:
        return rubric_title or "Совет логопеда дня"
    words = action.split()
    if len(words) > 7:
        action = " ".join(words[:7]).strip(" ,.;:")
    if action:
        action = action[0].upper() + action[1:]
    if len(action) < 8:
        return rubric_title or "Совет логопеда дня"
    return action


def strengthen_tip_of_day_title(plain_text: str, rubric_title: str) -> str:
    lines = (plain_text or "").replace("\r\n", "\n").split("\n")
    idx = _extract_first_nonempty_line_index(lines)
    if idx < 0:
        return plain_text
    current = norm_space(lines[idx])
    if not (_is_structural_heading(current) or _tip_of_day_title_too_generic(current)):
        return plain_text
    action_line = _extract_tip_of_day_section_action(lines) or _extract_tip_of_day_first_substantive_sentence(lines)
    stronger = _build_tip_of_day_title_from_action(action_line, plain_text, rubric_title)
    lines[idx] = stronger
    return "\n".join(lines).strip()


def validate_tip_of_day_editorial_fit(plain_text: str, rubric_title: str) -> Optional[str]:
    lines = (plain_text or "").replace("\r\n", "\n").split("\n")
    title = _first_nonempty_line(plain_text)
    if _is_structural_heading(title) or _tip_of_day_title_too_generic(title):
        return "tip_of_day_title_too_generic"

    first_sentence = _extract_tip_of_day_first_substantive_sentence(lines)
    if not first_sentence or not _looks_tip_of_day_action_sentence(first_sentence):
        return "tip_of_day_first_action_missing"

    blob = _topic_scan_normalize(plain_text)
    if any(marker in blob for marker in TIP_OF_DAY_BILINGUAL_MARKERS):
        action_line = _extract_tip_of_day_section_action(lines) or first_sentence
        if not action_line or not _looks_tip_of_day_action_sentence(action_line):
            return "tip_of_day_bilingual_too_general"

    return None


def _looks_generic_age_norms_title(title: str) -> bool:
    blob = _topic_scan_normalize(title)
    if not blob:
        return True
    if any(marker in blob for marker in AGE_GENERIC_TITLE_MARKERS):
        return True
    return not bool(re.search(r"\d", blob))


def _build_stronger_age_norms_title(age_value: str, plain_text: str) -> str:
    age_clean = norm_space(age_value).replace("-", "–")
    blob = _topic_scan_normalize(plain_text)
    if any(marker in blob for marker in ("слова", "фразы", "предложени", "вопрос", "понимает")):
        return f"Что обычно говорит ребёнок в {age_clean}"
    return f"Речевые ориентиры в {age_clean}"


def strengthen_age_norms_title(plain_text: str, rubric_title: str) -> str:
    lines = (plain_text or "").replace("\r\n", "\n").split("\n")
    age_value = _extract_age_value(lines)
    if not age_value:
        return plain_text
    idx = _extract_first_nonempty_line_index(lines)
    if idx < 0:
        return plain_text
    current = norm_space(lines[idx])
    stronger = _build_stronger_age_norms_title(age_value, plain_text)
    if _is_structural_heading(current) or _looks_generic_age_norms_title(current):
        lines[idx] = stronger
        return "\n".join(lines).strip()
    return plain_text


def validate_age_norms_age_consistency(plain_text: str) -> Optional[str]:
    lines = (plain_text or "").replace("\r\n", "\n").split("\n")
    age_value = _extract_age_value(lines)
    if not age_value:
        return "age_consistency_missing_age_line"

    bounds = _parse_age_bounds(age_value)
    if not bounds:
        return "age_consistency_unparsed_age_line"

    if _age_range_too_wide(age_value):
        return "age_consistency_range_too_wide"

    lo, hi = bounds
    blob = _topic_scan_normalize(plain_text)

    if hi <= 2.0 and any(marker in blob for marker in AGE_EARLY_COMPLEX_MARKERS):
        return "age_consistency_milestone_mismatch"

    if lo >= 3.0 and any(marker in blob for marker in AGE_LATE_SIMPLE_MARKERS):
        return "age_consistency_milestone_mismatch"

    return None


def validate_rubric_source_fit(rubric_id: str, candidate_title: str, canonical_url: str, evidence_text: str) -> Optional[str]:
    rid = (rubric_id or "").strip().lower()
    if rid != "age_norms":
        return None

    scan = "\n".join([candidate_title or "", canonical_url or "", (evidence_text or "")[:4000]])
    negative_marker = _find_negative_marker(scan, AGE_NORMS_NEGATIVE_MARKERS)
    if negative_marker:
        return "rubric_topic_mismatch_source"

    if _contains_any_marker(scan, AGE_NORMS_RED_FLAG_PHRASES):
        return "rubric_topic_mismatch_source"

    if not _contains_any_marker(scan, AGE_NORMS_POSITIVE_MARKERS):
        return "rubric_topic_mismatch_source"

    return None


def validate_rubric_post_fit(rubric_id: str, plain_text: str, rubric_title: str, source_url: str = "", evidence_text: str = "") -> Optional[str]:
    rid = (rubric_id or "").strip().lower()
    if rid != "age_norms":
        return None

    title = _first_nonempty_line(plain_text)
    scan = "\n".join([title or "", plain_text or "", source_url or "", (evidence_text or "")[:2500]])

    negative_marker = _find_negative_marker(scan, AGE_NORMS_NEGATIVE_MARKERS)
    if negative_marker:
        return "rubric_topic_mismatch_post_after_regen"

    if _contains_any_marker(scan, AGE_NORMS_RED_FLAG_PHRASES):
        return "rubric_topic_mismatch_post_after_regen"

    positive_scan = "\n".join([title or "", plain_text or "", rubric_title or ""])
    if not _contains_any_marker(positive_scan, AGE_NORMS_POSITIVE_MARKERS):
        return "rubric_topic_mismatch_post_after_regen"

    return None




def _attach_source_id(items: List[Dict[str, str]], source_id: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for it in items:
        row = dict(it)
        row["source_id"] = source_id
        out.append(row)
    return out


def _round_robin_candidates(
    items_by_source: Dict[str, List[Dict[str, str]]],
    seed: int,
    max_items: int,
    domain_cap: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    source_ids = list(items_by_source.keys())
    rng.shuffle(source_ids)

    queues: Dict[str, deque] = {}
    for sid in source_ids:
        bucket = list(items_by_source.get(sid) or [])
        rng.shuffle(bucket)
        queues[sid] = deque(bucket)

    selected: List[Dict[str, str]] = []
    overflow: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    domain_counts: Counter = Counter()

    progressed = True
    while len(selected) < max_items and progressed:
        progressed = False
        for sid in source_ids:
            q = queues.get(sid)
            if not q:
                continue
            picked = None
            while q:
                cand = q.popleft()
                url = (cand.get("link") or "").strip()
                if not url or url in seen_urls:
                    continue
                dom = safe_domain(url)
                if dom and domain_counts.get(dom, 0) >= domain_cap:
                    overflow.append(cand)
                    continue
                picked = cand
                break
            if picked is None:
                continue
            selected.append(picked)
            seen_urls.add((picked.get("link") or "").strip())
            dom = safe_domain((picked.get("link") or "").strip())
            if dom:
                domain_counts[dom] += 1
            progressed = True
            if len(selected) >= max_items:
                break

    if len(selected) < max_items:
        remainder: List[Dict[str, str]] = []
        remainder.extend(overflow)
        for sid in source_ids:
            q = queues.get(sid)
            if q:
                remainder.extend(list(q))
        rng.shuffle(remainder)
        for cand in remainder:
            url = (cand.get("link") or "").strip()
            if not url or url in seen_urls:
                continue
            selected.append(cand)
            seen_urls.add(url)
            if len(selected) >= max_items:
                break

    return selected


def diversify_candidates_for_rubric(
    rubric_id: str,
    items_by_source: Dict[str, List[Dict[str, str]]],
    seed: int,
    max_items: int,
) -> List[Dict[str, str]]:
    rid = (rubric_id or "").strip().lower()
    flattened: List[Dict[str, str]] = []
    for bucket in items_by_source.values():
        flattened.extend(bucket)

    if rid != "tip_of_day":
        rng = random.Random(seed)
        rng.shuffle(flattened)
        return flattened

    diversified = _round_robin_candidates(
        items_by_source=items_by_source,
        seed=seed,
        max_items=max(max_items, 60),
        domain_cap=max(1, TIP_OF_DAY_DOMAIN_CAP),
    )
    return diversified or flattened


def _describe_candidate_mix(items: List[Dict[str, str]], limit: int = 12) -> str:
    domains = Counter()
    sources = Counter()
    for it in items[:limit]:
        url = (it.get("link") or "").strip()
        dom = safe_domain(url) or "(none)"
        sid = (it.get("source_id") or "?")
        domains[dom] += 1
        sources[sid] += 1
    domain_part = ", ".join(f"{k}:{v}" for k, v in domains.most_common(6)) or "n/a"
    source_part = ", ".join(f"{k}:{v}" for k, v in sources.most_common(6)) or "n/a"
    return f"sources[{source_part}] domains[{domain_part}]"


# =========================
# Sources
# =========================

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
    out: Dict[str, Source] = {}
    for s in cfg.get("sources", []) or []:
        out[s["id"]] = Source(**s)
    return out


def fetch_rss(url: str) -> List[Dict[str, str]]:
    d = feedparser.parse(url)
    out: List[Dict[str, str]] = []
    for e in d.entries[:50]:
        out.append({
            "title": norm_space(getattr(e, "title", "")),
            "link": getattr(e, "link", ""),
            "summary": norm_space(re.sub("<.*?>", "", getattr(e, "summary", ""))),
        })
    return out


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


def _collect_links(base_url: str, soup: BeautifulSoup, selector: str, href_re: Optional[str] = None) -> List[Dict[str, str]]:
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
    out = []
    for it in items:
        link = it.get("link", "")
        if re.search(r"/news/\d{4}-\d{2}/?$", link):
            continue
        out.append(it)
    return out[:80]


def parse_logomag_lib(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    return _collect_links(url, soup, "main a, div.content a, a", r"/lib/[^\"']+")[:80]


def parse_logoportal_articles(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    return _collect_links(url, soup, "main a, div#content a, article a, a", r"(statya-|/statya-)")[:80]


def parse_logopedy_articles(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "div.content a, main a, a", r"logoped-article|logoped-literature|portal/[^#]+")
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
    items = [it for it in items if not re.search(r"/page/\d+/?$", it["link"])]
    return items[:120]


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


# =========================
# Evidence extraction
# =========================

_SKIP_EXT_RE = re.compile(r"\.(ppt|pptx|pdf|doc|docx|xls|xlsx|zip|rar|mp3|mp4)$", re.IGNORECASE)


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


_GENERIC_EVIDENCE_STOP_MARKERS = [
    "нравится статья",
    "расскажи друзьям",
    "поделиться",
    "поделитесь",
    "комментари",
    "похожие материалы",
    "похожие статьи",
    "рекомендуем также",
]

_LOGOPEDY_BOILERPLATE_MARKERS = [
    "логопеды россии",
    "логопедические центры",
    "полезные материалы",
    "рабочие материалы",
    "студентам",
    "новые статьи на сайте",
    "зонды постановочные",
    "присоединяйтесь к нам",
    "разместить материал",
    "вход/выход",
    "регистрация",
    "логопедические центры",
]


def _is_logopedy_useful_url(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
        return parsed.netloc.lower().endswith("logopedy.ru") and "/portal/logopeduseful/" in (parsed.path or "")
    except Exception:
        return False


def _looks_like_boilerplate_line(text: str) -> bool:
    low = (text or "").lower().replace("ё", "е")
    return any(marker in low for marker in _LOGOPEDY_BOILERPLATE_MARKERS)


def _select_article_root_from_h1(soup: BeautifulSoup, h1_tag: Any) -> Any:
    node = getattr(h1_tag, "parent", None)
    while node is not None and getattr(node, "name", None) not in (None, "body", "html"):
        try:
            blocks = node.find_all(["p", "li", "h2", "h3"])
            text_len = len(norm_space(node.get_text(" ", strip=True)))
            if len(blocks) >= 6 and text_len >= 900:
                return node
        except Exception:
            pass
        node = getattr(node, "parent", None)
    return getattr(h1_tag, "parent", None) or soup.body or soup


def _collect_text_chunks_from_root(root: Any, max_chars: int, stop_markers: List[str]) -> List[str]:
    chunks: List[str] = []
    total_len = 0
    for el in root.find_all(["h2", "h3", "p", "li"]):
        txt = norm_space(el.get_text(" ", strip=True))
        if not txt:
            continue
        low = txt.lower().replace("ё", "е")
        min_len = 20 if getattr(el, "name", "") in ("p", "li") else 8
        if len(txt) < min_len:
            continue
        if any(bad in low for bad in ["cookie", "privacy", "политик", "подпис", "реклама", "скачать", "регистрация"]):
            continue
        if _looks_like_boilerplate_line(low):
            continue
        if any(marker in low for marker in stop_markers):
            if len(chunks) >= 4:
                break
            continue
        chunks.append(txt)
        total_len += len(txt)
        if total_len > max_chars * 1.45:
            break
    return chunks


def _dedupe_preserve_order(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        key = line.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _extract_logopedy_useful_evidence(url: str, soup: BeautifulSoup, max_chars: int) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    h1 = soup.find("h1")
    if not h1:
        return ""

    title = norm_space(h1.get_text(" ", strip=True))
    root = _select_article_root_from_h1(soup, h1)
    chunks: List[str] = []
    if title:
        chunks.append(title)

    chunks.extend(_collect_text_chunks_from_root(root, max_chars=max_chars, stop_markers=_GENERIC_EVIDENCE_STOP_MARKERS))
    uniq = _dedupe_preserve_order(chunks)
    out = "\n".join(uniq).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0].strip()
    return out


def _extract_generic_evidence(url: str, soup: BeautifulSoup, max_chars: int) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    h1 = soup.find("h1")
    root = None
    title = ""
    if h1:
        title = norm_space(h1.get_text(" ", strip=True))
        root = _select_article_root_from_h1(soup, h1)

    if root is None:
        root = (
            soup.select_one("div#dle-content")
            or soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )

    chunks: List[str] = []
    if title:
        chunks.append(title)
    chunks.extend(_collect_text_chunks_from_root(root, max_chars=max_chars, stop_markers=_GENERIC_EVIDENCE_STOP_MARKERS))

    uniq = _dedupe_preserve_order(chunks)
    out = "\n".join(uniq).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0].strip()
    return out


def assess_evidence_quality(url: str, evidence_text: str) -> Optional[str]:
    text = (evidence_text or "").strip()
    if not text:
        return None

    low = text.lower().replace("ё", "е")
    lines = [norm_space(x) for x in text.splitlines() if norm_space(x)]
    tokens = re.findall(r"[a-zа-я0-9]+", low, flags=re.IGNORECASE)
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    if _is_logopedy_useful_url(url):
        boilerplate_hits = sum(1 for marker in _LOGOPEDY_BOILERPLATE_MARKERS if marker in low)
        first_lines_hits = sum(1 for line in lines[:12] if _looks_like_boilerplate_line(line))
        if boilerplate_hits >= 2 or first_lines_hits >= 2:
            return "evidence_boilerplate_extracted"
        if len(tokens) < 120 and unique_ratio < 0.38:
            return "evidence_low_information"
        if len(lines) < 5 and len(tokens) < 160:
            return "evidence_low_information"

    return None


def is_probable_false_semantic_collision(candidate_url: str, evidence_text: str, hit: Any) -> bool:
    if float(getattr(hit, "similarity", 0.0) or 0.0) < 0.999:
        return False
    cand_dom = safe_domain(candidate_url)
    hit_dom = safe_domain(getattr(hit, "canonical_url", ""))
    if not cand_dom or cand_dom != hit_dom:
        return False
    if _is_logopedy_useful_url(candidate_url):
        return True
    quality_reason = assess_evidence_quality(candidate_url, evidence_text)
    return quality_reason in {"evidence_boilerplate_extracted", "evidence_low_information"}


def extract_evidence_text(url: str, max_chars: int = 3600) -> str:
    r = requests.get(url, headers=HEADERS, timeout=35, verify=_verify_for_url(url))
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return ""

    soup = BeautifulSoup(r.text, "lxml")
    if _is_logopedy_useful_url(url):
        return _extract_logopedy_useful_evidence(url, soup, max_chars=max_chars)
    return _extract_generic_evidence(url, soup, max_chars=max_chars)


# =========================
# Telegram markup helpers
# =========================

_MD_V2_RE = re.compile(r"([_\*\[\]\(\)~`>#+\-=|{}.!\\])")


def escape_markdown_v2(text: str) -> str:
    s = (text or "")
    s = s.replace("\\", "\\\\")
    return _MD_V2_RE.sub(r"\\\1", s)


def escape_markdown_v2_url(url: str) -> str:
    s = (url or "").replace("\\", "\\\\")
    s = s.replace("(", "\\(")
    return s.replace(")", "\\)")


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
            if url.startswith(("http://", "https://")):
                out.append(_link_anchor(url, prefix="🔗 "))
            else:
                out.append(_escape(st))
            continue

        out.append(_escape(s))

    return "\n".join(out).strip()


def render_plain_to_telegram_markdown_v2(plain_text: str) -> str:
    lines = (plain_text or "").splitlines()
    if not lines:
        return ""

    out: List[str] = []
    for idx, raw in enumerate(lines):
        s = raw.rstrip("\n")
        st = s.strip()

        if idx == 0 and st:
            out.append(f"*{escape_markdown_v2(st)}*")
            continue

        if _is_structural_heading(st):
            out.append(f"*{escape_markdown_v2(st)}*")
            continue

        if st.startswith("🔗 "):
            url = st[2:].strip()
            if url.startswith(("http://", "https://")):
                out.append(f"🔗 [Читать оригинальный материал]({escape_markdown_v2_url(url)})")
            else:
                out.append(escape_markdown_v2(st))
            continue

        out.append(escape_markdown_v2(s))

    return "\n".join(out).strip()


def render_post_for_telegram(plain_text: str) -> str:
    if TELEGRAM_PARSE_MODE.lower() == "markdownv2":
        return render_plain_to_telegram_markdown_v2(plain_text)
    return render_plain_to_telegram_html(plain_text)


def render_semantic_alert_message(
    candidate_url: str,
    matched_url: str,
    score: float,
    audience: str,
    rubric_id: str,
    match_field: str,
) -> Tuple[str, str]:
    plain = (
        "⚠️ Semantic dedup alert\n"
        f"Материал отклонён: cosine similarity ≥ {SEMANTIC_THRESHOLD_SOURCE:.2f}\n"
        f"AUDIENCE={audience} | RUBRIC={rubric_id} | FIELD={match_field}\n\n"
        f"Новый кандидат: {candidate_url}\n"
        f"Похож на: {matched_url}\n"
        f"Cosine: {score:.3f}"
    )
    if TELEGRAM_PARSE_MODE.lower() == "markdownv2":
        rendered = (
            "⚠️ *Semantic dedup alert*\n"
            f"Материал отклонён: cosine similarity ≥ {escape_markdown_v2(f'{SEMANTIC_THRESHOLD:.2f}')}\n"
            f"AUDIENCE={escape_markdown_v2(audience)} \\| RUBRIC={escape_markdown_v2(rubric_id)} \\| FIELD={escape_markdown_v2(match_field)}\n\n"
            f"Новый кандидат: [ссылка]({escape_markdown_v2_url(candidate_url)})\n"
            f"Похож на: [ссылка]({escape_markdown_v2_url(matched_url)})\n"
            f"Cosine: *{escape_markdown_v2(f'{score:.3f}')}*"
        )
        return rendered, plain

    cand = _html.escape(candidate_url, quote=True)
    hit = _html.escape(matched_url, quote=True)
    html_text = (
        "⚠️ <b>Semantic dedup alert</b>\n"
        f"Материал отклонён: cosine similarity ≥ {SEMANTIC_THRESHOLD_SOURCE:.2f}\n"
        f"AUDIENCE={_escape(audience)} | RUBRIC={_escape(rubric_id)} | FIELD={_escape(match_field)}\n\n"
        f"Новый кандидат: <a href=\"{cand}\">{_escape(candidate_url)}</a>\n"
        f"Похож на: <a href=\"{hit}\">{_escape(matched_url)}</a>\n"
        f"Cosine: <b>{score:.3f}</b>"
    )
    return html_text, plain




def render_semantic_alert_summary_message(
    alerts: List[Dict[str, Any]],
    audience: str,
    rubric_id: str,
) -> Tuple[str, str]:
    total = len(alerts)
    top = sorted(alerts, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:5]

    plain_parts: List[str] = [
        f"⚠️ Semantic dedup summary — {rubric_id}",
        f"AUDIENCE={audience} | RUBRIC={rubric_id} | skipped={total}",
        "",
    ]
    for idx, item in enumerate(top, start=1):
        score_text = f"{float(item.get('score', 0.0)):.3f}"
        plain_parts.append(f"{idx}) candidate: {item.get('candidate_url', '')}")
        plain_parts.append(f"   matched:   {item.get('matched_url', '')}")
        plain_parts.append(f"   field: {item.get('match_field', '')} | cosine: {score_text}")
        plain_parts.append("")
    if total > len(top):
        plain_parts.append(f"И ещё: {total - len(top)}")
    plain = "\n".join(plain_parts).rstrip()

    if TELEGRAM_PARSE_MODE.lower() == "markdownv2":
        rendered_parts: List[str] = [
            f"⚠️ *Semantic dedup summary — {escape_markdown_v2(rubric_id)}*",
            f"AUDIENCE={escape_markdown_v2(audience)} \| RUBRIC={escape_markdown_v2(rubric_id)} \| skipped={escape_markdown_v2(str(total))}",
            "",
        ]
        for idx, item in enumerate(top, start=1):
            score_text = f"{float(item.get('score', 0.0)):.3f}"
            rendered_parts.append(f"{escape_markdown_v2(str(idx))}\) candidate: {escape_markdown_v2(item.get('candidate_url', ''))}")
            rendered_parts.append(f"   matched:   {escape_markdown_v2(item.get('matched_url', ''))}")
            rendered_parts.append(f"   field: {escape_markdown_v2(item.get('match_field', ''))} \| cosine: {escape_markdown_v2(score_text)}")
            rendered_parts.append("")
        if total > len(top):
            rendered_parts.append(f"И ещё: {escape_markdown_v2(str(total - len(top)))}")
        return "\n".join(rendered_parts).rstrip(), plain

    html_parts: List[str] = [
        f"⚠️ <b>Semantic dedup summary — {_escape(rubric_id)}</b>",
        f"AUDIENCE={_escape(audience)} | RUBRIC={_escape(rubric_id)} | skipped={_escape(str(total))}",
        "",
    ]
    for idx, item in enumerate(top, start=1):
        score_text = f"{float(item.get('score', 0.0)):.3f}"
        html_parts.append(f"<b>{_escape(str(idx))})</b> candidate: {_escape(item.get('candidate_url', ''))}")
        html_parts.append(f"&nbsp;&nbsp;&nbsp;matched: {_escape(item.get('matched_url', ''))}")
        html_parts.append(f"&nbsp;&nbsp;&nbsp;field: {_escape(item.get('match_field', ''))} | cosine: {_escape(score_text)}")
        html_parts.append("")
    if total > len(top):
        html_parts.append(f"И ещё: {_escape(str(total - len(top)))}")
    return "\n".join(html_parts).rstrip(), plain


def _build_posted_zero_alert_message(
    now: datetime,
    day: str,
    week_key: str,
    audience: str,
    provider: str,
    skip_reasons: Dict[str, int],
    samples: List[str],
    soft_skip_reasons: Dict[str, int],
    hard_skip_reasons: Dict[str, int],
    stop_events: List[str],
    attempted_rubrics: List[str],
) -> Tuple[str, str]:
    soft_top = sorted(soft_skip_reasons.items(), key=lambda x: x[1], reverse=True)[:8]
    hard_top = sorted(hard_skip_reasons.items(), key=lambda x: x[1], reverse=True)[:8]
    total_soft = sum(soft_skip_reasons.values())
    total_hard = sum(hard_skip_reasons.values())

    unique_attempted: List[str] = []
    for item in attempted_rubrics:
        if item not in unique_attempted:
            unique_attempted.append(item)

    unique_stops: List[str] = []
    for item in stop_events:
        if item not in unique_stops:
            unique_stops.append(item)

    plain_parts: List[str] = [
        "⚠️ Publisher diagnostic: пост не опубликован (Posted: 0)",
        f"Дата: {str(now.date())} | День: {day} | Неделя: {week_key}",
        f"AUDIENCE={audience} | PROVIDER={provider} | TARGET_CHANNEL={TARGET_CHANNEL} | STATE_SCOPE={STATE_SCOPE}",
        f"History DB: {PUBLICATION_DB_NAME}",
        f"Rubrics attempted: {', '.join(unique_attempted) if unique_attempted else 'n/a'}",
        f"Soft skips: {total_soft} | Hard skips: {total_hard}",
        "",
        "Диагноз:",
    ]
    if unique_stops:
        for item in unique_stops[:6]:
            plain_parts.append(f"• {item}")
    else:
        plain_parts.append("• publishable candidates were not found before run completion")

    if hard_top:
        plain_parts.append("")
        plain_parts.append("Hard skip reasons:")
        for reason, count in hard_top:
            plain_parts.append(f"• {reason}: {count}")

    if soft_top:
        plain_parts.append("")
        plain_parts.append("Soft skip reasons:")
        for reason, count in soft_top:
            plain_parts.append(f"• {reason}: {count}")

    if samples:
        plain_parts.append("")
        plain_parts.append("Examples:")
        plain_parts.extend(samples[:8])
    plain = "\n".join(plain_parts)

    if TELEGRAM_PARSE_MODE.lower() == "markdownv2":
        rendered_parts: List[str] = [
            "⚠️ *Publisher diagnostic: пост не опубликован \(Posted: 0\)*",
            f"Дата: {escape_markdown_v2(str(now.date()))} \| День: {escape_markdown_v2(day)} \| Неделя: {escape_markdown_v2(week_key)}",
            f"AUDIENCE={escape_markdown_v2(audience)} \| PROVIDER={escape_markdown_v2(provider)} \| TARGET\_CHANNEL={escape_markdown_v2(TARGET_CHANNEL)}",
            f"Rubrics attempted: {escape_markdown_v2(', '.join(unique_attempted) if unique_attempted else 'n/a')}",
            f"Soft skips: {escape_markdown_v2(str(total_soft))} \| Hard skips: {escape_markdown_v2(str(total_hard))}",
            "",
            "*Диагноз:*",
        ]
        if unique_stops:
            for item in unique_stops[:6]:
                rendered_parts.append(f"• {escape_markdown_v2(item)}")
        else:
            rendered_parts.append("• publishable candidates were not found before run completion")
        if hard_top:
            rendered_parts.append("")
            rendered_parts.append("*Hard skip reasons:*")
            for reason, count in hard_top:
                rendered_parts.append(f"• {escape_markdown_v2(reason)}: {escape_markdown_v2(str(count))}")
        if soft_top:
            rendered_parts.append("")
            rendered_parts.append("*Soft skip reasons:*")
            for reason, count in soft_top:
                rendered_parts.append(f"• {escape_markdown_v2(reason)}: {escape_markdown_v2(str(count))}")
        if samples:
            rendered_parts.append("")
            rendered_parts.append("*Examples:*")
            for sample in samples[:8]:
                rendered_parts.append(escape_markdown_v2(sample))
        return "\n".join(rendered_parts), plain

    html_parts: List[str] = [
        "⚠️ <b>Publisher diagnostic: пост не опубликован (Posted: 0)</b>",
        f"Дата: {_escape(str(now.date()))} | День: {_escape(day)} | Неделя: {_escape(week_key)}",
        f"AUDIENCE={_escape(audience)} | PROVIDER={_escape(provider)} | TARGET_CHANNEL={_escape(TARGET_CHANNEL)}",
        f"Rubrics attempted: {_escape(', '.join(unique_attempted) if unique_attempted else 'n/a')}",
        f"Soft skips: {_escape(str(total_soft))} | Hard skips: {_escape(str(total_hard))}",
        "",
        "<b>Диагноз:</b>",
    ]
    if unique_stops:
        for item in unique_stops[:6]:
            html_parts.append(f"• {_escape(item)}")
    else:
        html_parts.append("• publishable candidates were not found before run completion")
    if hard_top:
        html_parts.append("")
        html_parts.append("<b>Hard skip reasons:</b>")
        for reason, count in hard_top:
            html_parts.append(f"• {_escape(reason)}: {_escape(str(count))}")
    if soft_top:
        html_parts.append("")
        html_parts.append("<b>Soft skip reasons:</b>")
        for reason, count in soft_top:
            html_parts.append(f"• {_escape(reason)}: {_escape(str(count))}")
    if samples:
        html_parts.append("")
        html_parts.append("<b>Examples:</b>")
        for sample in samples[:8]:
            html_parts.append(_escape(sample))
    return "\n".join(html_parts), plain


# =========================
# Telegram send
# =========================

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
        description = ""
        if isinstance(payload, dict):
            description = payload.get("description", "") or ""
        if not description:
            description = r.text or ""
        raise RuntimeError(f"telegram_api_error:{r.status_code}:{description}")

    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(f"telegram_api_error:{r.status_code}:{payload.get('description', '')}")

    return payload or {}


def _markup_json(reply_markup: Optional[InlineKeyboardMarkup]) -> str:
    if not reply_markup:
        return ""
    return reply_markup.model_dump_json(exclude_none=True)


def send_message(
    chat_id: str,
    rendered_text: str,
    fallback_text: Optional[str] = None,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    if not chat_id:
        raise RuntimeError("chat_id is missing")

    base_data: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": rendered_text,
        "disable_web_page_preview": "true",
    }
    if reply_markup:
        base_data["reply_markup"] = _markup_json(reply_markup)

    if TELEGRAM_PARSE_MODE:
        try:
            data = dict(base_data)
            data["parse_mode"] = TELEGRAM_PARSE_MODE
            tg_request("sendMessage", data=data)
            return
        except Exception as e:
            if not _is_probably_parse_mode_error(e):
                raise

    fallback_data: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": fallback_text or _strip_html_tags_for_telegram(rendered_text),
        "disable_web_page_preview": "true",
    }
    if reply_markup:
        fallback_data["reply_markup"] = _markup_json(reply_markup)
    tg_request("sendMessage", data=fallback_data)


def _photo_file_tuple(photo_buffer: BytesIO) -> tuple[str, bytes, str]:
    filename = getattr(photo_buffer, "name", "cover.png")
    mime_type = getattr(photo_buffer, "mime_type", "image/png")
    return (filename, photo_buffer.getvalue(), mime_type)


def send_post_with_visual(
    chat_id: str,
    photo_buffer: BytesIO,
    plain_post: str,
    rendered_post: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    plain_bytes = len((plain_post or "").encode("utf-8"))
    file_tuple = _photo_file_tuple(photo_buffer)

    if plain_bytes <= TG_CAPTION_MAX_BYTES:
        try:
            data: Dict[str, Any] = {"chat_id": chat_id, "caption": rendered_post}
            if TELEGRAM_PARSE_MODE:
                data["parse_mode"] = TELEGRAM_PARSE_MODE
            if reply_markup:
                data["reply_markup"] = _markup_json(reply_markup)
            tg_request("sendPhoto", data=data, files={"photo": file_tuple})
            return
        except Exception:
            pass

    data: Dict[str, Any] = {"chat_id": chat_id, "caption": ""}
    if reply_markup:
        data["reply_markup"] = _markup_json(reply_markup)
    tg_request("sendPhoto", data=data, files={"photo": file_tuple})
    send_message(chat_id, rendered_post, fallback_text=plain_post)


def send_poll(chat_id: str, question: str, options: List[str]) -> None:
    tg_request(
        "sendPoll",
        data={
            "chat_id": chat_id,
            "question": question,
            "options": json.dumps(options, ensure_ascii=False),
            "is_anonymous": "true",
            "allows_multiple_answers": "false",
        },
    )


def send_semantic_alert(
    chat_id: str,
    candidate_url: str,
    matched_url: str,
    score: float,
    audience: str,
    rubric_id: str,
    match_field: str,
) -> None:
    rendered, plain = render_semantic_alert_message(
        candidate_url=candidate_url,
        matched_url=matched_url,
        score=score,
        audience=audience,
        rubric_id=rubric_id,
        match_field=match_field,
    )
    send_message(chat_id, rendered, fallback_text=plain)


def build_interactive_followup(day_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[InlineKeyboardMarkup]]:
    key = (day_key or "").upper()
    if key == "WE":
        return {
            "question": 'Этот формат "Миф / Факт" был полезен?',
            "options": ["Да, очень", "Немного", "Хочу ещё примеры"],
        }, None
    if key == "FR":
        return {
            "question": "Этот вопрос откликается вашей семье?",
            "options": ["Да", "Частично", "Пока нет"],
        }, None
    if MINI_APP_URL and key in {"TH", "SU"}:
        return None, build_mini_app_markup(MINI_APP_URL)
    return None, None


# =========================
# Main run
# =========================

async def amain() -> None:
    rub_cfg = load_yaml(CFG_DIR / "rubrics.yml")
    channel_cfg = rub_cfg.get("channel", {}) or {}
    pub_cfg = rub_cfg.get("publishing", {}) or {}

    tzname = channel_cfg.get("timezone", "Asia/Nicosia")
    now = get_local_now(tzname)
    run_started_monotonic = time.monotonic()
    print(f"[START] Publisher started at {now.isoformat()} target_channel={TARGET_CHANNEL} state_scope={STATE_SCOPE} db={PUBLICATION_DB_NAME}", flush=True)

    week_key = iso_week_key(now)
    day = weekday_key(now)

    max_posts = int(pub_cfg.get("max_posts_per_run", 1))
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    hashtags = channel_cfg.get("hashtags", []) or []

    sources = load_sources()
    store = PublicationStore(STATE_DIR / PUBLICATION_DB_NAME)
    recent_since_iso = _start_recent_window(now).isoformat()

    audiences_cfg = rub_cfg.get("audiences", {}) or {}
    if AUDIENCE == "both":
        aud_list = ["parents", "pros"]
    elif AUDIENCE in ("parents", "pros"):
        aud_list = [AUDIENCE]
    else:
        aud_list = ["parents"]

    posted = 0
    skip_reasons: Dict[str, int] = {}
    soft_skip_reasons: Dict[str, int] = {}
    hard_skip_reasons: Dict[str, int] = {}
    samples: List[str] = []
    stop_events: List[str] = []
    attempted_rubrics: List[str] = []
    seen_urls_this_run: set[str] = set()
    seen_body_hashes_this_run: set[str] = set()
    seen_evidence_hashes_this_run: set[str] = set()

    def note(reason: str, url: str, severity: Optional[str] = None) -> None:
        sev = severity or classify_skip_severity(reason)
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        bucket = hard_skip_reasons if sev == "hard" else soft_skip_reasons
        bucket[reason] = bucket.get(reason, 0) + 1
        if len(samples) < 8:
            samples.append(f"• [{sev}] {reason}: {url}")

    async def _generate_unique_post(
        rubric_id: str,
        rubric_title: str,
        rf: str,
        aud: str,
        title_suffix: str,
        sd: str,
        canon: str,
        evidence: str,
        day: str,
    ) -> Tuple[Optional[str], str]:
        duplicate_hint = ""
        for attempt in range(1, MAX_LLM_REGEN_ATTEMPTS + 1):
            temperature = min(0.95, 0.2 + (attempt - 1) * 0.18)
            variation_seed = random.randint(1000, 999999)
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
                        day_key=day,
                        temperature=temperature,
                        variation_seed=variation_seed,
                        regeneration_hint=duplicate_hint,
                    ),
                    timeout=MAX_LLM_SECONDS_PER_CANDIDATE,
                )
            except asyncio.TimeoutError:
                return None, "llm_timeout"

            if not ok or not plain_raw:
                duplicate_hint = "Сделай структуру и формулировки заметно более отличающимися от предыдущих публикаций."
                if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                    return None, llm_note
                continue

            plain = finalize_plain_post_for_publication(
                plain_text=plain_raw,
                day_key=day,
                source_domain=sd,
                source_url=canon,
                max_chars=POST_MAX_CHARS,
            )

            if (rubric_id or "").strip().lower() == "tip_of_day":
                plain = strengthen_tip_of_day_title(plain, rubric_title)
                tip_of_day_reason = validate_tip_of_day_editorial_fit(plain, rubric_title)
                if tip_of_day_reason:
                    duplicate_hint = (
                        "Для рубрики 'Совет логопеда дня' нужен один конкретный домашний приём на сегодня. "
                        "Заголовок должен быть прикладным, а первая смысловая фраза — сразу вести к действию. "
                        "Избегай обзорных тем вроде 'работа с...', 'развитие...', 'особенности...'. "
                        "Для bilingual-темы допустим один конкретный домашний приём, а не общая тема."
                    )
                    if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                        return None, tip_of_day_reason
                    continue

            if (rubric_id or "").strip().lower() == "age_norms":
                plain = strengthen_age_norms_title(plain, rubric_title)
                age_consistency_reason = validate_age_norms_age_consistency(plain)
                if age_consistency_reason:
                    duplicate_hint = (
                        "Для рубрики возрастных норм нужен узкий возрастной диапазон: например, 1–2, 2–3, 3–4 или 4–5 лет, а не слишком широкий интервал. "
                        "Заголовок должен быть конкретным и возрастным, а ориентиры — соответствовать именно этому возрасту. "
                        "Не смешивай milestones для слишком разных этапов развития в одном посте."
                    )
                    if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                        return None, age_consistency_reason
                    continue

            rubric_mismatch_reason = validate_rubric_post_fit(
                rubric_id=rubric_id,
                plain_text=plain,
                rubric_title=rubric_title,
                source_url=canon,
                evidence_text=evidence,
            )
            if rubric_mismatch_reason:
                duplicate_hint = (
                    "Материал не соответствует рубрике возрастных норм. "
                    "Нужны именно возрастные ориентиры и milestones, без патологий, диагнозов, коррекции и симптомов нарушений. "
                    "Сфокусируй текст на том, что обычно появляется по возрасту и какие ориентиры можно спокойно отслеживать."
                )
                if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                    return None, rubric_mismatch_reason
                continue

            body_hash = sha1(norm_space(plain))
            if body_hash in seen_body_hashes_this_run or store.has_body_hash(body_hash):
                duplicate_hint = "Перепиши пост в другом narrative-угле, без повторения уже опубликованных формулировок и структуры."
                if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                    return None, "dup_body_hash_after_regen"
                continue

            sem_body_hit = store.find_semantic_duplicate(
                plain,
                threshold=SEMANTIC_THRESHOLD_POST,
                since_iso=None,
                limit=500,
                compare="body",
            )
            if sem_body_hit:
                duplicate_hint = (
                    "Новый текст всё ещё слишком семантически похож на уже опубликованный. "
                    "Смени структуру, начальный ракурс, примеры и полезный следующий шаг. "
                    f"Ориентир similarity<{SEMANTIC_THRESHOLD_POST:.2f}."
                )
                if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                    return None, "dup_semantic_post_after_regen"
                continue

            return plain, "ok"

        return None, "llm_regen_exhausted"

    for aud in aud_list:
        if posted >= max_posts:
            break

        aud_cfg = audiences_cfg.get(aud, {}) or {}
        title_suffix = (aud_cfg.get("title_suffix", "") or "").strip()
        rubrics = aud_cfg.get("rubrics", []) or []

        for rubric in rubrics:
            if posted >= max_posts:
                break
            if not is_due(rubric, now):
                continue

            rf = (rubric.get("format") or "").strip().lower()
            if rf == "quality_dashboard":
                continue

            rubric_id = (rubric.get("id") or "").strip() or "unknown"
            rubric_title = rubric.get("title", "Рубрика") or "Рубрика"
            attempted_rubrics.append(rubric_id)
            rubric_hard_skips = 0
            rubric_soft_skips = 0
            rubric_stop_reason = ""
            rubric_posted = False
            semantic_alerts_for_rubric: List[Dict[str, Any]] = []

            def record_rubric_skip(reason: str, url: str, severity: Optional[str] = None) -> bool:
                nonlocal rubric_hard_skips, rubric_soft_skips, rubric_stop_reason
                sev = severity or classify_skip_severity(reason)
                note(reason, url, severity=sev)
                if sev == "hard":
                    rubric_hard_skips += 1
                    if rubric_hard_skips >= MAX_SKIPS_PER_RUBRIC:
                        rubric_stop_reason = f"hard_skip_budget_exceeded:{rubric_id}"
                        stop_events.append(rubric_stop_reason)
                        print(
                            f"[STOP] hard_skip_budget_exceeded rubric={rubric_id} hard_skips={rubric_hard_skips} soft_skips={rubric_soft_skips}",
                            flush=True,
                        )
                        return True
                else:
                    rubric_soft_skips += 1
                return False

            items_by_source: Dict[str, List[Dict[str, str]]] = {}
            for sid in rubric.get("sources", []) or []:
                src = sources.get(sid)
                if not src:
                    note("unknown_source_id", sid, severity="soft")
                    continue
                try:
                    fetched = _attach_source_id(fetch_source(src), sid)
                    items_by_source[sid] = fetched
                except Exception as e:
                    note("source_fetch_failed", f"{sid}: {e}", severity="hard")

            all_items = diversify_candidates_for_rubric(
                rubric_id=rubric_id,
                items_by_source=items_by_source,
                seed=int(hashlib.sha1(f"{now.date()}|{rubric_id}|{aud}".encode("utf-8")).hexdigest()[:8], 16),
                max_items=MAX_CANDIDATES_PER_RUBRIC,
            )

            if not all_items:
                note("no_candidates", rubric_id, severity="soft")
                stop_events.append(f"no_candidates:{rubric_id}")
                continue

            print(
                f"[RUBRIC] rubric={rubric_id} audience={aud} candidates_total={len(all_items)} max_scan={MAX_CANDIDATES_PER_RUBRIC} mix={_describe_candidate_mix(all_items)}",
                flush=True,
            )

            for cand in all_items[:MAX_CANDIDATES_PER_RUBRIC]:
                url = (cand.get("link") or "").strip()

                elapsed = time.monotonic() - run_started_monotonic
                if elapsed > MAX_RUN_SECONDS:
                    note("max_run_seconds", rubric_id, severity="hard")
                    rubric_stop_reason = f"max_run_seconds:{rubric_id}"
                    stop_events.append(rubric_stop_reason)
                    print(f"[STOP] max_run_seconds reached: {elapsed:.1f}s", flush=True)
                    break

                print(f"[CANDIDATE] rubric={rubric_id} audience={aud} url={url}", flush=True)

                if not url.startswith(("http://", "https://")):
                    if record_rubric_skip("bad_candidate_url", url or "(empty)", severity="soft"):
                        break
                    print(f"[SKIP][soft] bad_candidate_url url={url}", flush=True)
                    continue

                if _SKIP_EXT_RE.search(url):
                    if record_rubric_skip("skip_non_html_asset", url, severity="soft"):
                        break
                    print(f"[SKIP][soft] skip_non_html_asset url={url}", flush=True)
                    continue

                canon = get_canonical(url)
                if _SKIP_EXT_RE.search(canon):
                    if record_rubric_skip("skip_non_html_asset", canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] skip_non_html_asset canon={canon}", flush=True)
                    continue

                prefilter_reason = prefilter_candidate_url(rubric_id, canon)
                if prefilter_reason:
                    if record_rubric_skip(prefilter_reason, canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] {prefilter_reason} url={canon}", flush=True)
                    continue

                if canon in seen_urls_this_run:
                    if record_rubric_skip("dup_url_same_run", canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] dup_url_same_run url={canon}", flush=True)
                    continue

                if store.has_url(canon):
                    if record_rubric_skip("dup_url_db", canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] dup_url_db url={canon}", flush=True)
                    continue

                try:
                    evidence = extract_evidence_text(canon, max_chars=3600)
                except Exception as e:
                    should_stop = record_rubric_skip("evidence_fetch_failed", f"{canon} ({e})", severity="hard")
                    print(f"[SKIP][hard] evidence_fetch_failed url={canon} err={e}", flush=True)
                    if should_stop:
                        break
                    continue

                if len((evidence or "").strip()) < 260:
                    if record_rubric_skip("no_evidence_short", canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] no_evidence_short url={canon}", flush=True)
                    continue

                evidence_quality_reason = assess_evidence_quality(canon, evidence)
                if evidence_quality_reason:
                    if record_rubric_skip(evidence_quality_reason, canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] {evidence_quality_reason} url={canon}", flush=True)
                    continue

                source_fit_reason = validate_rubric_source_fit(
                    rubric_id=rubric_id,
                    candidate_title=cand.get("title", ""),
                    canonical_url=canon,
                    evidence_text=evidence,
                )
                if source_fit_reason:
                    if record_rubric_skip(source_fit_reason, canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] {source_fit_reason} url={canon}", flush=True)
                    continue

                evidence_hash = sha1(norm_space(evidence))
                if evidence_hash in seen_evidence_hashes_this_run:
                    if record_rubric_skip("dup_evidence_same_run", canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] dup_evidence_same_run url={canon}", flush=True)
                    continue

                if store.has_evidence_hash(evidence_hash):
                    if record_rubric_skip("dup_evidence_hash_db", canon, severity="soft"):
                        break
                    print(f"[SKIP][soft] dup_evidence_hash_db url={canon}", flush=True)
                    continue

                sem_source_hit = store.find_semantic_duplicate(
                    evidence,
                    threshold=SEMANTIC_THRESHOLD_SOURCE,
                    since_iso=None,
                    limit=500,
                    compare="evidence",
                )
                if sem_source_hit:
                    if is_probable_false_semantic_collision(canon, evidence, sem_source_hit):
                        if record_rubric_skip("extractor_collision_suspected", canon, severity="soft"):
                            break
                        print(
                            f"[SKIP][soft] extractor_collision_suspected url={canon} matched={sem_source_hit.canonical_url} score={sem_source_hit.similarity:.3f}",
                            flush=True,
                        )
                        continue

                    if record_rubric_skip("dup_semantic_source", canon, severity="soft"):
                        break
                    print(
                        f"[SKIP][soft] dup_semantic_source url={canon} matched={sem_source_hit.canonical_url} score={sem_source_hit.similarity:.3f}",
                        flush=True,
                    )
                    if not DRY_RUN and TELEGRAM_DRAFTS_CHAT_ID:
                        recent_hit = store.find_semantic_duplicate(
                            evidence,
                            threshold=SEMANTIC_THRESHOLD_SOURCE,
                            since_iso=recent_since_iso,
                            limit=120,
                            compare="evidence",
                        )
                        if recent_hit and not is_probable_false_semantic_collision(canon, evidence, recent_hit):
                            semantic_alerts_for_rubric.append({
                                "candidate_url": canon,
                                "matched_url": recent_hit.canonical_url,
                                "score": recent_hit.similarity,
                                "match_field": recent_hit.match_field,
                            })
                    continue

                sd = safe_domain(canon) or safe_domain(url) or "источник"

                plain, generation_note = await _generate_unique_post(
                    rubric_id=rubric_id,
                    rubric_title=rubric_title,
                    rf=rf,
                    aud=aud,
                    title_suffix=title_suffix,
                    sd=sd,
                    canon=canon,
                    evidence=evidence,
                    day=day,
                )
                if not plain:
                    severity = classify_skip_severity(generation_note)
                    should_stop = record_rubric_skip(generation_note, canon, severity=severity)
                    print(f"[SKIP][{severity}] {generation_note} url={canon}", flush=True)
                    if should_stop:
                        break
                    continue

                body_hash = sha1(norm_space(plain))

                h1_title = _extract_h1_from_plain_post(plain, fallback=rubric_title)
                image_prompt = ""
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

                visual_buffer, visual_meta = build_post_visual(
                    title=h1_title,
                    day_key=day,
                    image_prompt=image_prompt,
                    pollinations_token=POLLINATIONS_TOKEN,
                    fallback_title=rubric_title,
                )
                print(
                    f"[VISUAL] rubric={rubric_id} mode={visual_meta.get('mode')} reason={visual_meta.get('reason')} image_prompt_note={image_prompt_note}",
                    flush=True,
                )

                rendered_post = render_post_for_telegram(plain)

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
                    poll_payload, reply_markup = build_interactive_followup(day)
                    send_post_with_visual(target_chat_id, visual_buffer, plain, rendered_post, reply_markup=reply_markup)
                    if poll_payload:
                        try:
                            send_poll(target_chat_id, poll_payload["question"], poll_payload["options"])
                        except Exception as e:
                            print(f"[WARN] failed_to_send_poll err={e}", flush=True)

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
                rubric_posted = True
                print(f"[POSTED] rubric={rubric_id} audience={aud} url={canon}", flush=True)
                await asyncio.sleep(1.0)
                break

            if not rubric_posted and not rubric_stop_reason:
                rubric_stop_reason = f"candidates_exhausted:{rubric_id}"
                stop_events.append(rubric_stop_reason)

            if semantic_alerts_for_rubric and not DRY_RUN and TELEGRAM_DRAFTS_CHAT_ID:
                try:
                    rendered_summary, plain_summary = render_semantic_alert_summary_message(
                        alerts=semantic_alerts_for_rubric,
                        audience=aud,
                        rubric_id=rubric_id,
                    )
                    send_message(TELEGRAM_DRAFTS_CHAT_ID, rendered_summary, fallback_text=plain_summary)
                except Exception as e:
                    print(f"[WARN] failed_to_send_semantic_summary err={e}", flush=True)

            if (time.monotonic() - run_started_monotonic) > MAX_RUN_SECONDS:
                break
            if posted >= max_posts:
                break

        if (time.monotonic() - run_started_monotonic) > MAX_RUN_SECONDS:
            break
        if posted >= max_posts:
            break

    if posted == 0 and not DRY_RUN:
        if TELEGRAM_DRAFTS_CHAT_ID:
            try:
                send_message(
                    TELEGRAM_DRAFTS_CHAT_ID,
                    *_build_posted_zero_alert_message(
                        now=now,
                        day=day,
                        week_key=week_key,
                        audience=AUDIENCE,
                        provider=PROVIDER,
                        skip_reasons=skip_reasons,
                        samples=samples,
                        soft_skip_reasons=soft_skip_reasons,
                        hard_skip_reasons=hard_skip_reasons,
                        stop_events=stop_events,
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

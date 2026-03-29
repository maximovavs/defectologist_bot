from __future__ import annotations
"""
Publisher (cron/GitHub Actions) v4.3.0

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
"""

import asyncio
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

USER_AGENT = "logoped-channel-bot/4.3.0 (+https://github.com/)"
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
MAX_LLM_REGEN_ATTEMPTS = int(os.getenv("MAX_LLM_REGEN_ATTEMPTS", "3"))
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


def _resolve_publish_chat_id() -> str:
    if TARGET_CHANNEL == "test":
        return TELEGRAM_DRAFTS_CHAT_ID
    return TELEGRAM_CHAT_ID


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

    final_tags: List[str] = []
    for tag in [rubric_tag, age_tag, *thematic_tags]:
        tag = (tag or "").strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        if tag not in final_tags:
            final_tags.append(tag)

    body_text = "\n".join(body_lines).strip()
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


def extract_evidence_text(url: str, max_chars: int = 3600) -> str:
    r = requests.get(url, headers=HEADERS, timeout=35, verify=_verify_for_url(url))
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return ""

    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root = (
        soup.select_one("div#dle-content")
        or soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    chunks: List[str] = []
    h1 = soup.find("h1")
    if h1:
        chunks.append(norm_space(h1.get_text(" ", strip=True)))

    for el in root.select("h2, h3, p, li"):
        txt = norm_space(el.get_text(" ", strip=True))
        if len(txt) < 20:
            continue
        low = txt.lower()
        if any(bad in low for bad in ["cookie", "privacy", "политик", "подпис", "реклама", "скачать", "регистрация"]):
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
        f"Материал отклонён: cosine similarity ≥ {SEMANTIC_THRESHOLD:.2f}\n"
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
        f"Материал отклонён: cosine similarity ≥ {SEMANTIC_THRESHOLD:.2f}\n"
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
    for item in top:
        plain_parts.append(
            f"• {item.get('candidate_url', '')} -> {item.get('matched_url', '')} | FIELD={item.get('match_field', '')} | Cosine={float(item.get('score', 0.0)):.3f}"
        )
    if total > len(top):
        plain_parts.append("")
        plain_parts.append(f"И ещё: {total - len(top)}")
    plain = "\n".join(plain_parts)

    if TELEGRAM_PARSE_MODE.lower() == "markdownv2":
        rendered_parts: List[str] = [
            f"⚠️ *Semantic dedup summary — {escape_markdown_v2(rubric_id)}*",
            f"AUDIENCE={escape_markdown_v2(audience)} \\| RUBRIC={escape_markdown_v2(rubric_id)} \\| skipped={escape_markdown_v2(str(total))}",
            "",
        ]
        for item in top:
            rendered_parts.append(
                f"• [candidate]({escape_markdown_v2_url(item.get('candidate_url', ''))}) → [matched]({escape_markdown_v2_url(item.get('matched_url', ''))}) \\| FIELD={escape_markdown_v2(item.get('match_field', ''))} \\| Cosine={escape_markdown_v2(f"{float(item.get('score', 0.0)):.3f}")}"
            )
        if total > len(top):
            rendered_parts.append("")
            rendered_parts.append(f"И ещё: {escape_markdown_v2(str(total - len(top)))}")
        return "\n".join(rendered_parts), plain

    html_parts: List[str] = [
        f"⚠️ <b>Semantic dedup summary — {_escape(rubric_id)}</b>",
        f"AUDIENCE={_escape(audience)} | RUBRIC={_escape(rubric_id)} | skipped={_escape(str(total))}",
        "",
    ]
    for item in top:
        cand = _html.escape(item.get("candidate_url", ""), quote=True)
        hit = _html.escape(item.get("matched_url", ""), quote=True)
        html_parts.append(
            f'• <a href="{cand}">candidate</a> → <a href="{hit}">matched</a> | FIELD={_escape(item.get("match_field", ""))} | Cosine={_escape(f"{float(item.get("score", 0.0)):.3f}")}'
        )
    if total > len(top):
        html_parts.append("")
        html_parts.append(f"И ещё: {_escape(str(total - len(top)))}")
    return "\n".join(html_parts), plain

def _build_posted_zero_alert_message(
    now: datetime,
    day: str,
    week_key: str,
    audience: str,
    provider: str,
    skip_reasons: Dict[str, int],
    samples: List[str],
) -> Tuple[str, str]:
    top = sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:12]
    plain_parts: List[str] = [
        "⚠️ Publisher: не удалось опубликовать пост (Posted: 0)",
        f"Дата: {str(now.date())} | День: {day} | Неделя: {week_key}",
        f"AUDIENCE={audience} | PROVIDER={provider} | TARGET_CHANNEL={TARGET_CHANNEL}",
        "",
        "Причины пропуска (топ):",
    ]
    for reason, count in top:
        plain_parts.append(f"• {reason}: {count}")
    if samples:
        plain_parts.append("")
        plain_parts.append("Примеры:")
        plain_parts.extend(samples[:8])
    plain = "\n".join(plain_parts)

    if TELEGRAM_PARSE_MODE.lower() == "markdownv2":
        rendered_parts: List[str] = [
            "⚠️ *Publisher: не удалось опубликовать пост \\(Posted: 0\\)*",
            f"Дата: {escape_markdown_v2(str(now.date()))} \\| День: {escape_markdown_v2(day)} \\| Неделя: {escape_markdown_v2(week_key)}",
            f"AUDIENCE={escape_markdown_v2(audience)} \\| PROVIDER={escape_markdown_v2(provider)} \\| TARGET_CHANNEL={escape_markdown_v2(TARGET_CHANNEL)}",
            "",
            "*Причины пропуска \\(топ\\):*",
        ]
        for reason, count in top:
            rendered_parts.append(f"• {escape_markdown_v2(reason)}: {escape_markdown_v2(str(count))}")
        if samples:
            rendered_parts.append("")
            rendered_parts.append("*Примеры:*")
            for sample in samples[:8]:
                rendered_parts.append(escape_markdown_v2(sample))
        return "\n".join(rendered_parts), plain

    html_parts: List[str] = [
        "⚠️ <b>Publisher: не удалось опубликовать пост (Posted: 0)</b>",
        f"Дата: {_escape(str(now.date()))} | День: {_escape(day)} | Неделя: {_escape(week_key)}",
        f"AUDIENCE={_escape(audience)} | PROVIDER={_escape(provider)} | TARGET_CHANNEL={_escape(TARGET_CHANNEL)}",
        "",
        "<b>Причины пропуска (топ):</b>",
    ]
    for reason, count in top:
        html_parts.append(f"• {_escape(reason)}: {_escape(str(count))}")
    if samples:
        html_parts.append("")
        html_parts.append("<b>Примеры:</b>")
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
    print(f"[START] Publisher started at {now.isoformat()} target_channel={TARGET_CHANNEL}", flush=True)

    week_key = iso_week_key(now)
    day = weekday_key(now)

    max_posts = int(pub_cfg.get("max_posts_per_run", 1))
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    hashtags = channel_cfg.get("hashtags", []) or []

    sources = load_sources()
    store = PublicationStore(STATE_DIR / "publication_history.sqlite3")
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
    samples: List[str] = []
    seen_urls_this_run: set[str] = set()
    seen_body_hashes_this_run: set[str] = set()
    seen_evidence_hashes_this_run: set[str] = set()

    def note(reason: str, url: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if len(samples) < 8:
            samples.append(f"• {reason}: {url}")

    async def _generate_unique_post(
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

            body_hash = sha1(norm_space(plain))
            if body_hash in seen_body_hashes_this_run or store.has_body_hash(body_hash):
                duplicate_hint = "Перепиши пост в другом narrative-угле, без повторения уже опубликованных формулировок и структуры."
                if attempt >= MAX_LLM_REGEN_ATTEMPTS:
                    return None, "dup_body_hash_after_regen"
                continue

            sem_body_hit = store.find_semantic_duplicate(
                plain,
                threshold=SEMANTIC_THRESHOLD,
                since_iso=None,
                limit=500,
                compare="body",
            )
            if sem_body_hit:
                duplicate_hint = (
                    "Новый текст всё ещё слишком семантически похож на уже опубликованный. "
                    "Смени структуру, начальный ракурс, примеры и полезный следующий шаг. "
                    f"Ориентир similarity<{SEMANTIC_THRESHOLD:.2f}."
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
            rubric_skips = 0
            semantic_alerts_for_rubric: List[Dict[str, Any]] = []

            all_items: List[Dict[str, str]] = []
            for sid in rubric.get("sources", []) or []:
                src = sources.get(sid)
                if not src:
                    note("unknown_source_id", sid)
                    continue
                try:
                    all_items.extend(fetch_source(src))
                except Exception as e:
                    note("source_fetch_failed", f"{sid}: {e}")

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
                    note("max_run_seconds", rubric_id)
                    print(f"[STOP] max_run_seconds reached: {elapsed:.1f}s", flush=True)
                    break

                print(f"[CANDIDATE] rubric={rubric_id} audience={aud} url={url}", flush=True)

                if not url.startswith(("http://", "https://")):
                    note("bad_candidate_url", url or "(empty)")
                    rubric_skips += 1
                    print(f"[SKIP] bad_candidate_url url={url}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if _SKIP_EXT_RE.search(url):
                    note("skip_non_html_asset", url)
                    rubric_skips += 1
                    print(f"[SKIP] skip_non_html_asset url={url}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                canon = get_canonical(url)
                if _SKIP_EXT_RE.search(canon):
                    note("skip_non_html_asset", canon)
                    rubric_skips += 1
                    print(f"[SKIP] skip_non_html_asset canon={canon}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if canon in seen_urls_this_run:
                    note("dup_url_same_run", canon)
                    rubric_skips += 1
                    print(f"[SKIP] dup_url_same_run url={canon}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_url(canon):
                    note("dup_url_db", canon)
                    rubric_skips += 1
                    print(f"[SKIP] dup_url_db url={canon}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                try:
                    evidence = extract_evidence_text(canon, max_chars=3600)
                except Exception as e:
                    note("evidence_fetch_failed", f"{canon} ({e})")
                    rubric_skips += 1
                    print(f"[SKIP] evidence_fetch_failed url={canon} err={e}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if len((evidence or "").strip()) < 260:
                    note("no_evidence_short", canon)
                    rubric_skips += 1
                    print(f"[SKIP] no_evidence_short url={canon}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                evidence_hash = sha1(norm_space(evidence))
                if evidence_hash in seen_evidence_hashes_this_run:
                    note("dup_evidence_same_run", canon)
                    rubric_skips += 1
                    print(f"[SKIP] dup_evidence_same_run url={canon}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                if store.has_evidence_hash(evidence_hash):
                    note("dup_evidence_hash_db", canon)
                    rubric_skips += 1
                    print(f"[SKIP] dup_evidence_hash_db url={canon}", flush=True)
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
                    note("dup_semantic_source", canon)
                    rubric_skips += 1
                    print(
                        f"[SKIP] dup_semantic_source url={canon} matched={sem_source_hit.canonical_url} score={sem_source_hit.similarity:.3f}",
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
                            semantic_alerts_for_rubric.append({
                                "candidate_url": canon,
                                "matched_url": recent_hit.canonical_url,
                                "score": recent_hit.similarity,
                                "match_field": recent_hit.match_field,
                            })
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
                        break
                    continue

                sd = safe_domain(canon) or safe_domain(url) or "источник"

                plain, generation_note = await _generate_unique_post(
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
                    note(generation_note, canon)
                    rubric_skips += 1
                    print(f"[SKIP] {generation_note} url={canon}", flush=True)
                    if rubric_skips >= MAX_SKIPS_PER_RUBRIC:
                        note("max_skips_per_rubric", rubric_id)
                        print(f"[STOP] max_skips_per_rubric reached for {rubric_id}", flush=True)
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
                print(f"[POSTED] rubric={rubric_id} audience={aud} url={canon}", flush=True)
                await asyncio.sleep(1.0)
                break

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

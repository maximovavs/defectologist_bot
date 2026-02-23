from __future__ import annotations

"""
Publisher (cron/GitHub Actions)

Запуск:
  python -m src.publisher.run_publisher

Ключевые гарантии:
- Quality dashboard НИКОГДА не публикуется в публичный канал.
  Он отправляется ТОЛЬКО в TELEGRAM_DRAFTS_CHAT_ID.
  Если TELEGRAM_DRAFTS_CHAT_ID не задан — job падает (fail-closed).
- Основные посты отправляются в TELEGRAM_CHAT_ID (который workflow может подменять: main/drafts/test).
- Ротация рубрик: cadence/byweekday из config/rubrics.yml строго учитываются.
"""

import os
import re
import json
import time
import random
import hashlib
import shutil
import math
import html as _html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
import yaml
import feedparser
from bs4 import BeautifulSoup
from dateutil import tz


# =========================
# Paths / env
# =========================

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
ASSETS_DIR = ROOT / "assets"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/2.2.0 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()               # publish target (workflow may override)
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip() # technical channel

DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")

TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()  # HTML | Markdown | ""

REWRITE_PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()  # none|auto|groq|gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()  # parents|pros|both

POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1000"))
PARENTS_MAX_BODY_CHARS = int(os.getenv("PARENTS_MAX_BODY_CHARS", "900"))
PROS_MAX_BODY_CHARS = int(os.getenv("PROS_MAX_BODY_CHARS", "1050"))

TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))

INSECURE_TLS_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("INSECURE_TLS_DOMAINS", "") or "").split(",")
    if d.strip()
]


# =========================
# Services imports (already in repo)
# =========================

from src.services.image_builder import render_image_card  # card builder with textwrap wrapping
from src.services.llm_generator import (
    enforce_total_chars_keep_structure as _enforce_total_chars_keep_structure,
    has_required_structure_plain_v3 as _has_required_structure_plain_v3,
    rewrite_if_enabled_plain as _rewrite_if_enabled_plain,
)


# =========================
# Helpers
# =========================

def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def clamp_text(s: str, max_len: int) -> str:
    s = norm_space(s)
    if len(s) <= max_len:
        return s
    return (s[:max_len].rstrip(" .,:;—-") + "…").strip()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_local_now(tzname: str) -> datetime:
    return datetime.now(tz=tz.gettz(tzname))


def iso_week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def is_due(rubric: Dict[str, Any], now: datetime) -> bool:
    """
    Weekly matrix support (cadence/byweekday):
      - byweekday acts as a filter ALWAYS if present (even when cadence=DAILY).
      - cadence:
          DAILY  -> eligible every day (subject to byweekday)
          WEEKLY -> eligible only on byweekday; if byweekday absent -> eligible every day (avoid in config)
    """
    cadence = (rubric.get("cadence") or "DAILY").upper()
    byweekday = rubric.get("byweekday") or []

    if byweekday:
        map_wd = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
        if map_wd[now.weekday()] not in set(byweekday):
            return False

    if cadence == "DAILY":
        return True
    if cadence == "WEEKLY":
        return True  # already filtered by byweekday if provided
    return False


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


def norm_title_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(логопед|логопедия|логопедический|упражнение|упражнения)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180]


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


def load_sources() -> Tuple[Dict[str, Source], Dict[str, Any]]:
    cfg = load_yaml(CFG_DIR / "sources.yml")
    quality = cfg.get("quality", {}) or {}
    out: Dict[str, Source] = {}
    for s in cfg.get("sources", []) or []:
        out[s["id"]] = Source(**s)
    return out, quality


def get_canonical_and_soup(url: str) -> Tuple[str, Optional[BeautifulSoup]]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=25, verify=_verify_for_url(url))
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
        if canon and canon.get("href"):
            href = canon["href"].strip()
            if href.startswith("/"):
                href = urljoin(url, href)
            return href, soup
        return url, soup
    except Exception:
        return url, None


def extract_article_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return norm_space(og["content"])
    h1 = soup.find("h1")
    if h1:
        return norm_space(h1.get_text(" ", strip=True))
    if soup.title and soup.title.string:
        return norm_space(soup.title.string)
    return ""


def extract_article_summary(soup: BeautifulSoup) -> str:
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        return norm_space(md["content"])
    ogd = soup.find("meta", property="og:description")
    if ogd and ogd.get("content"):
        return norm_space(ogd["content"])
    paras: List[str] = []
    for p in soup.select("p"):
        txt = norm_space(p.get_text(" ", strip=True))
        if len(txt) < 60:
            continue
        if any(bad in txt.lower() for bad in ["cookie", "политик", "подпис", "реклама"]):
            continue
        paras.append(txt)
        if len(paras) >= 2:
            break
    return norm_space(" ".join(paras))[:420]


def enrich_article(item: Dict[str, str]) -> Dict[str, str]:
    link = item.get("link", "")
    canon, soup = get_canonical_and_soup(link)
    item["canonical"] = canon
    if soup:
        at = extract_article_title(soup)
        if at:
            item["article_title"] = at
        sm = extract_article_summary(soup)
        if sm:
            item["article_summary"] = sm
    return item


# ---------------------------
# html_site parsers
# ---------------------------

def _abs(base_url: str, href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(base_url, href)
    if href.startswith("http://") or href.startswith("https://"):
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
    # dedupe
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
    return _collect_links(url, soup, "article a, div.news a, a", r"/news/[\w\-]+/?$")[:80]


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


SITE_PARSERS = {
    "logorina_news": parse_logorina_news,
    "logomag_lib": parse_logomag_lib,
    "logoportal_articles": parse_logoportal_articles,
    "logopedy_articles": parse_logopedy_articles,
}


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


def fetch_html_site(url: str, parser_name: str) -> List[Dict[str, str]]:
    r = requests.get(url, headers=HEADERS, timeout=30, verify=_verify_for_url(url))
    r.raise_for_status()
    parser = SITE_PARSERS.get(parser_name)
    if not parser:
        raise ValueError(f"Unknown site parser: {parser_name}")
    items = parser(url, r.text)
    # dedupe by link
    uniq: Dict[str, Dict[str, str]] = {}
    for it in items:
        uniq[it["link"]] = it
    return list(uniq.values())


def fetch_source(src: Source) -> List[Dict[str, str]]:
    if src.type == "rss":
        return fetch_rss(src.url or "")
    if src.type == "html_site":
        return fetch_html_site(src.url or "", src.parser or "")
    if src.type == "static":
        return fetch_static(src.urls or [])
    raise ValueError(f"Unsupported source type: {src.type}")


# =========================
# State
# =========================

def load_state(name: str, default: Any) -> Any:
    p = STATE_DIR / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_state(name: str, data: Any) -> None:
    (STATE_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Post composition (minimal but rubric-aware)
# =========================

def make_nav_strip(rubric_format: str) -> List[str]:
    rf = (rubric_format or "").strip().lower()
    if rf in ("bilingual_parents", "myth_fact"):
        return [
            "🧠 Навык: уверенная русская фраза",
            "🎯 Цель: устойчивость билингва",
            "📌 Подсказка: мягко моделируйте",
            "📏 Критерий прогресса: фраза без подсказки",
        ]
    if rf in ("age_norms",):
        return [
            "🧠 Навык: понимание и фразы",
            "🎯 Цель: возрастная динамика",
            "📌 Подсказка: наблюдайте 2–4 недели",
            "📏 Критерий прогресса: новые слова/фразы",
        ]
    if rf in ("exercise_steps", "games_vocab"):
        return [
            "🧠 Навык: слова и действия",
            "🎯 Цель: лексика и грамматика",
            "📌 Подсказка: хвалите попытку",
            "📏 Критерий прогресса: 2–3 слова самостоятельно",
        ]
    return [
        "🧠 Навык: речь в короткой фразе",
        "🎯 Цель: связность и точность",
        "📌 Подсказка: коротко, без давления",
        "📏 Критерий прогресса: легче говорит сам",
    ]


def friendly_source_label(url: str) -> str:
    dom = safe_domain(url)
    if not dom:
        return "профессиональные материалы"
    if "logopedy" in dom:
        return "Материалы Logopedy.ru"
    if "logoportal" in dom:
        return "Материалы Logoportal"
    if "asha" in dom:
        return "ASHA (multilingual)"
    if "ncbi" in dom or "pubmed" in dom:
        return "PubMed/PMC"
    return dom


def _rewrite_if_enabled(plain: str, audience: str, rubric_format: str) -> Tuple[str, bool, str]:
    return _rewrite_if_enabled_plain(
        plain,
        audience,
        rubric_format,
        REWRITE_PROVIDER,
        GROQ_API_KEY,
        GEMINI_API_KEY,
        PARENTS_MAX_BODY_CHARS,
        PROS_MAX_BODY_CHARS,
        POST_MAX_CHARS,
    )


def compose_post_plain_v31(
    rubric_title: str,
    rubric_format: str,
    audience: str,
    channel_cfg: Dict[str, Any],
    picked: Dict[str, str],
    title_suffix: str,
) -> Tuple[str, Dict[str, Any]]:
    link = picked.get("canonical") or picked.get("link", "")
    picked_title = picked.get("article_title") or picked.get("title") or ""
    summary = picked.get("article_summary") or picked.get("summary") or ""

    disclaimer = channel_cfg.get("disclaimer", "") or ""
    tags = " ".join(channel_cfg.get("hashtags", []) or []).strip()

    aud = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()

    # Hooks vary by rubric
    if rf == "myth_fact":
        hook = "Миф о билингвизме звучит убедительно — но часто он просто пугает родителей. Разберём спокойно, по фактам."
    elif rf == "age_norms":
        hook = "Возрастные нормы — это ориентир, а не экзамен. Важно смотреть на динамику и контекст."
    elif rf == "bilingual_parents":
        hook = "Русский за границей — не про «идеальность», а про устойчивые привычки и тёплую практику в быту."
    elif rf == "exercise_steps":
        hook = "Сделаем 5–7 минут речи игрой: коротко, легко, без «переделывай»."
    else:
        hook = "Короткая практика на 5–7 минут помогает, если делать её мягко и регулярно."

    # Practice: 2 steps max (as per your prompt rule)
    practice = [
        "2–3 минуты: повтор слога/слова в игре (хвалим попытку).",
        "2–3 минуты: 6–10 глаголов по картинкам (кто что делает?).",
    ]

    age_tag = "3–6 лет"
    if "age" in (picked.get("notes") or "").lower():
        age_tag = "6–9 лет"

    src_label = friendly_source_label(link)
    source_lines = [
        f"Источник: {src_label}",
        "Основа: рекомендации логопедов",
        f"🔗 {link}",
    ]

    lines: List[str] = []
    lines.append(f"{rubric_title} {title_suffix}".strip())
    lines.append(f"👶 Возраст: {age_tag}".strip())
    lines.append("")
    lines.append(hook)
    if picked_title:
        lines.append(clamp_text(picked_title, 150))
    if summary:
        lines.append(clamp_text(summary, 220))
    lines.append("")
    lines.append("Практика на сегодня (5–7 минут)")
    for i, x in enumerate(practice[:2], start=1):
        lines.append(f"{i}) {x}")
    lines.append("")
    lines.extend(make_nav_strip(rf))
    lines.append("")
    lines.append("Норма / когда нужен специалист")
    lines.append("✅ Норма: есть понимание речи и постепенный прогресс.")
    lines.append("⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.")
    lines.append("")
    lines.append("Источник")
    lines.extend(source_lines)
    lines.append("")
    lines.append("💬 Что даётся сложнее: начать или закончить спокойно?")

    raw_plain = "\n".join(lines).strip()
    raw_plain = _enforce_total_chars_keep_structure(raw_plain, POST_MAX_CHARS)

    # Add disclaimer/tags only if they fit
    if disclaimer:
        cand = (raw_plain + "\n\n" + f"ℹ️ {norm_space(disclaimer)}").strip()
        if len(cand) <= POST_MAX_CHARS:
            raw_plain = cand
    if tags:
        cand = (raw_plain + "\n\n" + tags).strip()
        if len(cand) <= POST_MAX_CHARS:
            raw_plain = cand

    final_plain, used, note = _rewrite_if_enabled(raw_plain, aud, rf)
    final_plain = _enforce_total_chars_keep_structure(final_plain, POST_MAX_CHARS)

    meta: Dict[str, Any] = {"ok": True, "rewrite_used": used, "rewrite_note": note}
    if not _has_required_structure_plain_v3(final_plain):
        final_plain = raw_plain
        meta["rewrite_used"] = False
        meta["rewrite_note"] = "rewrite:force_raw_structure"

    return final_plain, meta


# =========================
# Plain -> Telegram HTML rendering (hide raw URL behind anchor)
# =========================

_HTML_HEADINGS = {"Практика на сегодня (5–7 минут)", "Норма / когда нужен специалист", "Источник"}

def _escape(s: str) -> str:
    return _html.escape(s or "", quote=False)


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
        stripped = s.strip()

        if idx == 0 and stripped:
            out.append(f"<b>{_escape(stripped)}</b>")
            continue
        if idx == 1 and stripped.startswith("👶 Возраст:"):
            out.append(f"<b>{_escape(stripped)}</b>")
            continue
        if stripped in _HTML_HEADINGS:
            out.append(f"<b>{_escape(stripped)}</b>")
            continue

        if stripped.startswith("🔗 "):
            url = stripped[2:].strip()
            if url.startswith(("http://", "https://")):
                out.append(_link_anchor(url, prefix="🔗 "))
            else:
                out.append(_escape(stripped))
            continue
        if stripped.startswith(("http://", "https://")):
            out.append(_link_anchor(stripped, prefix="🔗 "))
            continue

        if stripped.startswith("ℹ️ "):
            out.append(f"<i>{_escape(stripped)}</i>")
            continue

        out.append(_escape(s))

    return "\n".join(out).strip()


def build_card_theses_from_plain_v3(plain_post: str) -> Tuple[List[str], str]:
    lines = (plain_post or "").splitlines()
    age = ""
    if len(lines) >= 2 and lines[1].strip().startswith("👶 Возраст:"):
        age = lines[1].split(":", 1)[1].strip()

    bullets = [ln.strip() for ln in lines if ln.strip().startswith("• ")][:2]
    warn = ""
    for ln in lines:
        if ln.strip().startswith("⚠️"):
            warn = ln.strip()
            break

    a = clamp_text(bullets[0][2:].strip() if bullets else "Короткая практика без давления.", 92)
    b = clamp_text(bullets[1][2:].strip() if len(bullets) > 1 else "5 минут в игре — каждый день.", 92)
    c = clamp_text(warn.lstrip("⚠️").strip() if warn else "Если нет прогресса 4–6 недель — специалист.", 92)
    return [f"💡 {a}", f"🧩 {b}", f"⚠️ {c}"], age


# =========================
# Telegram send (NO-DUP)
# =========================

def tg_request(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, data=data, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def send_message(chat_id: str, html_text: str) -> None:
    data = {"chat_id": chat_id, "text": html_text}
    if TELEGRAM_PARSE_MODE:
        data["parse_mode"] = TELEGRAM_PARSE_MODE
    tg_request("sendMessage", data=data)


def send_post_with_card(chat_id: str, card_path: Path, plain_post: str, html_full_post: str) -> None:
    # If full post fits caption bytes -> send photo with caption; else photo + separate text.
    plain_bytes = len((plain_post or "").encode("utf-8"))

    if plain_bytes <= TG_CAPTION_MAX_BYTES:
        try:
            data: Dict[str, Any] = {"chat_id": chat_id, "caption": html_full_post}
            if TELEGRAM_PARSE_MODE:
                data["parse_mode"] = TELEGRAM_PARSE_MODE
            with card_path.open("rb") as f:
                tg_request("sendPhoto", data=data, files={"photo": f})
            return
        except Exception as e:
            print(f"[WARN] sendPhoto(full caption) failed -> fallback photo+text: {e}")

    with card_path.open("rb") as f:
        tg_request("sendPhoto", data={"chat_id": chat_id, "caption": ""}, files={"photo": f})
    send_message(chat_id, html_full_post)


# =========================
# Weekly stats + dashboard
# =========================

def load_weekly_stats() -> Dict[str, Any]:
    return load_state("stats_weekly.json", {})


def save_weekly_stats(stats: Dict[str, Any]) -> None:
    save_state("stats_weekly.json", stats)


def bump_weekly(stats: Dict[str, Any], week_key: str, field: str, amount: int = 1, reason: Optional[str] = None) -> None:
    wk = stats.get(week_key) or {"passed": 0, "rejected": 0, "reasons": {}}
    wk[field] = int(wk.get(field, 0)) + amount
    if reason:
        rs = wk.get("reasons") or {}
        rs[str(reason)] = int(rs.get(str(reason), 0)) + amount
        wk["reasons"] = rs
    stats[week_key] = wk


def format_dashboard(stats: Dict[str, Any], week_key: str, title: str) -> str:
    wk = stats.get(week_key) or {"passed": 0, "rejected": 0, "reasons": {}}
    passed = int(wk.get("passed", 0))
    rejected = int(wk.get("rejected", 0))
    reasons = wk.get("reasons") or {}
    top = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:6]

    lines = [
        f"{title} ({week_key})",
        "",
        f"✅ Прошло: {passed}",
        f"🗂️ В черновики/отсев: {rejected}",
        "",
    ]
    if top:
        lines.append("Причины отсева (топ):")
        for k, v in top:
            lines.append(f"• {k}: {v}")
    else:
        lines.append("Причины отсева: нет данных.")
    lines.append("")
    lines.append("ℹ️ Примечание: тех. статистика качества источников/фильтров.")
    return render_plain_to_telegram_html("\n".join(lines))


# =========================
# Run
# =========================

def run() -> None:
    rub_cfg = load_yaml(CFG_DIR / "rubrics.yml")
    channel_cfg = rub_cfg.get("channel", {}) or {}
    branding = rub_cfg.get("branding", {}) or {}
    tzname = channel_cfg.get("timezone", "Asia/Nicosia")
    now = get_local_now(tzname)
    week_key = iso_week_key(now)

    sources, _quality_cfg = load_sources()
    used_canon = set(load_state("used_canonical.json", []))
    used_titles = set(load_state("used_titles.json", []))

    pub_cfg = rub_cfg.get("publishing", {}) or {}
    max_posts = int(pub_cfg.get("max_posts_per_run", 1))

    stats = load_weekly_stats()

    audiences_cfg = rub_cfg.get("audiences", {}) or {}
    if AUDIENCE == "both":
        aud_list = ["parents", "pros"]
    elif AUDIENCE in ("parents", "pros"):
        aud_list = [AUDIENCE]
    else:
        aud_list = ["parents"]

    posted = 0

    for aud in aud_list:
        if posted >= max_posts:
            break

        aud_cfg = audiences_cfg.get(aud, {}) or {}
        title_suffix = (aud_cfg.get("title_suffix", "") or "").strip()
        rubrics = aud_cfg.get("rubrics", []) or []

        for rubric in rubrics:
            if posted >= max_posts:
                break

            # ✅ respect schedule
            if not is_due(rubric, now):
                continue

            # ✅ HOTFIX: dashboard always to drafts
            if (rubric.get("format") or "").strip().lower() == "quality_dashboard":
                dash_title = pub_cfg.get("dashboard_title", "Quality dashboard недели")
                dashboard_html = format_dashboard(stats, week_key, dash_title)

                if not TELEGRAM_DRAFTS_CHAT_ID:
                    raise RuntimeError(
                        "TELEGRAM_DRAFTS_CHAT_ID is missing. Refusing to post quality dashboard to public channel."
                    )

                if DRY_RUN:
                    print("[DRY_RUN] dashboard would be posted to DRAFTS chat only.")
                else:
                    send_message(TELEGRAM_DRAFTS_CHAT_ID, dashboard_html)

                time.sleep(0.4)
                continue

            # collect items
            all_items: List[Dict[str, str]] = []
            for sid in rubric.get("sources", []) or []:
                src = sources.get(sid)
                if not src:
                    continue
                try:
                    all_items.extend(fetch_source(src))
                except Exception as e:
                    print(f"[WARN] source {sid} failed: {e}")

            # pick first not used
            picked = None
            for it0 in all_items[:80]:
                it = enrich_article(dict(it0))
                canon = it.get("canonical") or it.get("link", "")
                if not canon or canon in used_canon:
                    continue
                raw_title = it.get("article_title") or it.get("title") or ""
                tkey = norm_title_key(raw_title)
                if tkey and tkey in used_titles:
                    continue
                picked = it
                break

            if not picked:
                continue

            title = rubric.get("title", "Рубрика")
            plain_post, meta = compose_post_plain_v31(title, rubric.get("format", ""), aud, channel_cfg, picked, title_suffix)
            html_full_post = render_plain_to_telegram_html(plain_post)

            theses, age_tag = build_card_theses_from_plain_v3(plain_post)
            card = render_image_card(title, theses, branding, age_tag=age_tag)

            if DRY_RUN:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                out_dir = STATE_DIR / "dry_run" / ts
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(card, out_dir / f"{posted+1:02d}_{aud}_card.png")
                (out_dir / f"{posted+1:02d}_{aud}_plain.txt").write_text(plain_post, encoding="utf-8")
                (out_dir / f"{posted+1:02d}_{aud}_html.txt").write_text(html_full_post, encoding="utf-8")
            else:
                if not TELEGRAM_CHAT_ID:
                    raise RuntimeError("TELEGRAM_CHAT_ID is missing (publish target).")
                send_post_with_card(TELEGRAM_CHAT_ID, card, plain_post, html_full_post)

                canon = picked.get("canonical") or picked.get("link", "")
                if canon:
                    used_canon.add(canon)
                tkey = norm_title_key(picked.get("article_title") or picked.get("title") or "")
                if tkey:
                    used_titles.add(tkey)

                bump_weekly(stats, week_key, "passed", 1)

            posted += 1
            time.sleep(1.0)

    if not DRY_RUN:
        save_state("used_canonical.json", sorted(list(used_canon))[-6000:])
        save_state("used_titles.json", sorted(list(used_titles))[-6000:])
        save_weekly_stats(stats)

    print(f"Publisher done. Posted: {posted}. Week: {week_key}.{' [DRY_RUN]' if DRY_RUN else ''}")


if __name__ == "__main__":
    run()

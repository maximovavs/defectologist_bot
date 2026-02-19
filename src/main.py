from __future__ import annotations

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
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================
# Paths / env
# =========================

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/1.9.0 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip()

# v1.8: dry run (no Telegram). Generate artifacts locally into .state/dry_run/<ts>/
DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")

# Use HTML for Telegram rendering (requested)
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()  # HTML | Markdown | ""

REWRITE_PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()  # none|auto|groq|gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()  # parents|pros|both

# v1.9: hard post length cap (requested). Applies to the whole post.
POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1000"))

# v1.6.1: style/length knobs (kept for backward-compat; capped by POST_MAX_CHARS)
PARENTS_MAX_BODY_CHARS = int(os.getenv("PARENTS_MAX_BODY_CHARS", "900"))
PROS_MAX_BODY_CHARS = int(os.getenv("PROS_MAX_BODY_CHARS", "1050"))

# v1.9: quality gate knobs (align with the new laconic structure)
MIN_MEANING_BULLETS = int(os.getenv("MIN_MEANING_BULLETS", "1"))
MIN_PRACTICE_STEPS = int(os.getenv("MIN_PRACTICE_STEPS", "1"))

# Telegram hard limits: captions are very strict (often ~1024 chars, but UTF-8 bytes matter).
# Use bytes-limit for safety.
TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))
TG_SEND_FULL_TEXT_AFTER_PHOTO = (
    os.getenv("TG_SEND_FULL_TEXT_AFTER_PHOTO", "1").strip().lower() in ("1", "true", "yes")
)

# Optional: treat these domains as insecure TLS (comma-separated), if you want to bypass bad certs
INSECURE_TLS_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("INSECURE_TLS_DOMAINS", "") or "").split(",")
    if d.strip()
]


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


def safe_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def domain_allowed(url: str, allow_domains: List[str]) -> bool:
    d = safe_domain(url)
    if not d:
        return False
    return any(d == ad or d.endswith("." + ad) for ad in allow_domains)


def _verify_for_url(url: str) -> bool:
    """Return TLS verification flag based on INSECURE_TLS_DOMAINS."""
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


def score_item(title: str, link: str, quality_cfg: Dict[str, Any]) -> Tuple[int, str]:
    t = (title or "").strip()
    u = (link or "").strip()
    if len(t) < 12 or len(t) > 240:
        return (-100, "bad_title_len")
    allow_domains = quality_cfg.get("allow_domains") or []
    if allow_domains and not domain_allowed(u, allow_domains):
        return (-100, "domain_not_allowed")
    tl, ul = t.lower(), u.lower()
    for k in [x.lower() for x in (quality_cfg.get("deny_keywords") or [])]:
        if k and (k in tl or k in ul):
            return (-100, f"deny_keyword:{k}")
    score = 10
    for k in [x.lower() for x in (quality_cfg.get("boost_keywords") or [])]:
        if k and k in tl:
            score += 2
    return (score, "ok")


def is_due(rubric: Dict[str, Any], now: datetime) -> bool:
    cadence = (rubric.get("cadence") or "DAILY").upper()
    if cadence == "DAILY":
        return True
    if cadence == "WEEKLY":
        byweekday = set(rubric.get("byweekday") or [])
        map_wd = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
        return map_wd[now.weekday()] in byweekday
    return False


def utf8_clip(text: str, max_bytes: int, add_ellipsis: bool = True) -> str:
    """
    Clip by UTF-8 bytes (Telegram caption errors are often byte-based).
    Guarantees result.encode('utf-8') <= max_bytes.
    """
    s = (text or "")
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    cut = b[:max_bytes]
    while cut:
        try:
            out = cut.decode("utf-8")
            break
        except UnicodeDecodeError:
            cut = cut[:-1]
    else:
        return "…"

    out = out.rstrip(" .,:;—-")
    if add_ellipsis:
        ell = "…"
        # ensure ellipsis fits
        while out and len((out + ell).encode("utf-8")) > max_bytes:
            out = out[:-1]
            out = out.rstrip(" .,:;—-")
        if len((out + ell).encode("utf-8")) <= max_bytes:
            out = out + ell
    # final guard
    while len(out.encode("utf-8")) > max_bytes and out:
        out = out[:-1]
    return out.strip() or "…"


def _is_sovet_dnya_format(rubric_format: str) -> bool:
    """
    v1.7.1+: 'Совет дня' must be strictly practice-oriented.
    v1.9: also used for age inference and colleague-note variants.
    """
    rf = (rubric_format or "").strip().lower()
    if rf in {
        "tip_day", "daily_tip", "advice_day", "tip_of_day",
        "sovet_dnya", "sovet_dnya_parents", "sovet_day",
        "sovet_logopeda", "logoped_tip_day",
    }:
        return True
    return ("совет" in rf) or ("tip" in rf and "day" in rf)


def mask_diagnostics_for_parents(text: str) -> str:
    """
    Hide heavy diagnostic labels in parents-facing copy (source masking).
    We still allow mild professional terms in the colleague note.
    """
    t = text or ""
    rep = {
        r"\bалал(ия|ии|ией|ию)\b": "речевые трудности",
        r"\bдизартри(я|и)\b": "трудности произношения",
        r"\bафази(я|и)\b": "трудности речи",
        r"\bринолали(я|и)\b": "трудности речи",
        r"\bзаикани(е|я)\b": "сбои плавности речи",
        r"\bдисграфи(я|и)\b": "трудности письма",
        r"\bдислекси(я|и)\b": "трудности чтения",
        r"\bОНР\b": "речевые трудности",
        r"\bФФН\b": "трудности звуков",
        r"\bЗРР\b": "задержка речи",
    }
    for pat, repl in rep.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE | re.UNICODE)
    return t


def infer_target_age(rubric_format: str, picked_title: str, summary: str, practice: List[str]) -> str:
    """
    Heuristic age inference. Output examples:
      '3–5 лет' or 'от 2 лет'
    """
    blob = " ".join([picked_title or "", summary or "", " ".join(practice or [])]).lower()

    # literacy / school
    if any(k in blob for k in ["дисграф", "дислекс", "письм", "чтени", "школ", "первокласс", "1 класс", "второкласс"]):
        return "6–9 лет"

    # sound setting / complex consonants
    if any(k in blob for k in ["звук р", "звук л", "шипящ", "свистящ", "ротац", "ламбдац", "постановк"]):
        return "4–7 лет"

    # phrases / grammar / narratives
    if any(k in blob for k in ["фраз", "предложен", "связн", "рассказ", "согласован", "падеж", "множествен"]):
        return "3–6 лет"

    # first words / delay
    if any(k in blob for k in ["первые слова", "лепет", "гулен", "не говорит", "молчит", "словар", "запуск речи"]):
        return "от 1,5 лет"

    # articulation gymnastics baseline
    rf = (rubric_format or "").strip().lower()
    if _is_sovet_dnya_format(rf):
        return "3–5 лет"
    if rf in ("exercise_steps", "age_norms", "bilingual_parents", "myth_fact", "question_week"):
        return "от 2 лет"

    return "3–6 лет"


def friendly_source_label(url: str) -> str:
    dom = safe_domain(url)
    if not dom:
        return "профессиональные материалы"
    if "logopedy" in dom:
        return "Материалы Logopedy.ru"
    if "logopediya" in dom:
        return "Материалы Logopediya.ru"
    if "asha" in dom:
        return "ASHA (multilingual)"
    if "pubmed" in dom or "ncbi" in dom:
        return "PubMed/PMC"
    if "logoportal" in dom:
        return "Материалы Logoportal"
    if "logorina" in dom:
        return "Материалы Logorina"
    if "logomag" in dom:
        return "Материалы Logomag"
    return dom


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


def is_scientific_or_methodical(domain: str, title: str, summary: str, quality_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    scientific_domains = [d.lower() for d in (quality_cfg.get("scientific_domains") or [])]
    if any(domain == d or domain.endswith("." + d) for d in scientific_domains):
        return True, "scientific_domain"
    blob = f"{title}\n{summary}".lower()
    kws = [k.lower() for k in (quality_cfg.get("methodical_keywords") or [])]
    hits = sum(1 for k in kws if k and k in blob)
    if hits >= 2:
        return True, f"methodical_kw_hits:{hits}"
    return False, f"not_methodical_hits:{hits}"


def source_type_label_from_factcheck(factcheck_reason: str) -> str:
    r = (factcheck_reason or "").lower()
    if "scientific_domain" in r:
        return "научный/академический источник"
    return "методический/профессиональный материал"


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
# Site-specific parsers
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

    seen: set[str] = set()
    uniq: List[Dict[str, str]] = []
    for it in out:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)
    return uniq


def parse_logopediya_publ(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "div#dle-content a, div#dle-content h2 a, div#dle-content h3 a", r"/publ/[^\"']+")
    items = [it for it in items if not re.search(r"/page/\d+/?$", it["link"])]
    return items[:80]


def parse_logorina_news(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "article a, div.news a, a", r"/news/[\w\-]+/?$")
    return items[:80]


def parse_logomag_lib(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "main a, div.content a, a", r"/lib/[^\"']+")
    return items[:80]


def parse_logoportal_articles(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "main a, div#content a, article a, a", r"(statya-|/statya-)")
    return items[:80]


def parse_logopedy_articles(url: str, html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    items = _collect_links(url, soup, "div.content a, main a, a", r"logoped-article|logoped-literature|portal/[^#]+")
    items.sort(key=lambda x: len(x["title"]), reverse=True)
    return items[:80]


SITE_PARSERS = {
    "logopediya_publ": parse_logopediya_publ,
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
# LLM rewrite (plain text), with fallback on raw if rendering/structure breaks
# =========================

def _is_quota_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (402, 429) or any(k in t for k in ["quota", "rate limit", "exceeded", "insufficient_quota", "resource_exhausted"])


def rewrite_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.35,
        },
        timeout=45,
    )
    if r.status_code != 200 and _is_quota_error(r.status_code, r.text):
        raise RuntimeError(f"groq_quota:{r.status_code}")
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def rewrite_with_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=45,
    )
    if r.status_code != 200 and _is_quota_error(r.status_code, r.text):
        raise RuntimeError(f"gemini_quota:{r.status_code}")
    r.raise_for_status()
    return (r.json()["candidates"][0]["content"]["parts"][0]["text"] or "").strip()


def _aud_limits(audience: str) -> int:
    a = (audience or "parents").strip().lower()
    base = PARENTS_MAX_BODY_CHARS if a == "parents" else PROS_MAX_BODY_CHARS
    return min(int(base), int(POST_MAX_CHARS))


_REQUIRED_HEADINGS_V3 = [
    "Практика на сегодня (5–7 минут)",
    "Норма / когда нужен специалист",
    "Источник",
]


def _has_nav_panel_v3(text: str) -> bool:
    """Nav panel must be exactly 4 lines with fixed labels (order matters)."""
    lines = [x.rstrip() for x in (text or "").splitlines()]
    want = [
        "🧠 Навык:",
        "🎯 Цель:",
        "📌 Подсказка:",
        "📏 Критерий прогресса:",
    ]
    idx = 0
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        if s.startswith(want[idx]):
            idx += 1
            if idx == len(want):
                return True
    return False


def _build_rewrite_prompt_v3(body: str, audience: str, max_chars: int, rubric_format: str = "") -> str:
    a = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()
    is_tip = _is_sovet_dnya_format(rf)

    base_role = (
        "Роль: эмпатичный, современный логопед. Ты дружелюбно общаешься с уставшими родителями, "
        "но при этом показываешь методическую глубину и аккуратность.\n"
    )

    # Important: forbid robot-like stubs.
    hard_bans = (
        "ЖЁСТКИЕ запреты:\n"
        "• НЕ используй шаблонные заголовки-отбивки: «Суть», «Коротко», «Что это значит для вас», «Вывод», «Итог», «Резюме», «Важно».\n"
        "• НЕ добавляй новые подзаголовки и новые разделы.\n"
        "• НЕ ставь диагнозы, НЕ обещай лечения, НЕ назначай препараты.\n"
        "• Не добавляй новых фактов — только перефразируй то, что уже есть.\n"
    )

    structure = (
        "Структура должна сохраниться строго:\n"
        "1) Первая строка — название рубрики (как в исходнике).\n"
        "2) Вторая строка — строка возраста в формате: «👶 Возраст: ...».\n"
        "3) Далее — 1 короткий абзац: хук (ситуация/вопрос) → польза для родителей, без дополнительных подзаголовков.\n"
        f"4) Далее идут заголовки (каждый — отдельной строкой) строго из списка: {', '.join(_REQUIRED_HEADINGS_V3)}.\n"
        "5) В блоке «Практика…» — строго 1–2 коротких шага (1) 2)).\n"
        "6) Сразу после практики должна быть навигационная полоса для коллег: ровно 4 строки в порядке и формате:\n"
        "   🧠 Навык: … (3–5 слов)\n"
        "   🎯 Цель: … (3–5 слов)\n"
        "   📌 Подсказка: … (3–5 слов)\n"
        "   📏 Критерий прогресса: … (3–5 слов)\n"
        "7) В конце перед техническим дисклеймером — ровно одна короткая строка-вопрос, начинается с «💬 ».\n"
        f"8) Тело текста (до блока «Источник» включительно): старайся держать до {max_chars} символов.\n"
        "9) Весь пост должен укладываться в 1000 символов (лучше меньше).\n"
        "Форматирование: только обычный текст, без HTML/Markdown.\n"
    )

    tip_rules = ""
    if is_tip:
        tip_rules = (
            "\nДоп. правило для «Совет дня»:\n"
            "- Фокус на практике и поддержке мотивации. Никакой академической теории и классификаций.\n"
        )

    if a == "pros":
        style = (
            "Аудитория: специалисты (логопеды/дефектологи).\n"
            "Стиль: профессионально, точные термины, но без канцелярита. "
            "Термины и методическая суть — компактно в навигационной полосе (🧠/🎯/📌/📏).\n"
        )
    else:
        style = (
            "Аудитория: родители.\n"
            "Стиль: тёплый, поддерживающий, простые слова. Сложные термины — только в навигационной полосе (она для коллег).\n"
        )

    return (
        base_role
        + style
        + hard_bans
        + structure
        + tip_rules
        + "\nТЕКСТ ДЛЯ ПЕРЕФОРМУЛИРОВКИ:\n"
        + (body or "").strip()
    )


def _enforce_body_limit_v3(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()


def _enforce_total_limit_v3(text: str, max_total_chars: int) -> str:
    """Hard cap by total characters for the whole post."""
    t = (text or "").strip()
    if len(t) <= max_total_chars:
        return t

    lines = t.splitlines()
    # Prefer dropping optional tail lines (hashtags / disclaimer / empty lines)
    while lines and len("\n".join(lines).strip()) > max_total_chars:
        last = (lines[-1] or "").strip()
        if last == "" or last.startswith("#") or last.startswith("ℹ️"):
            lines.pop()
            continue
        break

    t2 = "\n".join(lines).strip()
    if len(t2) <= max_total_chars:
        return t2

    cut = t2[:max_total_chars]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()


def _has_required_structure_plain_v3(text: str) -> bool:
    raw_lines = (text or "").splitlines()
    nonempty = [ln.strip() for ln in raw_lines if ln.strip()]

    if len(nonempty) < 6:
        return False

    # line 2 must be age
    if not nonempty[1].startswith("👶 Возраст:"):
        return False

    # required headings must exist exactly as lines
    line_set = set([ln.strip() for ln in raw_lines])
    for h in _REQUIRED_HEADINGS_V3:
        if h not in line_set:
            return False

    # practice steps: laconic 1–2 steps
    in_practice = False
    practice_steps = []
    for ln in raw_lines:
        s = ln.strip()
        if s == "Практика на сегодня (5–7 минут)":
            in_practice = True
            continue
        if in_practice and s in ("Норма / когда нужен специалист", "Источник"):
            break
        if in_practice and re.match(r"^\d+\)\s+", s):
            practice_steps.append(s)
    if len(practice_steps) < max(1, int(MIN_PRACTICE_STEPS)):
        return False
    if len(practice_steps) > 2:
        return False

    # nav panel: 4 fixed lines, in order
    if not _has_nav_panel_v3(text):
        return False

    # engagement line (question)
    if not any((x or "").strip().startswith("💬 ") for x in raw_lines):
        return False

    return True


def rewrite_if_enabled_plain(full_plain_text: str, audience: str, rubric_format: str = "") -> Tuple[str, bool, str]:
    """
    Returns: (rewritten_or_raw, used_rewrite, note)
    Requested behavior:
      - if rewrite breaks structure -> fallback to raw, do NOT fail quality.
    """
    if REWRITE_PROVIDER == "none":
        return full_plain_text, False, "rewrite:none"

    max_chars = _aud_limits(audience)
    prompt = _build_rewrite_prompt_v3(full_plain_text, audience, max_chars, rubric_format=rubric_format)

    try:
        if REWRITE_PROVIDER in ("groq", "auto"):
            try:
                out = rewrite_with_groq(prompt)
                out = _enforce_body_limit_v3(out, max_chars)
                out = _enforce_total_limit_v3(out, POST_MAX_CHARS)
                if not _has_required_structure_plain_v3(out):
                    print("[WARN] rewrite broke structure (groq) -> fallback to raw")
                    return full_plain_text, False, "rewrite:fallback_raw_structure"
                return out, True, "rewrite:groq"
            except Exception as e:
                if REWRITE_PROVIDER == "groq":
                    raise
                if "groq_quota" in str(e):
                    print("[WARN] groq quota; fallback to gemini")
                else:
                    print(f"[WARN] groq rewrite failed: {e}")

        if REWRITE_PROVIDER in ("gemini", "auto"):
            out = rewrite_with_gemini(prompt)
            out = _enforce_body_limit_v3(out, max_chars)
            out = _enforce_total_limit_v3(out, POST_MAX_CHARS)
            if not _has_required_structure_plain_v3(out):
                print("[WARN] rewrite broke structure (gemini) -> fallback to raw")
                return full_plain_text, False, "rewrite:fallback_raw_structure"
            return out, True, "rewrite:gemini"

    except Exception as e:
        print(f"[WARN] rewrite failed ({REWRITE_PROVIDER}): {e}")
        return full_plain_text, False, "rewrite:fallback_raw_error"

    return full_plain_text, False, "rewrite:fallback_raw_unknown"


# =========================
# Post template v3.0 (parents + pros) + quality gate
# =========================

def make_question_week() -> str:
    questions = [
        "Малыш понимает обращённую речь, но говорит мало — что вы уже пробовали дома и что сработало хоть немного?",
        "В билингвальной семье: на каком языке ребёнку легче рассказывать истории — и в каких ситуациях это меняется?",
        "Какие звуки/слоги даются труднее всего — и в каких словах это заметнее?",
        "Что вызывает больше сопротивления: артикуляционная гимнастика, повторение слогов или чтение/письмо?",
        "Как выглядит ваш «идеальный результат» через 4 недели занятий — в одном предложении?",
    ]
    return random.choice(questions)


def make_engagement_question(audience: str, age_tag: str, rubric_format: str) -> str:
    a = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()
    if a == "pros":
        qs = [
            "Какие подсказки/опоры у вас лучше всего срабатывают для удержания мотивации на домашке?",
            "Какой критерий прогресса вы бы выбрали на ближайшие 2 недели для этого навыка?",
            "Какие варианты усложнения вы добавляете, когда базовая форма уже стабилизировалась?",
        ]
    else:
        qs = [
            "А ваш ребёнок любит корчить рожицы перед зеркалом — или скорее убегает? Как у вас обычно получается “включить игру”?",
            "Если попробуете сегодня — что окажется самым лёгким: повторять за вами или дуть в игре? Напишите пару слов в комментариях.",
            "Какая часть даётся сложнее всего: начать, удержать внимание или закончить спокойно? Расскажите, я подскажу мягкий первый шаг.",
        ]
    q = random.choice(qs)
    # small personalization with age
    if age_tag and "лет" in age_tag and a != "pros":
        q = q.replace("сегодня", f"сегодня (для возраста {age_tag})")
    return q


def _quality_gate(
    rubric_format: str,
    audience: str,
    link: str,
    hook: str,
    meaning: List[str],
    practice: List[str],
    nav_panel: List[str],
    norm_lines: List[str],
    age_tag: str,
) -> Tuple[bool, str]:
    rf = (rubric_format or "").strip().lower()
    aud = (audience or "parents").strip().lower()

    if not link or not link.startswith(("http://", "https://")):
        return False, "quality_gate:no_source_link"

    if not (age_tag or "").strip():
        return False, "quality_gate:no_age_tag"

    hook_len = len(norm_space(hook))
    # new laconic spec: hooks are shorter
    if rf != "question_week" and hook_len < 45:
        return False, f"quality_gate:weak_hook_len:{hook_len}"
    if rf == "question_week" and hook_len < 35:
        return False, f"quality_gate:weak_hook_len:{hook_len}"

    m = [x for x in meaning if norm_space(x)]
    if len(m) < MIN_MEANING_BULLETS:
        return False, f"quality_gate:meaning_bullets_lt_{MIN_MEANING_BULLETS}:{len(m)}"

    p = [x for x in practice if norm_space(x)]
    if len(p) < MIN_PRACTICE_STEPS:
        return False, f"quality_gate:practice_steps_lt_{MIN_PRACTICE_STEPS}:{len(p)}"

    if len(p) > 2:
        return False, "quality_gate:practice_too_long"

    # navigation panel (4 fixed lines)
    if not nav_panel or len(nav_panel) != 4:
        return False, "quality_gate:no_nav_panel"
    want = ["🧠 Навык:", "🎯 Цель:", "📌 Подсказка:", "📏 Критерий прогресса:"]
    for i, pref in enumerate(want):
        if not norm_space(nav_panel[i]).startswith(pref):
            return False, "quality_gate:bad_nav_panel"

    nl = "\n".join([norm_space(x) for x in norm_lines if norm_space(x)])
    if "✅" not in nl or "⚠️" not in nl:
        return False, "quality_gate:norm_block_missing_markers"

    # old pro-specific practice gate removed (nav panel carries the methodical signal now)

    return True, "ok"


def make_colleague_note(rubric_format: str, practice: List[str], picked_title: str, summary: str) -> str:
    rf = (rubric_format or "").strip().lower()
    blob = " ".join([picked_title or "", summary or "", " ".join(practice or [])]).lower()

    # Breathing / airflow
    if any(k in blob for k in ["дуем", "выдох", "пузы", "перыш", "ватн", "воздуш", "дых"]):
        return "Упражнение тренирует регуляцию воздушной струи и длительность фонационного выдоха; это поддерживает слоговую структуру и стабильность речевого темпа."

    # Articulation / lips-tongue
    if any(k in blob for k in ["лопаточ", "часик", "язык", "губ", "зеркал", "артикуляц"]):
        return "Практика направлена на развитие артикуляционной моторики и кинестетического контроля (язык/губы); это создаёт базу для более точной коартикуляции и постановки звуков."

    # Vocabulary / grammar
    if any(k in blob for k in ["глагол", "кто что делает", "словар", "категор", "описан", "предложен"]):
        return "Задание усиливает лексико‑семантические связи и грамматическое программирование высказывания (глагольная валентность/согласование), что повышает связность фразы."

    # Bilingual
    if rf == "bilingual_parents" or any(k in blob for k in ["билингв", "два языка", "переключ", "код"]):
        return "Мягкое моделирование фразы поддерживает формирование двуязычных лексических сетей без усиления тревоги; важен перенос навыка в естественные контексты обоих языков."

    # Default
    return "Короткая ежедневная практика усиливает нейропластичность: частые повторения с положительным подкреплением быстрее закрепляют моторно‑речевые программы и перенос в спонтанную речь."


def make_nav_panel(rubric_format: str, practice: List[str], picked_title: str, summary: str) -> List[str]:
    """Navigation panel for specialists: 4 compact lines with fixed labels."""
    rf = (rubric_format or "").strip().lower()
    blob = " ".join([picked_title or "", summary or "", " ".join(practice or [])]).lower()

    # Defaults (keep short)
    skill = "артикуляционная моторика"
    goal = "кинестетический контроль"
    hint = "зеркало, без напряжения"
    metric = "поза держится 5 сек"

    if any(k in blob for k in ["дуем", "выдох", "пузы", "перыш", "ватн", "воздуш", "дых"]):
        skill = "речевое дыхание"
        goal = "ровный фонационный выдох"
        hint = "щёки не надувать"
        metric = "выдох 4–6 сек"
    elif any(k in blob for k in ["фонемат", "различ", "слыш", "путает", "звук", "слог"]):
        skill = "фонематический слух"
        goal = "дифференциация фонем"
        hint = "начните с контраста"
        metric = "меньше замен звуков"
    elif rf == "bilingual_parents" or any(k in blob for k in ["билингв", "два языка", "переключ", "код"]):
        skill = "лексика в билингвизме"
        goal = "активация русского"
        hint = "моделируйте без давления"
        metric = "больше русских фраз"
    elif any(k in blob for k in ["глот", "жев", "прикус", "жевани", "пищ"]):
        skill = "оральная моторика"
        goal = "стабилизация челюсти"
        hint = "медленно, симметрично"
        metric = "меньше напряжения"

    def short(s: str, n: int = 5) -> str:
        parts = (s or "").split()
        return " ".join(parts[:n])

    return [
        f"🧠 Навык: {short(skill)}",
        f"🎯 Цель: {short(goal)}",
        f"📌 Подсказка: {short(hint)}",
        f"📏 Критерий прогресса: {short(metric)}",
    ]


def compose_post_plain_v30(
    rubric_title: str,
    rubric_format: str,
    audience: str,
    channel_cfg: Dict[str, Any],
    picked: Dict[str, str],
    title_suffix: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Produces STRICT PLAIN TEXT (no HTML/Markdown). Then rewrite (optional) with fallback on raw.
    Template v3.0:
      1) Title
      2) 👶 Возраст: ...
      3) Hook paragraph (no "Суть/Коротко/Что это значит...")
      4) Bullets with benefits
      5) Практика...
      6) 🔬 Заметка для коллег:
      7) Норма / когда нужен специалист
      8) Источник
      9) 💬 engagement question
      10) ℹ️ disclaimer + hashtags
    """
    link = picked.get("canonical") or picked.get("link", "")
    picked_title = picked.get("picked_title") or picked.get("title") or ""
    summary = picked.get("picked_summary") or picked.get("summary") or ""
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    tags = " ".join(channel_cfg.get("hashtags", []) or []).strip()

    aud = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()
    is_tip = _is_sovet_dnya_format(rf)

    # Mask heavy diagnostics in parents-facing preview text
    safe_title = picked_title
    safe_summary = summary
    if aud == "parents":
        safe_title = mask_diagnostics_for_parents(safe_title)
        safe_summary = mask_diagnostics_for_parents(safe_summary)

    picked_title_c = clamp_text(safe_title, 140) if safe_title else ""
    summary_c = clamp_text(safe_summary, 240) if safe_summary else ""

    # --- Practice (defines age too)
    if is_tip:
        practice = [
            "2 минуты «Эхо»: вы говорите слог/короткое слово, ребёнок повторяет (похвала за попытку).",
            "2 минуты «Кто что делает?»: 6–10 глаголов по картинкам/предметам (прыгает, моет, рисует…).",
            "1 минута «Дуем в игре»: пузыри/ватный шарик/перышко (ровный выдох).",
            "Вариант по возрасту: 3–4 года — 3–5 повторов; 5–6 лет — 6–10 повторов; 7+ — добавьте короткую фразу.",
        ]
    elif rf == "exercise_steps":
        practice = [
            "Перед зеркалом: «Лопаточка» — 5 раз по 5 секунд.",
            "«Часики» — 10 плавных движений вправо-влево.",
            "1 минута: дуем на ватный шарик/мыльные пузыри (в игре).",
            "В конце — короткая похвала за попытки, без “переделывай”.",
        ]
    elif rf == "bilingual_parents":
        practice = [
            "Игра “Два варианта”: повторите фразу ребёнка по-русски (спокойно, без оценок), затем спросите «как по-русски?».",
            "5 минут “островка русского”: книжка/картинки/комментирование действий дома.",
            "В конце дня: ребёнок выбирает 3 предмета и называет их по-русски (можно с подсказкой).",
        ]
    elif rf == "question_week":
        practice = [
            "Запишите 3 примера фраз ребёнка (как есть) и ситуации, где они прозвучали.",
            "Отметьте: понимает ли ребёнок просьбы без жестов (2–3 примера).",
            "Выберите 1 мини-игру на речь на 5 минут (картинки/описание предмета/пузыри).",
        ]
    elif rf == "age_norms":
        practice = [
            "5 минут “описательной речи”: предмет → цвет/форма/назначение/действие.",
            "Игра “Кто что делает?”: 10 глаголов по картинкам (прыгает, рисует, моет…).",
            "Если ребёнок билингв — дайте ответить, затем мягко повторите модель по-русски.",
        ]
    elif rf == "myth_fact":
        practice = [
            "Выберите 1 ситуацию для спокойного “моделирования”: повторите фразу ребёнка правильно, без оценки.",
            "5 минут игры на словарь (категории: еда/одежда/игрушки).",
            "В конце — один открытый вопрос: «Что было самым интересным?»",
        ]
    elif rf in ("pro_friendly", "case_digest") and aud != "parents":
        practice = [
            "Сформулируйте цель на 2 недели и 1–2 измеримых критерия (частота/точность/самоконтроль).",
            "Соберите мини-протокол: стимул → подсказка → самостоятельное выполнение → перенос в спонтанную речь.",
            "Подготовьте короткий чек-лист для родителей (до 6 пунктов) + как фиксировать прогресс (1 минуту в день).",
        ]
    else:
        practice = [
            "5 минут артикуляционной гимнастики (в игре, перед зеркалом).",
            "5 минут словарной игры: категории/противоположности/описание предметов.",
            "1 минута дыхательной игры (пузыри/ватный шарик/перышко).",
        ]

    age_tag = infer_target_age(rf, picked_title_c, summary_c, practice)

    # --- Hook + meaning (no robot headings)
    if rf == "question_week":
        q = make_question_week()
        hook = (
            f"Замечаете, что в речи ребёнка есть “затыки” или вы сомневаетесь, нормально ли это для возраста? "
            f"Давайте без давления — через один понятный вопрос.\n{q}"
        )
        meaning = [
            "Так вы увидите реальную картину (не по ощущениям), и станет яснее, что тренировать в первую очередь.",
            "Даже 5 минут в день помогают, если шаги стабильные и добрые — без «экзаменов».",
        ]
    else:
        if aud == "parents":
            hook = (
                "Малыш отказывается делать гимнастику и убегает от зеркала? Знакомая ситуация! "
                "Давайте превратим это в маленькую игру на 5–7 минут — без борьбы и без «переделывай»."
            )
        else:
            hook = (
                "Нужна короткая, воспроизводимая практика, которую легко перенести в домашку без сопротивления? "
                "Ниже — компактный протокол на 5–7 минут + методическая логика."
            )

        meaning = [
            "Регулярность важнее идеальности: короткие ежедневные шаги дают устойчивее результат.",
            "Сначала комфорт и желание говорить, потом точность — так меньше сопротивления и больше повторов.",
            "Если ребёнок устал — заканчиваем раньше, чтобы не закреплять отрицательный опыт.",
        ]

        # adapt by rubric
        if rf == "bilingual_parents":
            meaning = [
                "Смешивание языков часто бывает частью нормы — важнее динамика и желание общаться.",
                "Мягкое моделирование фразы по-русски помогает без давления и «запретов».",
                "Регулярный «островок русского» лучше редких “марафонов”.",
            ]
        elif rf == "age_norms":
            meaning = [
                "Возрастные нормы — ориентир, а не «экзамен»: диапазон вариантов широкий.",
                "Смотрите динамику 2–4 недели: растёт ли понимание, инициатива общения, новые слова/фразы.",
                "Если сомневаетесь — фиксируйте примеры и обсуждайте по фактам, так спокойнее.",
            ]
        elif rf == "myth_fact":
            meaning = [
                "Мифы пугают, а нам важны наблюдаемые признаки и динамика, а не «страшные слова».",
                "Обычно полезнее поддерживать коммуникацию и регулярную практику, чем искать один «симптом».",
                "Если тревожно — лучше оценивать ситуацию комплексно и мягко.",
            ]
        elif rf in ("pro_friendly", "case_digest") and aud != "parents":
            meaning = [
                "Стройте связку: задача → критерий → шаги → контроль (2–4 недели), затем проверяйте перенос.",
                "Опирайтесь на мотивацию: простая структура занятия снижает отказ и повышает число повторов.",
                "Отдельно фиксируйте контекст и качество подсказок (вербальные/визуальные/тактильные).",
            ]

    # Normalize and keep it short (new laconic spec)
    meaning = [norm_space(x) for x in meaning if norm_space(x)][:2]
    practice = [norm_space(x) for x in practice if norm_space(x)][:2]

    # --- Navigation panel for colleagues (4 fixed lines)
    nav_panel = make_nav_panel(rf, practice, picked_title_c, summary_c)

    # --- Norm block
    if rf in ("pro_friendly", "case_digest") and aud != "parents":
        norm_lines = [
            "✅ Норма: сохранён контакт, понимание инструкций, позитивная динамика по критериям в 2–4 недели.",
            "⚠️ Обсудить со специалистом: регресс навыков, стойкое отсутствие прогресса при регулярной практике 4–6 недель, выраженные трудности понимания.",
        ]
    else:
        norm_lines = [
            "✅ Норма: ребёнок понимает обращённую речь, общается (жестами/словами), и есть постепенный прогресс по неделям.",
            "⚠️ Обсудить со специалистом: если ребёнок часто не понимает простые просьбы, резко “теряет” навыки или прогресса нет при регулярной практике 4–6 недель.",
        ]

    # --- Source block (masked for parents)
    factcheck = picked.get("fact_check") or ""
    stype = picked.get("source_type") or source_type_label_from_factcheck(factcheck)

    src_label = friendly_source_label(link)

    if aud == "parents":
        source_lines = [
            f"Источник: {src_label}",
            "Основа: рекомендации логопедов",
            f"🔗 {link}",
        ]
    else:
        # pros can see title and type
        title_line = clamp_text(picked_title, 160) if picked_title else ""
        if title_line:
            source_lines = [f"Материал: {title_line}", f"Тип: {stype}", f"🔗 {link}"]
        else:
            source_lines = [f"Тип: {stype}", f"🔗 {link}"]

    # --- Engagement (before disclaimer)
    engage = make_engagement_question(aud, age_tag, rf)
    engage_line = f"💬 {norm_space(engage)}"

    ok, q_reason = _quality_gate(rf, aud, link, hook, meaning, practice, nav_panel, norm_lines, age_tag)
    meta: Dict[str, Any] = {
        "ok": ok,
        "reason": q_reason,
        "rubric_format": rf,
        "audience": aud,
        "source_type": stype,
        "age": age_tag,
    }
    if not ok:
        return "", meta

    # Strict plain text structure
    lines: List[str] = []
    lines.append(f"{rubric_title} {title_suffix}".strip())
    lines.append(f"👶 Возраст: {age_tag}".strip())
    lines.append("")
    lines.append(hook.strip())
    lines.append("")
    for x in meaning:
        x = norm_space(x)
        if x:
            lines.append(f"• {x}")
    lines.append("")
    lines.append("Практика на сегодня (5–7 минут)")
    for i, x in enumerate(practice, start=1):
        x = norm_space(x)
        if x:
            lines.append(f"{i}) {x}")
    # Navigation panel for colleagues (4 fixed lines)
    for x in nav_panel:
        x = norm_space(x)
        if x:
            lines.append(x)
    lines.append("")
    lines.append("Норма / когда нужен специалист")
    for x in norm_lines:
        x = norm_space(x)
        if x:
            lines.append(x)
    lines.append("")
    lines.append("Источник")
    for x in source_lines:
        lines.append(norm_space(x))

    lines.append("")
    lines.append(engage_line)

    if disclaimer:
        lines.append("")
        lines.append(f"ℹ️ {norm_space(disclaimer)}")
    if tags:
        lines.append("")
        lines.append(tags)

    raw_plain = "\n".join(lines).strip()

    # Rewrite with requested fallback behavior
    final_plain, used_rewrite, note = rewrite_if_enabled_plain(raw_plain, aud, rubric_format=rf)
    meta["rewrite_used"] = used_rewrite
    meta["rewrite_note"] = note

    # Hard total length cap (requested)
    final_plain = _enforce_total_limit_v3(final_plain, POST_MAX_CHARS)

    # Absolute guard
    if not _has_required_structure_plain_v3(final_plain):
        print("[WARN] final structure broken unexpectedly -> force raw")
        final_plain = _enforce_total_limit_v3(raw_plain, POST_MAX_CHARS)
        meta["rewrite_used"] = False
        meta["rewrite_note"] = "rewrite:force_raw_structure"

    return final_plain, meta


# =========================
# Plain -> Telegram HTML rendering
# =========================

_HTML_HEADINGS = set(_REQUIRED_HEADINGS_V3 + ["Норма / когда нужен специалист", "Источник", "Практика на сегодня (5–7 минут)"])

def _escape(s: str) -> str:
    return _html.escape(s or "", quote=False)


def _strip_html_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    return _html.unescape(s)


def _looks_like_html(s: str) -> bool:
    return bool(re.search(r"</?(b|i|a)\b", s or ""))


def render_plain_to_telegram_html(plain_text: str) -> str:
    """
    Convert our strict plain post to Telegram HTML:
      - First line as <b>title</b>
      - Age line bolded
      - Headings as <b>Heading</b>
      - Link line "🔗 URL" as clickable <a href="URL">domain</a>
      - Disclaimer line starting with "ℹ️ " italicized
    """
    lines = (plain_text or "").splitlines()
    if not lines:
        return ""

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
                dom = safe_domain(url) or url
                out.append(f"🔗 <a href=\"{_html.escape(url, quote=True)}\">{_escape(dom)}</a>")
            else:
                out.append(_escape(stripped))
            continue

        if stripped.startswith("ℹ️ "):
            out.append(f"<i>{_escape(stripped)}</i>")
            continue

        out.append(_escape(s))

    return "\n".join(out).strip()


def parse_plain_sections_v3(plain_post: str) -> Tuple[str, str, Dict[str, List[str]], List[str]]:
    """
    Parse strict v3.0 post.
    Returns: (title, age, sections_dict, pre_practice_lines)
      - title = first line
      - age = second line (without prefix)
      - sections: keys are headings in _REQUIRED_HEADINGS_V3 plus Norm/Source/Practice
      - pre_practice_lines: lines between age line and "Практика..." heading
    """
    lines = (plain_post or "").splitlines()
    title = (lines[0].strip() if lines else "").strip()
    age = ""
    if len(lines) >= 2 and lines[1].strip().startswith("👶 Возраст:"):
        age = lines[1].split(":", 1)[1].strip()

    sec: Dict[str, List[str]] = {}
    cur = ""
    pre_practice: List[str] = []

    headings = set(_REQUIRED_HEADINGS_V3 + ["Норма / когда нужен специалист", "Источник", "Практика на сегодня (5–7 минут)"])

    NAV_PREFIXES = ("🧠 Навык:", "🎯 Цель:", "📌 Подсказка:", "📏 Критерий прогресса:")

    for i, line in enumerate(lines[2:], start=2):
        s = line.strip()
        if s in headings:
            cur = s
            sec[cur] = []
            continue
        if cur:
            if s.startswith(NAV_PREFIXES):
                sec.setdefault("__NAV__", []).append(s)
            else:
                sec[cur].append(line.rstrip("\n"))
        else:
            if line.strip():
                pre_practice.append(line.rstrip("\n"))

    return title, age, sec, pre_practice


def build_card_theses_from_plain_v3(plain_post: str) -> Tuple[List[str], str]:
    """
    Card must show exactly 3 short theses derived from the post:
      1) key benefit
      2) practice
      3) when to a specialist
    Also returns age tag.
    """
    _, age, sec, pre = parse_plain_sections_v3(plain_post)

    def _benefit() -> str:
        # prefer first bullet line in pre_practice
        for x in pre:
            s = x.strip()
            if s.startswith("•"):
                return s.lstrip("•").strip()
        # else first sentence of hook (first non-empty pre line)
        for x in pre:
            s = x.strip()
            if s:
                # take first sentence
                m = re.split(r"(?<=[.!?])\s+", s)
                return m[0].strip()
        return "Маленький шаг сегодня — больше спокойствия завтра."

    def _practice() -> str:
        arr = sec.get("Практика на сегодня (5–7 минут)", []) or []
        for x in arr:
            s = x.strip()
            s = re.sub(r"^\d+\)\s*", "", s)
            if s:
                return s
        return "Сделайте один маленький шаг (5 минут) — в игре."

    def _specialist() -> str:
        arr = sec.get("Норма / когда нужен специалист", []) or []
        for x in arr:
            s = x.strip()
            if s.startswith("⚠️"):
                return s.lstrip("⚠️").strip()
        for x in arr:
            s = x.strip()
            if s:
                return s.lstrip("✅").lstrip("⚠️").strip()
        return "Если нет прогресса 4–6 недель — обсудите со специалистом."

    a = clamp_text(_benefit(), 92)
    b = clamp_text(_practice(), 92)
    c = clamp_text(_specialist(), 92)
    return [f"💡 {a}", f"🧩 {b}", f"⚠️ {c}"], age


def build_caption_plain_v3(plain_post: str, max_bytes: int) -> str:
    """
    Build a compact caption that keeps required semantics and fits max_bytes (UTF-8 bytes).
    """
    title, age, sec, pre = parse_plain_sections_v3(plain_post)

    # Take 1 hook line and 1 bullet (if exists)
    hook_line = ""
    for x in pre:
        s = x.strip()
        if s and not s.startswith("•") and not s.startswith("💬"):
            hook_line = s
            break

    bullet_line = ""
    for x in pre:
        s = x.strip()
        if s.startswith("•"):
            bullet_line = s
            break

    practice_lines = [x.strip() for x in (sec.get("Практика на сегодня (5–7 минут)", []) or []) if re.match(r"^\d+\)\s+", x.strip())]
    norm_lines = [x.strip() for x in (sec.get("Норма / когда нужен специалист", []) or []) if x.strip().startswith(("✅", "⚠️"))]
    source_lines = [x.strip() for x in (sec.get("Источник", []) or []) if x.strip()]
    src_url = ""
    for x in source_lines:
        if x.startswith("🔗 "):
            src_url = x[2:].strip()
            break

    lines: List[str] = []
    if title:
        lines.append(title)
    if age:
        lines.append(f"👶 Возраст: {age}")
    lines.append("")
    if hook_line:
        lines.append(clamp_text(hook_line, 220))
    if bullet_line:
        lines.append(clamp_text(bullet_line, 180))
    lines.append("")
    lines.append("Практика на сегодня (5–7 минут)")
    for x in practice_lines[:2]:
        lines.append(clamp_text(x, 175))
    lines.append("")
    lines.append("Норма / когда нужен специалист")
    for x in norm_lines[:2]:
        lines.append(clamp_text(x, 210))
    lines.append("")
    lines.append("Источник")
    if src_url:
        lines.append(f"🔗 {src_url}")
    else:
        lines.append("🔗 (см. полный текст)")

    caption = "\n".join(lines).strip()
    return utf8_clip(caption, max_bytes=max_bytes, add_ellipsis=True)


def render_caption_html_from_plain(plain_caption: str) -> str:
    return render_plain_to_telegram_html(plain_caption)


# =========================
# Card rendering
# =========================

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    ttf = FONTS_DIR / "DejaVuSans.ttf"
    if ttf.exists():
        return ImageFont.truetype(str(ttf), size=size)
    return ImageFont.load_default()


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = (h or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join([c + c for c in h])
    if len(h) != 6:
        return (74, 144, 226)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_image_card(rubric_title: str, subtitle: Any, branding: Dict[str, Any], age_tag: str = "") -> Path:
    """
    v1.9: renders an image card. If age_tag provided, render it as small line under title.
    """
    theme = (branding or {}).get("card_theme", "minimal") or "minimal"
    theme = str(theme).strip().lower()

    W, H = 1280, 720
    accent = _hex_to_rgb((branding or {}).get("card_accent", "#4A90E2"))

    if theme == "kids":
        bg_top = (252, 246, 255)
        bg_bottom = (240, 252, 255)
        panel_fill = (255, 255, 255)
        panel_outline = (236, 230, 244)
        title_color = (32, 36, 46)
        sub_color = (78, 86, 104)
        footer_color = (120, 126, 140)
        wave_alpha = 30
    elif theme == "scientific":
        bg_top = (245, 247, 250)
        bg_bottom = (232, 236, 244)
        panel_fill = (255, 255, 255)
        panel_outline = (220, 226, 235)
        title_color = (16, 20, 30)
        sub_color = (54, 62, 78)
        footer_color = (98, 104, 118)
        wave_alpha = 22
        if sum(accent) > 560:
            accent = (36, 79, 166)
    else:
        bg_top = (245, 247, 250)
        bg_bottom = (235, 240, 246)
        panel_fill = (255, 255, 255)
        panel_outline = (235, 238, 242)
        title_color = (24, 32, 44)
        sub_color = (70, 78, 92)
        footer_color = (110, 118, 132)
        wave_alpha = 26

    img = Image.new("RGB", (W, H), bg_top)
    draw = ImageDraw.Draw(img)

    for y in range(H):
        t = y / (H - 1)
        r = int(bg_top[0] + (bg_bottom[0] - bg_top[0]) * t)
        g = int(bg_top[1] + (bg_bottom[1] - bg_top[1]) * t)
        b = int(bg_top[2] + (bg_bottom[2] - bg_top[2]) * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    if theme in ("minimal", "scientific"):
        for i in range(3):
            y0 = 440 + i * 55
            pts = []
            for x in range(0, W + 1, 40):
                yy = y0 + int(12 * math.sin((x / 140.0) + i))
                pts.append((x, yy))
            ld.line(pts, fill=(*accent, wave_alpha), width=6 if theme == "minimal" else 5)

        if theme == "scientific":
            gx0, gy0, gx1, gy1 = 760, 60, 1240, 300
            step = 34
            grid_col = (accent[0], accent[1], accent[2], 16)
            for x in range(gx0, gx1, step):
                ld.line([(x, gy0), (x, gy1)], fill=grid_col, width=2)
            for y in range(gy0, gy1, step):
                ld.line([(gx0, y), (gx1, y)], fill=grid_col, width=2)

    elif theme == "kids":
        seed = int(hashlib.sha1((rubric_title or "").encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        dot_col = (accent[0], accent[1], accent[2], 22)
        for _ in range(120):
            x = rng.randint(60, W - 60)
            y = rng.randint(60, H - 60)
            rr = rng.randint(3, 9)
            ld.ellipse([x - rr, y - rr, x + rr, y + rr], fill=dot_col)
        for cx, cy, rr in [(220, 160, 110), (1120, 520, 140)]:
            ld.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(accent[0], accent[1], accent[2], 18))

    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(img)

    panel = (70, 90, W - 70, H - 110)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([panel[0] + 6, panel[1] + 10, panel[2] + 6, panel[3] + 10], radius=28, fill=(0, 0, 0, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(panel, radius=28, fill=panel_fill, outline=panel_outline, width=2)

    ax = panel[0] + 28
    ay = panel[1] + 28
    draw.rounded_rectangle([ax, ay, ax + 10, panel[3] - 28], radius=6, fill=accent)

    f_title = _load_font(56 if theme != "scientific" else 54)
    f_age = _load_font(28 if theme != "scientific" else 26)
    f_small = _load_font(24)

    x_text = ax + 28
    y_text = panel[1] + 44
    max_w = panel[2] - x_text - 28

    def fit_one_line(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        t = norm_space(text or "")
        if not t:
            return ""
        if draw.textlength(t, font=font) <= max_width:
            return t
        base = t.rstrip(" .,:;—-")
        ell = "…"
        while base and draw.textlength(base + ell, font=font) > max_width:
            base = base[:-1].rstrip(" .,:;—-")
        return (base + ell).strip() if base else "…"

    def wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        words = (text or "").split()
        if not words:
            return []
        lines2: List[str] = []
        cur: List[str] = []
        for w in words:
            test = " ".join(cur + [w])
            if draw.textlength(test, font=font) <= max_width:
                cur.append(w)
            else:
                if cur:
                    lines2.append(" ".join(cur))
                cur = [w]
        if cur:
            lines2.append(" ".join(cur))
        return lines2

    # Title
    for ln in wrap(rubric_title, f_title, max_w)[:3]:
        draw.text((x_text, y_text), ln, fill=title_color, font=f_title)
        y_text += 68

    # Age line under title (small)
    if age_tag:
        y_text += 2
        age_line = fit_one_line(f"👶 {age_tag}", f_age, max_w)
        draw.text((x_text, y_text), age_line, fill=sub_color, font=f_age)
        y_text += 44

    y_text += 10

    # v1.7+: if subtitle is a list -> render exactly 3 theses (one line each)
    if isinstance(subtitle, (list, tuple)):
        theses = [norm_space(str(x)) for x in subtitle if norm_space(str(x))]
        theses = theses[:3]
        f_th = _load_font(36 if theme != "scientific" else 34)
        for t in theses:
            one = fit_one_line(t, f_th, max_w)
            if one:
                draw.text((x_text, y_text), one, fill=sub_color, font=f_th)
                y_text += 52
    else:
        # legacy
        f_sub = _load_font(32 if theme != "scientific" else 30)
        sub_txt = str(subtitle or "")
        for ln in wrap(sub_txt, f_sub, max_w)[:3]:
            draw.text((x_text, y_text), ln, fill=sub_color, font=f_sub)
            y_text += 44

    footer = (branding or {}).get("card_footer", "")
    if footer:
        draw.text((panel[0] + 28, panel[3] - 48), footer, fill=footer_color, font=f_small)

    # ---- subtitle_key FIX: stable hash input even when subtitle is list/tuple ----
    if isinstance(subtitle, (list, tuple)):
        subtitle_key = " | ".join([norm_space(str(x)) for x in subtitle if norm_space(str(x))])
    else:
        subtitle_key = norm_space(str(subtitle or ""))
    subtitle_key = subtitle_key[:320]
    # ---------------------------------------------------------------------------

    out = STATE_DIR / f"card_{sha1(theme + rubric_title + subtitle_key + age_tag)[:10]}.png"
    img.save(out)
    return out


# =========================
# Telegram sending with “safe send”
# =========================

def tg_request(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, data=data, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def _tg_error_description(resp_text: str) -> str:
    try:
        j = json.loads(resp_text or "")
        if isinstance(j, dict):
            return str(j.get("description") or "") or (resp_text or "")
    except Exception:
        pass
    return (resp_text or "").strip()


def _safe_retry_plain_text(data: Dict[str, Any], caption_max_bytes: Optional[int] = None) -> Dict[str, Any]:
    """
    Remove parse_mode and strip markup (HTML/Markdown) from text/caption.
    For captions, also enforce UTF-8 bytes limit.
    """
    data2 = dict(data)
    data2.pop("parse_mode", None)

    if "text" in data2:
        data2["text"] = _strip_html_tags(str(data2.get("text", "")))

    if "caption" in data2:
        plain = _strip_html_tags(str(data2.get("caption", "")))
        cap_limit = int(caption_max_bytes or TG_CAPTION_MAX_BYTES)
        data2["caption"] = utf8_clip(plain, max_bytes=cap_limit, add_ellipsis=True)

    return data2


def tg_request_safe(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Requested behavior:
      - if Telegram returns 400 (Bad Request), retry once without markup (plain text),
        and log the reason.
      - extra hardening: if caption too long even after retry, clip more and retry once more.
    """
    try:
        return tg_request(method, data=data, files=files)
    except requests.exceptions.HTTPError as e:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        text = ""
        try:
            text = (resp.text or "") if resp is not None else ""
        except Exception:
            text = ""

        if status == 400:
            desc = _tg_error_description(text)
            print(f"[WARN] Telegram 400 on {method}. Will retry plain. Description: {desc}")

            # 1) plain retry
            try:
                data_plain = _safe_retry_plain_text(data)
                return tg_request(method, data=data_plain, files=files)
            except requests.exceptions.HTTPError as e2:
                resp2 = getattr(e2, "response", None)
                status2 = getattr(resp2, "status_code", None)
                text2 = ""
                try:
                    text2 = (resp2.text or "") if resp2 is not None else ""
                except Exception:
                    text2 = ""
                desc2 = _tg_error_description(text2)

                # 2) If caption too long, clip more and retry once more
                if status2 == 400 and "caption is too long" in (desc2 or "").lower() and "caption" in data:
                    smaller = min(780, TG_CAPTION_MAX_BYTES)
                    print(f"[WARN] Telegram still says caption too long. Retrying with smaller caption bytes={smaller}.")
                    data_plain2 = _safe_retry_plain_text(data, caption_max_bytes=smaller)
                    return tg_request(method, data=data_plain2, files=files)

                print(f"[ERROR] Telegram safe retry failed for {method}: {e2}. Description: {desc2}")
                raise
        raise


def send_message(chat_id: str, html_text: str) -> None:
    data = {"chat_id": chat_id, "text": html_text}
    if TELEGRAM_PARSE_MODE:
        data["parse_mode"] = TELEGRAM_PARSE_MODE
    tg_request_safe("sendMessage", data=data)


def send_photo(chat_id: str, photo_path: Path, caption_html: str, caption_plain_fallback: Optional[str] = None) -> None:
    """
    Robust sendPhoto:
      - try HTML caption if it fits and looks like HTML
      - fallback to plain caption
      - if Telegram says caption too long -> progressively shrink caption
      - as last resort: send without caption (and rely on full text message after)
    """
    plain_fb = caption_plain_fallback or _strip_html_tags(caption_html or "")
    plain_fb = plain_fb or ""

    # Prepare candidates (caption, parse_mode)
    candidates: List[Tuple[str, Optional[str], str]] = []

    # 1) HTML caption only if it fits bytes and contains safe tags
    cap_html = (caption_html or "").strip()
    if cap_html and _looks_like_html(cap_html) and len(cap_html.encode("utf-8")) <= TG_CAPTION_MAX_BYTES:
        candidates.append((cap_html, TELEGRAM_PARSE_MODE or "HTML", "html"))

    # 2) Plain caption (full limit)
    cap_plain = utf8_clip(plain_fb, max_bytes=TG_CAPTION_MAX_BYTES, add_ellipsis=True)
    candidates.append((cap_plain, None, "plain"))

    # 3) Smaller plain captions (for stubborn Telegram limits)
    for mb in [min(780, TG_CAPTION_MAX_BYTES), 650, 520, 400]:
        candidates.append((utf8_clip(plain_fb, max_bytes=mb, add_ellipsis=True), None, f"plain_{mb}"))

    # 4) No caption fallback
    candidates.append(("", None, "no_caption"))

    last_err: Optional[Exception] = None

    for cap, pm, label in candidates:
        data: Dict[str, Any] = {"chat_id": chat_id}
        if cap:
            data["caption"] = cap
        if pm and cap:
            data["parse_mode"] = pm

        try:
            with photo_path.open("rb") as f:
                tg_request("sendPhoto", data=data, files={"photo": f})
            return
        except requests.exceptions.HTTPError as e:
            last_err = e
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            txt = ""
            try:
                txt = (resp.text or "") if resp is not None else ""
            except Exception:
                txt = ""
            desc = _tg_error_description(txt)

            if status == 400:
                # we keep trying next candidate (this is expected for parse errors / caption length)
                print(f"[WARN] sendPhoto failed (attempt={label}) 400: {desc}")
                continue

            # non-400: stop immediately
            raise

    # If all attempts failed, raise last error
    if last_err:
        raise last_err


def send_post_with_card(chat_id: str, card_path: Path, plain_post: str, html_full_post: str) -> None:
    """v1.9 sending logic:

    - If full post fits caption limits -> send ONE message: photo + full caption.
    - If too long -> send TWO messages: photo (no caption) + full text message.
    """

    full_html = (html_full_post or "").strip()
    can_caption = TG_ALLOW_CAPTION and full_html and len(full_html.encode("utf-8")) <= TG_CAPTION_MAX_BYTES

    try:
        send_photo(chat_id, card_path, (full_html if can_caption else ""))
    except Exception as e:
        print(f"[ERROR] sendPhoto failed, fallback to sendMessage only. Reason: {e}")
        if full_html:
            send_message(chat_id, full_html)
        return

    if full_html and not can_caption:
        send_message(chat_id, full_html)


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:60] or "item"


def write_dry_run_outputs(
    out_dir: Path,
    idx: int,
    aud: str,
    rubric_id: str,
    rubric_title: str,
    plain_post: str,
    html_full_post: str,
    card_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{idx:02d}_{_slug(aud)}_{_slug(rubric_id or rubric_title)}"

    # card
    card_out = out_dir / f"{base}.png"
    try:
        shutil.copyfile(card_path, card_out)
    except Exception:
        Image.open(card_path).save(card_out)

    (out_dir / f"{base}.plain.txt").write_text(plain_post, encoding="utf-8")
    (out_dir / f"{base}.full.html.txt").write_text(html_full_post, encoding="utf-8")

    full_html = (html_full_post or "").strip()
    full_bytes = len(full_html.encode("utf-8")) if full_html else 0
    can_caption = TG_ALLOW_CAPTION and full_html and full_bytes <= TG_CAPTION_MAX_BYTES

    # What would be sent
    (out_dir / f"{base}.send.photo_caption.html.txt").write_text((full_html if can_caption else ""), encoding="utf-8")
    (out_dir / f"{base}.send.message.html.txt").write_text(("" if can_caption else full_html), encoding="utf-8")

    meta = {
        "idx": idx,
        "audience": aud,
        "rubric_id": rubric_id,
        "rubric_title": rubric_title,
        "card": str(card_out),
        "caption_bytes_max": TG_CAPTION_MAX_BYTES,
        "full_bytes": full_bytes,
        "caption_used": bool(can_caption),
    }
    (out_dir / f"{base}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Weekly stats + drafts
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

    plain = "\n".join(lines).strip()
    return render_plain_to_telegram_html(plain)


def handle_draft(pub_cfg: Dict[str, Any], entry: Dict[str, Any], stats: Dict[str, Any], week_key: str) -> None:
    mode = (pub_cfg.get("drafts_mode") or "skip").strip()
    drafts_chat_id = ""
    if mode == "post_to_drafts_chat":
        env_name = pub_cfg.get("drafts_chat_id_env") or "TELEGRAM_DRAFTS_CHAT_ID"
        drafts_chat_id = os.getenv(env_name, "").strip() or TELEGRAM_DRAFTS_CHAT_ID

    drafts = load_state("drafts.json", [])
    drafts.append(entry)
    save_state("drafts.json", drafts[-2000:])

    bump_weekly(stats, week_key, "rejected", 1, reason=str(entry.get("reason", "unknown")))

    if mode == "post_to_drafts_chat" and drafts_chat_id:
        lines = [
            "Черновик/пропуск",
            "",
            f"Причина: {entry.get('reason')}",
            f"Рубрика: {entry.get('rubric_title', '')}",
            f"Аудитория: {entry.get('audience', '')}",
            f"Заголовок: {entry.get('title')}",
            f"Ссылка: {entry.get('link')}",
        ]
        send_message(drafts_chat_id, render_plain_to_telegram_html("\n".join(lines)))


# =========================
# Selection
# =========================

def pick_item(
    items: List[Dict[str, str]],
    used_canon: set[str],
    used_titles: set[str],
    quality_cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
    ranked: List[Tuple[int, Dict[str, str]]] = []
    for it in items:
        t = norm_space(it.get("title", ""))
        l = it.get("link", "")
        if not l:
            continue
        s, _ = score_item(t or "(no title)", l, quality_cfg)
        if s >= 0:
            ranked.append((s, it))
    ranked.sort(key=lambda x: x[0], reverse=True)

    for _, it0 in ranked[:22]:
        it = enrich_article(dict(it0))
        canon = it.get("canonical") or it.get("link", "")
        if not canon or canon in used_canon:
            continue

        raw_title = it.get("article_title") or it.get("title") or ""
        tkey = norm_title_key(raw_title)
        if tkey and tkey in used_titles:
            continue

        dom = safe_domain(canon)
        summ = it.get("article_summary") or it.get("summary") or ""
        ok, reason = is_scientific_or_methodical(dom, raw_title, summ, quality_cfg)
        if not ok:
            return None, {
                "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                "reason": f"fact_check_failed:{reason}",
                "title": raw_title,
                "link": canon,
                "domain": dom,
            }

        it["picked_title"] = raw_title
        it["picked_summary"] = summ
        it["fact_check"] = reason
        it["source_type"] = source_type_label_from_factcheck(reason)
        return it, None

    return None, None


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

    sources, quality_cfg = load_sources()
    used_canon = set(load_state("used_canonical.json", []))
    used_titles = set(load_state("used_titles.json", []))

    pub_cfg = rub_cfg.get("publishing", {}) or {}
    max_posts = int(pub_cfg.get("max_posts_per_run", 3))
    max_per_aud = int(pub_cfg.get("max_posts_per_audience_per_run", 2))

    stats = load_weekly_stats()

    dry_out_dir: Optional[Path] = None
    if DRY_RUN:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dry_out_dir = STATE_DIR / "dry_run" / ts
        dry_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DRY_RUN] enabled: outputs -> {dry_out_dir}")

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
        aud_posted = 0

        for rubric in rubrics:
            if posted >= max_posts or aud_posted >= max_per_aud:
                break
            if not is_due(rubric, now):
                continue

            if (rubric.get("format") or "").strip().lower() == "quality_dashboard":
                dash_title = pub_cfg.get("dashboard_title", "Quality dashboard недели")
                dashboard_html = format_dashboard(stats, week_key, dash_title)
                dash_chat = (pub_cfg.get("dashboard_chat") or "main").strip().lower()
                chat_id = TELEGRAM_CHAT_ID
                if dash_chat == "drafts" and TELEGRAM_DRAFTS_CHAT_ID:
                    chat_id = TELEGRAM_DRAFTS_CHAT_ID
                if DRY_RUN:
                    print("[DRY_RUN] dashboard would be posted.")
                else:
                    send_message(chat_id, dashboard_html)
                time.sleep(0.7)
                continue

            all_items: List[Dict[str, str]] = []
            for sid in rubric.get("sources", []) or []:
                src = sources.get(sid)
                if not src:
                    continue
                try:
                    all_items.extend(fetch_source(src))
                except Exception as e:
                    print(f"[WARN] source {sid} failed: {e}")

            picked, draft = pick_item(all_items, used_canon, used_titles, quality_cfg)
            if draft:
                draft.update({"audience": aud, "rubric": rubric.get("id", ""), "rubric_title": rubric.get("title", "")})
                handle_draft(pub_cfg, draft, stats, week_key)
                continue
            if not picked:
                continue

            title = rubric.get("title", "Рубрика")
            plain_post, meta = compose_post_plain_v30(title, rubric.get("format", ""), aud, channel_cfg, picked, title_suffix)

            if not meta.get("ok", False) or not plain_post:
                draft_entry = {
                    "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                    "reason": str(meta.get("reason", "quality_gate_failed")),
                    "audience": aud,
                    "rubric": rubric.get("id", ""),
                    "rubric_title": title,
                    "title": picked.get("picked_title") or picked.get("title") or "",
                    "link": picked.get("canonical") or picked.get("link") or "",
                    "domain": safe_domain(picked.get("canonical") or picked.get("link") or ""),
                    "source_type": meta.get("source_type", ""),
                    "rewrite_note": meta.get("rewrite_note", ""),
                }
                handle_draft(pub_cfg, draft_entry, stats, week_key)
                continue

            html_full_post = render_plain_to_telegram_html(plain_post)

            theses, age_tag = build_card_theses_from_plain_v3(plain_post)
            card = render_image_card(title, theses, branding, age_tag=age_tag)

            if DRY_RUN and dry_out_dir is not None:
                write_dry_run_outputs(
                    dry_out_dir,
                    idx=posted + 1,
                    aud=aud,
                    rubric_id=str(rubric.get("id", "") or ""),
                    rubric_title=title,
                    plain_post=plain_post,
                    html_full_post=html_full_post,
                    card_path=card,
                )
            else:
                send_post_with_card(TELEGRAM_CHAT_ID, card, plain_post, html_full_post)

            if not DRY_RUN:
                bump_weekly(stats, week_key, "passed", 1)

            canon = picked.get("canonical") or picked.get("link", "")
            if not DRY_RUN:
                if canon:
                    used_canon.add(canon)
                tkey = norm_title_key(picked.get("picked_title") or picked.get("title") or "")
                if tkey:
                    used_titles.add(tkey)

            posted += 1
            aud_posted += 1
            time.sleep(1.2)

    if not DRY_RUN:
        save_state("used_canonical.json", sorted(list(used_canon))[-6000:])
        save_state("used_titles.json", sorted(list(used_titles))[-6000:])
        save_weekly_stats(stats)

    print(
        "Done. "
        f"Posted: {posted}. Audience: {AUDIENCE}. Rewrite: {REWRITE_PROVIDER}. Week: {week_key}. "
        f"Parse: {TELEGRAM_PARSE_MODE}. CaptionMaxBytes: {TG_CAPTION_MAX_BYTES}. "
        f"FullTextAfterPhoto: {int(TG_SEND_FULL_TEXT_AFTER_PHOTO)}"
        f"{' [DRY_RUN]' if DRY_RUN else ''}"
    )


if __name__ == "__main__":
    run()

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

# v2.0: global hard length rule (requested)
POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1000"))

# v1.6.1: style/length knobs (legacy). We hard-cap by POST_MAX_CHARS anyway.
PARENTS_MAX_BODY_CHARS = int(os.getenv("PARENTS_MAX_BODY_CHARS", "900"))
PROS_MAX_BODY_CHARS = int(os.getenv("PROS_MAX_BODY_CHARS", "1050"))

# v1.6.1: quality gate knobs
MIN_MEANING_BULLETS = int(os.getenv("MIN_MEANING_BULLETS", "2"))
MIN_PRACTICE_STEPS = int(os.getenv("MIN_PRACTICE_STEPS", "2"))  # requested: practice 1–2 steps

# Telegram hard limits: captions are very strict (often ~1024 chars, but UTF-8 bytes matter).
# Use bytes-limit for safety.
TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))

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
    """Clip by UTF-8 bytes. Guarantees result.encode('utf-8') <= max_bytes."""
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
        while out and len((out + ell).encode("utf-8")) > max_bytes:
            out = out[:-1]
            out = out.rstrip(" .,:;—-")
        if len((out + ell).encode("utf-8")) <= max_bytes:
            out = out + ell
    while len(out.encode("utf-8")) > max_bytes and out:
        out = out[:-1]
    return out.strip() or "…"


def _is_sovet_dnya_format(rubric_format: str) -> bool:
    rf = (rubric_format or "").strip().lower()
    if rf in {
        "tip_day", "daily_tip", "advice_day", "tip_of_day",
        "sovet_dnya", "sovet_dnya_parents", "sovet_day",
        "sovet_logopeda", "logoped_tip_day",
    }:
        return True
    return ("совет" in rf) or ("tip" in rf and "day" in rf)


def mask_diagnostics_for_parents(text: str) -> str:
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
    blob = " ".join([picked_title or "", summary or "", " ".join(practice or [])]).lower()

    if any(k in blob for k in ["дисграф", "дислекс", "письм", "чтени", "школ", "первокласс", "1 класс", "второкласс"]):
        return "6–9 лет"
    if any(k in blob for k in ["звук р", "звук л", "шипящ", "свистящ", "ротац", "ламбдац", "постановк"]):
        return "4–7 лет"
    if any(k in blob for k in ["фраз", "предложен", "связн", "рассказ", "согласован", "падеж", "множествен"]):
        return "3–6 лет"
    if any(k in blob for k in ["первые слова", "лепет", "гулен", "не говорит", "молчит", "словар", "запуск речи"]):
        return "от 1,5 лет"

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
    per_aud = PARENTS_MAX_BODY_CHARS if a == "parents" else PROS_MAX_BODY_CHARS
    return min(int(per_aud), int(POST_MAX_CHARS))


_REQUIRED_HEADINGS_V3 = [
    "Практика на сегодня (5–7 минут)",
    "Норма / когда нужен специалист",
    "Источник",
]

_NAV_KEYS = [
    "🧠 Навык:",
    "🎯 Цель:",
    "📌 Подсказка:",
    "📏 Критерий прогресса:",
]


def _build_rewrite_prompt_v3(body: str, audience: str, max_chars: int, rubric_format: str = "") -> str:
    rf = (rubric_format or "").strip().lower()
    is_tip = _is_sovet_dnya_format(rf)

    base_role = (
        "Роль: эмпатичный, современный логопед. Ты дружелюбно общаешься с уставшими родителями, "
        "но пишешь точно и аккуратно.\n"
    )

    hard_bans = (
        "ЖЁСТКИЕ правила:\n"
        "• Пиши максимально лаконично: меньше вводных слов, меньше воды.\n"
        f"• ВЕСЬ текст поста (целиком) не длиннее {max_chars} символов.\n"
        "• «Практика на сегодня» — 1–2 коротких шага.\n"
        "• НЕ используй шаблонные заголовки-отбивки: «Суть», «Коротко», «Что это значит для вас», «Вывод», «Итог», «Резюме», «Важно».\n"
        "• НЕ добавляй новые подзаголовки и новые разделы.\n"
        "• НЕ ставь диагнозы, НЕ обещай лечения, НЕ назначай препараты.\n"
        "• Не добавляй новых фактов — только перефразируй и сожми то, что уже есть.\n"
    )

    structure = (
        "Структура должна быть строго такой:\n"
        "1) Первая строка — название рубрики (как в исходнике).\n"
        "2) Вторая строка — строка возраста в формате: «👶 Возраст: ...».\n"
        "3) Далее — хук (жизненная ситуация/вопрос) + 2–3 короткие строки пользы, без отдельных подзаголовков.\n"
        f"4) Затем идут заголовки (каждый — отдельной строкой) строго из списка: {', '.join(_REQUIRED_HEADINGS_V3)}.\n"
        "5) Внутри «Практика на сегодня» — 1–2 шага, нумерацией 1) 2).\n"
        "6) Сразу после «Практика на сегодня (5–7 минут)» вставь навигационную полосу РОВНО из 4 строк:\n"
        "🧠 Навык: (3–5 слов)\n"
        "🎯 Цель: (3–5 слов)\n"
        "📌 Подсказка: (3–5 слов)\n"
        "📏 Критерий прогресса: (3–5 слов)\n"
        "Никаких других строк в этом блоке.\n"
        "7) В конце перед техническим дисклеймером должна быть ровно одна вовлекающая строка, начинающаяся с «💬 ».\n"
        "Форматирование: только обычный текст, без HTML/Markdown.\n"
    )

    tip_rules = ""
    if is_tip:
        tip_rules = (
            "\nДоп. правило для «Совет дня»:\n"
            "- Только практика и поддержка мотивации. Никакой академической теории и классификаций.\n"
        )

    return (
        base_role
        + hard_bans
        + structure
        + tip_rules
        + "\nТЕКСТ ДЛЯ ПЕРЕФОРМУЛИРОВКИ:\n"
        + (body or "").strip()
    )


def _count_words_3_5(s: str) -> bool:
    part = (s.split(":", 1)[1] if ":" in s else "").strip()
    words = [w for w in re.split(r"\s+", part) if w]
    return 3 <= len(words) <= 5


def _has_required_structure_plain_v3(text: str) -> bool:
    lines = [(x or "").rstrip("\n") for x in (text or "").splitlines()]
    if len(lines) < 8:
        return False

    if not (len(lines) >= 2 and lines[1].strip().startswith("👶 Возраст:")):
        return False

    sset = set([x.strip() for x in lines])
    for h in _REQUIRED_HEADINGS_V3:
        if h not in sset:
            return False

    if not any(x.strip().startswith("💬 ") for x in lines):
        return False

    nav_lines = [x.strip() for x in lines if any(x.strip().startswith(k) for k in _NAV_KEYS)]
    if len(nav_lines) != 4:
        return False
    for need, got in zip(_NAV_KEYS, nav_lines):
        if not got.startswith(need):
            return False
        if not _count_words_3_5(got):
            return False

    if not any(re.match(r"^\s*1\)\s+\S+", x) for x in lines):
        return False

    return True


def _enforce_total_chars_keep_structure(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t

    lines = [(x or "").rstrip("\n") for x in t.splitlines()]

    def _looks_like_tags(line: str) -> bool:
        s = line.strip()
        return s.startswith("#") or (" #" in s)

    while lines and _looks_like_tags(lines[-1]):
        lines.pop()
    while lines and lines[-1].strip().startswith("ℹ️ "):
        lines.pop()

    clamped: List[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0:
            clamped.append(clamp_text(s, 90))
            continue
        if i == 1 and s.startswith("👶 Возраст:"):
            clamped.append(clamp_text(s, 28))
            continue
        if s.startswith("💬 "):
            clamped.append("💬 " + clamp_text(s[2:].strip(), 120))
            continue
        if s.startswith("• "):
            clamped.append("• " + clamp_text(s[2:].strip(), 120))
            continue
        if re.match(r"^\d+\)\s+", s):
            n, rest = s.split(")", 1)
            clamped.append(f"{n}) {clamp_text(rest.strip(), 140)}")
            continue
        if s.startswith(("✅", "⚠️")):
            clamped.append(s[0] + clamp_text(s[1:].strip(), 160))
            continue
        if s.startswith("🔗 "):
            clamped.append(s)
            continue
        if any(s == h for h in _REQUIRED_HEADINGS_V3) or any(s.startswith(k) for k in _NAV_KEYS):
            clamped.append(s)
            continue
        clamped.append(clamp_text(line, 220))

    out = "\n".join(clamped).strip()
    if len(out) <= max_chars:
        return out

    cut = out[:max_chars]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()


def rewrite_if_enabled_plain(full_plain_text: str, audience: str, rubric_format: str = "") -> Tuple[str, bool, str]:
    if REWRITE_PROVIDER == "none":
        final_raw = _enforce_total_chars_keep_structure(full_plain_text, POST_MAX_CHARS)
        return final_raw, False, "rewrite:none"

    max_chars = _aud_limits(audience)
    prompt = _build_rewrite_prompt_v3(full_plain_text, audience, max_chars, rubric_format=rubric_format)

    try:
        if REWRITE_PROVIDER in ("groq", "auto"):
            try:
                out = rewrite_with_groq(prompt)
                out = _enforce_total_chars_keep_structure(out, max_chars)
                if not _has_required_structure_plain_v3(out):
                    print("[WARN] rewrite broke structure (groq) -> fallback to raw")
                    raw2 = _enforce_total_chars_keep_structure(full_plain_text, max_chars)
                    return raw2, False, "rewrite:fallback_raw_structure"
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
            out = _enforce_total_chars_keep_structure(out, max_chars)
            if not _has_required_structure_plain_v3(out):
                print("[WARN] rewrite broke structure (gemini) -> fallback to raw")
                raw2 = _enforce_total_chars_keep_structure(full_plain_text, max_chars)
                return raw2, False, "rewrite:fallback_raw_structure"
            return out, True, "rewrite:gemini"

    except Exception as e:
        print(f"[WARN] rewrite failed ({REWRITE_PROVIDER}): {e}")
        raw2 = _enforce_total_chars_keep_structure(full_plain_text, max_chars)
        return raw2, False, "rewrite:fallback_raw_error"

    raw2 = _enforce_total_chars_keep_structure(full_plain_text, max_chars)
    return raw2, False, "rewrite:fallback_raw_unknown"


# =========================
# Post template v3.1 + quality gate
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
    if a == "pros":
        qs = [
            "Какой критерий прогресса взяли бы на 2 недели?",
            "Какая подсказка чаще срабатывает: вербальная или визуальная?",
            "Как вы снижаете отказы на домашке?",
        ]
    else:
        qs = [
            "Что даётся сложнее: начать или закончить спокойно?",
            "Что проще: повторять за вами или играть с выдохом?",
            "Что влияет сильнее: время суток или формат игры?",
        ]
    q = random.choice(qs)
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
    norm_lines: List[str],
    age_tag: str,
) -> Tuple[bool, str]:
    rf = (rubric_format or "").strip().lower()

    if not link or not link.startswith(("http://", "https://")):
        return False, "quality_gate:no_source_link"

    if not (age_tag or "").strip():
        return False, "quality_gate:no_age_tag"

    hook_len = len(norm_space(hook))
    if rf != "question_week" and hook_len < 70:
        return False, f"quality_gate:weak_hook_len:{hook_len}"
    if rf == "question_week" and hook_len < 50:
        return False, f"quality_gate:weak_hook_len:{hook_len}"

    m = [x for x in meaning if norm_space(x)]
    if len(m) < MIN_MEANING_BULLETS:
        return False, f"quality_gate:meaning_bullets_lt_{MIN_MEANING_BULLETS}:{len(m)}"

    p = [x for x in practice if norm_space(x)]
    if len(p) < MIN_PRACTICE_STEPS:
        return False, f"quality_gate:practice_steps_lt_{MIN_PRACTICE_STEPS}:{len(p)}"

    nl = "\n".join([norm_space(x) for x in norm_lines if norm_space(x)])
    if "✅" not in nl or "⚠️" not in nl:
        return False, "quality_gate:norm_block_missing_markers"

    return True, "ok"


def make_nav_strip(rubric_format: str, picked_title: str, summary: str, practice: List[str]) -> List[str]:
    blob = " ".join([picked_title or "", summary or "", " ".join(practice or [])]).lower()
    rf = (rubric_format or "").strip().lower()

    if rf == "bilingual_parents" or any(k in blob for k in ["билингв", "два языка", "код", "переключ"]):
        return [
            "🧠 Навык: переключение без тревоги",
            "🎯 Цель: сеть русского языка",
            "📌 Подсказка: повторяйте мягко по-русски",
            "📏 Критерий прогресса: русская фраза в быту",
        ]

    if any(k in blob for k in ["дуем", "выдох", "пузы", "перыш", "ватн", "дых"]):
        return [
            "🧠 Навык: ровный длительный выдох",
            "🎯 Цель: контроль воздушной струи",
            "📌 Подсказка: играйте, считайте выдох",
            "📏 Критерий прогресса: выдох 5–7 секунд",
        ]

    if any(k in blob for k in ["зеркал", "артикуляц", "язык", "губ", "лопаточ", "часик"]):
        return [
            "🧠 Навык: координация язык губы",
            "🎯 Цель: точная коартикуляция звуков",
            "📌 Подсказка: медленно, без давления",
            "📏 Критерий прогресса: движения ровные стабильно",
        ]

    return [
        "🧠 Навык: глаголы в короткой фразе",
        "🎯 Цель: лексика и грамматика",
        "📌 Подсказка: моделируйте фразу, хвалите",
        "📏 Критерий прогресса: 2–3 слова самостоятельно",
    ]


def compose_post_plain_v31(
    rubric_title: str,
    rubric_format: str,
    audience: str,
    channel_cfg: Dict[str, Any],
    picked: Dict[str, str],
    title_suffix: str,
) -> Tuple[str, Dict[str, Any]]:
    link = picked.get("canonical") or picked.get("link", "")
    picked_title = picked.get("picked_title") or picked.get("title") or ""
    summary = picked.get("picked_summary") or picked.get("summary") or ""
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    tags = " ".join(channel_cfg.get("hashtags", []) or []).strip()

    aud = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()
    is_tip = _is_sovet_dnya_format(rf)

    safe_title = picked_title
    safe_summary = summary
    if aud == "parents":
        safe_title = mask_diagnostics_for_parents(safe_title)
        safe_summary = mask_diagnostics_for_parents(safe_summary)

    picked_title_c = clamp_text(safe_title, 140) if safe_title else ""
    summary_c = clamp_text(safe_summary, 240) if safe_summary else ""

    if is_tip:
        practice = [
            "2 минуты «Эхо»: слог/слово → повтор (хвалим попытку).",
            "2 минуты «Кто что делает?»: 6–10 глаголов по картинкам.",
            "1 минута «Дуем»: пузыри/ватный шарик/перышко.",
        ]
    elif rf == "bilingual_parents":
        practice = [
            "Игра «Два варианта»: ребёнок → вы по-русски → ребёнок.",
            "5 минут «островка русского»: книжка/картинки/комментирование.",
            "В конце дня: 3 предмета назвать по-русски.",
        ]
    elif rf == "age_norms":
        practice = [
            "5 минут описания: предмет → цвет/форма/назначение/действие.",
            "Игра «Кто что делает?»: 10 глаголов по картинкам.",
            "Если билингв — ответить, затем мягко повторить по-русски.",
        ]
    else:
        practice = [
            "5 минут артикуляции в игре перед зеркалом.",
            "5 минут словаря: категории/описание предметов.",
            "1 минута дыхательной игры: пузыри/ватный шарик.",
        ]

    practice = [x for x in practice if norm_space(x)][:2]

    age_tag = infer_target_age(rf, picked_title_c, summary_c, practice)

    if aud == "parents":
        hook = (
            "Малыш убегает от зеркала и не хочет заниматься? Давайте превратим это в игру на 5–7 минут — "
            "без борьбы и без «переделывай»."
        )
    else:
        hook = "Нужна короткая домашка без сопротивления? Ниже — простой протокол и критерий прогресса."

    meaning = [
        "Регулярность важнее идеальности.",
        "Комфорт → мотивация → точность.",
    ]
    if rf == "age_norms":
        meaning = [
            "Норма — ориентир, не «экзамен».",
            "Смотрите динамику 2–4 недели.",
        ]

    norm_lines = [
        "✅ Норма: есть понимание речи и постепенный прогресс.",
        "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.",
    ]

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
        title_line = clamp_text(picked_title, 160) if picked_title else ""
        if title_line:
            source_lines = [f"Материал: {title_line}", f"Тип: {stype}", f"🔗 {link}"]
        else:
            source_lines = [f"Тип: {stype}", f"🔗 {link}"]

    engage_line = f"💬 {norm_space(make_engagement_question(aud, age_tag, rf))}"

    ok, q_reason = _quality_gate(rf, aud, link, hook, meaning, practice, norm_lines, age_tag)
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

    lines: List[str] = []
    lines.append(f"{rubric_title} {title_suffix}".strip())
    lines.append(f"👶 Возраст: {age_tag}".strip())
    lines.append("")
    lines.append(hook.strip())
    lines.append("")
    for x in meaning[:2]:
        x = norm_space(x)
        if x:
            lines.append(f"• {x}")
    lines.append("")
    lines.append("Практика на сегодня (5–7 минут)")
    for i, x in enumerate(practice, start=1):
        x = norm_space(x)
        if x:
            lines.append(f"{i}) {x}")
    lines.append("")
    for nav_line in make_nav_strip(rf, picked_title_c, summary_c, practice):
        lines.append(nav_line)
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

    raw_plain = "\n".join(lines).strip()
    raw_plain = _enforce_total_chars_keep_structure(raw_plain, POST_MAX_CHARS)

    if disclaimer:
        candidate = (raw_plain + "\n\n" + f"ℹ️ {norm_space(disclaimer)}").strip()
        if len(candidate) <= POST_MAX_CHARS:
            raw_plain = candidate

    if tags:
        candidate = (raw_plain + "\n\n" + tags).strip()
        if len(candidate) <= POST_MAX_CHARS:
            raw_plain = candidate

    final_plain, used_rewrite, note = rewrite_if_enabled_plain(raw_plain, aud, rubric_format=rf)
    final_plain = _enforce_total_chars_keep_structure(final_plain, POST_MAX_CHARS)

    meta["rewrite_used"] = used_rewrite
    meta["rewrite_note"] = note

    if not _has_required_structure_plain_v3(final_plain):
        print("[WARN] final structure broken unexpectedly -> force raw")
        final_plain = raw_plain
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


def render_plain_to_telegram_html(plain_text: str) -> str:
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
    lines = (plain_post or "").splitlines()
    title = (lines[0].strip() if lines else "").strip()
    age = ""
    if len(lines) >= 2 and lines[1].strip().startswith("👶 Возраст:"):
        age = lines[1].split(":", 1)[1].strip()

    sec: Dict[str, List[str]] = {}
    cur = ""
    pre_practice: List[str] = []

    headings = set(_REQUIRED_HEADINGS_V3 + ["Норма / когда нужен специалист", "Источник", "Практика на сегодня (5–7 минут)"])

    for line in lines[2:]:
        s = line.strip()
        if s in headings:
            cur = s
            sec[cur] = []
            continue
        if cur:
            sec[cur].append(line.rstrip("\n"))
        else:
            if line.strip():
                pre_practice.append(line.rstrip("\n"))

    return title, age, sec, pre_practice


def build_card_theses_from_plain_v3(plain_post: str) -> Tuple[List[str], str]:
    _, age, sec, pre = parse_plain_sections_v3(plain_post)

    def _benefit() -> str:
        for x in pre:
            s = x.strip()
            if s.startswith("•"):
                return s.lstrip("•").strip()
        for x in pre:
            s = x.strip()
            if s:
                m = re.split(r"(?<=[.!?])\s+", s)
                return m[0].strip()
        return "Маленький шаг сегодня — больше спокойствия завтра."

    def _practice() -> str:
        arr = sec.get("Практика на сегодня (5–7 минут)", []) or []
        for x in arr:
            s = x.strip()
            s2 = re.sub(r"^\d+\)\s*", "", s)
            if s2 and s != s2:
                return s2
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


# =========================
# Card rendering (same as before)
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

    for ln in wrap(rubric_title, f_title, max_w)[:3]:
        draw.text((x_text, y_text), ln, fill=title_color, font=f_title)
        y_text += 68

    if age_tag:
        y_text += 2
        age_line = fit_one_line(f"👶 {age_tag}", f_age, max_w)
        draw.text((x_text, y_text), age_line, fill=sub_color, font=f_age)
        y_text += 44

    y_text += 10

    if isinstance(subtitle, (list, tuple)):
        theses = [norm_space(str(x)) for x in subtitle if norm_space(str(x))][:3]
        f_th = _load_font(36 if theme != "scientific" else 34)
        for t in theses:
            one = fit_one_line(t, f_th, max_w)
            if one:
                draw.text((x_text, y_text), one, fill=sub_color, font=f_th)
                y_text += 52
    else:
        f_sub = _load_font(32 if theme != "scientific" else 30)
        sub_txt = str(subtitle or "")
        for ln in wrap(sub_txt, f_sub, max_w)[:3]:
            draw.text((x_text, y_text), ln, fill=sub_color, font=f_sub)
            y_text += 44

    footer = (branding or {}).get("card_footer", "")
    if footer:
        draw.text((panel[0] + 28, panel[3] - 48), footer, fill=footer_color, font=f_small)

    if isinstance(subtitle, (list, tuple)):
        subtitle_key = " | ".join([norm_space(str(x)) for x in subtitle if norm_space(str(x))])
    else:
        subtitle_key = norm_space(str(subtitle or ""))
    subtitle_key = subtitle_key[:320]

    out = STATE_DIR / f"card_{sha1(theme + rubric_title + subtitle_key + age_tag)[:10]}.png"
    img.save(out)
    return out


# =========================
# Telegram sending (NO-DUP)
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


def send_post_with_card(chat_id: str, card_path: Path, plain_post: str, html_full_post: str) -> None:
    plain_bytes = len((plain_post or "").encode("utf-8"))

    if plain_bytes <= TG_CAPTION_MAX_BYTES:
        try:
            data: Dict[str, Any] = {"chat_id": chat_id, "caption": html_full_post}
            if TELEGRAM_PARSE_MODE:
                data["parse_mode"] = TELEGRAM_PARSE_MODE
            with card_path.open("rb") as f:
                tg_request_safe("sendPhoto", data=data, files={"photo": f})
            return
        except Exception as e:
            print(f"[WARN] sendPhoto(full caption) failed -> fallback to photo+text. Reason: {e}")

    try:
        with card_path.open("rb") as f:
            tg_request_safe("sendPhoto", data={"chat_id": chat_id, "caption": ""}, files={"photo": f})
    except Exception as e:
        print(f"[ERROR] sendPhoto(no caption) failed окончательно, fallback to sendMessage only. Reason: {e}")
        send_message(chat_id, html_full_post)
        return

    send_message(chat_id, html_full_post)


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:60] or "item"


def _dry_send_mode(plain_post: str) -> str:
    return "photo_caption_full" if len((plain_post or "").encode("utf-8")) <= TG_CAPTION_MAX_BYTES else "photo_then_text"


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

    card_out = out_dir / f"{base}.png"
    try:
        shutil.copyfile(card_path, card_out)
    except Exception:
        Image.open(card_path).save(card_out)

    mode = _dry_send_mode(plain_post)

    (out_dir / f"{base}.plain.txt").write_text(plain_post, encoding="utf-8")
    (out_dir / f"{base}.full.html.txt").write_text(html_full_post, encoding="utf-8")

    if mode == "photo_caption_full":
        (out_dir / f"{base}.caption.html.txt").write_text(html_full_post, encoding="utf-8")
        (out_dir / f"{base}.caption.plain.txt").write_text(plain_post, encoding="utf-8")
    else:
        (out_dir / f"{base}.caption.html.txt").write_text("", encoding="utf-8")
        (out_dir / f"{base}.caption.plain.txt").write_text("", encoding="utf-8")

    meta = {
        "idx": idx,
        "audience": aud,
        "rubric_id": rubric_id,
        "rubric_title": rubric_title,
        "card": str(card_out),
        "send_mode": mode,
        "tg_caption_max_bytes": TG_CAPTION_MAX_BYTES,
        "post_max_chars": POST_MAX_CHARS,
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
            plain_post, meta = compose_post_plain_v31(title, rubric.get("format", ""), aud, channel_cfg, picked, title_suffix)

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
            if not DRY_RUN and canon:
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
        f"PostMaxChars: {POST_MAX_CHARS}. "
        f"{' [DRY_RUN]' if DRY_RUN else ''}"
    )


if __name__ == "__main__":
    run()

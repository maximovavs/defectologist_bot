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

USER_AGENT = "logoped-channel-bot/2.1.0 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()               # main publish target (can be overridden by workflow)
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip() # technical channel (dashboard, questions, etc.)

DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")

# Telegram HTML parse mode (we rely on <a href=\"\"> links to hide scary URLs)
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()  # HTML | Markdown | ""

REWRITE_PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()  # none|auto|groq|gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()  # parents|pros|both

POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1000"))
PARENTS_MAX_BODY_CHARS = int(os.getenv("PARENTS_MAX_BODY_CHARS", "900"))
PROS_MAX_BODY_CHARS = int(os.getenv("PROS_MAX_BODY_CHARS", "1050"))

MIN_MEANING_BULLETS = int(os.getenv("MIN_MEANING_BULLETS", "2"))
MIN_PRACTICE_STEPS = int(os.getenv("MIN_PRACTICE_STEPS", "2"))

TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))

INSECURE_TLS_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("INSECURE_TLS_DOMAINS", "") or "").split(",")
    if d.strip()
]


# =========================
# Services imports (you already have these in src/services/)
# =========================

from src.services.image_builder import render_image_card  # PIL card builder with textwrap wrapping
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


def utf8_clip(text: str, max_bytes: int, add_ellipsis: bool = True) -> str:
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
            out = out[:-1].rstrip(" .,:;—-")
        if len((out + ell).encode("utf-8")) <= max_bytes:
            out = out + ell
    while len(out.encode("utf-8")) > max_bytes and out:
        out = out[:-1]
    return out.strip() or "…"


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


def fetch_source(src: Source) -> List[Dict[str, str]]:
    if src.type == "rss":
        return fetch_rss(src.url or "")
    raise ValueError(f"Unsupported source type in this trimmed publisher: {src.type}")


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
# Post (minimal v3.1)
# =========================

def _is_sovet_dnya_format(rubric_format: str) -> bool:
    rf = (rubric_format or "").strip().lower()
    return ("совет" in rf) or ("tip" in rf and "day" in rf)


def make_nav_strip(rubric_format: str) -> List[str]:
    rf = (rubric_format or "").strip().lower()
    if rf == "bilingual_parents":
        return [
            "🧠 Навык: переключение без тревоги",
            "🎯 Цель: сеть русского языка",
            "📌 Подсказка: повторяйте мягко по-русски",
            "📏 Критерий прогресса: русская фраза в быту",
        ]
    return [
        "🧠 Навык: глаголы в короткой фразе",
        "🎯 Цель: лексика и грамматика",
        "📌 Подсказка: моделируйте фразу, хвалите",
        "📏 Критерий прогресса: 2–3 слова самостоятельно",
    ]


def friendly_source_label(url: str) -> str:
    dom = safe_domain(url)
    if not dom:
        return "профессиональные материалы"
    if "logopedy" in dom:
        return "Материалы Logopedy.ru"
    if "logoportal" in dom:
        return "Материалы Logoportal"
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
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    tags = " ".join(channel_cfg.get("hashtags", []) or []).strip()

    aud = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()

    if aud == "parents":
        hook = "Малыш избегает занятий? Сделаем это игрой на 5–7 минут — без борьбы и «переделывай»."
    else:
        hook = "Короткий протокол на 5–7 минут + критерий прогресса."

    practice = [
        "2–3 минуты: повтор слога/слова в игре (хвалим попытку).",
        "2–3 минуты: 6–10 глаголов по картинкам (кто что делает?).",
    ][:2]

    age_tag = "3–6 лет" if not _is_sovet_dnya_format(rf) else "3–5 лет"

    src_label = friendly_source_label(link)
    source_lines = [f"Источник: {src_label}", "Основа: рекомендации логопедов", f"🔗 {link}"]

    lines: List[str] = []
    lines.append(f"{rubric_title} {title_suffix}".strip())
    lines.append(f"👶 Возраст: {age_tag}".strip())
    lines.append("")
    lines.append(hook)
    lines.append("")
    lines.append("• Регулярность важнее идеальности.")
    lines.append("• Комфорт → мотивация → точность.")
    lines.append("")
    lines.append("Практика на сегодня (5–7 минут)")
    for i, x in enumerate(practice, start=1):
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

    meta: Dict[str, Any] = {"ok": True, "age": age_tag, "rewrite_used": used, "rewrite_note": note}
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


def tg_request_safe(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return tg_request(method, data=data, files=files)


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
        except Exception:
            pass

    with card_path.open("rb") as f:
        tg_request_safe("sendPhoto", data={"chat_id": chat_id, "caption": ""}, files={"photo": f})
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

    sources, quality_cfg = load_sources()

    used_canon = set(load_state("used_canonical.json", []))
    used_titles = set(load_state("used_titles.json", []))

    pub_cfg = rub_cfg.get("publishing", {}) or {}
    max_posts = int(pub_cfg.get("max_posts_per_run", 3))

    stats = load_weekly_stats()

    audiences_cfg = rub_cfg.get("audiences", {}) or {}
    aud_cfg = audiences_cfg.get("parents", {}) or {}
    rubrics = aud_cfg.get("rubrics", []) or []

    posted = 0

    for rubric in rubrics:
        if posted >= max_posts:
            break

        # HOTFIX: dashboard tech only
        if (rubric.get("format") or "").strip().lower() == "quality_dashboard":
            dash_title = pub_cfg.get("dashboard_title", "Quality dashboard недели")
            dashboard_html = format_dashboard(stats, week_key, dash_title)

            if not TELEGRAM_DRAFTS_CHAT_ID:
                raise RuntimeError("TELEGRAM_DRAFTS_CHAT_ID is missing. Refusing to post quality dashboard to public channel.")

            if DRY_RUN:
                print("[DRY_RUN] dashboard would be posted to DRAFTS chat only.")
            else:
                send_message(TELEGRAM_DRAFTS_CHAT_ID, dashboard_html)
            time.sleep(0.5)
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

        # pick first not used (trimmed selection for sprint)
        picked = None
        for it0 in all_items[:30]:
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
        title_suffix = (aud_cfg.get("title_suffix", "") or "").strip()
        plain_post, meta = compose_post_plain_v31(title, rubric.get("format", ""), "parents", channel_cfg, picked, title_suffix)
        html_full_post = render_plain_to_telegram_html(plain_post)

        theses, age_tag = build_card_theses_from_plain_v3(plain_post)
        card = render_image_card(title, theses, branding, age_tag=age_tag)

        if DRY_RUN:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_dir = STATE_DIR / "dry_run" / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(card, out_dir / f"{posted+1:02d}_card.png")
            (out_dir / f"{posted+1:02d}_plain.txt").write_text(plain_post, encoding="utf-8")
            (out_dir / f"{posted+1:02d}_html.txt").write_text(html_full_post, encoding="utf-8")
        else:
            if not TELEGRAM_CHAT_ID:
                raise RuntimeError("TELEGRAM_CHAT_ID is missing (publish target)." )
            send_post_with_card(TELEGRAM_CHAT_ID, card, plain_post, html_full_post)
            bump_weekly(stats, week_key, "passed", 1)

        canon = picked.get("canonical") or picked.get("link", "")
        if canon and not DRY_RUN:
            used_canon.add(canon)
        tkey = norm_title_key(picked.get("article_title") or picked.get("title") or "")
        if tkey and not DRY_RUN:
            used_titles.add(tkey)

        posted += 1
        time.sleep(1.0)

    if not DRY_RUN:
        save_state("used_canonical.json", sorted(list(used_canon))[-6000:])
        save_state("used_titles.json", sorted(list(used_titles))[-6000:])
        save_weekly_stats(stats)

    print(f"Publisher done. Posted: {posted}. Week: {week_key}.{' [DRY_RUN]' if DRY_RUN else ''}")


if __name__ == "__main__":
    run()

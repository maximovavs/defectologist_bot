from __future__ import annotations
"""
Publisher (cron/GitHub Actions) v3.0.0

Что изменено:
1) LLM теперь получает weekday-aware prompt, но контент строится как abstract summary, а не сухой список.
2) Добавлена SQLite-дедупликация публикаций:
   - canonical URL
   - хеш текста
   - similarity check по тексту
3) Если новый текст совпадает с публикацией со вчерашнего дня на >= 90%,
   пост отклоняется и шлётся отдельный alert в техчат.
"""

import asyncio
import os
import re
import json
import random
import hashlib
import shutil
import html as _html
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin

import requests
import yaml
import feedparser
from bs4 import BeautifulSoup
from dateutil import tz
import urllib3

from src.services.image_builder import render_image_card
from src.services.llm_generator import generate_post_plain_from_evidence_async
from src.services.publication_store import PublicationStore


# =========================
# Paths / env
# =========================

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/3.0.0 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip()

DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()

PROVIDER = os.getenv("REWRITE_PROVIDER", "auto").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()
POST_MAX_CHARS = int(os.getenv("POST_MAX_CHARS", "1150"))
TG_CAPTION_MAX_BYTES = int(os.getenv("TG_CAPTION_MAX_BYTES", "950"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

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
    "Почему это важно",
    "Что делать",
    "Что это даст",
    "Мини-практика",
    "Источник",
    "Введение",
    "Методы",
    "Главные выводы",
    "Практическое применение",
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


def _start_of_yesterday(now: datetime) -> datetime:
    yesterday = (now - timedelta(days=1)).date()
    return datetime.combine(yesterday, dtime.min, tzinfo=now.tzinfo)


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
    items = _collect_links(url, soup, "div#dle-content a, div#dle-content h2 a, div#dle-content h3 a", r"/publ/[^\"']+")
    items = [it for it in items if not re.search(r"/page/\d+/?$", it["link"])]
    return items[:80]


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


def extract_evidence_text(url: str, max_chars: int = 3200) -> str:
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
        if len(txt) < 25:
            continue
        low = txt.lower()
        if any(bad in low for bad in ["cookie", "privacy", "политик", "подпис", "реклама"]):
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
# Plain -> Telegram HTML rendering
# =========================

def _is_structural_heading(line: str) -> bool:
    st = (line or "").strip()
    if not st:
        return False
    if st.startswith("👶 Возраст:") or st.startswith("👩‍⚕️ Аудитория:"):
        return True
    if st in SECTION_HEADERS:
        return True
    return False


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

        if st.startswith("ℹ️ "):
            out.append(f"<i>{_escape(st)}</i>")
            continue

        out.append(_escape(s))

    return "\n".join(out).strip()


# =========================
# Telegram send
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
    plain_bytes = len((plain_post or "").encode("utf-8"))
    if plain_bytes <= TG_CAPTION_MAX_BYTES:
        try:
            data: Dict[str, Any] = {"chat_id": chat_id, "caption": html_full_post}
            if TELEGRAM_PARSE_MODE:
                data["parse_mode"] = TELEGRAM_PARSE_MODE
            with card_path.open("rb") as f:
                tg_request("sendPhoto", data=data, files={"photo": f})
            return
        except Exception:
            pass

    with card_path.open("rb") as f:
        tg_request("sendPhoto", data={"chat_id": chat_id, "caption": ""}, files={"photo": f})
    send_message(chat_id, html_full_post)


def send_similarity_alert(
    chat_id: str,
    candidate_url: str,
    matched_url: str,
    score: float,
    audience: str,
    rubric_id: str,
) -> None:
    cand = _html.escape(candidate_url, quote=True)
    hit = _html.escape(matched_url, quote=True)
    html_text = (
        "⚠️ <b>Dedup alert</b>\n"
        f"Новый текст отклонён: similarity >= {SIMILARITY_THRESHOLD:.2f}\n"
        f"AUDIENCE={_escape(audience)} | RUBRIC={_escape(rubric_id)}\n\n"
        f"Новый кандидат: <a href=\"{cand}\">{_escape(candidate_url)}</a>\n"
        f"Похож на: <a href=\"{hit}\">{_escape(matched_url)}</a>\n"
        f"Similarity: <b>{score:.3f}</b>"
    )
    send_message(chat_id, html_text)


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


# =========================
# Main run
# =========================

async def amain() -> None:
    rub_cfg = load_yaml(CFG_DIR / "rubrics.yml")
    channel_cfg = rub_cfg.get("channel", {}) or {}
    branding = rub_cfg.get("branding", {}) or {}
    pub_cfg = rub_cfg.get("publishing", {}) or {}

    tzname = channel_cfg.get("timezone", "Asia/Nicosia")
    now = get_local_now(tzname)
    week_key = iso_week_key(now)
    day = weekday_key(now)

    max_posts = int(pub_cfg.get("max_posts_per_run", 1))
    disclaimer = channel_cfg.get("disclaimer", "") or ""
    hashtags = channel_cfg.get("hashtags", []) or []

    sources = load_sources()
    store = PublicationStore(STATE_DIR / "publication_history.sqlite3")
    yesterday_since = _start_of_yesterday(now).isoformat()

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
    seen_hashes_this_run: set[str] = set()

    def note(reason: str, url: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if len(samples) < 8:
            samples.append(f"• {reason}: {url}")

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

            for cand in all_items[:90]:
                url = (cand.get("link") or "").strip()
                if not url.startswith(("http://", "https://")):
                    continue
                if _SKIP_EXT_RE.search(url):
                    note("skip_non_html_asset", url)
                    continue

                canon = get_canonical(url)
                if _SKIP_EXT_RE.search(canon):
                    note("skip_non_html_asset", canon)
                    continue

                if canon in seen_urls_this_run:
                    note("dup_url_same_run", canon)
                    continue

                if store.has_url(canon):
                    note("dup_url_db", canon)
                    continue

                try:
                    evidence = extract_evidence_text(canon, max_chars=3200)
                except Exception as e:
                    note("evidence_fetch_failed", f"{canon} ({e})")
                    continue

                if len((evidence or "").strip()) < 260:
                    note("no_evidence_short", canon)
                    continue

                sd = safe_domain(canon) or safe_domain(url) or "источник"

                plain, ok, llm_note = await generate_post_plain_from_evidence_async(
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
                )

                if not ok or not plain:
                    note(llm_note, canon)
                    continue

                body_hash = sha1(norm_space(plain))
                if body_hash in seen_hashes_this_run:
                    note("dup_body_same_run", canon)
                    continue

                if store.has_body_hash(body_hash):
                    note("dup_body_db", canon)
                    continue

                yesterday_hit = store.find_similar(
                    plain,
                    threshold=SIMILARITY_THRESHOLD,
                    since_iso=yesterday_since,
                    limit=60,
                )
                if yesterday_hit:
                    note("dup_yesterday_90", canon)
                    if not DRY_RUN and TELEGRAM_DRAFTS_CHAT_ID:
                        send_similarity_alert(
                            TELEGRAM_DRAFTS_CHAT_ID,
                            canon,
                            yesterday_hit.canonical_url,
                            yesterday_hit.similarity,
                            aud,
                            rubric_id,
                        )
                    continue

                historic_hit = store.find_similar(
                    plain,
                    threshold=SIMILARITY_THRESHOLD,
                    since_iso=None,
                    limit=400,
                )
                if historic_hit:
                    note("dup_similar_db", canon)
                    continue

                html_full = render_plain_to_telegram_html(plain)
                theses = ["💡 " + rubric_title, "📝 краткий разбор", "🧠 без воды и дублей"]
                card = render_image_card(rubric_title, theses, branding, age_tag="")

                if DRY_RUN:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    out = STATE_DIR / "dry_run" / ts
                    out.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(card, out / f"{posted+1:02d}_{aud}_{rubric_id}.png")
                    (out / f"{posted+1:02d}_{aud}_{rubric_id}.txt").write_text(plain, encoding="utf-8")
                else:
                    if not TELEGRAM_CHAT_ID:
                        raise RuntimeError("TELEGRAM_CHAT_ID missing")
                    send_post_with_card(TELEGRAM_CHAT_ID, card, plain, html_full)
                    store.record_publication(
                        canonical_url=canon,
                        body_hash=body_hash,
                        body_text=plain,
                        posted_at=now.isoformat(),
                        audience=aud,
                        rubric_id=rubric_id,
                        rubric_title=rubric_title,
                        source_domain=sd,
                    )

                seen_urls_this_run.add(canon)
                seen_hashes_this_run.add(body_hash)

                posted += 1
                await asyncio.sleep(1.0)
                break

            if posted >= max_posts:
                break

        if posted >= max_posts:
            break

    if posted == 0 and not DRY_RUN:
        if TELEGRAM_DRAFTS_CHAT_ID:
            top = sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
            lines = [
                "⚠️ Publisher: не удалось опубликовать пост (Posted: 0)",
                f"Дата: {now.date()} | День: {day} | Неделя: {week_key}",
                f"AUDIENCE={AUDIENCE} | PROVIDER={PROVIDER}",
                "",
                "Причины пропуска (топ):",
            ]
            for k, v in top:
                lines.append(f"• {k}: {v}")
            if samples:
                lines.append("")
                lines.append("Примеры:")
                lines.extend(samples)
            send_message(TELEGRAM_DRAFTS_CHAT_ID, "\n".join(lines))
        else:
            print("[WARN] Posted:0 but TELEGRAM_DRAFTS_CHAT_ID not set; no alert sent.")

    print(f"Publisher done. Posted: {posted}. Week: {week_key}.{' [DRY_RUN]' if DRY_RUN else ''}")


def run() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    run()

\
from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
import yaml
import feedparser
from bs4 import BeautifulSoup
from dateutil import tz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/1.1 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

REWRITE_PROVIDER = os.getenv("REWRITE_PROVIDER", "none").strip().lower()  # none|groq|gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

AUDIENCE = os.getenv("AUDIENCE", "parents").strip().lower()  # parents|pros|both


@dataclass
class Source:
    id: str
    name: str
    type: str
    url: Optional[str] = None
    urls: Optional[List[str]] = None
    selectors: Optional[Dict[str, str]] = None
    include_if: Optional[Dict[str, Any]] = None
    notes: str = ""


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def norm_title_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(логопед|логопедия|логопедический|упражнение|упражнения)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_local_now(tzname: str) -> datetime:
    zone = tz.gettz(tzname)
    return datetime.now(tz=zone)


def is_due(rubric: Dict[str, Any], now: datetime) -> bool:
    cadence = (rubric.get("cadence") or "DAILY").upper()
    if cadence == "DAILY":
        return True
    if cadence == "WEEKLY":
        byweekday = set((rubric.get("byweekday") or []))
        map_wd = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
        return map_wd[now.weekday()] in byweekday
    return False


def load_sources() -> Tuple[Dict[str, Source], Dict[str, Any]]:
    cfg = load_yaml(CFG_DIR / "sources.yml")
    quality = cfg.get("quality", {})
    out: Dict[str, Source] = {}
    for s in cfg.get("sources", []):
        out[s["id"]] = Source(**s)
    return out, quality


def state_path(name: str) -> Path:
    return STATE_DIR / name


def load_state(name: str, default: Any) -> Any:
    p = state_path(name)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_state(name: str, data: Any) -> None:
    p = state_path(name)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def score_item(title: str, link: str, quality_cfg: Dict[str, Any]) -> Tuple[int, str]:
    t = (title or "").strip()
    u = (link or "").strip()
    if len(t) < 12 or len(t) > 220:
        return (-100, "bad_title_len")

    allow_domains = (quality_cfg.get("allow_domains") or [])
    if allow_domains and not domain_allowed(u, allow_domains):
        return (-100, "domain_not_allowed")

    tl = t.lower()
    ul = u.lower()

    deny = [k.lower() for k in (quality_cfg.get("deny_keywords") or [])]
    for k in deny:
        if k and (k in tl or k in ul):
            return (-100, f"deny_keyword:{k}")

    score = 10
    boosts = [k.lower() for k in (quality_cfg.get("boost_keywords") or [])]
    for k in boosts:
        if k and k in tl:
            score += 2

    return (score, "ok")


def get_canonical(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        canon = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
        if canon and canon.get("href"):
            href = canon["href"].strip()
            if href.startswith("/"):
                href = urljoin(url, href)
            return href
    except Exception:
        pass
    return url


def fetch_rss(url: str) -> List[Dict[str, str]]:
    d = feedparser.parse(url)
    items = []
    for e in d.entries[:40]:
        items.append({
            "title": norm_space(getattr(e, "title", "")),
            "link": getattr(e, "link", ""),
            "summary": norm_space(re.sub("<.*?>", "", getattr(e, "summary", ""))),
        })
    return items


def fetch_html_latest(url: str, selectors: Dict[str, str], include_if: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    item_sel = selectors.get("item", "a")
    nodes = soup.select(item_sel)

    items: List[Dict[str, str]] = []
    link_contains = set((include_if or {}).get("link_contains_any", []) or [])

    for a in nodes[:600]:
        href = a.get("href") or ""
        title = norm_space(a.get_text(" ", strip=True))
        if not href or not title:
            continue
        if href.startswith("/"):
            href = urljoin(url, href)
        if link_contains and not any(x in href for x in link_contains):
            continue
        items.append({"title": title, "link": href, "summary": ""})

    uniq = {}
    for it in items:
        uniq[it["link"]] = it
    return list(uniq.values())[:60]


def fetch_static(urls: List[str]) -> List[Dict[str, str]]:
    return [{"title": "", "link": u, "summary": ""} for u in urls]


def fetch_source(src: Source) -> List[Dict[str, str]]:
    if src.type == "rss":
        return fetch_rss(src.url or "")
    if src.type == "html_latest":
        return fetch_html_latest(src.url or "", src.selectors or {}, src.include_if)
    if src.type == "static":
        return fetch_static(src.urls or [])
    raise ValueError(f"Unsupported source type: {src.type}")


def pick_item(items: List[Dict[str, str]],
              used_canon: set[str],
              used_titles: set[str],
              quality_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    random.shuffle(items)
    best: Optional[Tuple[int, Dict[str, str]]] = None

    for it in items:
        title = norm_space(it.get("title", ""))
        link = it.get("link", "")
        if not link:
            continue

        canon = get_canonical(link)
        it["canonical"] = canon

        tkey = norm_title_key(title)
        if tkey and tkey in used_titles:
            continue
        if canon in used_canon:
            continue

        score, _ = score_item(title or "(no title)", canon, quality_cfg)
        if score < 0:
            continue

        if best is None or score > best[0]:
            best = (score, it)

    return best[1] if best else None


def rewrite_if_enabled(text: str) -> str:
    if REWRITE_PROVIDER == "none":
        return text

    parts = text.split("Источник:", 1)
    body = parts[0].strip()
    tail = ("Источник:" + parts[1]) if len(parts) == 2 else ""

    prompt = (
        "Переформулируй текст ниже по-русски: разговорный, нейтрально-научный, без диагнозов и обещаний лечения. "
        "Сохрани структуру и списки. Не добавляй факты.\n\n"
        f"ТЕКСТ:\n{body}\n"
    )

    try:
        if REWRITE_PROVIDER == "groq" and GROQ_API_KEY:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                },
                timeout=40,
            )
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            return out.strip() + ("\n\n" + tail if tail else "")

        if REWRITE_PROVIDER == "gemini" and GEMINI_API_KEY:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                params={"key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=40,
            )
            r.raise_for_status()
            out = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return out.strip() + ("\n\n" + tail if tail else "")
    except Exception as e:
        print(f"[WARN] rewrite failed ({REWRITE_PROVIDER}): {e}")

    return text


def make_text(title: str, rubric_format: str, channel_cfg: Dict[str, Any], picked: Dict[str, str], title_suffix: str) -> str:
    link = picked.get("canonical") or picked.get("link", "")
    summary = picked.get("summary", "")
    disclaimer = channel_cfg.get("disclaimer", "")
    tags = " ".join(channel_cfg.get("hashtags", []))

    if rubric_format == "myth_fact":
        body = (
            "Миф: «Если ребёнок слышит два языка, он обязательно начнёт говорить позже или “перепутается”.»\n\n"
            "Факт: многоязычие само по себе не “ломает” речь. Смешение языков может быть нормальной частью развития.\n"
            "Если есть сомнения, оценивают коммуникацию ребёнка и развитие по обоим языкам.\n"
        )
    elif rubric_format == "age_norms":
        body = (
            "Короткая ориентация по возрасту (варианты нормы бывают широкими):\n"
            "• 2 года: растёт словарь, появляются простые фразы.\n"
            "• 3 года: фразы длиннее, больше вопросов «что/где/почему».\n"
            "Если речи мало, нет понимания обращённой речи или нет прогресса — лучше обсудить со специалистом.\n"
        )
    elif rubric_format == "exercise_steps":
        body = (
            "Упражнение на 3–5 минут:\n"
            "1) Перед зеркалом: «Лопаточка» — 5 раз по 5 секунд.\n"
            "2) «Часики» — 10 повторов.\n"
            "3) В конце — похвала и короткая игра (пузыри/дуем на ватный шарик).\n"
        )
    elif rubric_format == "bilingual_parents":
        body = (
            "Если ребёнок живёт в другой языковой среде:\n"
            "• Сделайте “островки русского” (дома/с мамой/на сказках).\n"
            "• Нормально, если ребёнок иногда вставляет иностранные слова.\n"
            "• Важно: регулярность контакта с русской речью и позитивная мотивация.\n"
        )
    elif rubric_format == "pro_friendly":
        body = (
            "Для практики:\n"
            "• Сохраните материал/подборку в методкопилку.\n"
            "• Подумайте, как адаптировать под онлайн-сессию (демонстрация, ДЗ, чек-лист).\n"
        )
    elif rubric_format == "case_digest":
        body = (
            "Кейс-дайджест (обобщённо, без персональных данных):\n"
            "• Запрос: трудности звукопроизношения/лексико-грамматики в билингвальной среде.\n"
            "• Фокус: коммуникация, фонематические процессы, перенос между языками.\n"
            "• Идея: план с измеримыми шагами на 2–4 недели.\n"
        )
    else:
        body = (
            "Небольшая практика на сегодня:\n"
            "• 5 минут артикуляционной гимнастики.\n"
            "• 5 минут «описательной речи» (цвет, форма, назначение, действия).\n"
        )

    text = f"**{title} {title_suffix}**\n\n{body}\n"
    if summary:
        text += f"Контекст: {summary}\n\n"
    if link:
        text += f"Источник: {link}\n"
    if disclaimer:
        text += f"\n_{disclaimer}_\n"
    if tags:
        text += f"\n{tags}\n"

    return rewrite_if_enabled(text)


def render_image_card(title: str, subtitle: str = "") -> Path:
    img = Image.new("RGB", (1280, 720), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    def wrap(text: str, width: int = 42) -> List[str]:
        words = text.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) >= width:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        return lines

    y = 120
    for line in wrap(title, 40)[:4]:
        draw.text((80, y), line, fill=(20, 20, 20), font=font_title)
        y += 30

    if subtitle:
        y += 20
        for line in wrap(subtitle, 60)[:3]:
            draw.text((80, y), line, fill=(60, 60, 60), font=font_body)
            y += 24

    out = STATE_DIR / f"card_{sha1(title)[:10]}.png"
    img.save(out)
    return out


def tg_request(method: str, data: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, data=data, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def send_photo(photo_path: Path, caption: str) -> None:
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID is missing.")
    with photo_path.open("rb") as f:
        tg_request(
            "sendPhoto",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": f},
        )


def run() -> None:
    rub_cfg = load_yaml(CFG_DIR / "rubrics.yml")
    channel_cfg = rub_cfg.get("channel", {})
    tzname = channel_cfg.get("timezone", "Asia/Nicosia")
    now = get_local_now(tzname)

    sources, quality_cfg = load_sources()

    used_canon = set(load_state("used_canonical.json", []))
    used_titles = set(load_state("used_titles.json", []))

    pub_cfg = rub_cfg.get("publishing", {})
    max_posts = int(pub_cfg.get("max_posts_per_run", 2))
    max_per_aud = int(pub_cfg.get("max_posts_per_audience_per_run", 1))

    audiences_cfg = rub_cfg.get("audiences", {})
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

        aud_cfg = audiences_cfg.get(aud, {})
        title_suffix = aud_cfg.get("title_suffix", "")
        rubrics = aud_cfg.get("rubrics", []) or []
        aud_posted = 0

        for rubric in rubrics:
            if posted >= max_posts or aud_posted >= max_per_aud:
                break
            if not is_due(rubric, now):
                continue

            all_items: List[Dict[str, str]] = []
            for sid in rubric.get("sources", []):
                src = sources.get(sid)
                if not src:
                    continue
                try:
                    all_items.extend(fetch_source(src))
                except Exception as e:
                    print(f"[WARN] source {sid} failed: {e}")

            picked = pick_item(all_items, used_canon, used_titles, quality_cfg)
            if not picked:
                continue

            title = rubric.get("title", "Рубрика")
            text = make_text(title, rubric.get("format", ""), channel_cfg, picked, title_suffix)

            card = render_image_card(title, "Коротко и по делу")
            send_photo(card, text[:950])

            canon = picked.get("canonical") or picked.get("link", "")
            if canon:
                used_canon.add(canon)
            tkey = norm_title_key(picked.get("title", ""))
            if tkey:
                used_titles.add(tkey)

            posted += 1
            aud_posted += 1
            time.sleep(1.2)

    save_state("used_canonical.json", sorted(list(used_canon))[-4000:])
    save_state("used_titles.json", sorted(list(used_titles))[-4000:])
    print(f"Done. Posted: {posted}. Audience: {AUDIENCE}. Rewrite: {REWRITE_PROVIDER}.")


if __name__ == "__main__":
    run()

\
from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

USER_AGENT = "logoped-channel-bot/1.0 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Optional: LLM rewrite (not required for v1).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


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


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_local_now(tzname: str) -> datetime:
    zone = tz.gettz(tzname)
    return datetime.now(tz=zone)


def is_due(rubric: Dict[str, Any], now: datetime) -> bool:
    cadence = rubric.get("cadence", "DAILY").upper()
    if cadence == "DAILY":
        return True
    if cadence == "WEEKLY":
        byweekday = set((rubric.get("byweekday") or []))
        # RFC5545 weekday abbreviations: MO, TU, WE, TH, FR, SA, SU
        map_wd = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
        return map_wd[now.weekday()] in byweekday
    return False


def load_sources() -> Dict[str, Source]:
    cfg = load_yaml(CFG_DIR / "sources.yml")
    out: Dict[str, Source] = {}
    for s in cfg.get("sources", []):
        out[s["id"]] = Source(**s)
    return out


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


def fetch_rss(url: str) -> List[Dict[str, str]]:
    d = feedparser.parse(url)
    items = []
    for e in d.entries[:30]:
        items.append({
            "title": norm_space(getattr(e, "title", "")),
            "link": getattr(e, "link", ""),
            "summary": norm_space(re.sub("<.*?>", "", getattr(e, "summary", ""))),
        })
    return items


def fetch_html_list(url: str, selectors: Dict[str, str], include_if: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    item_sel = selectors.get("item", "a")
    nodes = soup.select(item_sel)
    items: List[Dict[str, str]] = []
    link_contains = set((include_if or {}).get("link_contains_any", []) or [])
    for a in nodes[:300]:
        href = a.get("href") or ""
        title = norm_space(a.get_text(" ", strip=True))
        if not href or not title:
            continue
        if href.startswith("/"):
            href = requests.compat.urljoin(url, href)
        if link_contains and not any(x in href for x in link_contains):
            continue
        # Basic de-dupe and noise filtering
        if len(title) < 12:
            continue
        items.append({"title": title, "link": href, "summary": ""})
    # De-dupe by link
    uniq = {}
    for it in items:
        uniq[it["link"]] = it
    return list(uniq.values())[:30]


def fetch_static(urls: List[str]) -> List[Dict[str, str]]:
    return [{"title": "", "link": u, "summary": ""} for u in urls]


def fetch_source(src: Source) -> List[Dict[str, str]]:
    if src.type == "rss":
        return fetch_rss(src.url or "")
    if src.type == "html_list":
        return fetch_html_list(src.url or "", src.selectors or {}, src.include_if)
    if src.type == "static":
        return fetch_static(src.urls or [])
    raise ValueError(f"Unsupported source type: {src.type}")


def pick_item(items: List[Dict[str, str]], used_links: set[str]) -> Optional[Dict[str, str]]:
    # Prefer unseen links
    random.shuffle(items)
    for it in items:
        if it["link"] and it["link"] not in used_links:
            return it
    return items[0] if items else None


def make_text(rubric: Dict[str, Any], channel_cfg: Dict[str, Any], picked: Dict[str, str]) -> str:
    title = rubric["title"]
    link = picked.get("link", "")
    summary = picked.get("summary", "")
    disclaimer = channel_cfg.get("disclaimer", "")
    tags = " ".join(channel_cfg.get("hashtags", []))

    if rubric.get("format") == "myth_fact":
        body = (
            "Миф: «Если ребёнок слышит два языка, он обязательно начнёт говорить позже или “перепутается”.»\n\n"
            "Факт: многоязычие само по себе не “ломает” речь. Смешение языков (code-mixing) может быть нормой развития.\n"
            "Если есть реальная задержка — оценивают развитие по обоим языкам и смотрят общую коммуникацию.\n"
        )
    elif rubric.get("format") == "age_norms":
        body = (
            "Короткая ориентация по возрасту (варианты нормы бывают широкими):\n"
            "• 2 года: растёт словарь, появляются простые фразы.\n"
            "• 3 года: фразы длиннее, больше вопросов «что/где/почему».\n"
            "Если речи мало, нет понимания обращённой речи или нет прогресса — лучше обсудить со специалистом.\n"
        )
    elif rubric.get("format") == "exercise_steps":
        body = (
            "Упражнение на 3–5 минут:\n"
            "1) Перед зеркалом: «Лопаточка» (широкий язык на нижней губе) — 5 раз по 5 секунд.\n"
            "2) «Часики» (язык влево-вправо) — 10 повторов.\n"
            "3) В конце — похвала и короткая игра (пузыри/дуем на ватный шарик).\n"
        )
    elif rubric.get("format") == "bilingual_parents":
        body = (
            "Если ребёнок живёт в другой языковой среде:\n"
            "• Договоритесь о “островках русского” (дома/с мамой/на сказках).\n"
            "• Нормально, если ребёнок иногда вставляет иностранные слова.\n"
            "• Критично — регулярность контакта с русской речью и позитивная мотивация.\n"
        )
    else:
        body = (
            "Небольшая практика на сегодня:\n"
            "• 5 минут артикуляционной гимнастики перед зеркалом.\n"
            "• 5 минут «описательной речи»: попросите ребёнка описать предмет (цвет, форма, что умеет).\n"
        )

    text = (
        f"**{title}**\n\n"
        f"{body}\n"
    )
    if summary:
        text += f"Подборка/контекст: {summary}\n\n"
    if link:
        text += f"Источник: {link}\n"
    if disclaimer:
        text += f"\n_{disclaimer}_\n"
    if tags:
        text += f"\n{tags}\n"
    return text


def render_image_card(title: str, subtitle: str = "") -> Path:
    img = Image.new("RGB", (1280, 720), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Use default PIL font to avoid missing system fonts.
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    # Simple wrapping
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


def send_message(text: str) -> None:
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID is missing.")
    tg_request("sendMessage", data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    })


def send_photo(photo_path: Path, caption: str) -> None:
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID is missing.")
    with photo_path.open("rb") as f:
        tg_request("sendPhoto",
                   data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
                   files={"photo": f})


def run() -> None:
    rub_cfg = load_yaml(CFG_DIR / "rubrics.yml")
    channel_cfg = rub_cfg.get("channel", {})
    tzname = channel_cfg.get("timezone", "Asia/Nicosia")
    now = get_local_now(tzname)

    sources = load_sources()
    used = set(load_state("used_links.json", []))

    posted = 0
    for rubric in rub_cfg.get("rubrics", []):
        if not is_due(rubric, now):
            continue

        # Pull from all sources listed for this rubric
        all_items: List[Dict[str, str]] = []
        for sid in rubric.get("sources", []):
            src = sources.get(sid)
            if not src:
                continue
            try:
                items = fetch_source(src)
                all_items.extend(items)
            except Exception as e:
                # Do not fail the whole run on one bad source
                print(f"[WARN] source {sid} failed: {e}")

        picked = pick_item(all_items, used)
        if not picked:
            continue

        text = make_text(rubric, channel_cfg, picked)

        # Optional image card
        if "image_card" in (rubric.get("media") or []):
            card = render_image_card(rubric["title"], "Коротко и по делу • для родителей и специалистов")
            send_photo(card, text[:1000])  # Telegram caption limit safety
        else:
            send_message(text)

        if picked.get("link"):
            used.add(picked["link"])
        posted += 1
        time.sleep(1.2)

        # Limit posts per run to avoid flooding
        if posted >= 2:
            break

    save_state("used_links.json", sorted(list(used))[-2000:])
    print(f"Done. Posted: {posted}")


if __name__ == "__main__":
    run()

from __future__ import annotations

import os, re, json, time, random, hashlib, math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests, yaml, feedparser
from bs4 import BeautifulSoup
from dateutil import tz
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "config"
STATE_DIR = ROOT / ".state"
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
STATE_DIR.mkdir(exist_ok=True)

USER_AGENT = "logoped-channel-bot/1.6.1 (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","").strip()
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID","").strip()

REWRITE_PROVIDER = os.getenv("REWRITE_PROVIDER","auto").strip().lower()  # none|auto|groq|gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY","").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","").strip()

AUDIENCE = os.getenv("AUDIENCE","parents").strip().lower()  # parents|pros|both

# v1.6.1: style/length knobs (Telegram photo caption is limited; we keep a conservative target)
PARENTS_MAX_BODY_CHARS = int(os.getenv("PARENTS_MAX_BODY_CHARS","860"))
PROS_MAX_BODY_CHARS = int(os.getenv("PROS_MAX_BODY_CHARS","980"))

# v1.6.1: quality gate knobs
MIN_MEANING_BULLETS = int(os.getenv("MIN_MEANING_BULLETS","2"))
MIN_PRACTICE_STEPS = int(os.getenv("MIN_PRACTICE_STEPS","3"))

@dataclass
class Source:
    id: str
    name: str
    type: str
    url: Optional[str] = None
    urls: Optional[List[str]] = None
    parser: Optional[str] = None
    notes: str = ""

def load_yaml(path: Path) -> Dict[str,Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def norm_space(s: str) -> str:
    return re.sub(r"\s+"," ",(s or "").strip())

def clamp_text(s: str, max_len: int) -> str:
    s = norm_space(s)
    if len(s) <= max_len:
        return s
    return (s[:max_len].rstrip(" .,:;—-") + "…").strip()

def markdown_to_plain(text: str) -> str:
    """
    Telegram Markdown часто падает на URL с '_' и прочими символами.
    Поэтому при ошибке парсинга мы отправляем plain-text без parse_mode.
    """
    t = text or ""
    # убираем типовые маркеры Markdown
    t = t.replace("**", "")
    t = t.replace("__", "")
    t = t.replace("`", "")
    # одиночные маркеры
    t = t.replace("*", "")
    t = t.replace("_", "")
    # иногда встречаются "###" и т.п.
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)
    return t.strip()

def norm_title_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]+"," ",s,flags=re.UNICODE)
    s = re.sub(r"\s+"," ",s).strip()
    s = re.sub(r"\b(логопед|логопедия|логопедический|упражнение|упражнения)\b","",s).strip()
    s = re.sub(r"\s+"," ",s).strip()
    return s[:180]

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def get_local_now(tzname: str) -> datetime:
    return datetime.now(tz=tz.gettz(tzname))

def iso_week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"

def is_due(rubric: Dict[str,Any], now: datetime) -> bool:
    cadence = (rubric.get("cadence") or "DAILY").upper()
    if cadence == "DAILY":
        return True
    if cadence == "WEEKLY":
        byweekday = set(rubric.get("byweekday") or [])
        map_wd = ["MO","TU","WE","TH","FR","SA","SU"]
        return map_wd[now.weekday()] in byweekday
    return False

def load_sources() -> Tuple[Dict[str,Source], Dict[str,Any]]:
    cfg = load_yaml(CFG_DIR/"sources.yml")
    quality = cfg.get("quality",{})
    out: Dict[str,Source] = {}
    for s in cfg.get("sources",[]):
        out[s["id"]] = Source(**s)
    return out, quality

def load_state(name: str, default: Any) -> Any:
    p = STATE_DIR/name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_state(name: str, data: Any) -> None:
    (STATE_DIR/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

def safe_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""

def domain_allowed(url: str, allow_domains: List[str]) -> bool:
    d = safe_domain(url)
    return bool(d) and any(d==ad or d.endswith("."+ad) for ad in allow_domains)

def score_item(title: str, link: str, quality_cfg: Dict[str,Any]) -> Tuple[int,str]:
    t = (title or "").strip()
    u = (link or "").strip()
    if len(t) < 12 or len(t) > 240:
        return (-100,"bad_title_len")
    allow_domains = quality_cfg.get("allow_domains") or []
    if allow_domains and not domain_allowed(u, allow_domains):
        return (-100,"domain_not_allowed")
    tl, ul = t.lower(), u.lower()
    for k in [x.lower() for x in (quality_cfg.get("deny_keywords") or [])]:
        if k and (k in tl or k in ul):
            return (-100,f"deny_keyword:{k}")
    score = 10
    for k in [x.lower() for x in (quality_cfg.get("boost_keywords") or [])]:
        if k and k in tl:
            score += 2
    return (score,"ok")

def http_get(url: str, timeout: int = 25) -> requests.Response:
    """
    Безопасный GET: для logorina.ru при CERT_VERIFY делаем единичный ретрай verify=False.
    """
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.exceptions.SSLError:
        d = safe_domain(url)
        if d.endswith("logorina.ru"):
            print("[WARN] SSL verify failed for logorina.ru; retry with verify=False")
            return requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
        raise

def get_canonical_and_soup(url: str) -> Tuple[str, Optional[BeautifulSoup]]:
    try:
        r = http_get(url, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text,"lxml")
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
        return norm_space(h1.get_text(" ",strip=True))
    if soup.title and soup.title.string:
        return norm_space(soup.title.string)
    return ""

def extract_article_summary(soup: BeautifulSoup) -> str:
    md = soup.find("meta", attrs={"name":"description"})
    if md and md.get("content"):
        return norm_space(md["content"])
    ogd = soup.find("meta", property="og:description")
    if ogd and ogd.get("content"):
        return norm_space(ogd["content"])
    paras=[]
    for p in soup.select("p"):
        txt = norm_space(p.get_text(" ",strip=True))
        if len(txt) < 60:
            continue
        if any(bad in txt.lower() for bad in ["cookie","политик","подпис","реклама"]):
            continue
        paras.append(txt)
        if len(paras)>=2:
            break
    return norm_space(" ".join(paras))[:420]

def is_scientific_or_methodical(domain: str, title: str, summary: str, quality_cfg: Dict[str,Any]) -> Tuple[bool,str]:
    scientific_domains = [d.lower() for d in (quality_cfg.get("scientific_domains") or [])]
    if any(domain==d or domain.endswith("."+d) for d in scientific_domains):
        return True,"scientific_domain"
    blob = f"{title}\n{summary}".lower()
    kws = [k.lower() for k in (quality_cfg.get("methodical_keywords") or [])]
    hits = sum(1 for k in kws if k and k in blob)
    if hits >= 2:
        return True,f"methodical_kw_hits:{hits}"
    return False,f"not_methodical_hits:{hits}"

def source_type_label_from_factcheck(factcheck_reason: str) -> str:
    r = (factcheck_reason or "").lower()
    if "scientific_domain" in r:
        return "научный/академический источник"
    return "методический/профессиональный материал"

# ---------------------------
# Site-specific parsers (v1.4)
# ---------------------------

def _abs(url: str, href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(url, href)
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return urljoin(url, href)

def _collect_links(base_url: str, soup: BeautifulSoup, selector: str, href_re: Optional[str]=None) -> List[Dict[str,str]]:
    pat = re.compile(href_re) if href_re else None
    out=[]
    for a in soup.select(selector):
        href = _abs(base_url, a.get("href",""))
        if not href:
            continue
        if pat and not pat.search(href):
            continue
        title = norm_space(a.get_text(" ", strip=True))
        if not title or len(title) < 8:
            continue
        out.append({"title": title, "link": href, "summary": ""})
    seen=set(); uniq=[]
    for it in out:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)
    return uniq

def parse_logopediya_publ(url: str, html: str) -> List[Dict[str,str]]:
    soup = BeautifulSoup(html, "lxml")
    items = _collect_links(url, soup, "div#dle-content a, div#dle-content h2 a, div#dle-content h3 a", r"/publ/[^\"']+")
    items = [it for it in items if not re.search(r"/page/\d+/?$", it["link"])]
    return items[:80]

def parse_logorina_news(url: str, html: str) -> List[Dict[str,str]]:
    soup = BeautifulSoup(html, "lxml")
    items = _collect_links(url, soup, "article a, div.news a, a", r"/news/[\w\-]+/?$")
    return items[:80]

def parse_logomag_lib(url: str, html: str) -> List[Dict[str,str]]:
    soup = BeautifulSoup(html, "lxml")
    items = _collect_links(url, soup, "main a, div.content a, a", r"/lib/[^\"']+")
    return items[:80]

def parse_logoportal_articles(url: str, html: str) -> List[Dict[str,str]]:
    soup = BeautifulSoup(html, "lxml")
    items = _collect_links(url, soup, "main a, div#content a, article a, a", r"(statya-|/statya-)")
    return items[:80]

def parse_logopedy_articles(url: str, html: str) -> List[Dict[str,str]]:
    soup = BeautifulSoup(html, "lxml")
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

def fetch_rss(url: str) -> List[Dict[str,str]]:
    d = feedparser.parse(url)
    out=[]
    for e in d.entries[:50]:
        out.append({
            "title": norm_space(getattr(e,"title","")),
            "link": getattr(e,"link",""),
            "summary": norm_space(re.sub("<.*?>","",getattr(e,"summary",""))),
        })
    return out

def fetch_static(urls: List[str]) -> List[Dict[str,str]]:
    return [{"title":"","link":u,"summary":""} for u in urls]

def fetch_html_site(url: str, parser_name: str) -> List[Dict[str,str]]:
    r = http_get(url, timeout=30)
    r.raise_for_status()
    parser = SITE_PARSERS.get(parser_name)
    if not parser:
        raise ValueError(f"Unknown site parser: {parser_name}")
    items = parser(url, r.text)
    uniq={}
    for it in items:
        uniq[it["link"]] = it
    return list(uniq.values())

def fetch_source(src: Source) -> List[Dict[str,str]]:
    if src.type=="rss":
        return fetch_rss(src.url or "")
    if src.type=="html_site":
        return fetch_html_site(src.url or "", src.parser or "")
    if src.type=="static":
        return fetch_static(src.urls or [])
    raise ValueError(f"Unsupported source type: {src.type}")

def enrich_article(item: Dict[str,str]) -> Dict[str,str]:
    link = item.get("link","")
    canon, soup = get_canonical_and_soup(link)
    item["canonical"]=canon
    if soup:
        at = extract_article_title(soup)
        if at: item["article_title"]=at
        sm = extract_article_summary(soup)
        if sm: item["article_summary"]=sm
    return item

# ---------------------------
# LLM rewriting (v1.6.1 prompts v2)
# ---------------------------

def _is_quota_error(status: int, text: str) -> bool:
    t=(text or "").lower()
    return status in (402,429) or any(k in t for k in ["quota","rate limit","exceeded","insufficient_quota","resource_exhausted"])

def rewrite_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY: raise RuntimeError("GROQ_API_KEY missing")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type":"application/json"},
        json={
            "model":"llama-3.1-8b-instant",
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.35
        },
        timeout=45
    )
    if r.status_code!=200 and _is_quota_error(r.status_code,r.text):
        raise RuntimeError(f"groq_quota:{r.status_code}")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def rewrite_with_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY: raise RuntimeError("GEMINI_API_KEY missing")
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents":[{"parts":[{"text":prompt}]}]},
        timeout=45
    )
    if r.status_code!=200 and _is_quota_error(r.status_code,r.text):
        raise RuntimeError(f"gemini_quota:{r.status_code}")
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

def _aud_limits(audience: str) -> int:
    a = (audience or "parents").strip().lower()
    return PARENTS_MAX_BODY_CHARS if a == "parents" else PROS_MAX_BODY_CHARS

def _build_rewrite_prompt_v2(body: str, audience: str, max_chars: int) -> str:
    a = (audience or "parents").strip().lower()

    common_rules = (
        "Требования:\n"
        "1) Русский язык. Нейтрально-научный, бережный тон.\n"
        "2) НЕ ставь диагнозы, НЕ обещай лечения, НЕ назначай препараты.\n"
        "3) Не добавляй новых фактов. Только перефразируй.\n"
        "4) Сохрани структуру и порядок секций и списков.\n"
        "5) Не меняй названия секций и не удаляй их.\n"
        "6) Не добавляй новых разделов.\n"
        f"7) Длина тела текста (без блока «Источник» и без дисклеймера/хештегов): до {max_chars} символов.\n"
        "Секции должны быть ровно такими:\n"
        "«Суть», «Что это значит для вас», «Практика на сегодня (5–7 минут)», «Норма / когда нужен специалист».\n"
    )

    if a == "pros":
        style = (
            "Аудитория: специалисты (логопеды/дефектологи).\n"
            "Стиль: профессионально, точнее термины, но без канцелярита. "
            "Можно использовать умеренную терминологию (фонематический слух, артикуляционная моторика, лексико-грамматический строй), "
            "но формулировки должны оставаться ясными.\n"
        )
    else:
        style = (
            "Аудитория: родители.\n"
            "Стиль: разговорный, простые слова, поддерживающий тон. "
            "Убирай канцелярит и ‘умные’ обороты. "
            "Если встречается термин — кратко объясни простыми словами в той же фразе.\n"
        )

    return (
        style
        + common_rules
        + "\nТЕКСТ ДЛЯ ПЕРЕФОРМУЛИРОВКИ:\n"
        + body.strip()
    )

def _enforce_body_limit_v2(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    # аккуратно урезаем конец, стараясь не ломать блоки: обрезаем по последней границе строки
    cut = t[:max_chars]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()

def rewrite_if_enabled(text: str, audience: str) -> str:
    if REWRITE_PROVIDER=="none":
        return text

    # v1.6+: protect Source + disclaimer + hashtags from rewriting
    marker = "\n**Источник**\n"
    idx = text.find(marker)

    if idx != -1:
        body = text[:idx].strip()
        tail = text[idx:].strip()
    else:
        # legacy fallback (older template)
        parts = text.split("Источник:",1)
        body = parts[0].strip()
        tail = ("Источник:"+parts[1]).strip() if len(parts)==2 else ""

    max_chars = _aud_limits(audience)
    prompt = _build_rewrite_prompt_v2(body, audience, max_chars)

    try:
        if REWRITE_PROVIDER in ("groq","auto"):
            try:
                out = rewrite_with_groq(prompt)
                out = _enforce_body_limit_v2(out, max_chars)
                return out + ("\n\n"+tail if tail else "")
            except Exception as e:
                if REWRITE_PROVIDER=="groq":
                    raise
                if "groq_quota" in str(e):
                    print("[WARN] groq quota; fallback to gemini")
                else:
                    print(f"[WARN] groq rewrite failed: {e}")
        if REWRITE_PROVIDER in ("gemini","auto"):
            out = rewrite_with_gemini(prompt)
            out = _enforce_body_limit_v2(out, max_chars)
            return out + ("\n\n"+tail if tail else "")
    except Exception as e:
        print(f"[WARN] rewrite failed ({REWRITE_PROVIDER}): {e}")
        return text

    return text

# ---------------------------
# Post template v2 + quality gate (v1.6.1)
# ---------------------------

def make_question_week() -> str:
    questions = [
        "Ребёнок понимает обращённую речь, но говорит мало: какие шаги вы уже пробовали дома?",
        "В билингвальной семье: на каком языке ребёнку легче рассказывать истории и почему?",
        "Какие звуки/слоги даются труднее всего — и в каких словах это заметнее?",
        "Что вызывает больше сопротивления: артикуляционная гимнастика, повторение слогов или чтение/письмо?",
        "Как выглядит ваш «идеальный результат» через 4 недели занятий — в одном предложении?",
    ]
    return random.choice(questions)

def _bullets(lines: List[str]) -> str:
    clean = [norm_space(x) for x in lines if norm_space(x)]
    return "\n".join([f"• {x}" for x in clean])

def _numbered(lines: List[str]) -> str:
    clean = [norm_space(x) for x in lines if norm_space(x)]
    return "\n".join([f"{i+1}) {x}" for i,x in enumerate(clean)])

def _has_required_headings(text: str) -> bool:
    # v1.6.1: ensure structure is intact after rewrite (or no-rewrite)
    required = [
        "**Суть**",
        "**Что это значит для вас**",
        "**Практика на сегодня (5–7 минут)**",
        "**Норма / когда нужен специалист**",
        "**Источник**",
    ]
    return all(r in text for r in required)

def _quality_gate(
    rubric_format: str,
    audience: str,
    link: str,
    essence: str,
    meaning: List[str],
    practice: List[str],
    norm_lines: List[str],
) -> Tuple[bool, str]:
    rf = (rubric_format or "").strip().lower()
    aud = (audience or "parents").strip().lower()

    if not link or not link.startswith(("http://","https://")):
        return False, "quality_gate:no_source_link"

    # essence should not be empty/too short (except question_week which can be compact)
    ess_len = len(norm_space(essence))
    if rf != "question_week" and ess_len < 40:
        return False, f"quality_gate:weak_essence_len:{ess_len}"
    if rf == "question_week" and ess_len < 25:
        return False, f"quality_gate:weak_question_len:{ess_len}"

    # meaning bullets
    m = [x for x in meaning if norm_space(x)]
    if len(m) < MIN_MEANING_BULLETS:
        return False, f"quality_gate:meaning_bullets_lt_{MIN_MEANING_BULLETS}:{len(m)}"

    # practice steps
    p = [x for x in practice if norm_space(x)]
    if len(p) < MIN_PRACTICE_STEPS:
        return False, f"quality_gate:practice_steps_lt_{MIN_PRACTICE_STEPS}:{len(p)}"

    # norm lines: must include both normal and consult hint
    nl = "\n".join([norm_space(x) for x in norm_lines if norm_space(x)])
    if "✅" not in nl or "⚠️" not in nl:
        return False, "quality_gate:norm_block_missing_markers"

    # audience nuance: for pros, require at least one actionable pro-oriented line in practice for pro rubrics
    if aud == "pros" and rf in ("pro_friendly","case_digest"):
        blob = " ".join(p).lower()
        if not any(k in blob for k in ["цель", "критер", "чек", "контрол", "онлайн", "план"]):
            return False, "quality_gate:pros_practice_too_generic"

    return True, "ok"

def compose_post_v2(
    rubric_title: str,
    rubric_format: str,
    audience: str,
    channel_cfg: Dict[str,Any],
    picked: Dict[str,str],
    title_suffix: str
) -> Tuple[str, Dict[str,Any]]:
    """
    v1.6 — post_template_v2 with mandatory blocks for all rubrics (except quality_dashboard).
    v1.6.1 — prompts v2 + quality gate + structure validation.
    """
    link = picked.get("canonical") or picked.get("link","")
    picked_title = picked.get("picked_title") or picked.get("title") or ""
    summary = picked.get("picked_summary") or picked.get("summary") or ""
    disclaimer = channel_cfg.get("disclaimer","")
    tags = " ".join(channel_cfg.get("hashtags",[]))

    aud = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()

    # clamp article title/summary to keep posts compact and consistent
    picked_title_c = clamp_text(picked_title, 140) if picked_title else ""
    summary_c = clamp_text(summary, 240) if summary else ""

    # --- Суть
    if rf == "question_week":
        q = make_question_week()
        essence = (
            "Небольшой “вопрос недели” — чтобы мягко понять текущую ситуацию и выбрать следующий шаг.\n\n"
            f"**{q}**"
        )
        if not picked_title_c:
            picked_title_c = "Рубрика канала (вопрос для самонаблюдения)"
        if not summary_c:
            summary_c = "Формат: наблюдение, маленький шаг, без давления."
    else:
        essence_lines = []
        if picked_title_c:
            essence_lines.append(f"Материал: {picked_title_c}")
        if summary_c:
            essence_lines.append(f"Коротко: {summary_c}")
        essence = "\n".join(essence_lines).strip() or "Коротко и по делу о развитии речи."

    # --- Что это значит для вас (2–3 пункта)
    meaning: List[str]
    if rf == "bilingual_parents":
        meaning = [
            "Смешивание языков и “вставки” слов второго языка часто бывают частью нормы в билингвизме.",
            "Запреты и давление обычно снижают мотивацию говорить — лучше поддерживать русский регулярно и спокойно.",
            "Важнее смотреть на понимание и динамику, а не на идеальную “чистоту” языка в каждый момент.",
        ]
    elif rf == "exercise_steps":
        meaning = [
            "Короткая регулярная практика эффективнее редких “длинных” занятий.",
            "Зеркало и игра помогают удержать внимание и сделать упражнение привычкой.",
            "Если ребёнок устал — лучше остановиться раньше, чем закрепить сопротивление.",
        ]
    elif rf == "myth_fact":
        meaning = [
            "Полезно отделять популярные мифы от того, что реально наблюдается в развитии речи.",
            "Обычно важнее понимание, коммуникация и динамика, чем единичные признаки.",
            "Если тревожно — лучше смотреть на ситуацию комплексно, а не по одному симптому.",
        ]
    elif rf == "age_norms":
        meaning = [
            "Возрастные нормы — ориентир, а не “экзамен”: варианты нормы бывают широкими.",
            "Главное — динамика: растёт ли понимание и инициатива общения, появляются ли новые слова/фразы.",
            "Удобнее обсуждать сомнения по конкретным примерам, а не “по ощущениям”.",
        ]
    elif rf in ("pro_friendly","case_digest"):
        if aud == "parents":
            meaning = [
                "Ниже — идея, как превратить материал в понятный домашний шаг без перегруза.",
                "Если ребёнку сложно — начинайте с малого и фиксируйте маленький прогресс.",
                "Системность важнее идеальности выполнения.",
            ]
        else:
            meaning = [
                "Переносите материал в практику: цель → критерий → шаги → контроль.",
                "Для онлайн особенно важны демонстрация, простое ДЗ и короткий чек-лист.",
                "Учитывайте билингвальную среду и перенос навыков между языками.",
            ]
    else:
        meaning = [
            "Самый надёжный прогресс в речи — регулярные маленькие шаги, а не разовые “рывки”.",
            "Коммуникация важнее идеальной артикуляции: сначала смысл и желание говорить, потом точность.",
            "Лучше опираться на проверенные источники и наблюдать динамику 2–4 недели.",
        ]

    # --- Практика 5–7 минут
    practice: List[str]
    if rf == "exercise_steps":
        practice = [
            "Перед зеркалом: «Лопаточка» — 5 раз по 5 секунд.",
            "«Часики» — 10 плавных движений вправо-влево.",
            "1 минута: дуем на ватный шарик/мыльные пузыри (в игре).",
            "В конце — короткая похвала за попытки, без “переделывай”.",
        ]
    elif rf == "bilingual_parents":
        practice = [
            "Игра “Два варианта”: повторите фразу ребёнка по-русски (спокойно, без исправлений), затем спросите «как по-русски?».",
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
            "5 минут “описательной речи”: выберите предмет и проговорите (цвет/форма/назначение/действие).",
            "Игра “Кто что делает?”: 10 глаголов по картинкам (прыгает, рисует, моет…).",
            "Если ребёнок билингв — дайте ответить, затем мягко повторите модель по-русски.",
        ]
    elif rf == "myth_fact":
        practice = [
            "Выберите 1 ситуацию для спокойного “моделирования”: повторите фразу ребёнка правильно, без оценки.",
            "5 минут игры на словарь (категории: еда/одежда/игрушки).",
            "В конце — один открытый вопрос: «Что было самым интересным?»",
        ]
    elif rf in ("pro_friendly","case_digest") and aud != "parents":
        practice = [
            "Сформулируйте цель на 2 недели (1–2 измеримых критерия).",
            "Подготовьте демонстрацию 1 упражнения + чек-лист для родителей (до 6 пунктов).",
            "Продумайте перенос в билингвальной среде: где закреплять русский ежедневно 5–10 минут.",
        ]
    else:
        practice = [
            "5 минут артикуляционной гимнастики (в игре, перед зеркалом).",
            "5 минут “словарной игры”: категории/противоположности/описание предметов.",
            "1 минута дыхательной игры (пузыри/ватный шарик/дуем на перышко).",
        ]

    # --- Норма / когда нужен специалист
    if rf in ("pro_friendly","case_digest") and aud != "parents":
        norm_lines = [
            "✅ Норма: есть стабильный контакт, понимание инструкций, постепенная динамика по целям.",
            "⚠️ Обсудить со специалистом: выраженная регрессия навыков или отсутствие прогресса при регулярной практике 4–6 недель.",
        ]
    else:
        norm_lines = [
            "✅ Норма: ребёнок понимает обращённую речь, общается (жестами/словами), и есть постепенный прогресс по неделям.",
            "⚠️ Обсудить со специалистом: если ребёнок часто не понимает простые просьбы, резко “теряет” навыки или прогресса нет при регулярной практике 4–6 недель.",
        ]

    factcheck = picked.get("fact_check") or ""
    stype = picked.get("source_type") or source_type_label_from_factcheck(factcheck)

    ok, q_reason = _quality_gate(rf, aud, link, essence, meaning, practice, norm_lines)
    meta = {
        "ok": ok,
        "reason": q_reason,
        "rubric_format": rf,
        "audience": aud,
        "source_type": stype,
    }
    if not ok:
        return "", meta

    parts: List[str] = []
    parts.append(f"**{rubric_title} {title_suffix}**")
    parts.append("")
    parts.append("**Суть**")
    parts.append(essence)
    parts.append("")
    parts.append("**Что это значит для вас**")
    parts.append(_bullets(meaning))
    parts.append("")
    parts.append("**Практика на сегодня (5–7 минут)**")
    parts.append(_numbered(practice))
    parts.append("")
    parts.append("**Норма / когда нужен специалист**")
    parts.append("\n".join(norm_lines))
    parts.append("")
    parts.append("**Источник**")
    parts.append(f"🔗 {link}")
    parts.append(f"Тип: {stype}")

    if disclaimer:
        parts.append("")
        parts.append(f"_{disclaimer}_")
    if tags (:= tags):
        parts.append("")
        parts.append(tags)

    raw_text = "\n".join(parts).strip()

    final_text = rewrite_if_enabled(raw_text, aud)

    # final sanity check: rewrite must preserve structure
    if not _has_required_headings(final_text):
        meta["ok"] = False
        meta["reason"] = "quality_gate:rewrite_broke_structure"
        return "", meta

    return final_text, meta

# ---------------------------
# Card rendering (v1.4)
# ---------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    ttf = FONTS_DIR/"DejaVuSans.ttf"
    if ttf.exists():
        return ImageFont.truetype(str(ttf), size=size)
    return ImageFont.load_default()

def _hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = (h or "").strip().lstrip("#")
    if len(h)==3:
        h = "".join([c+c for c in h])
    if len(h)!=6:
        return (74,144,226)
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def render_image_card(rubric_title: str, subtitle: str, branding: Dict[str,Any]) -> Path:
    """
    Visual themes (switch in config/rubrics.yml -> branding.card_theme):
      - minimal: clean neutral, subtle waves
      - kids: softer palette, playful dots
      - scientific: stricter palette, grid accents
    """
    theme = (branding or {}).get("card_theme","minimal") or "minimal"
    theme = str(theme).strip().lower()

    W,H = 1280,720
    accent = _hex_to_rgb((branding or {}).get("card_accent","#4A90E2"))

    # Theme palettes
    if theme == "kids":
        bg_top = (252, 246, 255)
        bg_bottom = (240, 252, 255)
        panel_fill = (255,255,255)
        panel_outline = (236,230,244)
        title_color = (32, 36, 46)
        sub_color = (78, 86, 104)
        footer_color = (120, 126, 140)
        wave_alpha = 30
    elif theme == "scientific":
        bg_top = (245, 247, 250)
        bg_bottom = (232, 236, 244)
        panel_fill = (255,255,255)
        panel_outline = (220,226,235)
        title_color = (16, 20, 30)
        sub_color = (54, 62, 78)
        footer_color = (98, 104, 118)
        wave_alpha = 22
        if sum(accent) > 560:
            accent = (36, 79, 166)
    else:  # minimal
        bg_top = (245, 247, 250)
        bg_bottom = (235, 240, 246)
        panel_fill = (255,255,255)
        panel_outline = (235,238,242)
        title_color = (24, 32, 44)
        sub_color = (70, 78, 92)
        footer_color = (110, 118, 132)
        wave_alpha = 26

    img = Image.new("RGB",(W,H),bg_top)
    draw = ImageDraw.Draw(img)

    # gradient background
    for y in range(H):
        t = y/(H-1)
        r = int(bg_top[0] + (bg_bottom[0]-bg_top[0])*t)
        g = int(bg_top[1] + (bg_bottom[1]-bg_top[1])*t)
        b = int(bg_top[2] + (bg_bottom[2]-bg_top[2])*t)
        draw.line([(0,y),(W,y)], fill=(r,g,b))

    # accents layer
    layer = Image.new("RGBA",(W,H),(0,0,0,0))
    ld = ImageDraw.Draw(layer)

    if theme in ("minimal","scientific"):
        for i in range(3):
            y0 = 440 + i*55
            pts=[]
            for x in range(0,W+1,40):
                yy = y0 + int(12*math.sin((x/140.0) + i))
                pts.append((x,yy))
            ld.line(pts, fill=(*accent, wave_alpha), width=6 if theme=="minimal" else 5)

        if theme == "scientific":
            gx0, gy0, gx1, gy1 = 760, 60, 1240, 300
            step = 34
            grid_col = (accent[0], accent[1], accent[2], 16)
            for x in range(gx0, gx1, step):
                ld.line([(x,gy0),(x,gy1)], fill=grid_col, width=2)
            for y in range(gy0, gy1, step):
                ld.line([(gx0,y),(gx1,y)], fill=grid_col, width=2)

    elif theme == "kids":
        seed = int(hashlib.sha1((rubric_title or "").encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        dot_col = (accent[0], accent[1], accent[2], 22)
        for _ in range(120):
            x = rng.randint(60, W-60)
            y = rng.randint(60, H-60)
            rr = rng.randint(3, 9)
            ld.ellipse([x-rr,y-rr,x+rr,y+rr], fill=dot_col)
        for cx,cy,rr in [(220,160,110),(1120,520,140)]:
            ld.ellipse([cx-rr,cy-rr,cx+rr,cy+rr], fill=(accent[0],accent[1],accent[2],18))

    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(img)

    # panel + shadow
    panel = (70,90,W-70,H-110)
    shadow = Image.new("RGBA",(W,H),(0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([panel[0]+6,panel[1]+10,panel[2]+6,panel[3]+10], radius=28, fill=(0,0,0,60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(panel, radius=28, fill=panel_fill, outline=panel_outline, width=2)

    # accent bar
    ax = panel[0]+28
    ay = panel[1]+28
    draw.rounded_rectangle([ax, ay, ax+10, panel[3]-28], radius=6, fill=accent)

    # typography
    f_title = _load_font(56 if theme!="scientific" else 54)
    f_sub = _load_font(32 if theme!="scientific" else 30)
    f_small = _load_font(24)

    x_text = ax+28
    y_text = panel[1]+44
    max_w = panel[2]-x_text-28

    def wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        words = (text or "").split()
        if not words:
            return []
        lines=[]; cur=[]
        for w in words:
            test = " ".join(cur+[w])
            if draw.textlength(test, font=font) <= max_width:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                cur=[w]
        if cur:
            lines.append(" ".join(cur))
        return lines

    for ln in wrap(rubric_title, f_title, max_w)[:3]:
        draw.text((x_text, y_text), ln, fill=title_color, font=f_title)
        y_text += 68

    y_text += 12
    for ln in wrap(subtitle, f_sub, max_w)[:3]:
        draw.text((x_text, y_text), ln, fill=sub_color, font=f_sub)
        y_text += 44

    footer = (branding or {}).get("card_footer","")
    if footer:
        draw.text((panel[0]+28, panel[3]-48), footer, fill=footer_color, font=f_small)

    out = STATE_DIR/f"card_{sha1(theme+rubric_title+subtitle)[:10]}.png"
    img.save(out)
    return out

# ---------------------------
# Telegram helpers + stats + selection
# ---------------------------

def tg_request(method: str, data: Dict[str,Any], files: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, data=data, files=files, timeout=30)

    if not r.ok:
        desc = ""
        try:
            j = r.json()
            desc = j.get("description","") or str(j)
        except Exception:
            desc = (r.text or "")[:800]
        print(f"[TG] {method} failed: {r.status_code} :: {desc[:800]}")
        r.raise_for_status()

    return r.json()

def send_photo(chat_id: str, photo_path: Path, caption: str) -> None:
    if not chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID is missing/empty.")
    cap = (caption or "").strip()

    # 1) пробуем Markdown
    try:
        with photo_path.open("rb") as f:
            tg_request(
                "sendPhoto",
                data={"chat_id": chat_id, "caption": cap[:1000], "parse_mode":"Markdown"},
                files={"photo": f}
            )
        return
    except requests.exceptions.HTTPError as e:
        resp_text = ""
        if getattr(e, "response", None) is not None:
            resp_text = (e.response.text or "")
        if "parse entities" in resp_text.lower() or "can't parse" in resp_text.lower():
            print("[WARN] Telegram Markdown parse error in caption; retry without parse_mode")
            cap_plain = markdown_to_plain(cap)
            with photo_path.open("rb") as f:
                tg_request(
                    "sendPhoto",
                    data={"chat_id": chat_id, "caption": cap_plain[:1000]},
                    files={"photo": f}
                )
            return
        raise

def send_message(chat_id: str, text: str) -> None:
    if not chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID is missing/empty.")
    t = (text or "").strip()

    # 1) пробуем Markdown
    try:
        tg_request("sendMessage", data={"chat_id": chat_id, "text": t[:4000], "parse_mode":"Markdown"})
        return
    except requests.exceptions.HTTPError as e:
        resp_text = ""
        if getattr(e, "response", None) is not None:
            resp_text = (e.response.text or "")
        if "parse entities" in resp_text.lower() or "can't parse" in resp_text.lower():
            print("[WARN] Telegram Markdown parse error in message; retry without parse_mode")
            tp = markdown_to_plain(t)
            tg_request("sendMessage", data={"chat_id": chat_id, "text": tp[:4000]})
            return
        raise

def load_weekly_stats() -> Dict[str,Any]:
    return load_state("stats_weekly.json", {})

def save_weekly_stats(stats: Dict[str,Any]) -> None:
    save_state("stats_weekly.json", stats)

def bump_weekly(stats: Dict[str,Any], week_key: str, field: str, amount: int = 1, reason: Optional[str]=None) -> None:
    wk = stats.get(week_key) or {"passed": 0, "rejected": 0, "reasons": {}}
    wk[field] = int(wk.get(field,0)) + amount
    if reason:
        rs = wk.get("reasons") or {}
        rs[reason] = int(rs.get(reason,0)) + amount
        wk["reasons"] = rs
    stats[week_key] = wk

def format_dashboard(stats: Dict[str,Any], week_key: str, title: str) -> str:
    wk = stats.get(week_key) or {"passed": 0, "rejected": 0, "reasons": {}}
    passed = int(wk.get("passed",0))
    rejected = int(wk.get("rejected",0))
    reasons = wk.get("reasons") or {}
    top = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:6]
    lines = [f"**{title} ({week_key})**",
             "",
             f"✅ Прошло: {passed}",
             f"🗂️ В черновики/отсев: {rejected}",
             ""]
    if top:
        lines.append("Причины отсева (топ):")
        for k,v in top:
            lines.append(f"• {k}: {v}")
    else:
        lines.append("Причины отсева: нет данных.")
    lines.append("")
    lines.append("_Примечание: это тех. статистика качества источников/фильтров._")
    return "\n".join(lines)

def handle_draft(pub_cfg: Dict[str,Any], entry: Dict[str,Any], stats: Dict[str,Any], week_key: str) -> None:
    mode = (pub_cfg.get("drafts_mode") or "skip").strip()
    drafts_chat_id = ""
    if mode == "post_to_drafts_chat":
        env_name = pub_cfg.get("drafts_chat_id_env") or "TELEGRAM_DRAFTS_CHAT_ID"
        drafts_chat_id = os.getenv(env_name,"").strip() or TELEGRAM_DRAFTS_CHAT_ID

    drafts = load_state("drafts.json", [])
    drafts.append(entry)
    save_state("drafts.json", drafts[-2000:])

    bump_weekly(stats, week_key, "rejected", 1, reason=str(entry.get("reason","unknown")))

    if mode == "post_to_drafts_chat" and drafts_chat_id:
        msg = ("**Черновик/пропуск**\n\n"
               f"Причина: {entry.get('reason')}\n"
               f"Рубрика: {entry.get('rubric_title','')}\n"
               f"Аудитория: {entry.get('audience','')}\n"
               f"Заголовок: {entry.get('title')}\n"
               f"Ссылка: {entry.get('link')}\n")
        send_message(drafts_chat_id, msg)

def pick_item(
    items: List[Dict[str,str]],
    used_canon: set[str],
    used_titles: set[str],
    quality_cfg: Dict[str,Any]
) -> Tuple[Optional[Dict[str,str]], Optional[Dict[str,Any]]]:
    ranked=[]
    for it in items:
        t = norm_space(it.get("title",""))
        l = it.get("link","")
        if not l:
            continue
        s,_ = score_item(t or "(no title)", l, quality_cfg)
        if s>=0:
            ranked.append((s,it))
    ranked.sort(key=lambda x:x[0], reverse=True)

    for _,it in ranked[:22]:
        it = enrich_article(dict(it))
        canon = it.get("canonical") or it.get("link","")
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
                "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z"),
                "reason": f"fact_check_failed:{reason}",
                "title": raw_title,
                "link": canon,
                "domain": dom,
            }

        it["picked_title"]=raw_title
        it["picked_summary"]=summ
        it["fact_check"]=reason
        it["source_type"]=source_type_label_from_factcheck(reason)
        return it, None
    return None, None

def run() -> None:
    rub_cfg = load_yaml(CFG_DIR/"rubrics.yml")
    channel_cfg = rub_cfg.get("channel",{})
    branding = rub_cfg.get("branding",{})
    tzname = channel_cfg.get("timezone","Asia/Nicosia")
    now = get_local_now(tzname)
    week_key = iso_week_key(now)

    sources, quality_cfg = load_sources()
    used_canon = set(load_state("used_canonical.json",[]))
    used_titles = set(load_state("used_titles.json",[]))

    pub_cfg = rub_cfg.get("publishing",{})
    max_posts = int(pub_cfg.get("max_posts_per_run",3))
    max_per_aud = int(pub_cfg.get("max_posts_per_audience_per_run",2))

    stats = load_weekly_stats()

    audiences_cfg = rub_cfg.get("audiences",{})
    if AUDIENCE=="both":
        aud_list=["parents","pros"]
    elif AUDIENCE in ("parents","pros"):
        aud_list=[AUDIENCE]
    else:
        aud_list=["parents"]

    posted=0
    for aud in aud_list:
        if posted>=max_posts:
            break
        aud_cfg = audiences_cfg.get(aud,{})
        title_suffix = aud_cfg.get("title_suffix","")
        rubrics = aud_cfg.get("rubrics",[]) or []
        aud_posted=0

        for rubric in rubrics:
            if posted>=max_posts or aud_posted>=max_per_aud:
                break
            if not is_due(rubric, now):
                continue

            if rubric.get("format") == "quality_dashboard":
                dash_title = pub_cfg.get("dashboard_title","Quality dashboard недели")
                dashboard_text = format_dashboard(stats, week_key, dash_title)
                dash_chat = (pub_cfg.get("dashboard_chat") or "main").strip().lower()
                chat_id = TELEGRAM_CHAT_ID
                if dash_chat == "drafts" and TELEGRAM_DRAFTS_CHAT_ID:
                    chat_id = TELEGRAM_DRAFTS_CHAT_ID
                send_message(chat_id, dashboard_text)
                time.sleep(0.7)
                continue

            all_items=[]
            for sid in rubric.get("sources",[]):
                src = sources.get(sid)
                if not src:
                    continue
                try:
                    all_items.extend(fetch_source(src))
                except Exception as e:
                    print(f"[WARN] source {sid} failed: {e}")

            picked, draft = pick_item(all_items, used_canon, used_titles, quality_cfg)
            if draft:
                draft.update({"audience": aud, "rubric": rubric.get("id",""), "rubric_title": rubric.get("title","")})
                handle_draft(pub_cfg, draft, stats, week_key)
                continue
            if not picked:
                continue

            title = rubric.get("title","Рубрика")
            text, meta = compose_post_v2(title, rubric.get("format",""), aud, channel_cfg, picked, title_suffix)

            if not meta.get("ok", False):
                draft_entry = {
                    "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z"),
                    "reason": str(meta.get("reason","quality_gate_failed")),
                    "audience": aud,
                    "rubric": rubric.get("id",""),
                    "rubric_title": title,
                    "title": picked.get("picked_title") or picked.get("title") or "",
                    "link": picked.get("canonical") or picked.get("link") or "",
                    "domain": safe_domain(picked.get("canonical") or picked.get("link") or ""),
                    "source_type": meta.get("source_type",""),
                }
                handle_draft(pub_cfg, draft_entry, stats, week_key)
                continue

            subtitle = "Коротко и по делу"
            summ = (picked.get("picked_summary") or "").strip()
            if summ:
                subtitle = summ[:110].rstrip(" .,:;—-") + "…"

            card = render_image_card(title, subtitle, branding)

            # Telegram sometimes fails on Markdown entities in caption; fallback to plain text + sendMessage
            try:
                send_photo(TELEGRAM_CHAT_ID, card, text[:950])
            except Exception as e:
                print(f"[WARN] send_photo failed; fallback to send_message. Err: {e}")
                try:
                    send_message(TELEGRAM_CHAT_ID, markdown_to_plain(text)[:4000])
                except Exception as e2:
                    print(f"[ERROR] send_message fallback also failed: {e2}")
                    raise

            bump_weekly(stats, week_key, "passed", 1)

            canon = picked.get("canonical") or picked.get("link","")
            if canon: used_canon.add(canon)
            tkey = norm_title_key(picked.get("picked_title") or picked.get("title") or "")
            if tkey: used_titles.add(tkey)

            posted += 1
            aud_posted += 1
            time.sleep(1.2)

    save_state("used_canonical.json", sorted(list(used_canon))[-6000:])
    save_state("used_titles.json", sorted(list(used_titles))[-6000:])
    save_weekly_stats(stats)

    print(f"Done. Posted: {posted}. Audience: {AUDIENCE}. Rewrite: {REWRITE_PROVIDER}. Week: {week_key}")

if __name__=="__main__":
    run()

from __future__ import annotations

"""Read-only runtime probe for configured myth_fact evidence sources.

The probe intentionally does not import the publisher, Telegram, publication
history, or visual/provider clients. It mirrors the publisher's HTTP decoding
and evidence extraction semantics, then calls the existing P1B myth/fact
evidence validator.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

from src.services.llm_generator import (
    MYTH_FACT_REFUTATION_PATTERNS,
    _myth_fact_families,
    validate_myth_fact_evidence_for_generation,
)
from src.services.topic_policy import RUBRIC_TOPIC_ROTATION


ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "config"

# Exact production publisher evidence-fetch identity/limits.
USER_AGENT = "logoped-channel-bot/4.3.2-safe (+https://github.com/)"
HEADERS = {"User-Agent": USER_AGENT}
CANONICAL_TIMEOUT_SECONDS = 25
EVIDENCE_TIMEOUT_SECONDS = 35
MAX_EVIDENCE_CHARS = 3600

INSECURE_TLS_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("INSECURE_TLS_DOMAINS", "") or "").split(",")
    if d.strip()
]

TELEMETRY_PHRASES: Tuple[str, ...] = (
    "myth",
    "not necessarily",
    "isn't necessarily",
    "no evidence",
    "no scientific evidence",
    "does not mean",
    "not caused by",
    "do not indicate",
)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


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


def _decode_candidate_score(text: str) -> tuple[int, int]:
    mojibake_markers = (
        "Р°",
        "Рµ",
        "Рё",
        "Рѕ",
        "С‚",
        "СЏ",
        "Ð",
        "Ñ",
        "Ã",
        "Â",
        "â",
    )
    replacement_count = text.count("�")
    mojibake_count = sum(text.count(marker) for marker in mojibake_markers)
    return (replacement_count * 10 + mojibake_count * 6, mojibake_count)


def _explicit_charset_from_headers(headers: object) -> str:
    content_type = ""
    if headers is not None:
        content_type = str(getattr(headers, "get", lambda *_args: "")("Content-Type", "") or "")
    match = re.search(
        r"(?:^|;)\s*charset\s*=\s*['\"]?([^;'\"]+)",
        content_type,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _decode_response_text(response: requests.Response) -> str:
    """Exact production response-decoding semantics used by run_publisher.py."""
    encodings: List[str] = []
    explicit_charset = _explicit_charset_from_headers(response.headers)
    if explicit_charset:
        encodings.append(explicit_charset)

    try:
        apparent_encoding = response.apparent_encoding
    except Exception:
        apparent_encoding = ""
    if apparent_encoding:
        encodings.append(apparent_encoding)

    encodings.extend(["utf-8", "windows-1251"])

    candidates: List[tuple[str, str]] = []
    seen: set[str] = set()
    for encoding in encodings:
        normalized = (encoding or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            candidates.append((normalized, response.content.decode(encoding)))
        except (LookupError, UnicodeDecodeError):
            continue

    if not candidates:
        return response.text

    explicit_candidate = next(
        (text for encoding, text in candidates if encoding == explicit_charset.strip().lower()),
        "",
    )
    if explicit_candidate and _decode_candidate_score(explicit_candidate) == (0, 0):
        return explicit_candidate

    return min(candidates, key=lambda item: _decode_candidate_score(item[1]))[1]


def _root_descriptor(root: BeautifulSoup) -> str:
    tag = getattr(root, "name", None) or "[document]"
    root_id = root.get("id", "") if hasattr(root, "get") else ""
    classes = root.get("class", []) if hasattr(root, "get") else []
    if isinstance(classes, str):
        classes = [classes]
    class_text = ".".join(str(item) for item in classes if str(item))
    return f"tag={tag} id={root_id or '-'} class={class_text or '-'}"


def select_extraction_root(soup: BeautifulSoup):
    """Exact production root priority, with a diagnostic descriptor."""
    root = (
        soup.select_one("div#dle-content")
        or soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )
    return root, _root_descriptor(root)


def extract_evidence_from_html(html_text: str, max_chars: int = MAX_EVIDENCE_CHARS) -> tuple[str, str]:
    """Mirror production h1 + h2/h3/p/li evidence extraction exactly."""
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root, root_descriptor = select_extraction_root(soup)

    chunks: List[str] = []
    h1 = soup.find("h1")
    if h1:
        chunks.append(norm_space(h1.get_text(" ", strip=True)))

    fallback_root = root is soup.body or root is soup
    elements = (
        h1.find_all_next(["h2", "h3", "p", "li"])
        if fallback_root and h1
        else root.select("h2, h3, p, li")
    )
    for el in elements:
        txt = norm_space(el.get_text(" ", strip=True))
        if len(txt) < 20:
            continue
        low = txt.lower()
        if any(
            bad in low
            for bad in ["cookie", "privacy", "политик", "подпис", "реклама", "скачать", "регистрация"]
        ):
            continue
        chunks.append(txt)
        if sum(len(item) for item in chunks) > max_chars * 1.35:
            break

    seen: set[str] = set()
    uniq: List[str] = []
    for chunk in chunks:
        key = chunk.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(chunk)

    out = "\n".join(uniq).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0].strip()
    return out, root_descriptor


def _find_myth_fact_rubric_sources(rubrics_cfg: Dict[str, Any]) -> List[str]:
    parents = ((rubrics_cfg.get("audiences") or {}).get("parents") or {})
    for rubric in parents.get("rubrics", []) or []:
        if str(rubric.get("id", "")).strip().lower() == "myth_fact":
            return [str(item).strip() for item in (rubric.get("sources") or []) if str(item).strip()]
    raise ValueError("myth_fact rubric not found")


def load_probe_contract() -> tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, str]]:
    sources_cfg = load_yaml(CFG_DIR / "sources.yml")
    topics_cfg = load_yaml(CFG_DIR / "topics.yml")
    rubrics_cfg = load_yaml(CFG_DIR / "rubrics.yml")

    source_map = {
        str(item.get("id", "")).strip(): item
        for item in (sources_cfg.get("sources") or [])
        if str(item.get("id", "")).strip()
    }
    myth_source_ids = _find_myth_fact_rubric_sources(rubrics_cfg)

    rotation = tuple(RUBRIC_TOPIC_ROTATION.get("myth_fact", ()))
    topics = topics_cfg.get("topics") or {}
    topic_by_source: Dict[str, str] = {}
    for source_id in myth_source_ids:
        matches = [
            topic_id
            for topic_id in rotation
            if source_id in set(((topics.get(topic_id) or {}).get("source_ids") or []))
        ]
        if len(matches) != 1:
            raise ValueError(f"source {source_id!r} maps to {len(matches)} myth_fact rotation topics: {matches}")
        if source_id not in source_map:
            raise ValueError(f"source {source_id!r} is missing from config/sources.yml")
        source = source_map[source_id]
        if str(source.get("type", "")).strip().lower() != "static":
            raise ValueError(f"source {source_id!r} must be static for this probe")
        urls = source.get("urls") or []
        if not urls:
            raise ValueError(f"source {source_id!r} has no configured static URLs")
        topic_by_source[source_id] = matches[0]

    return source_map, myth_source_ids, topic_by_source


def resolve_source_ids(requested: Sequence[str] | None = None) -> List[str]:
    _source_map, allowed, _topic_by_source = load_probe_contract()
    if not requested:
        return list(allowed)
    requested_clean = [str(item).strip() for item in requested if str(item).strip()]
    invalid = [item for item in requested_clean if item not in allowed]
    if invalid:
        raise ValueError(f"source IDs are not in configured myth_fact.sources: {invalid}")
    return requested_clean


def _canonical_url_for_probe(url: str) -> tuple[str, Dict[str, Any]]:
    """Mirror production get_canonical(): on failure use only the same original URL."""
    telemetry: Dict[str, Any] = {
        "canonical_http_status": None,
        "canonical_final_url": "",
        "canonical_fetch_error": "",
    }
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=CANONICAL_TIMEOUT_SECONDS,
            verify=_verify_for_url(url),
        )
        telemetry["canonical_http_status"] = response.status_code
        telemetry["canonical_final_url"] = response.url
        response.raise_for_status()
        soup = BeautifulSoup(_decode_response_text(response), "lxml")
        canonical = soup.find("link", rel=lambda value: value and "canonical" in value.lower())
        if canonical and canonical.get("href"):
            href = canonical["href"].strip()
            if href.startswith("/"):
                href = urljoin(url, href)
            return href, telemetry
        return url, telemetry
    except Exception as exc:
        telemetry["canonical_fetch_error"] = str(exc)
        return url, telemetry


def _phrase_presence(text: str) -> Dict[str, bool]:
    blob = (text or "").lower()
    return {phrase: phrase in blob for phrase in TELEMETRY_PHRASES}


def _existing_refutation_patterns(text: str) -> List[str]:
    blob = norm_space(text).replace("ё", "е").lower()
    return [
        pattern
        for pattern in MYTH_FACT_REFUTATION_PATTERNS
        if re.search(pattern, blob, flags=re.IGNORECASE)
    ]


def _evidence_snippets(evidence: str, limit: int = 4, width: int = 260) -> List[str]:
    snippets: List[str] = []
    for line in (evidence or "").splitlines():
        item = norm_space(line)
        if not item:
            continue
        snippets.append(item[:width])
        if len(snippets) >= limit:
            break
    return snippets


def probe_url(source_id: str, topic_id: str, configured_url: str) -> Dict[str, Any]:
    canonical_url, canonical_telemetry = _canonical_url_for_probe(configured_url)
    result: Dict[str, Any] = {
        "source_id": source_id,
        "configured_url": configured_url,
        "canonical_url": canonical_url,
        "effective_topic": topic_id,
        **canonical_telemetry,
        "http_status": None,
        "fetch_error": "",
        "final_response_url": "",
        "content_type": "",
        "response_html_chars": 0,
        "selected_extraction_root": "",
        "evidence_chars": 0,
        "evidence_snippets": [],
        "existing_refutation_patterns_found": [],
        "myth_fact_topic_families_found": [],
        "phrase_presence_html": {phrase: False for phrase in TELEMETRY_PHRASES},
        "phrase_presence_evidence": {phrase: False for phrase in TELEMETRY_PHRASES},
        "validator": {"ok": False, "reason": "not_run"},
    }

    response: requests.Response | None = None
    try:
        response = requests.get(
            canonical_url,
            headers=HEADERS,
            timeout=EVIDENCE_TIMEOUT_SECONDS,
            verify=_verify_for_url(canonical_url),
        )
        result["http_status"] = response.status_code
        result["final_response_url"] = response.url
        result["content_type"] = (response.headers.get("Content-Type") or "").lower()
        response.raise_for_status()

        content_type = result["content_type"]
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            ok, reason = validate_myth_fact_evidence_for_generation("", topic_id)
            result["validator"] = {"ok": bool(ok), "reason": reason}
            return result

        html_text = _decode_response_text(response)
        result["response_html_chars"] = len(html_text)
        result["phrase_presence_html"] = _phrase_presence(html_text)

        evidence, root_descriptor = extract_evidence_from_html(html_text, MAX_EVIDENCE_CHARS)
        result["selected_extraction_root"] = root_descriptor
        result["evidence_chars"] = len(evidence)
        result["evidence_snippets"] = _evidence_snippets(evidence)
        result["existing_refutation_patterns_found"] = _existing_refutation_patterns(evidence)
        result["myth_fact_topic_families_found"] = sorted(_myth_fact_families(evidence))
        result["phrase_presence_evidence"] = _phrase_presence(evidence)

        ok, reason = validate_myth_fact_evidence_for_generation(evidence, topic_id)
        result["validator"] = {"ok": bool(ok), "reason": reason}
        return result
    except Exception as exc:
        if response is not None:
            result["http_status"] = response.status_code
            result["final_response_url"] = response.url
            result["content_type"] = (response.headers.get("Content-Type") or "").lower()
        result["fetch_error"] = str(exc)
        return result


def run_probe(source_ids: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    source_map, allowed, topic_by_source = load_probe_contract()
    selected = resolve_source_ids(source_ids)
    # The requested set is validated against the exact configured myth_fact pool.
    if any(source_id not in allowed for source_id in selected):
        raise ValueError("probe source selection escaped configured myth_fact.sources")

    results: List[Dict[str, Any]] = []
    for source_id in selected:
        source = source_map[source_id]
        for configured_url in source.get("urls") or []:
            results.append(probe_url(source_id, topic_by_source[source_id], str(configured_url)))
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only probe for configured myth_fact sources")
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Optional configured myth_fact source ID; repeat to select more than one.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = run_probe(args.source_id or None)
    except Exception as exc:
        print(json.dumps({"probe_error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2

    for item in results:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

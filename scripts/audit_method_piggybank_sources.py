from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.publisher.run_publisher import (
    fetch_source,
    get_canonical,
    load_sources,
    norm_space,
    extract_evidence_text,
)
from src.services.llm_generator import validate_pro_evidence_for_generation


def _method_piggybank_source_ids() -> list[str]:
    cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8")) or {}
    for rubric in cfg.get("audiences", {}).get("pros", {}).get("rubrics", []):
        if rubric.get("id") == "method_piggybank":
            return list(rubric.get("sources") or [])
    raise RuntimeError("method_piggybank rubric not found")


def _iter_source_urls(source_ids: Iterable[str], max_per_source: int) -> Iterable[tuple[str, str]]:
    sources = load_sources()
    for source_id in source_ids:
        src = sources.get(source_id)
        if not src:
            print(f"{source_id}\t-\tsource_missing\t0\tFAIL\tunknown_source_id\t")
            continue
        try:
            candidates = fetch_source(src)
        except Exception as exc:
            print(f"{source_id}\t-\tsource_fetch_failed\t0\tFAIL\t{type(exc).__name__}: {exc}\t")
            continue
        for candidate in candidates[:max_per_source]:
            url = (candidate.get("link") or "").strip()
            if url:
                yield source_id, url


def audit(max_per_source: int) -> int:
    source_ids = _method_piggybank_source_ids()
    passed = 0
    print("source_id\turl\tfetch_status\tevidence_len\tprefilter\treason\tevidence_preview")
    for source_id, url in _iter_source_urls(source_ids, max_per_source=max_per_source):
        canon = get_canonical(url)
        try:
            evidence = extract_evidence_text(canon, max_chars=3600)
            fetch_status = "ok"
        except Exception as exc:
            print(f"{source_id}\t{canon}\tevidence_fetch_failed\t0\tFAIL\t{type(exc).__name__}: {exc}\t")
            continue

        ok, reason = validate_pro_evidence_for_generation(evidence)
        if ok:
            passed += 1
        preview = norm_space(evidence[:300]).replace("\t", " ")
        print(
            f"{source_id}\t{canon}\t{fetch_status}\t{len(evidence.strip())}\t"
            f"{'PASS' if ok else 'FAIL'}\t{reason}\t{preview}"
        )
    return passed


def exit_code_for_pass_count(pass_count: int, min_pass: int) -> int:
    return 0 if pass_count >= min_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-source", type=int, default=25)
    parser.add_argument("--min-pass", type=int, default=5)
    args = parser.parse_args()
    passed = audit(max_per_source=args.max_per_source)
    print(f"PASS_COUNT={passed}")
    raise SystemExit(exit_code_for_pass_count(passed, args.min_pass))


if __name__ == "__main__":
    main()

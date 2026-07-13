from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.publisher.run_publisher import (  # noqa: E402
    extract_evidence_text,
    fetch_source,
    get_canonical,
    load_sources,
    norm_space,
)
from src.services.llm_generator import validate_pro_evidence_for_generation  # noqa: E402


def _method_piggybank_source_ids() -> list[str]:
    config = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8")) or {}
    for rubric in config.get("audiences", {}).get("pros", {}).get("rubrics", []):
        if rubric.get("id") == "method_piggybank":
            return list(rubric.get("sources") or [])
    raise RuntimeError("method_piggybank rubric not found")


def _iter_source_urls(source_ids: Iterable[str], max_per_source: int) -> Iterable[tuple[str, str]]:
    sources = load_sources()
    for source_id in source_ids:
        source = sources.get(source_id)
        if not source:
            print(f"{source_id}\t-\tsource_missing\t0\tFAIL\tunknown_source_id\t")
            continue
        try:
            candidates = fetch_source(source)
        except Exception as exc:
            print(f"{source_id}\t-\tsource_fetch_failed\t0\tFAIL\t{type(exc).__name__}: {exc}\t")
            continue
        for candidate in candidates[:max_per_source]:
            url = (candidate.get("link") or "").strip()
            if url:
                yield source_id, url


def audit(max_per_source: int) -> int:
    source_ids = _method_piggybank_source_ids()
    pass_count = 0
    print("source_id\turl\tfetch_status\tevidence_len\tstatus\treason\tevidence_preview")
    for source_id, url in _iter_source_urls(source_ids, max_per_source=max_per_source):
        canonical_url = get_canonical(url)
        try:
            evidence = extract_evidence_text(canonical_url, max_chars=3600)
        except Exception as exc:
            print(
                f"{source_id}\t{canonical_url}\tevidence_fetch_failed\t0\tFAIL\t"
                f"{type(exc).__name__}: {exc}\t"
            )
            continue

        ok, reason = validate_pro_evidence_for_generation(evidence)
        if ok:
            pass_count += 1
        preview = norm_space(evidence[:300]).replace("\t", " ")
        print(
            f"{source_id}\t{canonical_url}\tok\t{len(evidence.strip())}\t"
            f"{'PASS' if ok else 'FAIL'}\t{reason}\t{preview}"
        )
    return pass_count


def exit_code_for_pass_count(pass_count: int, min_pass: int) -> int:
    return 0 if pass_count >= min_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit method_piggybank source evidence.")
    parser.add_argument("--max-per-source", type=int, default=25)
    parser.add_argument("--min-pass", type=int, default=5)
    args = parser.parse_args()
    if args.max_per_source < 1:
        parser.error("--max-per-source must be positive")
    if args.min_pass < 0:
        parser.error("--min-pass must be non-negative")

    pass_count = audit(max_per_source=args.max_per_source)
    print(f"PASS_COUNT={pass_count}")
    raise SystemExit(exit_code_for_pass_count(pass_count, args.min_pass))


if __name__ == "__main__":
    main()

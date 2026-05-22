"""
Deduplication policy helpers for publisher-level checks.

This module intentionally stays small and dependency-light:
- no Telegram logic
- no DB/store access
- no LLM logic
- no publisher orchestration

Goal:
Keep rubric-specific dedup thresholds out of the large run_publisher.py file,
so future dedup changes can be made safely in a small full-file replacement.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: str) -> float:
    """Read a float env var safely, falling back to default on invalid values."""
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


SEMANTIC_THRESHOLD = _env_float("SEMANTIC_THRESHOLD", "0.95")

SEMANTIC_THRESHOLD_POST_AGE_NORMS = _env_float(
    "SEMANTIC_THRESHOLD_POST_AGE_NORMS",
    "0.985",
)


def normalize_rubric_id(rubric_id: str | None) -> str:
    """Normalize rubric id for stable policy comparisons."""
    return (rubric_id or "").strip().lower()


def semantic_post_threshold_for_rubric(rubric_id: str | None) -> float:
    """
    Return semantic body/post dedup threshold for the current rubric.

    age_norms naturally produces similar milestone-style texts every Sunday.
    For this rubric we keep exact/hash dedup active, but make semantic post
    dedup stricter so only near-identical final posts are blocked.

    Other rubrics keep the global SEMANTIC_THRESHOLD.
    """
    if normalize_rubric_id(rubric_id) == "age_norms":
        return SEMANTIC_THRESHOLD_POST_AGE_NORMS

    return SEMANTIC_THRESHOLD


def should_bypass_source_semantic_dedup(rubric_id: str | None) -> bool:
    """
    Return True when source-level semantic dedup should not block candidates.

    method_piggybank can legitimately reuse the same broad source/article theme
    while still producing a different practical method. It should pass through
    to LLM and remain protected by final dedup checks:
    - dup_url_db
    - dup_evidence_hash_db
    - dup_body_hash_db
    - dup_semantic_post
    """
    return normalize_rubric_id(rubric_id) == "method_piggybank"

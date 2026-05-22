"""
Deduplication policy helpers for publisher-level checks.

This module intentionally stays small and dependency-light:
- no Telegram logic
- no DB/store access
- no LLM logic
- no publisher orchestration

Goal:
Keep rubric-specific dedup thresholds and bypass rules out of the large
run_publisher.py file, so future dedup changes can be made safely in a small
full-file replacement.
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

    Some rubrics naturally reuse the same broad evidence topics while still
    producing different final posts.

    We only bypass source-level semantic dedup. Safer final checks remain active:
    - dup_url_db
    - dup_evidence_hash_db
    - dup_body_hash_db
    - dup_semantic_post

    Rationale by rubric:
    - method_piggybank: professional method articles often share the same
      terminology while still producing different practical techniques.
    - question_week: parent-question posts often reuse the same recurring
      bilingualism / language-delay evidence topics, but the final Q&A framing
      can still be meaningfully different.
    """
    return normalize_rubric_id(rubric_id) in {
        "method_piggybank",
        "question_week",
    }

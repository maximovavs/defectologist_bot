# Project State

Repository: `maximovavs/defectologist_bot`

Purpose: evidence-grounded automated content generation and publishing for the speech-development channel, with strict source, quality, freshness, and safety checks.

## Operating contract

Permission and action boundaries are defined in `AGENTS.md`. This file grants no authorization.

## Last verified baseline

Verified read-only on 2026-09-19 before this documentation-only Draft PR:

- live `main`: `0cd302c47a26da677e07309079d1c28f578f550e`
- tree: `9d5571edac4f1cb642843e5c050ae4341eb895b0`
- sole parent: `25cf2fdb09a2195e2dbec8d87085e0c90525f00f`
- verification: `verified=true`, reason `valid`
- subject: `Give the question_week age repair a deterministic allowed-age hint (#68)`

This baseline is provenance only. Re-read live state before any future write.

## Current state

- The repository is a mature production system with publisher, validation, freshness, source-policy, and research surfaces.
- Current `main` includes PR #68, which adds a deterministic allowed-age hint for the existing single repair path when `question_week` fails specifically with `parent_age_not_grounded`; the existing validator remains authoritative.
- Stage 5 LLM S3 research assets and isolated calibration workflows are present and remain separate from production behavior.
- The root `README.md` is not a complete current-state handoff; use this file plus live GitHub evidence for active work.

## Current blocker

No single global blocker is encoded here.

The active blocker is stage-specific and must be established from the user's latest handoff plus fresh repository/run evidence. Do not infer it from an old PR description or from this file.

## Evidence pointers

Use these on demand:

- `research/RESEARCH_STATE.md` — lightweight index of committed research state.
- `research/llm_s3/` — frozen Stage 5 prompt/corpus/gold inputs.
- `scripts/calibrate_llm_s3.py` — isolated LLM S3 calibration/evaluation logic.
- `.github/workflows/llm_s3_calibration.yml` and `.github/workflows/llm_s3_groq_calibration.yml` — owner-only research execution surfaces.
- `tests/` — executable contracts for production and research behavior.
- `.github/workflows/post.yml` — production publisher workflow; do not dispatch without explicit authorization.

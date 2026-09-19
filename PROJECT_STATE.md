# Project State

Repository: `maximovavs/defectologist_bot`

Purpose: evidence-grounded automated content generation and publishing for the speech-development channel, with strict source, quality, freshness, and safety checks.

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

## Next allowed step

Read-only discovery/audit only, unless the user separately authorizes a specific mutation.

For any future stage:

1. read this file;
2. inspect the live GitHub state relevant to the stage;
3. consult `research/RESEARCH_STATE.md` when the task depends on prior research;
4. read only the minimum necessary code/tests/workflow evidence;
5. confirm the exact permission boundary;
6. mutate only when explicitly authorized.

## Authorization required

Explicit authorization is required for all writes and production-affecting actions, including commits, branch/PR mutations, Ready/merge, workflow dispatch or rerun/retry, provider calls, Telegram calls, publication, state/cache/database mutation, and secret/config changes.

This file grants no such authorization.

## Evidence pointers

Use these on demand:

- `research/RESEARCH_STATE.md` — lightweight index of committed research state.
- `research/llm_s3/` — frozen Stage 5 prompt/corpus/gold inputs.
- `scripts/calibrate_llm_s3.py` — isolated LLM S3 calibration/evaluation logic.
- `.github/workflows/llm_s3_calibration.yml` and `.github/workflows/llm_s3_groq_calibration.yml` — owner-only research execution surfaces.
- `tests/` — executable contracts for production and research behavior.
- `.github/workflows/post.yml` — production publisher workflow; do not dispatch without explicit authorization.

## Context-efficiency rule

Do not turn this file into a chronological log.

Keep only the minimal current entry point here. Put durable research conclusions in `research/RESEARCH_STATE.md`; keep raw corpora, gold, logs, and workflow artifacts in their existing locations and read them only when a concrete task requires them.

# Project State

Repository: `maximovavs/defectologist_bot`

Purpose: evidence-grounded automated content generation and publishing for the
speech-development channel, with strict source, quality, freshness, and safety
checks.

## Operating contract

Permission and action boundaries are defined in `AGENTS.md`.

This file grants no write, workflow-dispatch, provider, Telegram, or production
authorization.

## Last verified baseline

Read-only state-maintenance audit: 2026-10-02.

Observed live `main`:

`d9cfb4dc03187f672a4b62816e5c0f493188acae`

Subject:

`Extend text-safe retry to games communication (#85)`

This SHA is provenance only.

Always re-read live repository and workflow state before any future mutation or
production action.

## Current production state

- The repository is a mature production system with evidence-grounded source
  selection, publisher validation, freshness/dedup controls, visual QA, bounded
  fallback behavior, and isolated research surfaces.

- Parent-facing output uses deterministic validation and bounded repair around
  evidence grounding, age/context coherence, required structural fields, and
  user-facing text-quality defects. Deterministic validators remain authoritative
  over LLM output.

- The visual pipeline uses deterministic scene/category routing and QA. For
  explicitly eligible text-prone categories, an exact `object_contains_text`
  rejection may redirect only the second/final object attempt to a text-safe
  scene. The object-attempt budget remains bounded and the existing terminal
  text fallback remains available.

- Research and calibration surfaces remain isolated from production publishing
  behavior unless a separate production change is explicitly authorized and
  merged.

- The root `README.md` is not a complete current-state handoff. Use this file
  together with fresh GitHub evidence for active work.

## Durable source pointers

- Agent/authorization contract: `AGENTS.md`
- Production publisher: `src/publisher/run_publisher.py`
- Parent/LLM generation and deterministic output validation:
  `src/services/llm_generator.py`
- Visual routing, QA and bounded fallback:
  `src/services/visual_pipeline.py`
- Production workflow: `.github/workflows/post.yml`
- Publisher-policy CI contract:
  `.github/workflows/publisher_policy_pr_checks.yml`
- Research-state index: `research/RESEARCH_STATE.md`
- Stage 5 research assets: `research/llm_s3/`
- Isolated calibration logic: `scripts/calibrate_llm_s3.py`
- Isolated calibration workflows:
  `.github/workflows/llm_s3_calibration.yml` and
  `.github/workflows/llm_s3_groq_calibration.yml`
- Executable production/research contracts: `tests/`

## Current-state maintenance rule

Do not turn this file into a chronological log.

Keep here only durable production capabilities, canonical operating contracts,
stable source pointers, and a dated provenance baseline.

Do not store individual PRs, workflow runs, temporary failures, intermediate
RCA findings, or transient blockers here.

A merged change does not by itself prove natural-production behavior. Keep
`merged`, `CI-verified`, and `natural-production-proven` as distinct evidence
states.

Replace or remove current-state statements when a material merged change makes
them stale.

## Active work rule

Do not infer current blockers, open work, merge readiness, or production
validation from this file.

Establish those from fresh live GitHub state, relevant workflow evidence, and
the user's latest handoff.

## Next step

Establish the next concrete task from fresh GitHub state plus the user's current
handoff.

Default to read-only until a specific mutation or production stage is explicitly
authorized.

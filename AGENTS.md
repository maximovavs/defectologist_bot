# Agent Operating Contract

Repository: `maximovavs/defectologist_bot`

## Source of truth

- Treat live GitHub state as authoritative for repository, branch, commit, PR, workflow-run, and publication facts.
- Do not infer current state from an old chat, handoff, copied SHA, or historical run.
- Start with `PROJECT_STATE.md`, then read only the specific code, test, workflow, research file, or run evidence needed for the task.

## Default safety mode

- Default to fresh read-only audit.
- Commits, branch mutations, PR mutations, Ready transitions, merges, workflow dispatch, rerun/retry, provider calls, Telegram calls, publication, state/cache/database mutation, secret/config changes, and other production side effects require explicit user authorization for that exact action.
- Before any authorized write, verify fresh guards against live GitHub state.
- If a required guard is missing, ambiguous, or differs from the authorized contract: STOP and report the mismatch.
- Never convert a research/diagnostic stage into a production mutation implicitly.

## Fresh-guard principle

For writes, verify the smallest relevant live set immediately before mutation, such as:

- exact repository;
- live `main` commit;
- tree / sole parent / verification when required by the stage;
- target branch / PR / workflow / run state;
- exact allowed file scope.

Historical SHAs are provenance, not permission to act.

## Research discipline

- Research experiments and production changes are separate stages.
- Do not promote a hypothesis, calibration result, or model behavior into a production rule without explicit evidence and a separately authorized production stage.
- Preserve negative findings and failed approaches; do not repeat a completed research path without new evidence that justifies reopening it.
- Use `research/RESEARCH_STATE.md` as the lightweight index of committed research evidence.
- Read large corpora, gold files, logs, artifacts, and run output only when a concrete question requires them.

## Context discipline

Keep always-loaded context small.

- `PROJECT_STATE.md` is the lightweight current-state entry point.
- `research/RESEARCH_STATE.md` summarizes durable research state and points to evidence.
- `research/llm_s3/` contains frozen Stage 5 inputs; do not load the full corpus or gold by default.
- Tests, workflow logs, and historical PR descriptions are on-demand evidence, not startup context.

Do not duplicate long history into this file.

## Operational principles

- Separate diagnostics from mutations.
- Prefer bounded, reversible, least-privilege actions.
- Do not perform provider, Telegram, or publication calls merely to investigate a problem when offline/read-only evidence is sufficient.
- Do not expose or copy secret values, tokens, credentials, or private payloads into documentation.
- Do not invent a current blocker or next step. If the active stage is unclear, perform read-only discovery or ask for the missing authorization/context.

## Scope of this file

This file defines persistent agent behavior only. It grants no authorization and should not contain fast-changing PR numbers, run IDs, temporary branches, current blockers, or transient provider quota state.

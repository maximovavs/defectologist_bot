# Research State

This file is a lightweight index of durable, committed research evidence.

It is not a run log. Do not copy large artifacts, provider responses, or full calibration corpora here. Read those only when a concrete question requires them.

## Research / production boundary

- Research-only workflows, scripts, corpora, and evaluation logic do not authorize production changes.
- A research result must be supported by committed evidence or an explicitly cited run/artifact before it is treated as established.
- Negative findings and execution blocks are durable information; do not silently reinterpret them as model-quality failures.
- Reopening a completed research path requires new evidence or a new research question.

## Stage 5 — LLM S3 classification calibration

### Question

Can an isolated LLM classifier recover the preregistered S3 structure reliably enough to satisfy the frozen quality gates without leaking gold labels or coupling the experiment to production code?

### Committed evidence

Primary research inputs:

- `research/llm_s3/classification_prompt.txt`
- `research/llm_s3/validation_corpus.json`
- `research/llm_s3/validation_gold.json`

Research execution/evaluation:

- `scripts/calibrate_llm_s3.py`
- `.github/workflows/llm_s3_calibration.yml`
- `.github/workflows/llm_s3_groq_calibration.yml`
- `tests/test_llm_s3_calibration_surface.py`

### Frozen evaluation structure

PR #55, merged at commit `a6029a5b99c355f10ba4f5671da5a3cffaa31cb2`, established the isolated Stage 5 surface with:

- 124 primary scoring items;
- 30 paraphrase groups;
- 49 hard-negative pairs;
- 11 non-scoring ambiguity IDs;
- classifier jobs isolated from gold labels;
- evaluator isolated from provider credentials;
- no retry/repair/fallback inside the primary classification contract.

The evaluator measures joint/per-frame quality, MODEL_WAIT behavior, hard-negative collapse, paraphrase consistency, and unsafe incomplete-signature collisions.

### Execution-block semantics

PR #56, merged at commit `68061056053721748ec3c0ed1b64e506df2add22`, supersedes any interpretation of transport exhaustion as model quality.

The observed run shape recorded by that commit had:

- Groq: 116 of 124 provider interactions not complete;
- Gemini: 118 of 124 provider interactions not complete.

Those results are **EXECUTION BLOCKED / NOT SCORABLE**, not evidence of poor model accuracy.

The evaluator therefore separates:

- execution-blocking failures: provider interaction did not yield a meaningful model output;
- model-output failures: the provider answered but the model violated the frozen output contract.

If any primary item is execution-blocked, provider quality metrics are not scored.

### Groq-only execution surface

PR #64, merged at commit `16c42ff2c1f55690a4e31f8692bb146e79495224`, added a separate Groq-only Stage 5 workflow:

- manual `workflow_dispatch` only;
- repository-owner guard;
- `contents: read`;
- full frozen corpus;
- 16-second minimum request interval;
- no Gemini surface;
- no retry, fallback, repair, cache, publisher, Telegram, or production state surface.

This is an execution/calibration surface only. Its existence is not a production recommendation.

## Do not repeat / do not infer

- Do not quote the original low joint-accuracy numbers from the transport-exhausted multi-provider run as model-quality results; commit #56 explicitly invalidates that interpretation.
- Do not alter frozen Stage 5 prompt/corpus/gold while claiming comparability with the preregistered calibration.
- Do not add provider retries, fallback, repair, or hidden pacing changes to a calibration run without treating that as a new research contract.
- Do not promote S3 research findings into production guards without a separate validated production stage.
- Do not infer the latest provider feasibility, quota, or final calibration verdict from this summary. Those are time-sensitive and must be re-read from the relevant workflow run/artifact or later committed evidence.

## Other research

This index currently records only research conclusions that were verified from committed repository evidence while creating it.

If another completed study matters to future work, add a compact entry only after its evidence location and verdict are verified. Prefer:

- QUESTION
- METHOD
- RESULT
- VERDICT
- EVIDENCE LOCATION
- DO NOT REPEAT
- OPEN QUESTION

over chronological transcripts.

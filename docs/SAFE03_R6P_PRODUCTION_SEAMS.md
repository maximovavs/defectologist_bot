# SAFE03 R6-P — offline-only integration and CI seam contract

Base: `maximovavs/defectologist_bot` main `0e96a0ae494f1b8c80d098f65d7c8c824712c8b0`, tree `c6529e546b2311fc620d2db19279254c24b74882` (26 Oct 2026).

No production code is changed. This is NOT approval to activate SAFE03 or trust verbal confirmations. Public SAFE03 returns `NO_POST`. Ordinary publisher route is unchanged. This workflow is a LOCAL PROPOSAL, not installed GitHub CI.

| Production seam | Current contract | Future bounded implementation (separate authorization) | Current verdict |
|---|---|---|---|
| Selector | `src/publisher/run_publisher.py` has no SAFE03 import/call | trusted, explicit route selector **before** LLM/publisher side-effects; normal route untouched | UNWIRED |
| Trust & grounding | `safe03_r6h_trust_offline` roots empty; `USER_ATTESTED_CLINICAL_OK` is user report only | independent proof and source review for each whole-entry digest; no post-supplied roots | BLOCKED |
| Text formatting | `finalize_plain_post_for_publication` may add title/source/hashtags | one immutable approved exact caption; source button outside caption | INCOMPATIBLE until separate path |
| Visual | current `visual_pipeline.py` generates/mutates fallback objects and text covers | STATIC visual, exact title, QA of layout/safe-zone, no generative provider or dangerous objects | UNWIRED |
| Telegram | `send_post_with_visual` may HTML-fallback or split into multiple messages | ONE `sendPhoto`, `parse_mode` omitted, exact caption, hard reject on any failure; never split | INCOMPATIBLE |
| Dedup | publisher uses `sha1(norm_space(plain))`; global `store.has_body_hash` blocks repeats | maintain global dedup; any exception requires separately authenticated permit + event-window audit | BLOCKED |
| Delivery | `PublicationStore` has active delivery quarantine; SAFE03 has separate trial ledger | transaction/idempotency boundary integrated with durable publisher store and conservative quarantine after lost response | UNWIRED |
| CI | canonical policy workflow ignores all SAFE03 filenames | local proposed PR check, Python 3.11, no secrets/net, reproducible fixtures | LOCAL PROPOSAL |

## Release gates (all required, not achieved)

1. independently verified source and approval authority bound to immutable full-entry digest (or separately approved owner editorial risk model); oral expert reports never treated as signatures;
2. signed-off production SAFE03 sender design that **never** falls back to split delivery; conservative lost-receipt reconciliation;
3. correct visual QA/static text cards; caption UTF-16 <=1000 and unchanged; hearing/age guard and source grounding;
4. preservation of ordinary publisher routing, source policy, global dedup, and `NO_POST` on all unsupported input;
5. real Python 3.11 CI PASS and natural production evidence after explicitly permitted, separately guarded integration; offline synthetic receipts alone do not meet gate.

## Technical verification limitations

- Python 3.11 grammar via `ast.parse(..., feature_version=(3,11))` under Python 3.13 is a syntax compatibility screen, NOT a 3.11 runtime test.
- The CI candidate installs dependencies from the current canonical workflow during a future CI run; there was no GitHub workflow invocation in R6-P.
- Mocked source receipts and editorial attestations cannot confer actual production authority.
- SQLite offline ledger cannot establish atomicity across SQLite and Telegram; sender must quarantine uncertain outcomes.

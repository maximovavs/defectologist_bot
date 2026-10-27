# SAFE03 R6-Q — offline source + sender interface acceptance

Base: GitHub `main` `0e96a0ae494f1b8c80d098f65d7c8c824712c8b0`, tree `c6529e546b2311fc620d2db19279254c24b74882`; effective SAFE03 candidate from R6-P. **LOCAL ONLY. Not a publish permit.**

## Verdict / hard boundaries

- Trust/source **fixture-compatible only**: `public_source()` always returns `NO_POST`; no independently enrolled reviewer/source root. `OWNER_ATTESTED_EDITORIAL` or user-relayed telephone opinion is *not* a signing key or verified clinical source.
- Sender **interface-compatible only**: `public_sender()` always returns `NO_POST`. No Telegram calls, no transport, no chat ID, no bot key, no provider, no scheduler.
- `PublicationStore` compatibility: **schema and semantics confirmed on an isolated, artificial SQLite history**, with zero production history accesses; its current `delivery_attempts` quarantine is conservative. But using a snapshot from one DB and a separate SAFE03 ledger is not an atomic reservation against concurrent regular publisher runs. **Production atomicity STOP.**

## Proposed interface 1: source/clinical grounding

`ClaimMap(catalog_id, whole_entry_sha256, evidence_sha256, claim_ids, clinical_review_receipt_sha256)` is bound to the **entire approved entry** (text, age, hazards, metadata). Required claim IDs are enumerated per catalog entry. Each source must be pinned to the captured source document bytes/version/date, signed or authenticated clinician/source review, HTTPS canonical host, evidence payload SHA, and a claim-to-evidence map. A URL or unverified digest is **not evidence that the source supports a particular claim**. Fixture-only `SourceFinding` cannot authorize publication.

Real admission must involve an **independent trust root** outside catalog JSON and user/LLM supplied values: authenticated reviewer identity or an explicitly accepted owner editorial authority policy, validation of dated source evidence and provenance, two-person/clinical sign-off where required, revocation/expiry handling and audit trail. R6-Q implements NONE of those real trust capabilities; all public registries remain empty. The fixture hashes and local-process seal detect accidental test object tampering but **are not signatures, cross-process authentication, or authority**.

## Proposed interface 2: exact, single photo sender

- Input: an independently grounded and independently authorized exact `catalog_id + whole_entry_sha256` bound to approved `caption` bytes, approved static PNG bytes and SHA-256, validated visual safe-zone/layout/title, age/hearing/hazard semantics, and HTTPS source reference.
- One Telegram `sendPhoto` (never `sendMessage`, HTML, another caption, a second send, delete/retry or split). `parse_mode` **omitted**; `caption` == exact immutable catalog text, UTF-16 units <=1000; external source shown only as inline URL button, not in the caption.
- The current generic `send_post_with_visual` cannot be used for SAFE03, as it may HTML-fallback or split on error. The prepared `PhotoIntent` records requirements only; there is deliberately **no executable Telegram sender**.
- A PNG signature/hash and dimensions do not verify that a card is visually legible or accurate; production requires independently verified QA/approved static asset receipt. The local pure model marks this GAP.
- Before sending: one transaction/locking domain with the existing `PublicationStore` for both `body_hash` dedup and `delivery_attempts` reservation, synchronised with SAFE03 permit/ledger; reserve first, never send on any unknown state. Treat any row in `delivery_attempts` as a quarantine.
- On valid single Telegram message receipt, persist `message_id` plus source/caption/image digest and clear quarantine only after durable publication record. If request timing or outcome is unknown, **quarantine with zero automatic retries** and require explicit read-only reconciliation, never falsely infer delivered or not delivered.
- A network call cannot be made atomic with a local SQLite transaction. Exactly-once **cannot be claimed** without a reconciliable server idempotency mechanism (Telegram sendPhoto lacks a general client idempotency key). Contract guarantees only fail-closed local *at-most-one-attempt*, not confirmed exactly-once delivery.
- Current global `sha1(norm_space(plain))` duplicate gate remains active. The separate 90-day-repeat proposal is not a production exception and no bypass is authorized.

## Readiness matrix

| Requirement | Status |
| --- | --- |
| 5 exact catalog entries including warning v4/v3 | Verified in earlier R6-N/O/P candidate; rechecked in test |
| Trust/claim binding input validation | OFFLINE fixture PASS |
| Independent authenticated source/reviewer enrollment | GAP / STOP |
| Static media format/sha/dimensions | OFFLINE PASS; visual layout QA GAP |
| Exact one-message intended payload | OFFLINE contract PASS; actual Telegram sender NOT implemented |
| Actual ProductionStore schema read-only compatibility | OFFLINE PASS |
| Cross-ledger atomic dedup/idempotency | STOP (not implemented) |
| Confirmation / uncertain delivery policy | OFFLINE modeled; real receipts NOT VERIFIED |
| Public SAFE03 activation | OFF / NO_POST |

**Release prohibition:** Do NOT merge, set variables, call Telegram, or install a sender based on this document. Next bounded stage should produce a **local transaction-boundary + source-root acceptance design** that solves the two STOP items and is reviewed before proposing a production PR. All actual permissions and release gates must be separately granted.

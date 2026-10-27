# Logopedia39 SAFE03 R6-S — local dual-lane reservation (DISABLED)

Scope: `maximovavs/defectologist_bot`, live `main` at audit time `0e96a0ae494f1b8c80d098f65d7c8c824712c8b0` / tree `c6529e546b2311fc620d2db19279254c24b74882`. NOT a production rollout.

## Explicit invariant: activation is impossible here

- `JOINT_RESERVATION_PRODUCTION_ENABLED = False` and NO environment-based enablement, network, credentials or Telegram sender.
- `public_normal_reservation(...) -> LEGACY_UNCHANGED` is only a marker: actual publisher does not import it, so normal routing, text formatting, dedup and delivery behavior remain untouched.
- `public_safe03_reservation(...) -> NO_POST` always. Unverified owner-attested expert consultations and R6-R synthetic source permits are not trusted production roots.
- Only explicit `normal_fixture` / `safe03_fixture` create process-local sealed **test proposals**; both work only on disposable `test_history.sqlite3` created with real `PublicationStore`. `safe03_fixture` accepts the pre-existing R6-R synthetic source review + R6-Q photo intent, not a real source approval.

## Proposed future shared interface (NOT production-implemented)

`reserve(route, immutable_post, verified_source_receipt, visual_receipt, delivery_window)` must validate all immutable inputs and acquire **the same production store transaction** for BOTH routes. In one `BEGIN IMMEDIATE` transaction, verify schema, check **all** delivery_attempts (even a single `confirmed` row means unreconciled), global `publications.body_hash`, exact URL for normal publisher, and any prior reservations. Insert a shared reservation and existing-schema `delivery_attempts.pending`, then COMMIT before attempting network delivery. Never sleep or perform network inside the SQLite lock. A lost/unknown response means `ambiguous` quarantine, no retry. On a confirmed, independently trusted delivery receipt, write publication and clear the attempt in the **same transaction**. Fail closed on any inconsistent state; do not add a 90-day dedup bypass.

`DisabledJointFixtureStore` simulates this protocol for `NORMAL` and `SAFE03`, using `PublicationStore`'s real tables; both routes therefore contend for one writer lock. An attempted reservation after any other unresolved attempt is rejected. It is intentionally serial across unrelated posts, favoring safety over throughput. The R6-R table is quarantined if old fixture reservations exist. This is not a migration mechanism.

## Normal publisher backward compatibility

Current production behavior stays *exactly* as-is; no existing publisher files changed. The real `run_publisher.py` still checks `store.has_body_hash(...)` and later calls `send_post_with_visual(...)` independently. It is NOT using the shared transaction. Therefore, **cross-publisher race protection in real production is NOT proven**. The offline tests prove only the behavior of two hypothetical cooperating callers. No claim of exact-once Telegram delivery can be made, because SQLite and the Telegram API cannot participate in one distributed atomic transaction. A successful REST response itself still needs durable receipt validation; a lost response must be manually reconciled.

## Full release gate (not met)

1. Separate authorized, reviewed production wiring for both normal and SAFE03 publisher paths; preserve legacy behavior with feature OFF, and explicitly handle migration/open attempts and rollback.
2. Independently authenticated and versioned source/clinician review roots, source snapshots and evidence-to-claim mapping; user-reported oral agreement cannot populate verified roots automatically.
3. A separate one-photo SAFE03 Telegram sender, exact caption and static visual QA; no HTML changes or split fallback.
4. Real Python 3.11 CI execution, network blocked, concurrency/duplicate/failure/restore tests, config/flag fail-closed checks and source/visual QA gates.
5. Separate authorization for any GitHub mutation, activation, or publication. No approval is implied here.

## Test boundaries

The fixture has process-local HMAC seals, not secure signatures or privileged authority. A malicious programmer with repository access could create test tokens; they cannot activate any public SAFE03 route. SQLite test database path checks refuse production filename/symlinks. Fixture acknowledgements are synthetic, and tests must never call real Telegram or providers. Local CI under Python 3.13 and Python 3.11 AST parsing do not establish 3.11 runtime compatibility.

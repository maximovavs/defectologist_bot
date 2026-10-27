# SAFE03 R6-T — hard-OFF ordinary publisher seam (offline candidate)

## Scope and authority
- Built against `maximovavs/defectologist_bot` main `0e96a0ae494f1b8c80d098f65d7c8c824712c8b0` (tree `c6529e546b2311fc620d2db19279254c24b74882`).
- GitHub STRICT READ-ONLY. No publication, dispatch, secrets/config, provider, or Telegram calls. All test SQLite files are disposable.
- **NO production activation.** This local work is not approval for a commit, PR, merge or flag activation.

## Ordinary publisher seam
- `src/publisher/run_publisher.py` adds only a constant `_R6T_SHARED_RESERVATION_PRODUCTION_ENABLED = False` and a hard stop immediately BEFORE `send_post_with_visual` when that constant is true.
- The historical OFF control flow, dedup, visual QA, age/source checks, title/footer, engagement, scheduling, Telegram fallback and delivery hooks are otherwise unchanged. Removing the two insertions reproduces the exact original publisher bytes and SHA-256.
- The active production publisher imports no SAFE03 modules; `.github/workflows/post.yml` is unchanged.
- No environment variable, flag, permission or oral approval enables this seam. The developer must separately change source code to activate — **not authorized**.

## Local-only hypothetical ON behavior
- `safe03_r6t_publisher_seam_offline.simulate_pre_send_normal` uses an R6-S normal fixture and its `DisabledJointFixtureStore` in a temporary database named `test_history.sqlite3` only.
- `simulate_pre_send_safe03` uses the same SQLite transaction with an R6-S fixture proposal, which still requires **synthetic** expert/source/photo evidence. These return `OFFLINE_RESERVED_ONLY`, NEVER a send-capable token.
- One SQLite `BEGIN IMMEDIATE` transaction checks existing publications and delivery attempts, records a mock claim, and reserves a pending attempt; unknown outcomes quarantine and block further attempts.
- Fixture ON tests exercise contention NORMAL/SAFE03, duplicate body hashes and unknown states, not actual Telegram sending.
- The normal publisher's actual ON branch is intentionally blocked before Telegram. This is a safety gate, not a working joint sender.

## Unresolved blockers / STOP
1. `PublicationStore._install_publisher_delivery_hooks` independently calls `begin_delivery_attempt` just before live `send_post_with_visual` and uses SHA-256 of display text; normal publisher's previous duplicate check uses SHA-1 of normalized plain text. This lifecycle is not unified with R6-S claims. Naively enabling both creates a double-reservation / inconsistent-body-hash hazard.
2. Normal publisher checks `store.has_body_hash` separately. It does not participate in the R6-S shared transaction; OFF does not remove the original race.
3. Real Telegram transport is not transactionally coupled with local SQLite. Ambiguous outcomes must quarantine, not auto-retry; exactly-once external delivery is not established.
4. There is no independently trusted SAFE03 clinical source registry or approvals. Public SAFE03 always yields NO_POST, and the exact-caption single-photo sender remains unwired.
5. Real GitHub Python 3.11 CI, natural production and delivery receipts were not exercised.

Verdict: OFF-PATH PASS; **hypothetical mock ON transaction PASS**; **real ON integration STOP**. Do not merge or enable this candidate until a new guarded stage redesigns the delivery-hook boundary atomically and tests both production routes.

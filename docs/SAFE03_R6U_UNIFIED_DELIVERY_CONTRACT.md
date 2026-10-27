# SAFE03 R6-U — unified delivery lifecycle, offline-only candidate

Status: **LOCAL SIMULATION PASS / LIVE ON STOP**. This is not a Telegram sender,
not a production migration, not an approval/source root, and not activation.

## Boundaries

- `run_publisher.py` contains one new immutable `False` guard. The legacy R6-T
  OFF path remains otherwise unchanged. If the new guard is changed, it throws
  before `send_post_with_visual`; it never calls a network transport.
- `PublicationStore` retains its original hook behavior when OFF; a new
  immutable hard-STOP flag prevents combining early reservation with its legacy
  `begin_delivery_attempt` hook until the handoff has been fully designed and
  independently accepted. The actual hook does **not** consume R6-U reservations.
- `safe03_r6u_unified_lifecycle_offline.py` can open only a non-symlink, already
  existing disposable `test_history.sqlite3`. Its NORMAL and SAFE03 producers
  require an R6-S local sealed synthetic proposal; no public approval or source
  root is populated. Public SAFE03 returns `NO_POST`.

## One SQLite lifecycle (only in disposable fixture)

1. `reserve`: one `BEGIN IMMEDIATE` checks **all** prior delivery attempts,
   `publications.body_hash` (existing SHA-1 of normalized, pre-footer body),
   normal canonical URL, and other prototype claims. It inserts one claim and
   exactly one `delivery_attempts.pending` in the same transaction.
2. Fields remain distinct: `normalized_sha1` is the current publisher's dedup
   key; `display_sha256` is SHA-256 of actual *display text* bound to chat ID
   and to the existing delivery-hook hash semantics. They MUST NOT be substituted.
   SAFE03 requires `display_plain == approved proposal.caption` exactly.
3. `mock_hook_begin` validates chat, display bytes and claim seal, then updates
   the existing pending attempt to `in_flight` **without inserting a second**.
   This is a **mock hook**, not the real `PublicationStore` hook.
4. `mock_unknown` permanently quarantines the attempt as `ambiguous`; the
   fixture deliberately has no retry, stale-lease or automatic unlock API.
5. `mock_ack` atomically writes a **synthetic** `publications` row and deletes the
   one attempt in one SQLite transaction. A mock message ID is **not** Telegram
   evidence. A failed insert rolls back the removal and leaves the attempt.

## Explicitly not proved

- Production ordinary publisher uses `has_body_hash` before send outside a
  transaction, and the existing delivery hook independently calls
  `begin_delivery_attempt`. They are not integrated with the new fixture.
- There is no atomic transaction spanning SQLite and the Telegram API. After a
  lost Telegram response, a manual receipt reconciliation is needed; automatic
  retry risks a second publication.
- Real medical source admission, authenticated reviewer signatures, real
  Telegram receipts, static graphic visual QA, and Python 3.11 CI remain GAP.

## Acceptance barrier

Never enable either flag, add a production activation variable, or move this
prototype into the live publisher without a separately reviewed integration
that: (i) passes a single pre-reserved attempt identity into the REAL hook;
(ii) proves no second reservation; (iii) keeps unknown outcomes durable;
(iv) preserves OFF path and dedup; (v) passes Python 3.11 CI; and (vi) obtains
separate authorization. No external calls are made in R6-U.

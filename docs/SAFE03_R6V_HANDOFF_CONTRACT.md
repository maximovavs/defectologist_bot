# SAFE03 R6-V — TEST ONLY delivery hook handoff

- The R6-V adapter is **not a production sender**. The hard-coded gate `_R6V_TEST_ONLY_HANDOFF_ENABLED = False` is not configured by environment; real OFF publisher retains the legacy path.
- Tests monkeypatch the gate **only** for an isolated `test_history.sqlite3` and replace the provider transport with an in-memory fake. Existing `PublicationStore._install_publisher_delivery_hooks()` code performs its receipt wrapper using that fake.
- When pre-reserved, hook consumes a single HMAC-validated, pending `R6-U` fixture attempt via `mock_hook_begin`; it must never call `begin_delivery_attempt` again.
- SHA-1 of normalized content remains in `publications.body_hash`; SHA-256 of exact caption remains in `delivery_attempts.body_hash`. Receipt is simulated, not authenticated as Telegram delivery.
- Test-only acknowledgement atomically inserts a publication and removes the attempt. Unknown/failed/missing/multiple receipts remain quarantined; never automatically retry.
- Production hard STOP remains, independent reviewer/source approvals are absent, and no user-authorized publication exists.
- **RISK:** SQLite and Telegram cannot commit atomically. Lack of response or uncertain outcome requires manual reconciliation; false confirmation from fake transport is not evidence of actual Telegram delivery. No claim of real exactly-once delivery.

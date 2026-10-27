# SAFE03 R6-R — shared transaction and source admission (LOCAL / NO_POST)

**Scope:** `maximovavs/defectologist_bot`, frozen `main` `0e96a0ae494f1b8c80d098f65d7c8c824712c8b0` (`c6529e546b2311fc620d2db19279254c24b74882`). No GitHub mutations, no sending. **NOT a release approval**.

## Mandatory source admission, before any publisher integration

A public HTTPS URL, an SHA-256 hash, a claim identifier and a user-transmitted oral clinical opinion are **not** proof of independently authenticated medical source review. The live root registries are deliberately empty. The R6-R source reviewer accepts **synthetic fixtures only**; it is not a verifier of reviewer identity or the truth of a claim.

Proposed real admission workflow (NOT implemented in production):

1. Freeze each catalog entry byte-for-byte including body, safety caveats, age metadata and full-entry SHA-256; preserve previous immutable versions and date of owner-attested verbal approval. Reject a different version.
2. Capture an official clinical guideline **snapshot** with acquisition time, canonical URL, version/revision date, document digest, publisher/domain and retention/access restrictions. A live URL alone is insufficient; unknown HTTP/TLS or a missing snapshot means HOLD.
3. Maintain per-claim mapping `claim_id -> exact snapshot byte offsets / quote digest -> cited source section -> clinical relevance reviewed by a responsible human`. Verify exact bytes; excerpt existence is NOT clinical support. The human reviewer must attest to support, contraindications, exclusions and jurisdictional applicability separately.
4. Authenticate/enroll reviewer and independent checker through a **separate** trusted mechanism (not user- or LLM-supplied IDs). Bind their attested receipt to **entry SHA, snapshot SHA, all claim anchors, review date, expiry and any exclusion/caveat**. An owner-editorial lane may be used only under an explicitly approved distinct governance policy; never present it as a verified physician signature.
5. For high-risk topics (battery honey instructions / magnets), perform qualified clinical review of all mandatory caveats; fail closed on discrepancies. Example canonical sources: `https://www.poison.org/battery/guideline`, `https://www.rch.org.au/clinicalguide/guideline_index/Foreign_body_ingestion/`. Their provenance, content and quote support must be captured in future admission receipts; R6-R's synthetic fixture is not that review.
6. Record decision `APPROVED`/`HOLD`/`REVOKED` with audit log and revocation reason. Expiry, source drift, revoked reviewer, version change, unsupported claim or unavailable receipt -> `NO_POST`. Freshness interval is a separate clinical governance decision, not determined by this local prototype.
7. Validate actual static PNG asset and a signed/recorded visual QA receipt (text correctness, legibility, age, safe zone), check raw caption digest and <=1000 UTF-16 units. Preserve exact original text; put source URL only in an approved inline button, not in caption.
8. Keep the public registry empty until a separate specifically authorized production stage with source-root enrollment and a separately authorized technical release. No user attestation alone can populate the authenticated root.

## Shared SQLite transaction model, no transport

**Fixture:** initialize a brand-new temporary `test_history.sqlite3` using the real `PublicationStore` schema. Reject the production DB basename and symlinks. `BEGIN IMMEDIATE` on that **same SQLite file** encompasses:

- Verify the frozen local entry and sealed **synthetic** source permit and `PhotoIntent` for exactly one `sendPhoto` without parse mode or caption mutation.
- Query **all** existing `delivery_attempts`: any entry, even `confirmed`, is treated as quarantine (the existing publisher uses the same conservative rule).
- Check `publications.body_hash` against `sha1(normalize_whitespace(plain_text))`; **no** 90-day exception and **no** bypass of legacy dedup.
- Check any SAFE03 reserved/in-flight/unknown or historical identical hash/window in a separate fixture table.
- Commit the fixture reservation and a new existing-schema `delivery_attempts.pending` **atomically**. Roll back both on any schema/DB error. Connection closes after commit; no network side effect while holding SQLite lock.
- In another atomic transaction, mock `pending -> in_flight`. If outcome unknown: `unknown` + `delivery_attempts.ambiguous`; **no automatic retry**. If explicit mock receipt with positive ID: insert the mock publication, mark fixture `mock_confirmed`, and clear delivery quarantine in **the same transaction**; deletion must never precede durable publication record. All state transitions are fixture-only; no Telegram invocation.

**Not guaranteed:** A local database and a Telegram request cannot form a distributed atomic transaction. If a Telegram response is lost, status is unknown, not failed or successful. Existing `run_publisher.py` prechecks duplicate separately and does not use this reservation lock for the normal pipeline; R6-R only serializes SAFE03 fixture writers with each other. An ordinary publisher may precheck a duplicate, then race with an SAFE03 reservation before sending. Before production, both flows require a compatible shared single-authority reservation discipline, recovery protocol for stale/in-flight sends and hard-stop on ambiguous results; do not insert an experimental hook into the existing running publisher without its own explicitly authorized integration/rollout.

**Production enforcement:** both `public_source_admit` and `public_reserve` return `NO_POST`, and the local class refuses real production history filename. No Telegram transport, sender/chat IDs, imported provider clients, root credentials, webhook, scheduler, PR or CI run were created. The R6-R runner is a local-only CI candidate. Python 3.11 grammar check is not Python 3.11 runtime proof.

## Verdict / next gate

**OFFLINE TRANSACTION MODEL PASS, SOURCE-ROOT GAP, PRODUCTION CROSS-PUBLISHER ATOMICITY STOP, DELIVERY/ACTIVATION STOP.** One next bounded stage: code-level, offline, initially disabled joint reservation seam for normal publisher and SAFE03 + source-root receipt governance review (requires separate user permission and fresh guards), with CI on actual Python 3.11 before any merge/activation. No permission to relax dedup or publish.

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

WORKFLOW_FILE = "post.yml"
KNOWN_CONCLUSIONS = frozenset(
    {
        "success",
        "failure",
        "neutral",
        "cancelled",
        "skipped",
        "timed_out",
        "action_required",
        "stale",
        "startup_failure",
    }
)
PRE_PUBLISHER_SKIPPABLE = frozenset({"failure", "cancelled", "timed_out", "skipped"})
JOB_NOT_STARTED_CONCLUSIONS = frozenset({"cancelled", "skipped"})
PROD_MARKER_RE = re.compile(r"(?:^|[•\s])channel=prod(?:$|[•\s])")
TEST_MARKER_RE = re.compile(r"(?:^|[•\s])channel=test(?:$|[•\s])")


class StateContinuityError(RuntimeError):
    """Production state continuity cannot be proven safely."""


# Diagnostics only. Run #514 (37486543863) requested the cache of #342
# (27962120533) although #513 (37347011941) was the successful prod predecessor;
# the upstream cause is not yet proven. These stderr lines make a recurrence
# distinguishable (what the runtime was, which history pages it saw, which newest
# candidates it ordered, what it skipped and selected). They never change a
# decision: they are written to stderr only -- stdout is the $GITHUB_OUTPUT
# contract -- carry run metadata only, never the token, headers or URLs, and any
# failure while building or writing them is swallowed.
DIAGNOSTIC_PREFIX = "[STATE_CONTINUITY]"
DIAGNOSTIC_CANDIDATE_LIMIT = 10
_DIAGNOSTIC_VALUE_MAX_CHARS = 64


def _diagnostic_value(value: object) -> str:
    text = "none" if value is None else str(value)
    # One line, no spaces, printable only: a value can never start a new log
    # line or a workflow command.
    cleaned = "".join(ch if ch.isprintable() and not ch.isspace() else "_" for ch in text)
    return cleaned[:_DIAGNOSTIC_VALUE_MAX_CHARS] or "empty"


def _emit_diagnostic(tag: str, **fields: object) -> None:
    try:
        details = " ".join(
            f"{name}={_diagnostic_value(value)}" for name, value in fields.items()
        )
        print(f"{DIAGNOSTIC_PREFIX}[{tag}] {details}".rstrip(), file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - diagnostics must never affect continuity
        pass


def _diagnostic_run_number(raw: object) -> int | None:
    """Same acceptance as resolution, but report-only: never raises."""

    if not isinstance(raw, Mapping):
        return None
    try:
        return _positive_int(raw.get("run_number"), "run_number")
    except Exception:  # noqa: BLE001 - report-only; resolution re-validates
        return None


def _emit_page_diagnostic(page: int, page_runs: Sequence[object]) -> None:
    try:
        numbers = [
            number
            for number in (_diagnostic_run_number(raw) for raw in page_runs)
            if number is not None
        ]
        _emit_diagnostic(
            "PAGE",
            page=page,
            count=len(page_runs),
            min_run_number=min(numbers) if numbers else None,
            max_run_number=max(numbers) if numbers else None,
            unparsed_run_number=len(page_runs) - len(numbers),
        )
    except Exception:  # noqa: BLE001 - diagnostics must never affect continuity
        pass


def _diagnostic_lineage(raw: Mapping[str, object]) -> str:
    title = raw.get("display_title")
    if not isinstance(title, str) or not title.strip():
        return "unknown"
    try:
        return _lineage_from_title(title.strip())
    except StateContinuityError:
        return "ambiguous"


def _emit_candidate_summary(
    runs: Sequence[Mapping[str, object]],
    ordered: Sequence[Tuple[int, int, Mapping[str, object]]],
) -> None:
    try:
        shown = list(ordered[:DIAGNOSTIC_CANDIDATE_LIMIT])
        _emit_diagnostic(
            "CANDIDATES",
            history_count=len(runs),
            prior_count=len(ordered),
            shown=len(shown),
            limit=DIAGNOSTIC_CANDIDATE_LIMIT,
        )
        for rank, (run_number, run_id, raw) in enumerate(shown, start=1):
            _emit_diagnostic(
                "CANDIDATE",
                rank=rank,
                run_number=run_number,
                run_id=run_id,
                event=raw.get("event"),
                head_branch=raw.get("head_branch"),
                run_attempt=raw.get("run_attempt"),
                status=raw.get("status"),
                conclusion=raw.get("conclusion"),
                lineage=_diagnostic_lineage(raw),
            )
    except Exception:  # noqa: BLE001 - diagnostics must never affect continuity
        pass


@dataclass(frozen=True)
class Predecessor:
    run_id: int
    run_number: int


@dataclass(frozen=True)
class ProvenPreMutationIncident:
    run_id: int
    run_number: int
    event: str
    head_branch: str
    head_sha: str
    conclusion: str


PROVEN_PRE_MUTATION_INCIDENT = ProvenPreMutationIncident(
    run_id=34389481621,
    run_number=470,
    event="workflow_dispatch",
    head_branch="main",
    head_sha="a1bf73dd1e647d1a2436a7f6896f55edc1a8412f",
    conclusion="failure",
)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise StateContinuityError(f"ambiguous_run_metadata:{field}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StateContinuityError(f"ambiguous_run_metadata:{field}") from exc
    if parsed <= 0:
        raise StateContinuityError(f"ambiguous_run_metadata:{field}")
    return parsed


def _required_text(run: Mapping[str, object], field: str) -> str:
    value = run.get(field)
    if not isinstance(value, str) or not value.strip():
        raise StateContinuityError(f"ambiguous_run_metadata:{field}")
    return value.strip()


def _lineage_from_title(display_title: str) -> str:
    prod = bool(PROD_MARKER_RE.search(display_title))
    test = bool(TEST_MARKER_RE.search(display_title))
    if prod == test:
        raise StateContinuityError("ambiguous_run_metadata:channel")
    return "prod" if prod else "test"


def _ordered_prior_runs(
    runs: Sequence[Mapping[str, object]],
    *,
    current_run_number: int,
    current_run_id: int,
) -> List[Tuple[int, int, Mapping[str, object]]]:
    """Return minimally parsed prior runs in newest-to-oldest order.

    Only run identity/order metadata is read here. All other metadata is deferred
    until the run is actually reached while walking toward the canonical
    predecessor. This prevents irrelevant ancient legacy metadata below an
    already-proven predecessor from blocking the current run.
    """

    ordered: List[Tuple[int, int, Mapping[str, object]]] = []
    for raw in runs:
        if not isinstance(raw, Mapping):
            raise StateContinuityError("ambiguous_run_metadata:run_object")
        run_id = _positive_int(raw.get("id"), "id")
        run_number = _positive_int(raw.get("run_number"), "run_number")
        if run_id == current_run_id:
            continue
        if run_number >= current_run_number:
            continue
        ordered.append((run_number, run_id, raw))

    ordered.sort(key=lambda item: item[0], reverse=True)
    return ordered


def _validate_ordered_prod_run(
    run: Mapping[str, object],
    *,
    ref_name: str,
    run_id: int,
    run_number: int,
) -> Dict[str, object] | None:
    head_branch = _required_text(run, "head_branch")
    if head_branch != ref_name:
        return None

    display_title = _required_text(run, "display_title")
    lineage = _lineage_from_title(display_title)
    if lineage != "prod":
        return None

    run_attempt = _positive_int(run.get("run_attempt"), "run_attempt")
    status = _required_text(run, "status")
    conclusion = _required_text(run, "conclusion")
    event = _required_text(run, "event")

    if event not in {"schedule", "workflow_dispatch"}:
        raise StateContinuityError(f"ambiguous_run_metadata:event={event}")

    if run_attempt != 1:
        raise StateContinuityError(
            f"production_predecessor_rerun_not_safe:run_id={run_id}:run_attempt={run_attempt}"
        )
    if status != "completed":
        raise StateContinuityError(
            f"production_predecessor_not_completed:run_id={run_id}:status={status}"
        )
    if conclusion not in KNOWN_CONCLUSIONS:
        raise StateContinuityError(
            f"ambiguous_run_metadata:conclusion={conclusion}:run_id={run_id}"
        )

    return {
        "id": run_id,
        "run_number": run_number,
        "conclusion": conclusion,
    }


def _matches_proven_pre_mutation_incident(
    run: Mapping[str, object], *, run_id: int, run_number: int
) -> bool:
    incident = PROVEN_PRE_MUTATION_INCIDENT
    if run_id != incident.run_id and run_number != incident.run_number:
        return False

    actual = (
        run_id,
        run_number,
        run.get("event"),
        run.get("head_branch"),
        run.get("head_sha"),
        run.get("conclusion"),
    )
    expected = (
        incident.run_id,
        incident.run_number,
        incident.event,
        incident.head_branch,
        incident.head_sha,
        incident.conclusion,
    )
    if actual != expected:
        raise StateContinuityError(
            f"proven_pre_mutation_incident_metadata_mismatch:run_id={run_id}"
        )
    return True


def _post_job_proves_publisher_not_executed(
    jobs_payload: Mapping[str, object], run_id: int
) -> bool:
    """Prove the mutation-capable Publisher stage never executed.

    The proof is deliberately narrow: either the whole `post` job was
    cancelled/skipped before it started, or the unique `Run Publisher` step is
    explicitly completed/skipped. Missing or ambiguous metadata fails closed.
    """

    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        raise StateContinuityError(f"ambiguous_job_metadata:run_id={run_id}")

    matches = [
        job
        for job in jobs
        if isinstance(job, Mapping) and str(job.get("name") or "").strip() == "post"
    ]
    if len(matches) != 1:
        raise StateContinuityError(f"ambiguous_post_job_identity:run_id={run_id}")

    job = matches[0]
    status = str(job.get("status") or "").strip()
    conclusion = str(job.get("conclusion") or "").strip()
    started_at = job.get("started_at")

    if (
        status == "completed"
        and conclusion in JOB_NOT_STARTED_CONCLUSIONS
        and started_at is None
    ):
        return True

    if status != "completed":
        raise StateContinuityError(f"ambiguous_post_job_status:run_id={run_id}")

    steps = job.get("steps")
    if not isinstance(steps, list):
        raise StateContinuityError(f"ambiguous_post_steps:run_id={run_id}")

    publisher_steps = [
        step
        for step in steps
        if isinstance(step, Mapping)
        and str(step.get("name") or "").strip() == "Run Publisher"
    ]
    if len(publisher_steps) != 1:
        raise StateContinuityError(
            f"ambiguous_publisher_step_identity:run_id={run_id}"
        )

    publisher_step = publisher_steps[0]
    publisher_status = str(publisher_step.get("status") or "").strip()
    publisher_conclusion = str(publisher_step.get("conclusion") or "").strip()
    if not publisher_status or not publisher_conclusion:
        raise StateContinuityError(
            f"ambiguous_publisher_step_metadata:run_id={run_id}"
        )

    return publisher_status == "completed" and publisher_conclusion == "skipped"


def _post_job_proves_incident_publisher_failure(
    jobs_payload: Mapping[str, object], run_id: int
) -> bool:
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        raise StateContinuityError(f"ambiguous_job_metadata:run_id={run_id}")

    matches = [
        job
        for job in jobs
        if isinstance(job, Mapping) and str(job.get("name") or "").strip() == "post"
    ]
    if len(matches) != 1:
        raise StateContinuityError(f"ambiguous_post_job_identity:run_id={run_id}")

    job = matches[0]
    if (
        str(job.get("status") or "").strip() != "completed"
        or str(job.get("conclusion") or "").strip() != "failure"
    ):
        return False

    steps = job.get("steps")
    if not isinstance(steps, list):
        raise StateContinuityError(f"ambiguous_post_steps:run_id={run_id}")
    publisher_steps = [
        step
        for step in steps
        if isinstance(step, Mapping)
        and str(step.get("name") or "").strip() == "Run Publisher"
    ]
    if len(publisher_steps) != 1:
        raise StateContinuityError(
            f"ambiguous_publisher_step_identity:run_id={run_id}"
        )

    publisher_step = publisher_steps[0]
    return (
        str(publisher_step.get("status") or "").strip() == "completed"
        and str(publisher_step.get("conclusion") or "").strip() == "failure"
    )


def resolve_predecessor(
    runs: Sequence[Mapping[str, object]],
    *,
    current_run_id: int,
    current_run_number: int,
    current_run_attempt: int,
    ref_name: str,
    jobs_loader: Callable[[int], Mapping[str, object]],
) -> Predecessor:
    current_run_id = _positive_int(current_run_id, "current_run_id")
    current_run_number = _positive_int(current_run_number, "current_run_number")
    current_run_attempt = _positive_int(current_run_attempt, "current_run_attempt")
    if current_run_attempt != 1:
        raise StateContinuityError(
            f"production_rerun_not_safe:run_attempt={current_run_attempt}"
        )
    if not ref_name:
        raise StateContinuityError("current_ref_missing")

    ordered = _ordered_prior_runs(
        runs,
        current_run_number=current_run_number,
        current_run_id=current_run_id,
    )
    _emit_candidate_summary(runs, ordered)
    if not ordered:
        raise StateContinuityError("production_predecessor_missing")

    index = 0
    while index < len(ordered):
        run_number = ordered[index][0]
        same_number: List[Tuple[int, int, Mapping[str, object]]] = []
        while index < len(ordered) and ordered[index][0] == run_number:
            same_number.append(ordered[index])
            index += 1

        # Ambiguity at or above the would-be canonical predecessor remains
        # fail-closed. Ancient ambiguity below a selected predecessor is never
        # reached because we return immediately once continuity is proven.
        if len(same_number) != 1:
            raise StateContinuityError("ambiguous_production_predecessor_order")

        _, run_id, raw = same_number[0]
        matches_proven_incident = _matches_proven_pre_mutation_incident(
            raw,
            run_id=run_id,
            run_number=run_number,
        )
        candidate = _validate_ordered_prod_run(
            raw,
            ref_name=ref_name,
            run_id=run_id,
            run_number=run_number,
        )
        if candidate is None:
            continue

        conclusion = str(candidate["conclusion"])
        if conclusion in PRE_PUBLISHER_SKIPPABLE:
            jobs_payload = jobs_loader(run_id)
            if _post_job_proves_publisher_not_executed(jobs_payload, run_id):
                _emit_diagnostic(
                    "SKIP",
                    run_id=run_id,
                    run_number=run_number,
                    conclusion=conclusion,
                    reason="publisher_not_executed",
                )
                continue
            if matches_proven_incident and _post_job_proves_incident_publisher_failure(
                jobs_payload, run_id
            ):
                _emit_diagnostic(
                    "SKIP",
                    run_id=run_id,
                    run_number=run_number,
                    conclusion=conclusion,
                    reason="proven_pre_mutation_incident",
                )
                continue

        _emit_diagnostic("SELECTED", run_id=run_id, run_number=run_number, conclusion=conclusion)
        return Predecessor(run_id=run_id, run_number=run_number)

    raise StateContinuityError("production_predecessor_missing_after_safe_skips")


class GitHubActionsClient:
    def __init__(self, *, api_url: str, repository: str, token: str) -> None:
        if not api_url or not repository or not token:
            raise StateContinuityError("github_actions_api_configuration_missing")
        self.api_url = api_url.rstrip("/")
        self.repository = repository
        self.token = token

    def _get_json(self, path: str) -> Mapping[str, object]:
        url = f"{self.api_url}{path}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise StateContinuityError(
                f"github_actions_api_read_failed:{exc.__class__.__name__}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise StateContinuityError("github_actions_api_payload_invalid")
        return payload

    def list_workflow_runs(self, ref_name: str, max_pages: int = 20) -> List[Mapping[str, object]]:
        runs: List[Mapping[str, object]] = []
        encoded_branch = urllib.parse.quote(ref_name, safe="")
        encoded_workflow = urllib.parse.quote(WORKFLOW_FILE, safe="")
        for page in range(1, max_pages + 1):
            payload = self._get_json(
                f"/repos/{self.repository}/actions/workflows/{encoded_workflow}/runs"
                f"?branch={encoded_branch}&per_page=100&page={page}"
            )
            page_runs = payload.get("workflow_runs")
            if not isinstance(page_runs, list):
                raise StateContinuityError("github_actions_runs_payload_invalid")
            _emit_page_diagnostic(page, page_runs)
            runs.extend(page_runs)
            if len(page_runs) < 100:
                return runs
        raise StateContinuityError("github_actions_history_truncated")

    def jobs_for_run(self, run_id: int) -> Mapping[str, object]:
        return self._get_json(
            f"/repos/{self.repository}/actions/runs/{run_id}/jobs?filter=all&per_page=100"
        )


def build_expected_cache_key(
    *, cache_version: str, ref_name: str, predecessor_run_id: int
) -> str:
    if not cache_version or not ref_name:
        raise StateContinuityError("cache_key_configuration_missing")
    predecessor_run_id = _positive_int(predecessor_run_id, "predecessor_run_id")
    return f"logoped-state-{cache_version}-prod-{ref_name}-{predecessor_run_id}"


def main() -> int:
    _emit_diagnostic(
        "CURRENT",
        run_id=os.getenv("GITHUB_RUN_ID"),
        run_number=os.getenv("GITHUB_RUN_NUMBER"),
        run_attempt=os.getenv("GITHUB_RUN_ATTEMPT"),
        ref=os.getenv("GITHUB_REF_NAME"),
    )
    try:
        current_run_id = _positive_int(os.getenv("GITHUB_RUN_ID"), "GITHUB_RUN_ID")
        current_run_number = _positive_int(
            os.getenv("GITHUB_RUN_NUMBER"), "GITHUB_RUN_NUMBER"
        )
        current_run_attempt = _positive_int(
            os.getenv("GITHUB_RUN_ATTEMPT"), "GITHUB_RUN_ATTEMPT"
        )
        ref_name = (os.getenv("GITHUB_REF_NAME") or "").strip()
        client = GitHubActionsClient(
            api_url=(os.getenv("GITHUB_API_URL") or "").strip(),
            repository=(os.getenv("GITHUB_REPOSITORY") or "").strip(),
            token=(os.getenv("GITHUB_TOKEN") or "").strip(),
        )
        predecessor = resolve_predecessor(
            client.list_workflow_runs(ref_name),
            current_run_id=current_run_id,
            current_run_number=current_run_number,
            current_run_attempt=current_run_attempt,
            ref_name=ref_name,
            jobs_loader=client.jobs_for_run,
        )
        expected_cache_key = build_expected_cache_key(
            cache_version=(os.getenv("STATE_CACHE_VERSION") or "").strip(),
            ref_name=ref_name,
            predecessor_run_id=predecessor.run_id,
        )
    except StateContinuityError as exc:
        print(f"production_state_continuity_blocked:{exc}", file=sys.stderr)
        return 1

    print(f"predecessor_run_id={predecessor.run_id}")
    print(f"expected_cache_key={expected_cache_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

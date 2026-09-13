from __future__ import annotations

"""Isolated research-only calibration surface for the LLM S3 classification study.

This module is a standalone research caller. It deliberately does not import the
production publisher, provider helpers with fallback semantics, Telegram, the
visual pipeline, publication state, or any cache. It issues exactly one isolated
request per corpus text against exactly one primary model per provider, parses
the response strictly, and never issues a repair, retry, or fallback call.

Gold labels are never reachable from the classification path: ``classify`` takes
no gold argument and refuses to run when a gold artifact is visible from its
working inputs. Evaluation runs as a separate step that receives frozen
predictions plus frozen gold and zero provider credentials.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# ---------------------------------------------------------------------------
# Frozen research inputs (fail closed on any drift)
# ---------------------------------------------------------------------------

FROZEN_PROMPT_SHA256 = "306c223f393ced968d6ddc89c5209f032907845be199a705894caa9185e787b8"
FROZEN_CORPUS_SHA256 = "9835d955d7ae2a83266a4db01e7b2e953676f870ca880d242f5ccae6a8febfd0"
FROZEN_GOLD_SHA256 = "5be9e5eb3a61acda38b186bbbaa57d465a6621494e8861419da7e06581872472"

GOLD_FILENAME = "validation_gold.json"

# ---------------------------------------------------------------------------
# Exact provider contract: one primary model per provider, no fallback.
# ---------------------------------------------------------------------------

GROQ_PROVIDER = "groq"
GEMINI_PROVIDER = "gemini"
PROVIDERS = (GROQ_PROVIDER, GEMINI_PROVIDER)

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.7-flash"
ALLOWED_MODELS = {GROQ_PROVIDER: GROQ_MODEL, GEMINI_PROVIDER: GEMINI_MODEL}

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent"
PROVIDER_ENDPOINTS = {GROQ_PROVIDER: GROQ_ENDPOINT, GEMINI_PROVIDER: GEMINI_ENDPOINT}

PROVIDER_SECRET_ENV = {
    GROQ_PROVIDER: "GROQ_API_KEY",
    GEMINI_PROVIDER: "GEMINI_API_KEY",
}

REQUEST_TIMEOUT_SECONDS = 60
MAX_PERSISTED_ERROR_CHARS = 400
SECRET_REDACTION = "[redacted]"

# ---------------------------------------------------------------------------
# Allowed labels (exactly the frozen prompt's label space)
# ---------------------------------------------------------------------------

INTERACTION_FRAMES = frozenset(
    {
        "MODEL_WAIT",
        "SEARCH",
        "RETRIEVAL_COMMAND",
        "OPEN_NARRATIVE",
        "TURN_TAKING",
        "FORCED_CHOICE",
        "CLOZE",
        "RHYTHMIC_SEGMENTATION",
        "CLASSIFICATION",
        "IMITATION",
        "BREATH_CONTROL",
        "LOCATIVE_LANGUAGE",
        "OTHER",
    }
)

CHILD_RESPONSES = frozenset(
    {
        "ATTEND_VOCALIZE_REPEAT",
        "FIND",
        "BRING",
        "DESCRIBE",
        "TURN_TAKE",
        "CHOOSE_POINT",
        "COMPLETE_UTTERANCE",
        "CLAP_SEGMENTS",
        "SORT",
        "IMITATE_ORAL_MOVEMENT",
        "RETELL",
        "BLOW",
        "LOCATION_ANSWER",
        "UNSPECIFIED_RESPONSE",
        "OTHER",
    }
)

REQUIRED_OUTPUT_KEYS = frozenset({"interaction_frame", "child_response"})

# ---------------------------------------------------------------------------
# Primary scoring contract
# ---------------------------------------------------------------------------

PRIMARY_ITEM_COUNT = 124
MODEL_WAIT_FRAME = "MODEL_WAIT"
MODEL_WAIT_SUPPORT = 36
PARAPHRASE_GROUP_COUNT = 30
HARD_NEGATIVE_PAIR_COUNT = 49
AMBIGUITY_ID_COUNT = 11

JOINT_ACCURACY_THRESHOLD = 0.95
PER_FRAME_ACCURACY_THRESHOLD = 0.90
PER_FRAME_MIN_SUPPORT = 5
MODEL_WAIT_ACCURACY_THRESHOLD = 0.95
PARAPHRASE_GROUP_THRESHOLD = 0.95
HARD_NEGATIVE_MAX_FALSE_POSITIVES = 0

STABILITY_EXPECTED_IDS = 20
STABILITY_EXPECTED_RUNS = 3

INVALID_FIELD_MARKER = "<invalid>"


class FrozenInputError(RuntimeError):
    """Raised when a frozen research input does not match its expected digest."""


class GoldIsolationError(RuntimeError):
    """Raised when gold labels are reachable from the classification path."""


class ClassificationFailure(Exception):
    """Raised when a provider response does not satisfy the strict output contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Frozen input handling
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_file(path: Path, expected_sha256: str, label: str) -> str:
    """Verify a frozen input digest, failing closed on any mismatch."""

    path = Path(path)
    if not path.is_file():
        raise FrozenInputError(f"{label}: missing frozen input at {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise FrozenInputError(
            f"{label}: sha256 mismatch (expected {expected_sha256}, got {actual})"
        )
    return actual


def assert_gold_isolated(*input_paths: Path) -> None:
    """Refuse to classify when a gold artifact is visible from the working inputs."""

    for raw in input_paths:
        path = Path(raw)
        candidates = [path.parent / GOLD_FILENAME]
        if path.is_dir():
            candidates.append(path / GOLD_FILENAME)
        for candidate in candidates:
            if candidate.exists():
                raise GoldIsolationError(
                    "gold labels are visible to the classification path: "
                    f"{candidate}"
                )


def load_prompt(path: Path) -> str:
    verify_frozen_file(path, FROZEN_PROMPT_SHA256, "classification_prompt")
    return Path(path).read_text(encoding="utf-8")


def load_corpus(path: Path) -> List[Dict[str, str]]:
    verify_frozen_file(path, FROZEN_CORPUS_SHA256, "validation_corpus")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise FrozenInputError("validation_corpus: expected a list of items")
    items: List[Dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry or "text" not in entry:
            raise FrozenInputError("validation_corpus: malformed corpus item")
        items.append({"id": str(entry["id"]), "text": str(entry["text"])})
    if len(items) != PRIMARY_ITEM_COUNT:
        raise FrozenInputError(
            f"validation_corpus: expected {PRIMARY_ITEM_COUNT} items, got {len(items)}"
        )
    return items


def load_gold(path: Path) -> Dict[str, Any]:
    verify_frozen_file(path, FROZEN_GOLD_SHA256, "validation_gold")
    gold = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(gold, dict):
        raise FrozenInputError("validation_gold: expected an object")
    verify_gold_contract(gold)
    return gold


def verify_gold_contract(gold: Mapping[str, Any]) -> None:
    """Fail closed when the frozen gold does not match the primary scoring contract."""

    items = gold.get("primary_scoring_items")
    groups = gold.get("paraphrase_groups")
    pairs = gold.get("hard_negative_pairs")
    ambiguity = gold.get("non_scoring_ambiguity_ids")
    if not isinstance(items, list) or len(items) != PRIMARY_ITEM_COUNT:
        raise FrozenInputError(
            f"validation_gold: expected {PRIMARY_ITEM_COUNT} primary scoring items"
        )
    if not isinstance(groups, list) or len(groups) != PARAPHRASE_GROUP_COUNT:
        raise FrozenInputError(
            f"validation_gold: expected {PARAPHRASE_GROUP_COUNT} paraphrase groups"
        )
    if not isinstance(pairs, list) or len(pairs) != HARD_NEGATIVE_PAIR_COUNT:
        raise FrozenInputError(
            f"validation_gold: expected {HARD_NEGATIVE_PAIR_COUNT} hard-negative pairs"
        )
    if not isinstance(ambiguity, list) or len(ambiguity) != AMBIGUITY_ID_COUNT:
        raise FrozenInputError(
            f"validation_gold: expected {AMBIGUITY_ID_COUNT} non-scoring ambiguity ids"
        )
    frames = Counter(str(item["interaction_frame"]) for item in items)
    if frames[MODEL_WAIT_FRAME] != MODEL_WAIT_SUPPORT:
        raise FrozenInputError(
            f"validation_gold: expected {MODEL_WAIT_SUPPORT} {MODEL_WAIT_FRAME} items"
        )


# ---------------------------------------------------------------------------
# Secret scrubbing
# ---------------------------------------------------------------------------


def scrub_secret(text: str, secret: str) -> str:
    """Remove the exact provider secret from any persisted response or error text."""

    if not text:
        return ""
    if secret:
        text = text.replace(secret, SECRET_REDACTION)
    return text


def _truncate(text: str) -> str:
    if len(text) <= MAX_PERSISTED_ERROR_CHARS:
        return text
    return text[:MAX_PERSISTED_ERROR_CHARS] + "...[truncated]"


# ---------------------------------------------------------------------------
# Request construction: exactly one corpus text per isolated provider request
# ---------------------------------------------------------------------------


def build_request(
    provider: str, prompt_text: str, corpus_text: str, api_key: str
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    if provider == GROQ_PROVIDER:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": corpus_text},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        return GROQ_ENDPOINT, headers, payload
    if provider == GEMINI_PROVIDER:
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "systemInstruction": {"parts": [{"text": prompt_text}]},
            "contents": [{"role": "user", "parts": [{"text": corpus_text}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        return GEMINI_ENDPOINT, headers, payload
    raise ValueError(f"unsupported provider: {provider}")


def _http_post(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: int
) -> Tuple[int, str]:
    """Single research-only transport seam. No retry, no fallback, no repair."""

    import requests  # imported lazily so offline tests never need the transport

    response = requests.post(url, headers=dict(headers), json=dict(payload), timeout=timeout)
    return response.status_code, response.text


# ---------------------------------------------------------------------------
# Strict response parsing
# ---------------------------------------------------------------------------


def extract_provider_text(provider: str, body: str) -> str:
    try:
        envelope = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise ClassificationFailure("malformed_envelope", str(exc)) from exc
    if not isinstance(envelope, dict):
        raise ClassificationFailure("malformed_envelope", "envelope is not an object")
    try:
        if provider == GROQ_PROVIDER:
            choices = envelope["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ClassificationFailure(
                    "malformed_envelope", "expected exactly one choice"
                )
            text = choices[0]["message"]["content"]
        elif provider == GEMINI_PROVIDER:
            candidates = envelope["candidates"]
            if not isinstance(candidates, list) or len(candidates) != 1:
                raise ClassificationFailure(
                    "malformed_envelope", "expected exactly one candidate"
                )
            parts = candidates[0]["content"]["parts"]
            if not isinstance(parts, list) or len(parts) != 1:
                raise ClassificationFailure(
                    "malformed_envelope", "expected exactly one part"
                )
            text = parts[0]["text"]
        else:
            raise ValueError(f"unsupported provider: {provider}")
    except ClassificationFailure:
        raise
    except (KeyError, IndexError, TypeError) as exc:
        raise ClassificationFailure("malformed_envelope", str(exc)) from exc
    if not isinstance(text, str):
        raise ClassificationFailure("malformed_envelope", "response text is not a string")
    return text


def parse_classification(raw_text: str) -> Dict[str, str]:
    """Strictly parse the allowed output: exactly interaction_frame and child_response."""

    try:
        parsed = json.loads(raw_text)
    except (ValueError, TypeError) as exc:
        raise ClassificationFailure("malformed_json", str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ClassificationFailure("malformed_json", "output is not a JSON object")
    keys = set(parsed)
    if keys != REQUIRED_OUTPUT_KEYS:
        raise ClassificationFailure(
            "malformed_json", f"unexpected output keys: {sorted(keys)}"
        )
    frame = parsed["interaction_frame"]
    response = parsed["child_response"]
    if not isinstance(frame, str) or not isinstance(response, str):
        raise ClassificationFailure("malformed_json", "label values must be strings")
    invalid: List[str] = []
    if frame not in INTERACTION_FRAMES:
        invalid.append("interaction_frame")
    if response not in CHILD_RESPONSES:
        invalid.append("child_response")
    if invalid:
        failure = ClassificationFailure(
            "invalid_label", f"invalid label fields: {','.join(invalid)}"
        )
        failure.partial = {  # type: ignore[attr-defined]
            "interaction_frame": frame if frame in INTERACTION_FRAMES else INVALID_FIELD_MARKER,
            "child_response": response if response in CHILD_RESPONSES else INVALID_FIELD_MARKER,
        }
        raise failure
    return {"interaction_frame": frame, "child_response": response}


# ---------------------------------------------------------------------------
# Classification (never receives gold)
# ---------------------------------------------------------------------------


def _failure_record(item_id: str, reason: str, detail: str, partial: Mapping[str, str] | None = None) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": item_id,
        "status": "failed",
        "failure_reason": reason,
        "detail": _truncate(detail),
    }
    if partial:
        record["partial"] = dict(partial)
    return record


def classify_item(
    provider: str,
    prompt_text: str,
    item: Mapping[str, str],
    api_key: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    post: Any = None,
) -> Dict[str, Any]:
    """Classify exactly one corpus text with exactly one isolated provider request."""

    transport = post or _http_post
    item_id = str(item["id"])
    url, headers, payload = build_request(provider, prompt_text, str(item["text"]), api_key)
    try:
        status, body = transport(url, headers, payload, timeout)
    except Exception as exc:  # research surface: a transport error is a failure, not a retry
        detail = scrub_secret(f"{type(exc).__name__}: {exc}", api_key)
        return _failure_record(item_id, "transport_error", detail)
    if status != 200:
        detail = scrub_secret(f"status={status} body={body}", api_key)
        return _failure_record(item_id, "http_error", detail)
    try:
        raw_text = extract_provider_text(provider, body)
        labels = parse_classification(raw_text)
    except ClassificationFailure as exc:
        detail = scrub_secret(exc.detail, api_key)
        partial = getattr(exc, "partial", None)
        return _failure_record(item_id, exc.reason, detail, partial)
    return {
        "id": item_id,
        "status": "ok",
        "interaction_frame": labels["interaction_frame"],
        "child_response": labels["child_response"],
    }


def run_classification(
    provider: str,
    prompt_path: Path,
    corpus_path: Path,
    api_key: str,
    *,
    only_ids: Sequence[str] | None = None,
    run_label: str = "",
    post: Any = None,
) -> Dict[str, Any]:
    """Run the classification pass. This function takes no gold argument by design."""

    if provider not in PROVIDERS:
        raise ValueError(f"unsupported provider: {provider}")
    if not api_key:
        raise RuntimeError(
            f"missing provider credential in {PROVIDER_SECRET_ENV[provider]}"
        )
    assert_gold_isolated(prompt_path, corpus_path)
    prompt_sha = verify_frozen_file(prompt_path, FROZEN_PROMPT_SHA256, "classification_prompt")
    corpus_sha = verify_frozen_file(corpus_path, FROZEN_CORPUS_SHA256, "validation_corpus")
    prompt_text = load_prompt(prompt_path)
    corpus = load_corpus(corpus_path)
    if only_ids:
        wanted = list(dict.fromkeys(str(i) for i in only_ids))
        by_id = {item["id"]: item for item in corpus}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise ValueError(f"unknown corpus ids requested: {missing}")
        selected = [by_id[i] for i in wanted]
    else:
        selected = list(corpus)

    predictions = [
        classify_item(provider, prompt_text, item, api_key, post=post) for item in selected
    ]
    return {
        "schema_version": 1,
        "provider": provider,
        "model": ALLOWED_MODELS[provider],
        "endpoint": PROVIDER_ENDPOINTS[provider],
        "run_label": run_label,
        "prompt_sha256": prompt_sha,
        "corpus_sha256": corpus_sha,
        "requested_item_count": len(selected),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Evaluation (receives frozen predictions + frozen gold, zero provider secrets)
# ---------------------------------------------------------------------------


def _signature(record: Mapping[str, Any]) -> Tuple[str, str]:
    if record.get("status") == "ok":
        return (str(record["interaction_frame"]), str(record["child_response"]))
    partial = record.get("partial") or {}
    return (
        str(partial.get("interaction_frame", INVALID_FIELD_MARKER)),
        str(partial.get("child_response", INVALID_FIELD_MARKER)),
    )


def _is_complete(record: Mapping[str, Any]) -> bool:
    return record.get("status") == "ok"


def evaluate_provider(
    gold: Mapping[str, Any], prediction_payload: Mapping[str, Any], *, require_full_corpus: bool = True
) -> Dict[str, Any]:
    """Compute the primary metrics for one provider, independently of other providers."""

    ambiguity_ids = set(str(i) for i in gold["non_scoring_ambiguity_ids"])
    gold_items = {
        str(item["id"]): (str(item["interaction_frame"]), str(item["child_response"]))
        for item in gold["primary_scoring_items"]
        if str(item["id"]) not in ambiguity_ids
    }
    predictions = {str(p["id"]): p for p in prediction_payload["predictions"]}

    excluded_ambiguity_ids = sorted(set(predictions) & ambiguity_ids)
    scored_ids = sorted(set(gold_items) & set(predictions) - ambiguity_ids)

    if require_full_corpus and len(scored_ids) != PRIMARY_ITEM_COUNT:
        raise FrozenInputError(
            "primary evaluation requires all "
            f"{PRIMARY_ITEM_COUNT} scoring items, got {len(scored_ids)}"
        )

    correct_ids = {
        item_id
        for item_id in scored_ids
        if _is_complete(predictions[item_id]) and _signature(predictions[item_id]) == gold_items[item_id]
    }
    joint_accuracy = len(correct_ids) / len(scored_ids) if scored_ids else 0.0

    # Per-frame accuracy over every gold frame whose scored support is >= 5.
    by_frame: Dict[str, List[str]] = defaultdict(list)
    for item_id in scored_ids:
        by_frame[gold_items[item_id][0]].append(item_id)
    per_frame: Dict[str, Any] = {}
    frame_failures: List[str] = []
    for frame, ids in sorted(by_frame.items()):
        joint = sum(1 for i in ids if i in correct_ids) / len(ids)
        field = sum(
            1
            for i in ids
            if _is_complete(predictions[i]) and _signature(predictions[i])[0] == frame
        ) / len(ids)
        gated = len(ids) >= PER_FRAME_MIN_SUPPORT
        threshold = (
            MODEL_WAIT_ACCURACY_THRESHOLD
            if frame == MODEL_WAIT_FRAME
            else PER_FRAME_ACCURACY_THRESHOLD
        )
        passed = (not gated) or joint >= threshold
        per_frame[frame] = {
            "support": len(ids),
            "joint_accuracy": joint,
            "frame_field_accuracy": field,
            "gated": gated,
            "threshold": threshold if gated else None,
            "passed": passed,
        }
        if not passed:
            frame_failures.append(frame)

    model_wait = per_frame.get(MODEL_WAIT_FRAME, {"support": 0, "joint_accuracy": 0.0})
    model_wait_passed = (
        model_wait["support"] > 0
        and model_wait["joint_accuracy"] >= MODEL_WAIT_ACCURACY_THRESHOLD
    )

    # Hard negatives: a false positive is a prediction that collapses a frozen
    # contrast onto the paired item's gold signature.
    hard_negative_false_positives: List[Dict[str, str]] = []
    evaluated_pairs = 0
    for pair in gold["hard_negative_pairs"]:
        id_a, id_b = str(pair["id_a"]), str(pair["id_b"])
        if id_a not in scored_ids or id_b not in scored_ids:
            continue
        evaluated_pairs += 1
        sig_a = (str(pair["gold_a"]["interaction_frame"]), str(pair["gold_a"]["child_response"]))
        sig_b = (str(pair["gold_b"]["interaction_frame"]), str(pair["gold_b"]["child_response"]))
        if _is_complete(predictions[id_a]) and _signature(predictions[id_a]) == sig_b:
            hard_negative_false_positives.append(
                {"pair_id": str(pair["pair_id"]), "item_id": id_a, "collapsed_onto": id_b}
            )
        if _is_complete(predictions[id_b]) and _signature(predictions[id_b]) == sig_a:
            hard_negative_false_positives.append(
                {"pair_id": str(pair["pair_id"]), "item_id": id_b, "collapsed_onto": id_a}
            )

    # Paraphrase groups: a group passes only when every member is jointly correct.
    groups_evaluated = 0
    groups_correct = 0
    failed_groups: List[str] = []
    for group in gold["paraphrase_groups"]:
        member_ids = [str(m) for m in group["member_ids"]]
        if any(m not in scored_ids for m in member_ids):
            continue
        groups_evaluated += 1
        if all(m in correct_ids for m in member_ids):
            groups_correct += 1
        else:
            failed_groups.append(str(group["group_id"]))
    group_rate = groups_correct / groups_evaluated if groups_evaluated else 0.0

    # Unsafe collisions: distinct gold items collapsing to one incomplete/invalid signature.
    incomplete_by_signature: Dict[Tuple[str, str], set] = defaultdict(set)
    incomplete_members: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for item_id in scored_ids:
        record = predictions[item_id]
        if _is_complete(record):
            continue
        signature = _signature(record)
        incomplete_by_signature[signature].add(gold_items[item_id])
        incomplete_members[signature].append(item_id)
    unsafe_collisions = [
        {
            "signature": list(signature),
            "distinct_gold_signatures": len(gold_sigs),
            "item_ids": sorted(incomplete_members[signature]),
        }
        for signature, gold_sigs in sorted(incomplete_by_signature.items())
        if len(gold_sigs) > 1
    ]

    failure_reasons = Counter(
        str(predictions[i].get("failure_reason"))
        for i in scored_ids
        if not _is_complete(predictions[i])
    )

    checks = {
        "joint_accuracy": joint_accuracy >= JOINT_ACCURACY_THRESHOLD,
        "per_frame_accuracy": not frame_failures,
        "model_wait_accuracy": model_wait_passed,
        "hard_negative_false_positives": len(hard_negative_false_positives)
        <= HARD_NEGATIVE_MAX_FALSE_POSITIVES,
        "paraphrase_groups": group_rate >= PARAPHRASE_GROUP_THRESHOLD,
        "no_unsafe_collisions": not unsafe_collisions,
    }

    return {
        "provider": str(prediction_payload["provider"]),
        "model": str(prediction_payload["model"]),
        "run_label": str(prediction_payload.get("run_label", "")),
        "scored_items": len(scored_ids),
        "excluded_ambiguity_ids": excluded_ambiguity_ids,
        "correct_items": len(correct_ids),
        "joint_accuracy": joint_accuracy,
        "per_frame": per_frame,
        "failed_frames": frame_failures,
        "model_wait_accuracy": model_wait["joint_accuracy"],
        "hard_negative_pairs_evaluated": evaluated_pairs,
        "hard_negative_false_positives": hard_negative_false_positives,
        "paraphrase_groups_evaluated": groups_evaluated,
        "paraphrase_groups_correct": groups_correct,
        "paraphrase_group_rate": group_rate,
        "failed_paraphrase_groups": failed_groups,
        "unsafe_collisions": unsafe_collisions,
        "failure_reasons": dict(failure_reasons),
        "checks": checks,
        "passed": all(checks.values()),
    }


def load_predictions(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrozenInputError(f"{path}: predictions payload is not an object")
    provider = payload.get("provider")
    if provider not in PROVIDERS:
        raise FrozenInputError(f"{path}: unknown provider {provider!r}")
    if payload.get("model") != ALLOWED_MODELS[provider]:
        raise FrozenInputError(f"{path}: unexpected model {payload.get('model')!r}")
    if payload.get("endpoint") != PROVIDER_ENDPOINTS[provider]:
        raise FrozenInputError(f"{path}: unexpected endpoint {payload.get('endpoint')!r}")
    if payload.get("prompt_sha256") != FROZEN_PROMPT_SHA256:
        raise FrozenInputError(f"{path}: predictions built from a non-frozen prompt")
    if payload.get("corpus_sha256") != FROZEN_CORPUS_SHA256:
        raise FrozenInputError(f"{path}: predictions built from a non-frozen corpus")
    if not isinstance(payload.get("predictions"), list):
        raise FrozenInputError(f"{path}: predictions field is not a list")
    return payload


def evaluate_stability(
    prediction_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_runs: int = STABILITY_EXPECTED_RUNS,
    expected_ids: int = STABILITY_EXPECTED_IDS,
) -> Dict[str, Any]:
    """Repeatability view over repeated runs of the same deterministic id subset.

    This supports the preregistered stability audit without modifying the frozen
    corpus, prompt, or gold. It is not executed during the implementation stage.
    """

    if len(prediction_payloads) != expected_runs:
        raise FrozenInputError(
            f"stability audit expects {expected_runs} runs, got {len(prediction_payloads)}"
        )
    providers = {str(p["provider"]) for p in prediction_payloads}
    if len(providers) != 1:
        raise FrozenInputError("stability audit expects a single provider per report")
    id_sets = [{str(r["id"]) for r in p["predictions"]} for p in prediction_payloads]
    common = set.intersection(*id_sets)
    if any(ids != common for ids in id_sets):
        raise FrozenInputError("stability audit runs cover different corpus ids")
    if len(common) != expected_ids:
        raise FrozenInputError(
            f"stability audit expects {expected_ids} ids, got {len(common)}"
        )
    unstable: List[str] = []
    for item_id in sorted(common):
        signatures = set()
        for payload in prediction_payloads:
            record = next(r for r in payload["predictions"] if str(r["id"]) == item_id)
            signatures.add((_is_complete(record), _signature(record)))
        if len(signatures) != 1:
            unstable.append(item_id)
    return {
        "provider": providers.pop(),
        "runs": len(prediction_payloads),
        "ids": len(common),
        "unstable_ids": unstable,
        "stable_rate": (len(common) - len(unstable)) / len(common),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_verify_inputs(args: argparse.Namespace) -> int:
    verify_frozen_file(Path(args.prompt), FROZEN_PROMPT_SHA256, "classification_prompt")
    verify_frozen_file(Path(args.corpus), FROZEN_CORPUS_SHA256, "validation_corpus")
    print("frozen prompt sha256 OK")
    print("frozen corpus sha256 OK")
    if args.gold:
        load_gold(Path(args.gold))
        print("frozen gold sha256 OK")
        print("primary scoring contract OK")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    env_name = PROVIDER_SECRET_ENV[args.provider]
    api_key = os.environ.get(env_name, "")
    only_ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    payload = run_classification(
        args.provider,
        Path(args.prompt),
        Path(args.corpus),
        api_key,
        only_ids=only_ids,
        run_label=args.run_label,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ok = sum(1 for r in payload["predictions"] if r["status"] == "ok")
    failed = len(payload["predictions"]) - ok
    print(f"provider={payload['provider']} model={payload['model']}")
    print(f"classified={ok} failed={failed} total={len(payload['predictions'])}")
    print(f"predictions_sha256={sha256_file(out)}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    for name in PROVIDER_SECRET_ENV.values():
        if os.environ.get(name):
            raise RuntimeError(f"evaluator must not receive provider secret {name}")
    gold = load_gold(Path(args.gold))
    reports: Dict[str, Any] = {}
    for raw_path in args.predictions:
        payload = load_predictions(Path(raw_path))
        provider = str(payload["provider"])
        if provider in reports:
            raise FrozenInputError(f"duplicate predictions for provider {provider}")
        reports[provider] = evaluate_provider(
            gold, payload, require_full_corpus=not args.allow_partial_corpus
        )
    report = {
        "schema_version": 1,
        "gold_sha256": FROZEN_GOLD_SHA256,
        "prompt_sha256": FROZEN_PROMPT_SHA256,
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "thresholds": {
            "joint_accuracy": JOINT_ACCURACY_THRESHOLD,
            "per_frame_accuracy": PER_FRAME_ACCURACY_THRESHOLD,
            "per_frame_min_support": PER_FRAME_MIN_SUPPORT,
            "model_wait_accuracy": MODEL_WAIT_ACCURACY_THRESHOLD,
            "paraphrase_group_rate": PARAPHRASE_GROUP_THRESHOLD,
            "hard_negative_max_false_positives": HARD_NEGATIVE_MAX_FALSE_POSITIVES,
        },
        "providers": reports,
        "passed": all(r["passed"] for r in reports.values()) if reports else False,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for provider, result in sorted(reports.items()):
        print(f"--- {provider} ({result['model']}) ---")
        print(f"scored_items            {result['scored_items']}")
        print(f"joint_accuracy          {result['joint_accuracy']:.4f}")
        print(f"model_wait_accuracy     {result['model_wait_accuracy']:.4f}")
        print(f"paraphrase_group_rate   {result['paraphrase_group_rate']:.4f}")
        print(f"hard_negative_fp        {len(result['hard_negative_false_positives'])}")
        print(f"unsafe_collisions       {len(result['unsafe_collisions'])}")
        print(f"excluded_ambiguity_ids  {len(result['excluded_ambiguity_ids'])}")
        for check, passed in sorted(result["checks"].items()):
            print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def _cmd_stability(args: argparse.Namespace) -> int:
    payloads = [load_predictions(Path(p)) for p in args.predictions]
    result = evaluate_stability(
        payloads, expected_runs=args.expected_runs, expected_ids=args.expected_ids
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not result["unstable_ids"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_llm_s3",
        description="Isolated research-only LLM S3 calibration surface.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-inputs", help="verify frozen input digests")
    verify.add_argument("--prompt", required=True)
    verify.add_argument("--corpus", required=True)
    verify.add_argument("--gold", default="")
    verify.set_defaults(func=_cmd_verify_inputs)

    classify = sub.add_parser("classify", help="classify the frozen corpus with one provider")
    classify.add_argument("--provider", required=True, choices=list(PROVIDERS))
    classify.add_argument("--prompt", required=True)
    classify.add_argument("--corpus", required=True)
    classify.add_argument("--out", required=True)
    classify.add_argument("--ids", default="", help="optional comma-separated corpus id subset")
    classify.add_argument("--run-label", dest="run_label", default="")
    classify.set_defaults(func=_cmd_classify)

    evaluate = sub.add_parser("evaluate", help="score frozen predictions against frozen gold")
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--predictions", required=True, nargs="+")
    evaluate.add_argument("--out", default="")
    evaluate.add_argument("--allow-partial-corpus", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    stability = sub.add_parser("stability", help="repeatability view over repeated runs")
    stability.add_argument("--predictions", required=True, nargs="+")
    stability.add_argument("--expected-runs", type=int, default=STABILITY_EXPECTED_RUNS)
    stability.add_argument("--expected-ids", type=int, default=STABILITY_EXPECTED_IDS)
    stability.set_defaults(func=_cmd_stability)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

"""Offline contract tests for the isolated LLM S3 calibration surface.

Every provider interaction in this module is mocked. A module-scoped autouse
fixture makes any real outbound HTTP attempt fail loudly, so the suite proves
zero real Groq/Gemini calls.
"""

import ast
import copy
import hashlib
import inspect
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from scripts import calibrate_llm_s3 as cal


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "calibrate_llm_s3.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "llm_s3_calibration.yml"
RESEARCH_DIR = ROOT / "research" / "llm_s3"
PROMPT_PATH = RESEARCH_DIR / "classification_prompt.txt"
CORPUS_PATH = RESEARCH_DIR / "validation_corpus.json"
GOLD_PATH = RESEARCH_DIR / "validation_gold.json"

FROZEN_SHA256 = {
    PROMPT_PATH: "306c223f393ced968d6ddc89c5209f032907845be199a705894caa9185e787b8",
    CORPUS_PATH: "9835d955d7ae2a83266a4db01e7b2e953676f870ca880d242f5ccae6a8febfd0",
    GOLD_PATH: "5be9e5eb3a61acda38b186bbbaa57d465a6621494e8861419da7e06581872472",
}

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.7-flash"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent"

SURFACE_FILES = (SCRIPT_PATH, WORKFLOW_PATH)


def _forbidden_fallback_model_ids() -> tuple[str, ...]:
    return (
        "openai/gpt-oss-" + "20b",
        "gemini-" + "2.5-flash",
        "llama-" + "3.3-70b-versatile",
        "llama-" + "3.1-8b-instant",
    )


@pytest.fixture(autouse=True)
def _forbid_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything in this module attempts real network I/O."""

    import http.client

    import requests

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("real provider HTTP call attempted in offline tests")

    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "request", _boom)
    monkeypatch.setattr(requests.Session, "request", _boom)
    monkeypatch.setattr(http.client.HTTPConnection, "request", _boom)
    monkeypatch.setattr(http.client.HTTPSConnection, "request", _boom)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_only(path: Path) -> str:
    """Return the module source with comments and string literals removed.

    Isolation claims must hold for executable code, not for prose that merely
    names the surfaces the script stays away from.
    """

    import io
    import tokenize

    kept: List[str] = []
    with io.open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def _workflow() -> Dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _workflow_triggers(doc: Dict[str, Any]) -> Dict[str, Any]:
    # PyYAML resolves the bare `on:` key to the boolean True.
    return doc[True] if True in doc else doc["on"]


def _job_secret_names(job: Dict[str, Any]) -> set[str]:
    text = yaml.safe_dump(job, allow_unicode=True)
    return set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text))


def _gold() -> Dict[str, Any]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def _corpus() -> List[Dict[str, str]]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _perfect_predictions(provider: str) -> Dict[str, Any]:
    gold = _gold()
    return {
        "schema_version": 1,
        "provider": provider,
        "model": cal.ALLOWED_MODELS[provider],
        "endpoint": cal.PROVIDER_ENDPOINTS[provider],
        "run_label": "offline",
        "prompt_sha256": cal.FROZEN_PROMPT_SHA256,
        "corpus_sha256": cal.FROZEN_CORPUS_SHA256,
        "requested_item_count": len(gold["primary_scoring_items"]),
        "predictions": [
            {
                "id": item["id"],
                "status": "ok",
                "interaction_frame": item["interaction_frame"],
                "child_response": item["child_response"],
            }
            for item in gold["primary_scoring_items"]
        ],
    }


def _set_prediction(payload: Dict[str, Any], item_id: str, signature: tuple[str, str]) -> None:
    for record in payload["predictions"]:
        if record["id"] == item_id:
            record["status"] = "ok"
            record["interaction_frame"], record["child_response"] = signature
            return
    raise AssertionError(f"no prediction for {item_id}")


def _fail_prediction(payload: Dict[str, Any], item_id: str) -> None:
    for index, record in enumerate(payload["predictions"]):
        if record["id"] == item_id:
            payload["predictions"][index] = {
                "id": item_id,
                "status": "failed",
                "failure_reason": "malformed_json",
                "detail": "",
            }
            return
    raise AssertionError(f"no prediction for {item_id}")


def _gold_signature(side: Dict[str, str]) -> tuple[str, str]:
    return (side["interaction_frame"], side["child_response"])


def _isolated_hard_negative_pair(gold: Dict[str, Any]) -> Dict[str, Any]:
    """Pick a frozen pair whose members belong to no other pair and no group.

    Mutating such a pair's predictions cannot disturb any other pair, so the
    pair-level criterion can be asserted as an exact count.
    """

    membership: Dict[str, int] = {}
    for pair in gold["hard_negative_pairs"]:
        for item_id in (pair["id_a"], pair["id_b"]):
            membership[item_id] = membership.get(item_id, 0) + 1
    grouped = {m for group in gold["paraphrase_groups"] for m in group["member_ids"]}
    for pair in gold["hard_negative_pairs"]:
        ids = (pair["id_a"], pair["id_b"])
        if all(membership[i] == 1 for i in ids) and not set(ids) & grouped:
            assert _gold_signature(pair["gold_a"]) != _gold_signature(pair["gold_b"])
            return pair
    raise AssertionError("no isolated hard-negative pair in the frozen gold")


def _third_signature(gold: Dict[str, Any], *excluded: tuple[str, str]) -> tuple[str, str]:
    """A valid S3 signature drawn from gold, distinct from every excluded one."""

    for item in gold["primary_scoring_items"]:
        candidate = (item["interaction_frame"], item["child_response"])
        if candidate not in excluded:
            return candidate
    raise AssertionError("no third signature available")


def _groq_body(frame: str, response: str) -> str:
    payload = json.dumps({"interaction_frame": frame, "child_response": response})
    return json.dumps({"choices": [{"message": {"content": payload}}]})


def _gemini_body(frame: str, response: str) -> str:
    payload = json.dumps({"interaction_frame": frame, "child_response": response})
    return json.dumps({"candidates": [{"content": {"parts": [{"text": payload}]}}]})


class _RecordingTransport:
    """Mock transport seam: records every request, performs no network I/O."""

    def __init__(self, body: str = "", status: int = 200) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.body = body
        self.status = status

    def __call__(self, url, headers, payload, timeout):  # noqa: ANN001
        self.calls.append(
            {"url": url, "headers": dict(headers), "payload": copy.deepcopy(payload), "timeout": timeout}
        )
        return self.status, self.body


# ---------------------------------------------------------------------------
# Workflow trigger / permission / owner guard
# ---------------------------------------------------------------------------


def test_workflow_is_workflow_dispatch_only() -> None:
    triggers = _workflow_triggers(_workflow())
    assert list(triggers) == ["workflow_dispatch"]
    assert triggers["workflow_dispatch"] in (None, {})


def test_workflow_permissions_are_contents_read() -> None:
    assert _workflow()["permissions"] == {"contents": "read"}


def test_workflow_has_no_schedule_push_or_pull_request_trigger() -> None:
    triggers = _workflow_triggers(_workflow())
    for forbidden in ("schedule", "push", "pull_request", "pull_request_target", "issue_comment"):
        assert forbidden not in triggers
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    for forbidden in ("schedule:", "cron:", "pull_request:", "pull_request_target:"):
        assert forbidden not in raw


def test_every_job_carries_the_owner_only_guard() -> None:
    jobs = _workflow()["jobs"]
    assert set(jobs) == {"classify-groq", "classify-gemini", "evaluate"}
    for name, job in jobs.items():
        assert job.get("if") == "github.actor == github.repository_owner", name


def test_workflow_declares_the_three_contract_jobs_in_order() -> None:
    jobs = _workflow()["jobs"]
    assert list(jobs) == ["classify-groq", "classify-gemini", "evaluate"]
    assert sorted(jobs["evaluate"]["needs"]) == ["classify-gemini", "classify-groq"]


def test_workflow_avoids_cache_state_telegram_and_pollinations() -> None:
    raw = WORKFLOW_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "actions/cache",
        "cache/restore",
        "cache/save",
        ".state",
        "telegram",
        "pollinations",
        "run_publisher",
        "set -x",
    ):
        assert forbidden not in raw, forbidden


def test_workflow_checkouts_do_not_persist_credentials() -> None:
    for name, job in _workflow()["jobs"].items():
        checkouts = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")]
        assert checkouts, name
        for step in checkouts:
            assert step["with"]["persist-credentials"] is False, name


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", list(FROZEN_SHA256))
def test_frozen_research_inputs_match_expected_digests(path: Path) -> None:
    assert _sha256(path) == FROZEN_SHA256[path]


def test_script_pins_the_same_frozen_digests() -> None:
    assert cal.FROZEN_PROMPT_SHA256 == FROZEN_SHA256[PROMPT_PATH]
    assert cal.FROZEN_CORPUS_SHA256 == FROZEN_SHA256[CORPUS_PATH]
    assert cal.FROZEN_GOLD_SHA256 == FROZEN_SHA256[GOLD_PATH]


def test_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "classification_prompt.txt"
    tampered.write_text(PROMPT_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(cal.FrozenInputError):
        cal.verify_frozen_file(tampered, cal.FROZEN_PROMPT_SHA256, "classification_prompt")
    with pytest.raises(cal.FrozenInputError):
        cal.load_prompt(tampered)
    shutil.copyfile(CORPUS_PATH, tmp_path / "validation_corpus.json")
    with pytest.raises(cal.FrozenInputError):
        cal.run_classification(
            "groq", tampered, tmp_path / "validation_corpus.json", "k", post=_RecordingTransport()
        )


def test_missing_frozen_input_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(cal.FrozenInputError):
        cal.verify_frozen_file(tmp_path / "absent.txt", cal.FROZEN_PROMPT_SHA256, "prompt")


# ---------------------------------------------------------------------------
# Model and endpoint contract
# ---------------------------------------------------------------------------


def test_exactly_two_allowed_model_ids() -> None:
    assert cal.ALLOWED_MODELS == {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL}
    assert cal.GROQ_MODEL == GROQ_MODEL
    assert cal.GEMINI_MODEL == GEMINI_MODEL
    found: set[str] = set()
    pattern = re.compile(r"openai/gpt-oss-\d+b|gemini-[0-9.]+-[a-z]+|llama-[0-9.]+-[a-z0-9-]+")
    for path in SURFACE_FILES:
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert found == {GROQ_MODEL, GEMINI_MODEL}


def test_fallback_model_ids_are_absent_from_the_surface() -> None:
    for path in SURFACE_FILES:
        text = path.read_text(encoding="utf-8")
        for model_id in _forbidden_fallback_model_ids():
            assert model_id not in text, f"{path.name}: {model_id}"


def test_surface_exposes_no_model_or_provider_switch_input() -> None:
    triggers = _workflow_triggers(_workflow())
    assert triggers["workflow_dispatch"] in (None, {})
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "inputs:" not in raw
    for token in ("FALLBACK", "fallback"):
        assert token not in raw


def test_exact_provider_endpoints() -> None:
    assert cal.GROQ_ENDPOINT == GROQ_ENDPOINT
    assert cal.GEMINI_ENDPOINT == GEMINI_ENDPOINT
    assert cal.PROVIDER_ENDPOINTS == {"groq": GROQ_ENDPOINT, "gemini": GEMINI_ENDPOINT}
    urls = set(re.findall(r"https://[A-Za-z0-9./:_-]+", SCRIPT_PATH.read_text(encoding="utf-8")))
    assert urls == {GROQ_ENDPOINT, GEMINI_ENDPOINT}


def test_request_targets_the_exact_endpoint_per_provider() -> None:
    prompt = cal.load_prompt(PROMPT_PATH)
    groq_url, _headers, _payload = cal.build_request("groq", prompt, "текст", "key")
    gemini_url, _gheaders, _gpayload = cal.build_request("gemini", prompt, "текст", "key")
    assert groq_url == GROQ_ENDPOINT
    assert gemini_url == GEMINI_ENDPOINT


# ---------------------------------------------------------------------------
# Isolation from production code
# ---------------------------------------------------------------------------


def test_script_imports_no_production_modules() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if name.split(".")[0] == "src"], sorted(imported)


def test_script_never_references_publisher_or_side_effect_surfaces() -> None:
    code = _code_only(SCRIPT_PATH).lower()
    for forbidden in (
        "run_publisher",
        "src.publisher",
        "src . publisher",
        "src.services",
        "src . services",
        "telegram",
        "pollinations",
        "aiogram",
        "bot_token",
        "chat_id",
        ".state",
        "cache",
    ):
        assert forbidden not in code, forbidden


def test_script_has_no_retry_repair_or_fallback_machinery() -> None:
    code = _code_only(SCRIPT_PATH).lower()
    identifiers = {name.lower() for name in _identifiers(SCRIPT_PATH)}
    for forbidden in ("retry", "retries", "repair", "fallback", "backoff", "reattempt"):
        assert forbidden not in code, forbidden
        assert not [name for name in identifiers if forbidden in name], forbidden


# ---------------------------------------------------------------------------
# Secret isolation
# ---------------------------------------------------------------------------


def test_groq_job_receives_only_the_groq_secret() -> None:
    job = _workflow()["jobs"]["classify-groq"]
    assert _job_secret_names(job) == {"GROQ_API_KEY"}


def test_gemini_job_receives_only_the_gemini_secret() -> None:
    job = _workflow()["jobs"]["classify-gemini"]
    assert _job_secret_names(job) == {"GEMINI_API_KEY"}


def test_evaluator_job_receives_no_provider_secrets() -> None:
    job = _workflow()["jobs"]["evaluate"]
    assert _job_secret_names(job) == set()


def test_evaluator_refuses_to_run_with_a_provider_secret_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "groq.json"
    predictions.write_text(json.dumps(_perfect_predictions("groq")), encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "should-not-be-here")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        cal.main(["evaluate", "--gold", str(GOLD_PATH), "--predictions", str(predictions)])


def test_provider_secret_is_scrubbed_from_persisted_http_errors() -> None:
    fake_key = "offline-placeholder-not-a-credential"
    transport = _RecordingTransport(body=f'{{"error":"bad key {fake_key}"}}', status=401)
    record = cal.classify_item(
        "groq", "prompt", {"id": "v001", "text": "текст"}, fake_key, post=transport
    )
    persisted = json.dumps(record, ensure_ascii=False)
    assert fake_key not in persisted
    assert cal.SECRET_REDACTION in record["detail"]
    assert record["status"] == "failed"
    assert record["failure_reason"] == "http_error"


def test_provider_secret_is_scrubbed_from_persisted_transport_errors() -> None:
    fake_key = "offline-placeholder-not-a-credential"

    def _raising(url, headers, payload, timeout):  # noqa: ANN001
        raise RuntimeError(f"connection failed for key {fake_key}")

    record = cal.classify_item(
        "groq", "prompt", {"id": "v001", "text": "текст"}, fake_key, post=_raising
    )
    assert fake_key not in json.dumps(record, ensure_ascii=False)
    assert record["failure_reason"] == "transport_error"


def test_script_never_logs_headers_or_environment() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "print(headers" not in text
    assert "os.environ)" not in text
    assert "environ.items" not in text
    for match in re.findall(r"print\((.*)\)", text):
        assert "api_key" not in match
        assert "headers" not in match


# ---------------------------------------------------------------------------
# Gold isolation
# ---------------------------------------------------------------------------


def test_classification_entry_point_takes_no_gold_argument() -> None:
    params = set(inspect.signature(cal.run_classification).parameters)
    assert not [p for p in params if "gold" in p]
    assert not [p for p in set(inspect.signature(cal.classify_item).parameters) if "gold" in p]


def test_classifier_refuses_to_run_when_gold_is_reachable(tmp_path: Path) -> None:
    shutil.copyfile(PROMPT_PATH, tmp_path / "classification_prompt.txt")
    shutil.copyfile(CORPUS_PATH, tmp_path / "validation_corpus.json")
    shutil.copyfile(GOLD_PATH, tmp_path / "validation_gold.json")
    transport = _RecordingTransport(body=_groq_body("MODEL_WAIT", "ATTEND_VOCALIZE_REPEAT"))
    with pytest.raises(cal.GoldIsolationError):
        cal.run_classification(
            "groq",
            tmp_path / "classification_prompt.txt",
            tmp_path / "validation_corpus.json",
            "key",
            post=transport,
        )
    assert transport.calls == []


def test_classifier_runs_when_gold_is_absent(tmp_path: Path) -> None:
    shutil.copyfile(PROMPT_PATH, tmp_path / "classification_prompt.txt")
    shutil.copyfile(CORPUS_PATH, tmp_path / "validation_corpus.json")
    transport = _RecordingTransport(body=_groq_body("MODEL_WAIT", "ATTEND_VOCALIZE_REPEAT"))
    payload = cal.run_classification(
        "groq",
        tmp_path / "classification_prompt.txt",
        tmp_path / "validation_corpus.json",
        "key",
        only_ids=["v001"],
        post=transport,
    )
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["status"] == "ok"


def test_classifier_jobs_sparse_checkout_excludes_gold() -> None:
    jobs = _workflow()["jobs"]
    for name in ("classify-groq", "classify-gemini"):
        checkout = next(
            s for s in jobs[name]["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")
        )
        sparse = checkout["with"]["sparse-checkout"]
        assert "validation_gold.json" not in sparse, name
        assert "research/llm_s3/classification_prompt.txt" in sparse, name
        assert "research/llm_s3/validation_corpus.json" in sparse, name
        assert checkout["with"]["sparse-checkout-cone-mode"] is False, name
        steps = yaml.safe_dump(jobs[name], allow_unicode=True)
        assert "--gold" not in steps, name


def test_predictions_are_frozen_before_the_evaluator_sees_gold() -> None:
    jobs = _workflow()["jobs"]
    for name in ("classify-groq", "classify-gemini"):
        uploads = [s for s in jobs[name]["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
        assert uploads, name
        assert uploads[0]["with"]["if-no-files-found"] == "error", name
    evaluate_steps = yaml.safe_dump(jobs["evaluate"], allow_unicode=True)
    assert "download-artifact" in evaluate_steps
    assert "--gold" in evaluate_steps


# ---------------------------------------------------------------------------
# Request shape: one corpus text per isolated request, no repair
# ---------------------------------------------------------------------------


def test_groq_request_carries_exactly_one_corpus_text() -> None:
    prompt = cal.load_prompt(PROMPT_PATH)
    _url, headers, payload = cal.build_request("groq", prompt, "ровно один текст", "key")
    assert payload["model"] == GROQ_MODEL
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": prompt}
    assert payload["messages"][1] == {"role": "user", "content": "ровно один текст"}
    assert headers["Authorization"] == "Bearer key"


def test_gemini_request_carries_exactly_one_corpus_text() -> None:
    prompt = cal.load_prompt(PROMPT_PATH)
    _url, headers, payload = cal.build_request("gemini", prompt, "ровно один текст", "key")
    assert len(payload["contents"]) == 1
    assert len(payload["contents"][0]["parts"]) == 1
    assert payload["contents"][0]["parts"][0]["text"] == "ровно один текст"
    assert payload["systemInstruction"]["parts"][0]["text"] == prompt
    assert headers["x-goog-api-key"] == "key"
    assert "model" not in payload
    assert set(payload) == {"systemInstruction", "contents", "generationConfig"}


def test_gemini_generation_config_is_exactly_the_response_mime_type() -> None:
    """gemini-3.7-flash no longer accepts the legacy sampling controls."""

    prompt = cal.load_prompt(PROMPT_PATH)
    _url, _headers, payload = cal.build_request("gemini", prompt, "текст", "key")
    assert payload["generationConfig"] == {"responseMimeType": "application/json"}


@pytest.mark.parametrize("legacy", ["temperature", "topP", "topK", "top_p", "top_k"])
def test_gemini_request_carries_no_legacy_sampling_control(legacy: str) -> None:
    prompt = cal.load_prompt(PROMPT_PATH)
    _url, _headers, payload = cal.build_request("gemini", prompt, "текст", "key")
    assert legacy not in payload["generationConfig"]
    assert legacy not in payload
    assert legacy not in json.dumps(payload, ensure_ascii=False)


def test_groq_request_keeps_its_own_sampling_control() -> None:
    """Fix A is Gemini-only: the Groq request contract is unchanged."""

    prompt = cal.load_prompt(PROMPT_PATH)
    _url, _headers, payload = cal.build_request("groq", prompt, "текст", "key")
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}


def test_one_request_per_corpus_item_and_no_repair_call(tmp_path: Path) -> None:
    shutil.copyfile(PROMPT_PATH, tmp_path / "classification_prompt.txt")
    shutil.copyfile(CORPUS_PATH, tmp_path / "validation_corpus.json")
    transport = _RecordingTransport(body=_groq_body("MODEL_WAIT", "ATTEND_VOCALIZE_REPEAT"))
    payload = cal.run_classification(
        "groq",
        tmp_path / "classification_prompt.txt",
        tmp_path / "validation_corpus.json",
        "key",
        post=transport,
    )
    corpus = _corpus()
    assert len(transport.calls) == len(corpus) == cal.PRIMARY_ITEM_COUNT
    assert len(payload["predictions"]) == cal.PRIMARY_ITEM_COUNT
    sent_texts = [c["payload"]["messages"][1]["content"] for c in transport.calls]
    assert sent_texts == [item["text"] for item in corpus]
    assert {c["url"] for c in transport.calls} == {GROQ_ENDPOINT}
    assert {c["payload"]["model"] for c in transport.calls} == {GROQ_MODEL}


def test_malformed_response_triggers_no_second_request() -> None:
    transport = _RecordingTransport(body=json.dumps({"choices": [{"message": {"content": "не JSON"}}]}))
    record = cal.classify_item(
        "groq", "prompt", {"id": "v001", "text": "текст"}, "key", post=transport
    )
    assert len(transport.calls) == 1
    assert record["status"] == "failed"


# ---------------------------------------------------------------------------
# Strict output contract
# ---------------------------------------------------------------------------


def test_malformed_json_is_a_classification_failure() -> None:
    for raw in ("не JSON", "[]", "null", '{"interaction_frame": "MODEL_WAIT"}',
                '{"interaction_frame": "MODEL_WAIT", "child_response": "FIND", "note": "x"}',
                '{"interaction_frame": 1, "child_response": "FIND"}'):
        with pytest.raises(cal.ClassificationFailure) as excinfo:
            cal.parse_classification(raw)
        assert excinfo.value.reason == "malformed_json"


def test_invalid_label_is_a_classification_failure() -> None:
    with pytest.raises(cal.ClassificationFailure) as excinfo:
        cal.parse_classification('{"interaction_frame": "WAIT_MODEL", "child_response": "FIND"}')
    assert excinfo.value.reason == "invalid_label"
    with pytest.raises(cal.ClassificationFailure) as excinfo:
        cal.parse_classification('{"interaction_frame": "SEARCH", "child_response": "LOOKING"}')
    assert excinfo.value.reason == "invalid_label"


def test_valid_output_is_accepted() -> None:
    parsed = cal.parse_classification(
        '{"child_response": "ATTEND_VOCALIZE_REPEAT", "interaction_frame": "MODEL_WAIT"}'
    )
    assert parsed == {
        "interaction_frame": "MODEL_WAIT",
        "child_response": "ATTEND_VOCALIZE_REPEAT",
    }


def test_label_space_matches_the_frozen_prompt() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    for label in cal.INTERACTION_FRAMES | cal.CHILD_RESPONSES:
        assert label in prompt, label
    assert len(cal.INTERACTION_FRAMES) == 13
    assert len(cal.CHILD_RESPONSES) == 15


# ---------------------------------------------------------------------------
# Primary scoring contract
# ---------------------------------------------------------------------------


def test_corpus_holds_exactly_124_primary_scoring_items() -> None:
    corpus = _corpus()
    assert len(corpus) == 124 == cal.PRIMARY_ITEM_COUNT
    assert len({item["id"] for item in corpus}) == 124
    assert len(_gold()["primary_scoring_items"]) == 124


def test_model_wait_support_is_36() -> None:
    frames = [item["interaction_frame"] for item in _gold()["primary_scoring_items"]]
    assert frames.count("MODEL_WAIT") == 36 == cal.MODEL_WAIT_SUPPORT


def test_every_required_non_other_frame_has_support_at_least_seven() -> None:
    frames = [item["interaction_frame"] for item in _gold()["primary_scoring_items"]]
    for frame in cal.INTERACTION_FRAMES - {"OTHER"}:
        assert frames.count(frame) >= 7, frame


def test_thirty_paraphrase_groups() -> None:
    groups = _gold()["paraphrase_groups"]
    assert len(groups) == 30 == cal.PARAPHRASE_GROUP_COUNT
    assert len({g["group_id"] for g in groups}) == 30


def test_forty_nine_hard_negative_pairs() -> None:
    pairs = _gold()["hard_negative_pairs"]
    assert len(pairs) == 49 == cal.HARD_NEGATIVE_PAIR_COUNT
    assert len({p["pair_id"] for p in pairs}) == 49
    for pair in pairs:
        gold_a = (pair["gold_a"]["interaction_frame"], pair["gold_a"]["child_response"])
        gold_b = (pair["gold_b"]["interaction_frame"], pair["gold_b"]["child_response"])
        assert gold_a != gold_b, pair["pair_id"]


def test_eleven_ambiguity_ids_are_excluded_from_primary_scoring() -> None:
    gold = _gold()
    ambiguity = gold["non_scoring_ambiguity_ids"]
    assert len(ambiguity) == 11 == cal.AMBIGUITY_ID_COUNT
    scored_ids = {item["id"] for item in gold["primary_scoring_items"]}
    corpus_ids = {item["id"] for item in _corpus()}
    assert not scored_ids & set(ambiguity)
    assert not corpus_ids & set(ambiguity)


def test_evaluator_drops_ambiguous_ids_from_primary_metrics() -> None:
    gold = _gold()
    payload = _perfect_predictions("groq")
    payload["predictions"].append(
        {
            "id": gold["non_scoring_ambiguity_ids"][0],
            "status": "ok",
            "interaction_frame": "OTHER",
            "child_response": "OTHER",
        }
    )
    result = cal.evaluate_provider(gold, payload)
    assert result["scored_items"] == 124
    assert result["excluded_ambiguity_ids"] == [gold["non_scoring_ambiguity_ids"][0]]
    assert result["joint_accuracy"] == 1.0


def test_gold_contract_verification_fails_closed_on_drift() -> None:
    gold = _gold()
    gold["primary_scoring_items"] = gold["primary_scoring_items"][:-1]
    with pytest.raises(cal.FrozenInputError):
        cal.verify_gold_contract(gold)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_perfect_predictions_pass_every_primary_gate() -> None:
    gold = _gold()
    for provider in ("groq", "gemini"):
        result = cal.evaluate_provider(gold, _perfect_predictions(provider))
        assert result["passed"], provider
        assert result["joint_accuracy"] == 1.0
        assert result["model_wait_accuracy"] == 1.0
        assert result["hard_negative_false_positives"] == []
        assert result["paraphrase_groups_evaluated"] == 30
        assert result["paraphrase_group_rate"] == 1.0
        assert result["hard_negative_pairs_evaluated"] == 49
        assert result["unsafe_collisions"] == []
        assert all(result["checks"].values())


def test_thresholds_match_the_preregistered_contract() -> None:
    assert cal.JOINT_ACCURACY_THRESHOLD == 0.95
    assert cal.PER_FRAME_ACCURACY_THRESHOLD == 0.90
    assert cal.PER_FRAME_MIN_SUPPORT == 5
    assert cal.MODEL_WAIT_ACCURACY_THRESHOLD == 0.95
    assert cal.PARAPHRASE_GROUP_THRESHOLD == 0.95
    assert cal.HARD_NEGATIVE_MAX_FALSE_POSITIVES == 0


def test_hard_negative_case_a_perfect_predictions_have_no_false_positive() -> None:
    gold = _gold()
    result = cal.evaluate_provider(gold, _perfect_predictions("groq"))
    assert result["hard_negative_pairs_evaluated"] == 49
    assert result["hard_negative_false_positives"] == []
    assert result["checks"]["hard_negative_false_positives"] is True


def test_hard_negative_case_b_both_predict_gold_a_is_one_false_positive() -> None:
    gold = _gold()
    pair = _isolated_hard_negative_pair(gold)
    sig_a = _gold_signature(pair["gold_a"])
    payload = _perfect_predictions("groq")
    _set_prediction(payload, pair["id_a"], sig_a)
    _set_prediction(payload, pair["id_b"], sig_a)
    result = cal.evaluate_provider(gold, payload)
    assert len(result["hard_negative_false_positives"]) == 1
    entry = result["hard_negative_false_positives"][0]
    assert entry["pair_id"] == pair["pair_id"]
    assert tuple(entry["collapsed_signature"]) == sig_a
    assert result["checks"]["hard_negative_false_positives"] is False
    assert result["passed"] is False


def test_hard_negative_case_c_both_predict_gold_b_is_one_false_positive() -> None:
    gold = _gold()
    pair = _isolated_hard_negative_pair(gold)
    sig_b = _gold_signature(pair["gold_b"])
    payload = _perfect_predictions("groq")
    _set_prediction(payload, pair["id_a"], sig_b)
    _set_prediction(payload, pair["id_b"], sig_b)
    result = cal.evaluate_provider(gold, payload)
    assert len(result["hard_negative_false_positives"]) == 1
    entry = result["hard_negative_false_positives"][0]
    assert entry["pair_id"] == pair["pair_id"]
    assert tuple(entry["collapsed_signature"]) == sig_b
    assert result["checks"]["hard_negative_false_positives"] is False


def test_hard_negative_case_d_third_signature_collapse_is_one_false_positive() -> None:
    """The critical case: a collapse onto a signature that is neither pair gold."""

    gold = _gold()
    pair = _isolated_hard_negative_pair(gold)
    sig_a = _gold_signature(pair["gold_a"])
    sig_b = _gold_signature(pair["gold_b"])
    third = _third_signature(gold, sig_a, sig_b)
    assert third != sig_a and third != sig_b
    payload = _perfect_predictions("groq")
    _set_prediction(payload, pair["id_a"], third)
    _set_prediction(payload, pair["id_b"], third)
    result = cal.evaluate_provider(gold, payload)
    assert len(result["hard_negative_false_positives"]) == 1
    entry = result["hard_negative_false_positives"][0]
    assert entry["pair_id"] == pair["pair_id"]
    assert entry["id_a"] == pair["id_a"]
    assert entry["id_b"] == pair["id_b"]
    assert tuple(entry["collapsed_signature"]) == third
    assert result["checks"]["hard_negative_false_positives"] is False


def test_hard_negative_case_e_swapped_predictions_are_not_a_false_positive() -> None:
    """Swapped predictions stay distinct, so the pair criterion is a true negative."""

    gold = _gold()
    pair = _isolated_hard_negative_pair(gold)
    sig_a = _gold_signature(pair["gold_a"])
    sig_b = _gold_signature(pair["gold_b"])
    payload = _perfect_predictions("groq")
    _set_prediction(payload, pair["id_a"], sig_b)
    _set_prediction(payload, pair["id_b"], sig_a)
    result = cal.evaluate_provider(gold, payload)
    assert result["hard_negative_false_positives"] == []
    assert result["checks"]["hard_negative_false_positives"] is True
    # The swap still costs joint accuracy; only the pair criterion stays clean.
    assert result["joint_accuracy"] < 1.0


def test_hard_negative_criterion_ignores_incomplete_predictions() -> None:
    """Incomplete collapses belong to unsafe_collisions, not to the pair count."""

    gold = _gold()
    pair = _isolated_hard_negative_pair(gold)
    payload = _perfect_predictions("groq")
    _fail_prediction(payload, pair["id_a"])
    _fail_prediction(payload, pair["id_b"])
    result = cal.evaluate_provider(gold, payload)
    assert result["hard_negative_false_positives"] == []
    assert result["hard_negative_pairs_evaluated"] == 49
    assert len(result["unsafe_collisions"]) == 1
    assert sorted(result["unsafe_collisions"][0]["item_ids"]) == sorted([pair["id_a"], pair["id_b"]])


def test_paraphrase_group_failure_is_reported() -> None:
    gold = _gold()
    group = gold["paraphrase_groups"][0]
    payload = _perfect_predictions("groq")
    for record in payload["predictions"]:
        if record["id"] == group["member_ids"][0]:
            record["interaction_frame"] = "OTHER"
            record["child_response"] = "OTHER"
    result = cal.evaluate_provider(gold, payload)
    assert group["group_id"] in result["failed_paraphrase_groups"]
    assert result["paraphrase_group_rate"] < 1.0


def test_unsafe_collision_detection_for_incomplete_signatures() -> None:
    gold = _gold()
    payload = _perfect_predictions("groq")
    items = gold["primary_scoring_items"]
    first = items[0]
    other = next(
        i
        for i in items
        if (i["interaction_frame"], i["child_response"]) != (first["interaction_frame"], first["child_response"])
    )
    # Two distinct gold items collapsing onto one invalid signature.
    payload["predictions"] = [
        {
            "id": r["id"],
            "status": "failed",
            "failure_reason": "malformed_json",
            "detail": "",
            "partial": {"interaction_frame": cal.INVALID_FIELD_MARKER, "child_response": cal.INVALID_FIELD_MARKER},
        }
        if r["id"] in {first["id"], other["id"]}
        else r
        for r in payload["predictions"]
    ]
    result = cal.evaluate_provider(gold, payload)
    assert len(result["unsafe_collisions"]) == 1
    collision = result["unsafe_collisions"][0]
    assert collision["distinct_gold_signatures"] == 2
    assert sorted(collision["item_ids"]) == sorted([first["id"], other["id"]])
    assert result["checks"]["no_unsafe_collisions"] is False


def test_per_frame_gate_applies_to_frames_with_support_at_least_five() -> None:
    gold = _gold()
    result = cal.evaluate_provider(gold, _perfect_predictions("groq"))
    for frame, stats in result["per_frame"].items():
        assert stats["gated"] is (stats["support"] >= 5), frame
        expected = 0.95 if frame == "MODEL_WAIT" else 0.90
        if stats["gated"]:
            assert stats["threshold"] == expected, frame
    assert result["per_frame"]["MODEL_WAIT"]["support"] == 36


def test_partial_corpus_is_rejected_by_the_primary_evaluation() -> None:
    gold = _gold()
    payload = _perfect_predictions("groq")
    payload["predictions"] = payload["predictions"][:20]
    with pytest.raises(cal.FrozenInputError):
        cal.evaluate_provider(gold, payload)
    partial = cal.evaluate_provider(gold, payload, require_full_corpus=False)
    assert partial["scored_items"] == 20


def test_predictions_from_a_foreign_prompt_or_model_are_rejected(tmp_path: Path) -> None:
    for mutation in ("model", "endpoint", "prompt_sha256", "corpus_sha256"):
        payload = _perfect_predictions("groq")
        payload[mutation] = "tampered"
        path = tmp_path / f"{mutation}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(cal.FrozenInputError):
            cal.load_predictions(path)


# ---------------------------------------------------------------------------
# Stability audit support (not executed at this stage)
# ---------------------------------------------------------------------------


def test_stability_support_accepts_twenty_ids_across_three_runs() -> None:
    base = _perfect_predictions("groq")
    subset = base["predictions"][:20]
    runs = []
    for index in range(3):
        payload = dict(base)
        payload["run_label"] = f"run{index + 1}"
        payload["predictions"] = [dict(r) for r in subset]
        runs.append(payload)
    result = cal.evaluate_stability(runs)
    assert result["runs"] == 3
    assert result["ids"] == 20
    assert result["unstable_ids"] == []
    assert result["stable_rate"] == 1.0


def test_stability_support_flags_a_disagreeing_run() -> None:
    base = _perfect_predictions("groq")
    subset = base["predictions"][:20]
    runs = []
    for index in range(3):
        payload = dict(base)
        payload["predictions"] = [dict(r) for r in subset]
        runs.append(payload)
    runs[2]["predictions"][0] = dict(runs[2]["predictions"][0], interaction_frame="OTHER")
    result = cal.evaluate_stability(runs)
    assert result["unstable_ids"] == [subset[0]["id"]]


def test_stability_support_needs_no_corpus_prompt_or_gold_mutation() -> None:
    assert "gold" not in inspect.signature(cal.evaluate_stability).parameters
    assert _sha256(PROMPT_PATH) == FROZEN_SHA256[PROMPT_PATH]
    assert _sha256(CORPUS_PATH) == FROZEN_SHA256[CORPUS_PATH]
    assert _sha256(GOLD_PATH) == FROZEN_SHA256[GOLD_PATH]


# ---------------------------------------------------------------------------
# End-to-end offline pass with fully mocked transport
# ---------------------------------------------------------------------------


def test_full_offline_round_trip_with_mocked_providers(tmp_path: Path) -> None:
    shutil.copyfile(PROMPT_PATH, tmp_path / "classification_prompt.txt")
    shutil.copyfile(CORPUS_PATH, tmp_path / "validation_corpus.json")
    gold_by_id = {i["id"]: i for i in _gold()["primary_scoring_items"]}
    corpus_by_text = {i["text"]: i["id"] for i in _corpus()}

    def _oracle(provider: str):
        def _post(url, headers, payload, timeout):  # noqa: ANN001
            if provider == "groq":
                text = payload["messages"][1]["content"]
            else:
                text = payload["contents"][0]["parts"][0]["text"]
            item = gold_by_id[corpus_by_text[text]]
            body = (
                _groq_body(item["interaction_frame"], item["child_response"])
                if provider == "groq"
                else _gemini_body(item["interaction_frame"], item["child_response"])
            )
            return 200, body

        return _post

    prediction_files = []
    for provider in ("groq", "gemini"):
        payload = cal.run_classification(
            provider,
            tmp_path / "classification_prompt.txt",
            tmp_path / "validation_corpus.json",
            "key",
            post=_oracle(provider),
        )
        path = tmp_path / f"{provider}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        prediction_files.append(str(path))

    report_path = tmp_path / "report.json"
    exit_code = cal.main(
        [
            "evaluate",
            "--gold",
            str(GOLD_PATH),
            "--predictions",
            *prediction_files,
            "--out",
            str(report_path),
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert set(report["providers"]) == {"groq", "gemini"}
    for provider_report in report["providers"].values():
        assert provider_report["scored_items"] == 124
        assert provider_report["joint_accuracy"] == 1.0

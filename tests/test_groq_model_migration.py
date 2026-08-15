from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
GEMINI_PRIMARY_MODEL = "gemini-3.7-flash"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"


def _deprecated_model_ids() -> tuple[str, str]:
    return ("llama-" + "3.3-70b-versatile", "llama-" + "3.1-8b-instant")


def _tracked_runtime_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    exts = {".py", ".yml", ".yaml", ".md"}
    names = {".env.example", ".env"}
    files: list[Path] = []
    for rel in proc.stdout.splitlines():
        path = ROOT / rel
        if path.suffix in exts or path.name in names:
            files.append(path)
    return files


def test_no_deprecated_model_ids_in_runtime_files() -> None:
    offenders: list[str] = []
    deprecated = _deprecated_model_ids()
    for path in _tracked_runtime_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for model_id in deprecated:
            if model_id in text:
                offenders.append(str(path.relative_to(ROOT)))
                break
    assert not offenders, "deprecated Groq model IDs remain in: " + ", ".join(offenders)


def _import_llm_fresh():
    for name in (
        "GROQ_MODEL",
        "GROQ_FALLBACK_MODEL",
        "GROQ_MODELS",
        "GEMINI_MODEL",
        "GEMINI_FALLBACK_MODEL",
        "GEMINI_MODELS",
    ):
        os.environ.pop(name, None)
    sys.path.insert(0, str(ROOT))
    sys.modules.pop("src.services.llm_generator", None)
    services_package = sys.modules.get("src.services")
    if services_package is not None and hasattr(services_package, "llm_generator"):
        delattr(services_package, "llm_generator")
    from src.services import llm_generator  # type: ignore

    return llm_generator


class _FakeResponse:
    def __init__(self, status_code: int, text: str, content: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self._content = content

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self) -> None:
        raise RuntimeError(f"{self.status_code}: {self.text}")


class _FakeGeminiResponse(_FakeResponse):
    def json(self) -> dict:
        return {"candidates": [{"content": {"parts": [{"text": self._content}]}}]}


def test_default_groq_model_config() -> None:
    llm = _import_llm_fresh()
    assert llm.GROQ_MODEL == PRIMARY_MODEL
    assert llm.GROQ_FALLBACK_MODEL == FALLBACK_MODEL
    assert llm.GROQ_MODELS == [PRIMARY_MODEL, FALLBACK_MODEL]


def test_default_gemini_model_config() -> None:
    llm = _import_llm_fresh()
    assert llm.GEMINI_MODEL == GEMINI_PRIMARY_MODEL
    assert llm.GEMINI_FALLBACK_MODEL == GEMINI_FALLBACK_MODEL
    assert llm.GEMINI_MODELS == [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL]


def test_gemini_primary_unavailable_attempts_fallback_without_legacy_sampling() -> None:
    llm = _import_llm_fresh()
    calls: list[tuple[str, dict]] = []

    async def fake_post_json(url: str, headers: dict, payload: dict, timeout: int = 70):
        model = url.split("/models/", 1)[1].split(":", 1)[0]
        calls.append((model, payload))
        if model == GEMINI_PRIMARY_MODEL:
            return _FakeGeminiResponse(404, "model not found")
        return _FakeGeminiResponse(200, "ok", "Готовый текст")

    llm.LLM_MAX_RETRIES = 1
    llm.LLM_CALL_DELAY_SEC = 0
    llm._next_allowed_ts = 0.0
    llm._post_json = fake_post_json

    text = asyncio.run(llm.gemini_generate("prompt", "test-key"))

    assert [model for model, _payload in calls] == [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL]
    assert text == "Готовый текст"
    for _model, payload in calls:
        assert set(payload) == {"contents"}
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload


def test_workflow_uses_gemini_37_with_25_fallback() -> None:
    workflow = (ROOT / ".github/workflows/post.yml").read_text(encoding="utf-8")
    assert 'GEMINI_MODEL: "gemini-3.7-flash"' in workflow
    assert 'GEMINI_FALLBACK_MODEL: "gemini-2.5-flash"' in workflow
    assert 'GEMINI_MODELS: "gemini-3.7-flash,gemini-2.5-flash"' in workflow
    assert 'GEMINI_VISUAL_QA_MODEL: "gemini-3.7-flash"' in workflow


def test_primary_failure_attempts_fallback_model() -> None:
    llm = _import_llm_fresh()
    calls: list[str] = []

    async def fake_post_json(url: str, headers: dict, payload: dict, timeout: int = 70):
        calls.append(payload["model"])
        if payload["model"] == PRIMARY_MODEL:
            return _FakeResponse(503, "temporarily unavailable")
        return _FakeResponse(200, "ok", "Готовый текст")

    llm.LLM_MAX_RETRIES = 1
    llm.LLM_CALL_DELAY_SEC = 0
    llm._next_allowed_ts = 0.0
    llm._post_json = fake_post_json

    text = asyncio.run(llm.groq_chat("prompt", "test-key"))

    assert calls == [PRIMARY_MODEL, FALLBACK_MODEL]
    assert text == "Готовый текст"


def test_both_groq_models_fail_without_crashing_generation() -> None:
    llm = _import_llm_fresh()

    async def failing_groq_chat(prompt: str, api_key: str) -> str:
        raise RuntimeError("groq_failed_after_fallbacks: forced")

    llm.groq_chat = failing_groq_chat
    evidence = (
        "This source explains a practical speech-language activity for parents. "
        "It includes age guidance, simple materials, short steps, and an observable result. "
        "The activity can be done at home without pressure and should preserve a calm tone. "
        "The source also includes enough detail to build a concise Telegram post. "
    )

    out, ok, note = asyncio.run(
        llm.generate_post_plain_from_evidence_async(
            rubric_title="Совет логопеда дня",
            rubric_format="tip_of_day",
            audience="parents",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=evidence,
            disclaimer="",
            hashtags=["#логопед"],
            provider="groq",
            groq_key="test-key",
            gemini_key="",
            max_chars=1000,
            day_key="MO",
        )
    )

    assert isinstance(out, str)
    assert out == ""
    assert ok is False
    assert isinstance(note, str)
    assert "groq_failed" in note


def test_gemini_quota_error_disables_followup_http_requests() -> None:
    llm = _import_llm_fresh()
    calls = 0

    async def quota_response(url: str, headers: dict, payload: dict, timeout: int = 70):
        nonlocal calls
        calls += 1
        return _FakeResponse(429, '{"error":{"message":"quota exhausted"}}')

    llm._post_json = quota_response
    llm.LLM_MAX_RETRIES = 3
    llm.LLM_CALL_DELAY_SEC = 0
    llm._next_allowed_ts = 0.0

    with pytest.raises(RuntimeError, match="gemini_quota_exhausted"):
        asyncio.run(llm.gemini_generate("prompt", "test-key"))
    assert calls == 1
    assert llm.gemini_text_provider_status("test-key") == "quota_exhausted"

    with pytest.raises(RuntimeError, match="gemini_quota_exhausted_cached"):
        asyncio.run(llm.gemini_generate("prompt two", "test-key"))
    assert calls == 1

    fresh = _import_llm_fresh()
    assert fresh.gemini_text_provider_status("test-key") == "available"


def test_gemini_temporary_503_does_not_trip_quota_breaker() -> None:
    llm = _import_llm_fresh()

    async def temporary_response(url: str, headers: dict, payload: dict, timeout: int = 70):
        return _FakeResponse(503, "temporarily unavailable")

    llm._post_json = temporary_response
    llm.LLM_MAX_RETRIES = 1
    llm.LLM_CALL_DELAY_SEC = 0
    llm._next_allowed_ts = 0.0

    with pytest.raises(RuntimeError):
        asyncio.run(llm.gemini_generate("prompt", "test-key"))
    assert llm.gemini_text_provider_status("test-key") == "available"


def main() -> None:
    tests = [
        test_no_deprecated_model_ids_in_runtime_files,
        test_default_groq_model_config,
        test_default_gemini_model_config,
        test_gemini_primary_unavailable_attempts_fallback_without_legacy_sampling,
        test_workflow_uses_gemini_37_with_25_fallback,
        test_primary_failure_attempts_fallback_model,
        test_both_groq_models_fail_without_crashing_generation,
    ]
    for test in tests:
        test()
    print("PASS groq_model_migration")


if __name__ == "__main__":
    main()

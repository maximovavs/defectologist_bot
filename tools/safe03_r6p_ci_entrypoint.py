"""Candidate offline CI entrypoint; no real networking / providers / Telegram.

Runs existing immutable SAFE03 suites plus seam tests and selected publisher
regressions in a single offline process. Safe even with unusable API keys.
"""
from __future__ import annotations
import ast
from pathlib import Path
import os
import socket
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for key in ("GEMINI_API_KEY", "GEMINI_VISUAL_QA_API_KEY", "GROQ_API_KEY",
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_DRAFTS_CHAT_ID", "POLLINATIONS_TOKEN"):
    os.environ[key] = ""
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ[key] = "http://127.0.0.1:9"
os.environ["NO_PROXY"] = ""

# Force all socket-level attempts to fail, even if a future test accidentally
# reaches an external provider or Telegram codepath.
network_attempts = []
def _deny(*a, **kw):
    network_attempts.append("network_attempt_blocked")
    raise AssertionError("R6P: network access forbidden in offline CI")
socket.socket.connect = _deny
socket.socket.connect_ex = _deny
socket.create_connection = _deny
socket.getaddrinfo = _deny

# Mirror canonical policy workflow's unavailable semantic model: fail-on-use.
semantic = types.ModuleType("sentence_transformers")
class _UnavailableModel:
    def __init__(self, *a, **kw):
        raise RuntimeError("semantic model unavailable offline")
class _UnavailableUtil:
    @staticmethod
    def pytorch_cos_sim(*a, **kw):
        raise RuntimeError("semantic model unavailable offline")
semantic.SentenceTransformer = _UnavailableModel
semantic.util = _UnavailableUtil()
sys.modules["sentence_transformers"] = semantic

# This container lacks feedparser although the real GitHub CI candidate installs
# it. A fail-on-use stand-in permits source-level seam imports WITHOUT silently
# pretending feed parsing was tested here.
try:
    import feedparser  # noqa: F401
except ImportError:
    feedparser = types.ModuleType("feedparser")
    def _unavailable_feed(*args, **kwargs):
        raise RuntimeError("feedparser unavailable in local runtime; CI will install it")
    feedparser.parse = _unavailable_feed
    sys.modules["feedparser"] = feedparser

# Python 3.11 grammar compatibility check (not proof of 3.11 runtime behavior).
for folder in ("src", "tests", "tools"):
    for path in (ROOT / folder).rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))

TEST_MODULES = (
    "tests.test_safe03_r6d_offline_acceptance",
    "tests.test_safe03_r6g_optin_offline",
    "tests.test_safe03_r6g_production_guard_parity",
    "tests.test_safe03_r6h_contract_offline",
    "tests.test_safe03_r6n_warning_acceptance_offline",
    "tests.test_safe03_r6p_seam_policy",
    "tests.test_safe03_r6q_source_sender_offline",
    "tests.test_safe03_r6r_transaction_offline",
    "tests.test_safe03_r6s_joint_reservation_offline",
    "tests.test_safe03_r6t_publisher_seam_offline",
    "tests.test_safe03_r6u_unified_lifecycle_offline",
    "tests.test_safe03_r6v_hook_handoff_offline",
    "tests.test_visual_fallback_policy",
    "tests.test_visual_prompt_policy",
    "tests.test_publication_delivery_state",
    "tests.test_telegram_visual_delivery",
    "tests.test_content_grounding",
    "tests.test_parent_age_policy",
    "tests.test_parent_hearing_inference",
    "tests.test_parent_oral_safety",
)
suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(f"R6P offline test_count={result.testsRun} failures={len(result.failures)} "
      f"errors={len(result.errors)} network_attempts={len(network_attempts)}", flush=True)
if not result.wasSuccessful() or network_attempts:
    raise SystemExit(1)

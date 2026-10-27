"""OFFLINE-ONLY candidate for SAFE03 checks; NOT registered in any GitHub workflow.

Usage: python tools/safe03_r6h_ci_candidate.py safe03|canonical|all
No providers, Telegram calls, or network connections permitted.
Python 3.11 compatibility must still be confirmed by actual CI.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import re
import socket
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
connect_attempts = 0


def deny(*_args, **_kwargs):
    global connect_attempts
    connect_attempts += 1
    raise AssertionError('SAFE03_R6H_NO_NETWORK')


for item in ('connect','connect_ex'):
    setattr(socket.socket, item, deny)
for item in ('create_connection','getaddrinfo'):
    setattr(socket, item, deny)

fake = types.ModuleType('sentence_transformers')
class DeniedModel:
    def __init__(self,*_a,**_kw): raise RuntimeError('offline_semantic_model_disabled')
class DeniedUtil:
    @staticmethod
    def pytorch_cos_sim(*_a,**_kw): raise RuntimeError('offline_semantic_model_disabled')
fake.SentenceTransformer=DeniedModel
fake.util=DeniedUtil()
sys.modules['sentence_transformers']=fake
if importlib.util.find_spec('feedparser') is None:
    feed=types.ModuleType('feedparser')
    feed.parse=deny
    sys.modules['feedparser']=feed

for name in ('GROQ_API_KEY','GEMINI_API_KEY','GEMINI_VISUAL_QA_API_KEY',
             'TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID','TELEGRAM_DRAFTS_CHAT_ID',
             'POLLINATIONS_TOKEN'):
    os.environ[name]=''
for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    os.environ[name]='http://127.0.0.1:9'

mode=sys.argv[1] if len(sys.argv)>1 else 'safe03'
if mode == 'safe03':
    names=[
       'tests.test_safe03_r6d_offline_acceptance',
       'tests.test_safe03_r6g_optin_offline',
       'tests.test_safe03_r6g_production_guard_parity',
       'tests.test_safe03_r6h_contract_offline',
    ]
elif mode == 'canonical':
    content=(ROOT/'.github/workflows/publisher_policy_pr_checks.yml').read_text()
    region=content.split('loadTestsFromNames([',1)[1].split('])',1)[0]
    names=re.findall(r'"(tests\.test_[a-z0-9_]+)"',region)
    assert len(names)>=35
elif mode == 'all':
    import pytest
    result=pytest.main(['-q','tests/'])
    print('SAFE03_OFFLINE_SOCKET_ATTEMPTS',connect_attempts)
    sys.exit(int(result!=0 or connect_attempts!=0))
else:
    raise SystemExit('usage: safe03|canonical|all')

suite=unittest.defaultTestLoader.loadTestsFromNames(names)
res=unittest.TextTestRunner(verbosity=1).run(suite)
print('SAFE03_OFFLINE_SOCKET_ATTEMPTS',connect_attempts)
sys.exit(0 if res.wasSuccessful() and connect_attempts==0 else 1)

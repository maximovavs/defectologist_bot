"""Offline deadline contracts. No real clock waits or HTTP calls."""
import asyncio
import contextlib
import io
import time
import types
import unittest
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.services import llm_generator as llm


class BudgetTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 0.0
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(llm, 'time', types.SimpleNamespace(monotonic=lambda: self.now, time=time.time)))
        self.stack.enter_context(patch.object(llm.requests, 'post', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch.object(llm, '_gemini_quota_exhausted', False))
        self.stack.enter_context(patch.object(llm, '_gemini_region_blocked', False))
        self.stack.enter_context(patch.object(llm, '_throttle', AsyncMock()))
        self.stack.enter_context(patch.object(llm.random, 'uniform', side_effect=lambda a, b: a))
        # Explicit deployment settings; behavior tests never depend on shell env.
        for name, value in {
            'LLM_MAX_RETRIES': 3, 'LLM_CALL_DELAY_SEC': 1.5,
            'LLM_BACKOFF_MIN': 8, 'LLM_BACKOFF_MAX': 45,
            'GROQ_MODELS': ['openai/gpt-oss-120b', 'openai/gpt-oss-20b'],
            'GEMINI_MODELS': ['gemini-3.7-flash', 'gemini-2.5-flash'],
        }.items():
            self.stack.enter_context(patch.object(llm, name, value))
        self.logs = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.logs))
        token = llm._TEXT_BUDGET.set(None)
        self.addCleanup(llm._TEXT_BUDGET.reset, token)

    def budget(self, provider):
        b = llm._TextBudget(175.0)
        b.enter(provider)
        llm._TEXT_BUDGET.set(b)
        return b

    @staticmethod
    def response(status=200, text='temporary', output='valid'):
        return types.SimpleNamespace(status_code=status, text=text, json=lambda: {
            'choices': [{'message': {'content': output}}],
            'candidates': [{'content': {'parts': [{'text': output}]}}]},
            raise_for_status=lambda: None)

    async def post(self, provider='auto'):
        return await llm.generate_post_plain_from_evidence_async(
            'Title', 'tip_of_day', 'parents', '', 'example.org', 'https://example.org/a',
            'Evidence ' * 50, '', [], provider, 'fake-groq', 'fake-gemini', 3000, day_key='MO')

    def validation(self, side_effect):
        return self.stack.enter_context(patch.object(llm, '_validate_output', side_effect=side_effect))

    async def test_slow_groq_suppresses_backoff_and_preserves_gemini_phase(self):
        self.budget('groq')
        async def transport(*args, **kwargs):
            self.assertEqual(kwargs['timeout'], 80.0)
            self.now = 80
            return self.response(429, 'rate limit')
        http = self.stack.enter_context(patch.object(llm, '_post_json', side_effect=transport))
        sleep = self.stack.enter_context(patch.object(llm.asyncio, 'sleep', AsyncMock()))
        with self.assertRaisesRegex(llm._TextBudgetExceeded, 'previous=.*429'):
            await llm.groq_chat('private', 'fake')
        self.assertEqual(http.await_count, 1)
        sleep.assert_not_awaited()
        b = llm._TEXT_BUDGET.get()
        b.enter('gemini')
        self.assertEqual(b.remaining(), 90)
        self.assertIn('backoff_suppressed reason=deadline', self.logs.getvalue())

    async def test_gemini_reserve_suppresses_primary_retry_and_runs_fallback(self):
        self.budget('gemini')
        models = []
        async def transport(url, headers, payload, timeout):
            models.append(url.split('/models/')[1].split(':')[0])
            if len(models) == 1:
                self.assertEqual(timeout, 60)
                self.now = 55
                return self.response(503)
            self.assertEqual(timeout, 35)
            return self.response()
        self.stack.enter_context(patch.object(llm, '_post_json', side_effect=transport))
        self.assertEqual(await llm.gemini_generate('private', 'fake'), 'valid')
        self.assertEqual(models, llm.GEMINI_MODELS)
        self.assertIn('reason=30s_reserve', self.logs.getvalue())

    async def test_primary_elapsed_timeout_advances_to_fallback(self):
        self.budget('gemini')
        async def transport(url, headers, payload, timeout):
            if llm.GEMINI_MODELS[0] in url:
                self.now = 60
                raise asyncio.TimeoutError()
            return self.response()
        http = self.stack.enter_context(patch.object(llm, '_post_json', side_effect=transport))
        self.assertEqual(await llm.gemini_generate('private', 'fake'), 'valid')
        self.assertEqual(http.await_count, 2)

    async def test_repair_suppressed_at_each_provider_deadline(self):
        for provider, deadline in [('groq', 85), ('gemini', 90)]:
            self.now = 0
            self.budget(provider)
            self.now = deadline
            with patch.object(llm, 'groq_chat', AsyncMock()) as groq, patch.object(llm, 'gemini_generate', AsyncMock()) as gemini:
                with self.assertRaises(llm._TextBudgetExceeded):
                    await llm._text_provider_call(provider, 'private', 'fake', repair=True)
                groq.assert_not_awaited()
                gemini.assert_not_awaited()
        self.assertIn('repair_suppressed reason=deadline', self.logs.getvalue())

    async def test_fast_valid_auto_unchanged_and_context_reset(self):
        validate = self.validation(lambda *a, **k: (True, 'ok'))
        with patch.object(llm, 'groq_chat', AsyncMock(return_value='Valid output')) as groq, patch.object(llm, 'gemini_generate', AsyncMock()) as gemini:
            result = await self.post()
            self.assertTrue(result[1])
            self.assertEqual(result[2], 'ok:groq')
            self.assertIn('Valid output', result[0])
            self.assertEqual(groq.await_count, 1)
            gemini.assert_not_awaited()
            self.assertEqual(validate.call_count, 1)
        self.assertIsNone(llm._TEXT_BUDGET.get())

    async def test_initial_and_repair_share_phase_then_auto_falls_back(self):
        self.validation(lambda *a, **k: (False, 'too_short'))
        calls = []
        async def groq(prompt, key):
            calls.append(llm._TEXT_BUDGET.get().phase_deadline)
            if len(calls) == 1:
                self.now = 84
                return 'Invalid'
            self.now = 85
            raise llm._TextBudgetExceeded('groq_phase_deadline')
        with patch.object(llm, 'groq_chat', side_effect=groq), patch.object(llm, 'gemini_generate', AsyncMock(return_value='Invalid')) as gemini:
            out, ok, note = await self.post()
            self.assertFalse(ok)
            self.assertEqual(out, '')
            self.assertEqual(calls, [85, 85])
            self.assertGreater(gemini.await_count, 0)
        self.assertIsNone(llm._TEXT_BUDGET.get())

    async def test_explicit_groq_never_gains_gemini_fallback_or_budget(self):
        async def groq(*args):
            self.assertIsNone(llm._TEXT_BUDGET.get())
            raise RuntimeError('transport')
        with patch.object(llm, 'groq_chat', side_effect=groq), patch.object(llm, 'gemini_generate', AsyncMock()) as gemini:
            self.assertFalse((await self.post('groq'))[1])
            gemini.assert_not_awaited()

    async def test_explicit_gemini_has_no_budget(self):
        self.validation(lambda *a, **k: (True, 'ok'))
        async def gemini(*args):
            self.assertIsNone(llm._TEXT_BUDGET.get())
            return 'Valid'
        with patch.object(llm, 'gemini_generate', side_effect=gemini):
            self.assertTrue((await self.post('gemini'))[1])

    async def test_p2d_failure_still_blocks_provider_fallback(self):
        self.budget('gemini')
        tokens = [(v, v.set(value)) for v, value in [
            (llm._P2D_REQUESTED_PROVIDER, 'auto'),
            (llm._P2D_FAIL_REASON, 'exercise_coherence_violation'),
            (llm._P2D_FAIL_ORIGIN_PROVIDER, 'groq')]]
        try:
            with patch.object(llm, '_P2D_GEMINI_GENERATE_BASE', AsyncMock()) as base:
                with self.assertRaisesRegex(RuntimeError, 'p2d_provider_fallback_blocked'):
                    await llm._text_provider_call('gemini', 'private', 'fake')
                base.assert_not_awaited()
        finally:
            for var, token in reversed(tokens):
                var.reset(token)

    async def test_quota_cache_and_region_block_preserved(self):
        for status, text, flag in [(429, 'quota exceeded', '_gemini_quota_exhausted'), (400, 'User location is not supported', '_gemini_region_blocked')]:
            self.now = 0
            self.budget('gemini')
            with patch.object(llm, flag, False), patch.object(llm, '_post_json', AsyncMock(return_value=self.response(status, text))) as http:
                with self.assertRaises(RuntimeError):
                    await llm.gemini_generate('private', 'fake')
                with self.assertRaises(RuntimeError):
                    await llm.gemini_generate('private', 'fake')
                self.assertEqual(http.await_count, 1)

    async def test_no_transport_when_budget_zero(self):
        self.budget('groq')
        self.now = 85
        with patch.object(llm, '_post_json', AsyncMock()) as http:
            with self.assertRaises(llm._TextBudgetExceeded):
                await llm.groq_chat('private', 'fake')
            http.assert_not_awaited()

    async def test_image_provider_after_text_does_not_inherit_budget(self):
        self.validation(lambda *a, **k: (True, 'ok'))
        with patch.object(llm, 'groq_chat', AsyncMock(return_value='Valid')):
            await self.post()
        with patch.object(llm, '_post_json', AsyncMock(return_value=self.response())) as http:
            await llm.groq_chat('image prompt', 'fake')
            self.assertEqual(http.call_args.kwargs['timeout'], 80)
            self.assertIsNone(llm._TEXT_BUDGET.get())

    async def test_cancellation_resets_budget(self):
        with patch.object(llm, 'groq_chat', AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await self.post()
        self.assertIsNone(llm._TEXT_BUDGET.get())

    def test_candidate_safe_ceiling_and_gemini_reserve(self):
        b = self.budget('groq')
        self.now = 100
        b.enter('gemini')
        self.assertEqual(b.remaining(), 75)
        self.assertEqual(b.remaining(llm.GEMINI_MODELS[0]), 45)
        self.assertEqual(b.remaining(llm.GEMINI_MODELS[1]), 75)

    def test_frozen_outer_backstop_models_and_constants(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / '.github/workflows/post.yml').read_text()
        publisher = (root / 'src/publisher/run_publisher.py').read_text()
        self.assertIn('asyncio.wait_for(', publisher)
        self.assertIn('timeout=MAX_LLM_SECONDS_PER_CANDIDATE', publisher)
        for text in ['MAX_LLM_SECONDS_PER_CANDIDATE: "180"', 'MAX_RUN_SECONDS: "1500"', 'LLM_MAX_RETRIES: "3"']:
            self.assertIn(text, workflow)
        self.assertEqual(llm.GROQ_MODELS, ['openai/gpt-oss-120b', 'openai/gpt-oss-20b'])
        self.assertEqual(llm.GEMINI_MODELS, ['gemini-3.7-flash', 'gemini-2.5-flash'])
        deployment = yaml.safe_load(workflow)
        env = next(job['env'] for job in deployment['jobs'].values() if 'env' in job)
        expected = {
            'LLM_MAX_RETRIES': '3', 'LLM_CALL_DELAY_SEC': '1.5',
            'LLM_BACKOFF_MIN': '8', 'LLM_BACKOFF_MAX': '45',
            'MAX_RUN_SECONDS': '1500', 'MAX_LLM_SECONDS_PER_CANDIDATE': '180',
            'GROQ_MODEL': 'openai/gpt-oss-120b', 'GROQ_FALLBACK_MODEL': 'openai/gpt-oss-20b',
            'GEMINI_MODEL': 'gemini-3.7-flash', 'GEMINI_FALLBACK_MODEL': 'gemini-2.5-flash',
            'GEMINI_MODELS': 'gemini-3.7-flash,gemini-2.5-flash',
        }
        for name, value in expected.items():
            self.assertEqual(env[name], value, name)

    async def test_provider_context_updates_reach_validation(self):
        # Real P2D provider wrapper, mocked transport only: regression for Task isolation.
        self.validation(lambda *a, **k: (False, 'exercise_coherence_violation'))
        def validation(*args, **kwargs):
            self.assertEqual(llm._P2D_LAST_TEXT_PROVIDER.get(), 'groq')
            self.assertEqual(llm._P2D_LAST_RAW_OUTPUT.get(), 'Invalid exercise')
            llm._P2D_FAIL_REASON.set('exercise_coherence_violation')
            llm._P2D_FAIL_ORIGIN_PROVIDER.set('groq')
            return False, 'exercise_coherence_violation'
        with patch.object(llm, '_validate_output', side_effect=validation), \
             patch.object(llm, '_P2D_GROQ_CHAT_BASE', AsyncMock(return_value='Invalid exercise')), \
             patch.object(llm, '_P2D_GEMINI_GENERATE_BASE', AsyncMock()) as gemini:
            text, ok, note = await self.post()
            self.assertFalse(ok)
            self.assertEqual(text, '')
            self.assertIn('p2d_fail_closed:exercise_coherence_violation', note)
            gemini.assert_not_awaited()

    async def test_real_image_entrypoint_after_auto_has_no_budget(self):
        self.validation(lambda *a, **k: (True, 'ok'))
        with patch.object(llm, 'groq_chat', AsyncMock(return_value='Valid')):
            await self.post()
        async def image_provider(*args):
            self.assertIsNone(llm._TEXT_BUDGET.get())
            return '{"action":"look","setting":"room","props":[]}'
        with patch.object(llm, 'groq_chat', side_effect=image_provider) as provider:
            await llm.generate_image_prompt_async('Title', 'Body', 'parents', 'groq', 'fake', '')
            self.assertGreater(provider.call_count, 0)

    async def test_requests_timeout_preserves_gemini_fallback(self):
        self.budget('gemini')
        async def transport(url, headers, payload, timeout):
            if llm.GEMINI_MODELS[0] in url:
                self.now = 60
                raise llm.requests.Timeout('mocked transport timeout')
            self.assertEqual(timeout, 30)
            return self.response()
        with patch.object(llm, '_post_json', side_effect=transport) as http:
            self.assertEqual(await llm.gemini_generate('private', 'fake'), 'valid')
            self.assertEqual(http.await_count, 2)

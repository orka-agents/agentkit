"""Provider bookkeeping tests use stub modules and no model or MCP dependency."""

import asyncio
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class MCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self):
        return lambda function: function

    def custom_route(self, *args, **kwargs):
        return lambda function: function


class JSONResponse:
    def __init__(self, data, status_code=200):
        self.data, self.status_code = data, status_code


class Request:
    async def json(self):
        return {"model": "qwen-3.5-2b", "messages": [], "stream": False}


class HTTPError(Exception):
    pass


class Provider:
    def __init__(self, *, result=None, stalled=False, failure=False):
        self.result, self.stalled, self.failure = result, stalled, failure
        self.started = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, *, json):
        self.started.set()
        if self.stalled:
            await asyncio.Event().wait()
        if self.failure:
            raise HTTPError("transport failure")
        return types.SimpleNamespace(status_code=200, json=lambda: self.result)


def load_fixture(provider):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "httpx",
            "mcp",
            "mcp.server",
            "mcp.server.fastmcp",
            "starlette",
            "starlette.requests",
            "starlette.responses",
        )
    }
    modules["httpx"].HTTPError = HTTPError
    modules["httpx"].AsyncClient = lambda **kwargs: provider
    modules["mcp.server.fastmcp"].FastMCP = MCP
    modules["starlette.requests"].Request = Request
    modules["starlette.responses"].JSONResponse = JSONResponse
    spec = importlib.util.spec_from_file_location("eval_fixture", Path(__file__).with_name("fixture.py"))
    fixture = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {"EVAL_CASE_ID": "stock", "EVAL_TRIAL": "1"}):
        spec.loader.exec_module(fixture)
    return fixture


class ProviderAccounting(unittest.IsolatedAsyncioTestCase):
    async def test_valid_inference_counts_completed_response(self):
        fixture = load_fixture(
            Provider(result={"choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 4}})
        )
        response = await fixture.model(Request())
        self.assertEqual(response.status_code, 200)
        state = fixture.world.snapshot()
        self.assertEqual(state["providerRequests"], 1)
        self.assertEqual(state["providerCompletions"], 1)
        self.assertEqual(state["providerInflight"], 0)
        self.assertEqual(state["providerFailures"], 0)
        self.assertEqual(state["usage"]["total_tokens"], 4)
        self.assertFalse(fixture.provider_tasks)

    async def test_settle_accounts_for_stalled_first_inference(self):
        provider = Provider(stalled=True)
        fixture = load_fixture(provider)
        task = asyncio.create_task(fixture.model(Request()))
        await provider.started.wait()
        self.assertEqual(fixture.world.snapshot()["providerInflight"], 1)
        response = await fixture.settle(Request())
        self.assertEqual(response.data["providerRequests"], 1)
        self.assertEqual(response.data["providerCompletions"], 0)
        self.assertEqual(response.data["providerInflight"], 0)
        self.assertEqual(response.data["providerCancelled"], 1)
        self.assertEqual((await task).status_code, 499)
        self.assertFalse(fixture.provider_tasks)

    async def test_transport_failure_does_not_count_inference(self):
        fixture = load_fixture(Provider(failure=True))
        self.assertEqual((await fixture.model(Request())).status_code, 502)
        state = fixture.world.snapshot()
        self.assertEqual(state["providerCompletions"], 0)
        self.assertEqual(state["providerFailures"], 1)
        self.assertEqual(state["providerInflight"], 0)

    async def test_malformed_tool_calls_never_credit_completion(self):
        for calls in (
            False,
            0,
            "",
            {},
            [None],
            [{"function": None}],
            [{"function": {"name": "evals_get_stock", "arguments": {}}}],
            {"function": {}},
        ):
            with self.subTest(calls=calls):
                fixture = load_fixture(
                    Provider(result={"choices": [{"message": {"content": None, "tool_calls": calls}}]})
                )
                await fixture.model(Request())
                state = fixture.world.snapshot()
                self.assertEqual(state["providerCompletions"], 0)
                self.assertEqual(state["providerFailures"], 1)
                self.assertEqual(state["providerInflight"], 0)

    async def test_bad_protocol_response_does_not_count_inference(self):
        fixture = load_fixture(Provider(result={"not": "a completion"}))
        await fixture.model(Request())
        self.assertEqual(fixture.world.snapshot()["providerCompletions"], 0)
        self.assertEqual(fixture.world.snapshot()["providerFailures"], 1)


if __name__ == "__main__":
    unittest.main()

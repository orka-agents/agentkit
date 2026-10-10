from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from agentkit_serve_common.acp import _discard_runtime_session
from agentkit_serve_common.adapter_support import AsyncExitStackLifecycle, resolve_model_api
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.model_api_auto import (
    AutoModelAPIState, AutoModelRuntime, ModelAPIFallback, is_unsupported_responses_response,
)
from agentkit_serve_common.runtime import AgentRunError, RunResult


@pytest.mark.parametrize(("status", "body"), [
    (404, "404 page not found\n"), (404, {"detail": "Not Found"}),
    (404, {"error": "endpoint unsupported"}), (405, {"detail": "Method Not Allowed"}),
    (501, "Not Implemented"), (400, {"error": {"code": "unsupported_model_api"}}),
    (404, b'{"error":{"code":"unsupported_endpoint"}}'),
    (404, {"code": "unsupported_endpoint", "message": "Responses not supported"}),
])
def test_recognizable_endpoint_rejections_allow_auto_fallback(status, body):
    assert is_unsupported_responses_response(status, body)


@pytest.mark.parametrize(("status", "body"), [
    (404, None), (404, "model not found"), (404, {"error": {"code": "model_not_found"}}),
    (404, {"error": {"code": "DeploymentNotFound"}}), (404, {"error": {"message": "Not Found"}}),
    (400, {"detail": "Bad Request"}), (400, {"error": {"code": "unsupported_parameter"}}),
    (400, {"error": {"code": "invalid_api_key"}}), (200, {"detail": "Not Found"}),
    (401, {"error": {"code": "unsupported_endpoint"}}), (403, "Not Found"),
    (429, {"error": "endpoint unsupported"}), (500, "Not Implemented"),
    (503, "Not Implemented"), (404, "x" * (16 * 1024 + 1)),
])
def test_ambiguous_model_auth_or_operational_errors_never_allow_fallback(status, body):
    assert not is_unsupported_responses_response(status, body)


def test_auto_is_an_explicit_startup_choice(monkeypatch):
    monkeypatch.delenv("AGENTKIT_MODEL_API", raising=False)
    assert resolve_model_api() == "chat_completions"
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    assert resolve_model_api() == "auto"


class HTTPError(Exception):
    def __init__(self, status, body):
        self.status_code = status
        self.body = body
        self.request = SimpleNamespace(method="POST", url=SimpleNamespace(path="/v1/responses"))
        super().__init__("private-error-canary")


class Resource:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    @property
    def with_raw_response(self):
        return self

    async def create(self, *args, **kwargs):
        self.calls += 1
        result = self.results.pop(0) if self.results else {"text": "ok"}
        if isinstance(result, BaseException):
            raise result
        return result


def test_resource_guards_normal_and_raw_requests_before_acceptance():
    async def exercise():
        for raw in (False, True):
            state = AutoModelAPIState()
            resource = state.wrap_responses(Resource(HTTPError(404, {"detail": "Not Found"})))
            resource = resource.with_raw_response if raw else resource
            with pytest.raises(ModelAPIFallback) as captured:
                await resource.create()
            assert state.selected is None
            assert "private-error-canary" not in str(captured.value)
    asyncio.run(exercise())


@pytest.mark.parametrize("path", ["/token", "/oauth2/v2.0/token", "/metadata", "/v1/models"])
def test_credential_or_discovery_http_errors_cannot_negotiate(path):
    async def exercise():
        state = AutoModelAPIState()
        error = HTTPError(404, {"detail": "Not Found"})
        error.request.url.path = path
        resource = state.wrap_responses(Resource(error))
        with pytest.raises(HTTPError):
            await resource.create()
        assert state.selected is None
    asyncio.run(exercise())


@pytest.mark.parametrize("error", [
    HTTPError(401, {"detail": "Unauthorized"}),
    HTTPError(404, {"error": {"code": "model_not_found"}}),
])
def test_current_auth_or_model_error_overrules_older_route_error(error):
    async def exercise():
        state = AutoModelAPIState()
        error.__cause__ = HTTPError(404, {"detail": "Not Found"})
        resource = state.wrap_responses(Resource(error))
        with pytest.raises(HTTPError):
            await resource.create()
        assert state.selected is None
    asyncio.run(exercise())


def test_accepted_stream_or_decoding_failure_pins_responses():
    async def exercise():
        for accepted in (object(), HTTPError(200, {"status": "incomplete"})):
            state = AutoModelAPIState()
            resource = state.wrap_responses(Resource(accepted, HTTPError(404, {"detail": "Not Found"})))
            if isinstance(accepted, Exception):
                with pytest.raises(HTTPError):
                    await resource.create()
            else:
                assert await resource.create() is accepted
            assert state.selected == "responses"
            with pytest.raises(HTTPError):
                await resource.create()
            assert state.selected == "responses"
    asyncio.run(exercise())


def test_streaming_raw_context_is_guarded_and_closed():
    class Context:
        closed = False
        async def __aenter__(self):
            return "raw response"
        async def __aexit__(self, *args):
            self.closed = True

    class StreamingResource:
        def __init__(self):
            self.context = Context()
        @property
        def with_streaming_response(self):
            return self
        def create(self, **kwargs):
            return self.context

    async def exercise():
        state = AutoModelAPIState()
        native = StreamingResource()
        guarded = state.wrap_responses(native)
        async with guarded.with_streaming_response.create() as response:
            assert response == "raw response"
            assert state.selected == "responses"
        assert native.context.closed
    asyncio.run(exercise())


class Runtime:
    def __init__(self, api, state, events, results):
        self.api = api
        self.events = events
        self.resource = Resource(*results)
        if state is not None:
            self.resource = state.wrap_responses(self.resource)
        self.requests = []
        self.stack = AsyncExitStack()
        self.lifecycle = AsyncExitStackLifecycle(self.stack)

    async def __aenter__(self):
        async def start():
            self.events.append((self.api, "enter"))
            self.stack.push_async_callback(self.close)
            return self
        return await self.lifecycle.enter(start)

    async def close(self):
        self.events.append((self.api, "exit"))

    async def __aexit__(self, *args):
        return await self.lifecycle.exit(*args)

    async def run(self, request):
        self.requests.append(request)
        await self.resource.create()
        self.events.append((self.api, "tool"))
        return RunResult(text=self.api)


def test_first_request_fallback_is_cached_and_closes_old_runtime(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "auto")
    async def exercise():
        events, runtimes = [], {}
        def factory(api, state):
            results = [HTTPError(404, "404 page not found")] if api == "responses" else []
            runtimes[api] = Runtime(api, state, events, results)
            return runtimes[api]
        request = RunRequest("hello", session_id="session", env={"AGENTKIT_MODEL_API": "responses"})
        async with AutoModelRuntime(factory) as runtime:
            assert (await runtime.run(request)).text == "chat_completions"
            assert runtime.state.selected == "chat_completions"
            assert (await runtime.run(request)).text == "chat_completions"
            assert (await runtime.run(request)).text == "chat_completions"
            assert runtimes["responses"].requests == [request]
            assert runtimes["chat_completions"].requests == [request] * 3
            assert events[:3] == [("responses", "enter"), ("responses", "exit"), ("chat_completions", "enter")]
            assert ("responses", "tool") not in events
        assert events[-1] == ("chat_completions", "exit")
        assert events.count(("responses", "exit")) == 1
    asyncio.run(exercise())
    assert resolve_model_api() == "auto"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_runtime_does_not_fallback_on_unconfirmed_errors(status):
    async def exercise():
        built = []
        def factory(api, state):
            built.append(api)
            return Runtime(api, state, [], [HTTPError(status, {"error": {"code": "model_not_found"}})])
        async with AutoModelRuntime(factory) as runtime:
            with pytest.raises(HTTPError):
                await runtime.run(RunRequest("hello"))
        assert built == ["responses"]
    asyncio.run(exercise())


def test_api_is_not_changed_after_any_model_response_was_accepted():
    async def exercise():
        built = []
        def factory(api, state):
            built.append(api)
            return Runtime(api, state, [], [{"text": "ok"}, HTTPError(404, {"detail": "Not Found"})])
        async with AutoModelRuntime(factory) as runtime:
            assert (await runtime.run(RunRequest("one"))).text == "responses"
            with pytest.raises(HTTPError):
                await runtime.run(RunRequest("two"))
        assert built == ["responses"]
    asyncio.run(exercise())


def test_concurrent_initial_requests_share_one_negotiation():
    async def exercise():
        built, events = {}, []
        def factory(api, state):
            results = [HTTPError(404, {"detail": "Not Found"})] if api == "responses" else []
            built[api] = Runtime(api, state, events, results)
            return built[api]
        async with AutoModelRuntime(factory) as runtime:
            results = await asyncio.gather(*(runtime.run(RunRequest(str(n))) for n in range(5)))
            assert [result.text for result in results] == ["chat_completions"] * 5
            assert len(built["responses"].requests) == 1
            assert len(built["chat_completions"].requests) == 5
    asyncio.run(exercise())


def test_fallback_startup_failure_is_fatal_and_cleans_up():
    async def exercise():
        events = []
        def factory(api, state):
            if api == "chat_completions":
                raise ValueError("private-error-canary")
            return Runtime(api, state, events, [HTTPError(404, {"detail": "Not Found"})])
        async with AutoModelRuntime(factory) as runtime:
            for _ in range(2):
                with pytest.raises(AgentRunError) as captured:
                    await runtime.run(RunRequest("hello"))
                assert captured.value.fatal
                assert captured.value.code == "RuntimeStartFailed"
                assert "private-error-canary" not in str(captured.value)
        assert events == [("responses", "enter"), ("responses", "exit")]
    asyncio.run(exercise())


def test_auto_runtime_forwards_optional_session_rollback_hook():
    async def exercise():
        events, discarded = [], []
        def factory(api, state):
            runtime = Runtime(api, state, events, [])
            async def discard(session_id):
                discarded.append((api, session_id))
            runtime.discard_session = discard
            return runtime
        async with AutoModelRuntime(factory) as runtime:
            await runtime.run(RunRequest("hello", session_id="session"))
            await _discard_runtime_session(runtime, "session")
        assert discarded == [("responses", "session")]
    asyncio.run(exercise())


def test_accepted_headers_pin_api_before_untyped_body_read_failure():
    class Raw:
        status_code = 200
        async def parse(self):
            raise ValueError("body decoding failed without HTTP metadata")
        async def close(self):
            pass
    class Context:
        closed = False
        async def __aenter__(self):
            return Raw()
        async def __aexit__(self, *args):
            self.closed = True
    class Streaming:
        def __init__(self):
            self.context = Context()
        def create(self, **kwargs):
            return self.context
    class HeaderResource:
        def __init__(self):
            self.with_streaming_response = Streaming()
    async def exercise():
        native = HeaderResource()
        state = AutoModelAPIState()
        guarded = state.wrap_responses(native)
        with pytest.raises(ValueError):
            await guarded.create()
        assert state.selected == "responses"
        assert native.with_streaming_response.context.closed
        later = state.wrap_responses(Resource(HTTPError(404, {"detail": "Not Found"})))
        with pytest.raises(HTTPError):
            await later.create()
    asyncio.run(exercise())

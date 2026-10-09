"""Conservative, first-request model API negotiation without model SDK imports.

There is no portable capabilities endpoint across OpenAI-compatible servers.
Auto therefore uses the first real Responses request, not a synthetic inference
probe. Only a recognizable API rejection can trigger Chat before acceptance.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .adapter_support import AsyncExitStackLifecycle, ModelAPI, _exception_chain
from .conversation import RunRequest
from .runtime import AgentRunError, RunResult, RuntimeSession

# Generic model/deployment errors and unknown 404 bodies are deliberately absent.
_UNSUPPORTED_API_CODES = frozenset({
    "unsupported_endpoint", "unknown_endpoint", "unsupported_api",
    "responses_api_not_supported", "unsupported_model_api",
})
_ROUTE_ERRORS = {
    404: frozenset({"not found", "404 page not found", "endpoint unsupported"}),
    405: frozenset({"method not allowed"}),
    501: frozenset({"not implemented"}),
}
_MAX_ERROR_BODY_BYTES = 16 * 1024


def is_unsupported_responses_response(status: int, body: Any) -> bool:
    """Require endpoint-level evidence; a status alone does not identify a 404."""
    if type(status) is not int or status not in {400, 404, 405, 501}:
        return False
    if isinstance(body, bytes):
        if len(body) > _MAX_ERROR_BODY_BYTES:
            return False
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        if len(body) > _MAX_ERROR_BODY_BYTES:
            return False
        try:
            body = json.loads(body)
        except ValueError:
            return body.strip().lower() in _ROUTE_ERRORS.get(status, ())
    if not isinstance(body, Mapping):
        return False
    error = body.get("error")
    if isinstance(error, Mapping) or "code" in body:
        # The SDK exposes the inner error object; the raw HTTP loop sees its
        # outer envelope. Both carry the same explicit endpoint-level evidence.
        code = (error if isinstance(error, Mapping) else body).get("code")
        return isinstance(code, str) and code in _UNSUPPORTED_API_CODES
    # FastAPI/Starlette and older local servers use these route-level envelopes.
    detail = body.get("detail", error)
    return isinstance(detail, str) and detail.strip().lower() in _ROUTE_ERRORS.get(status, ())


class ModelAPIFallback(AgentRunError):
    """Internal signal raised only by the guarded first Responses SDK request."""

    def __init__(self) -> None:
        super().__init__("Responses API is not supported", code="ModelAPIUnsupported")


class AutoModelAPIState:
    """API choice scoped to one configured backend/model runtime lifetime."""

    def __init__(self) -> None:
        self.selected: ModelAPI | None = None

    def accept_responses(self) -> None:
        if self.selected is None:
            self.selected = "responses"

    def select_chat(self) -> None:
        if self.selected == "responses":
            raise RuntimeError("cannot change API after a Responses request was accepted")
        self.selected = "chat_completions"

    def wrap_responses(self, resource: Any, *, client: Any = None) -> Any:
        if client is not None:
            # The SDK copy shares the existing HTTP pool and credential hooks.
            # Its original owner still closes that pool; do not close the copy.
            resource = client.with_options(max_retries=0).responses
        return _AutoResponsesResource(resource, self)

    def _request_error(self, exc: Exception) -> None:
        for cause in _exception_chain(exc):
            if getattr(cause, "code", None) in ("ModelAuthMissing", "ModelAuthRejected"):
                return
            response = getattr(cause, "response", None)
            status = getattr(cause, "status_code", None)
            if status is None:
                status = getattr(response, "status_code", None)
            if status in (401, 403):
                return
            # SDK create() can also fail while fetching credentials. Such HTTP
            # errors are not evidence about the configured model endpoint.
            try:
                request = getattr(cause, "request", None) or getattr(response, "request", None)
                url = getattr(request, "url", None)
                path = getattr(url, "path", None)
                if path is None and isinstance(url, str):
                    path = urlsplit(url).path
            except (AttributeError, RuntimeError, ValueError):
                continue
            if request is None or not isinstance(path, str):
                continue
            if (
                getattr(request, "method", None) != "POST"
                or not path.rstrip("/").endswith("/responses")
            ):
                return
            if type(status) is not int:
                continue
            if 200 <= status < 300:
                # Strict SDK decoding may fail after the server accepted work.
                self.accept_responses()
                return
            if self.selected is not None:
                return
            if status not in {400, 404, 405, 501}:
                return
            body = getattr(cause, "body", None)
            if body is None and response is not None:
                try:
                    body = getattr(response, "content", None)
                except RuntimeError:
                    continue
            if is_unsupported_responses_response(status, body):
                raise ModelAPIFallback() from exc
            # A known model/auth/parameter rejection must not be replaced by an
            # older generic HTTP error in a wrapped exception's context.
            return


class _AutoResponseContext:
    def __init__(self, context: Any, state: AutoModelAPIState) -> None:
        self._context = context
        self._state = state

    async def __aenter__(self) -> Any:
        try:
            result = await self._context.__aenter__()
        except Exception as exc:
            self._state._request_error(exc)
            raise
        self._state.accept_responses()
        return result

    async def __aexit__(self, *args: Any) -> Any:
        return await self._context.__aexit__(*args)


class _AutoParsedStream:
    """Keep the header-reading context alive until native stream consumption ends."""

    def __init__(self, stream: Any, raw: _AutoRawResponse) -> None:
        self._stream = stream
        self._iterator = aiter(stream)
        self._raw = raw
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        try:
            return await anext(self._iterator)
        except BaseException:
            await self.close()
            raise

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.close()
        finally:
            await self._raw.close()


class _AutoRawResponse:
    """Preserve legacy raw-response sync parse() after eager, guarded SDK parsing."""

    def __init__(self, response: Any, context: Any) -> None:
        self._response = response
        self._context = context
        self._closed = False
        self._parsed: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def parse(self) -> Any:
        return self._parsed

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._context.__aexit__(None, None, None)


class _AutoRawResource:
    def __init__(self, parent: _AutoResponsesResource) -> None:
        self._parent = parent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent._resource.with_raw_response, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        headers = dict(kwargs.get("extra_headers") or {})
        headers["X-Stainless-Raw-Response"] = "true"
        kwargs["extra_headers"] = headers
        return self._parent.create(*args, **kwargs)


class _AutoResponsesResource:
    def __init__(self, resource: Any, state: AutoModelAPIState) -> None:
        self._resource = resource
        self._state = state

    def __getattr__(self, name: str) -> Any:
        if name == "with_raw_response":
            return _AutoRawResource(self)
        value = getattr(self._resource, name)
        if name == "with_streaming_response":
            return _AutoStreamingResource(value, self._state)
        return value

    def create(self, *args: Any, **kwargs: Any) -> Any:
        streaming = getattr(self._resource, "with_streaming_response", None)
        if streaming is not None:
            return self._create_with_headers(streaming, args, kwargs)
        # Small in-process resource doubles need no separate HTTP read boundary.
        try:
            value = self._resource.create(*args, **kwargs)
        except Exception as exc:
            self._state._request_error(exc)
            raise
        if inspect.isawaitable(value):
            return self._await_response(value)
        self._state.accept_responses()
        return value

    async def _create_with_headers(self, streaming: Any, args: tuple, kwargs: dict) -> Any:
        headers = kwargs.get("extra_headers") or {}
        raw_mode = isinstance(headers, Mapping) and headers.get("X-Stainless-Raw-Response") in {"true", "stream"}
        context = streaming.create(*args, **kwargs)
        try:
            response = await context.__aenter__()
        except Exception as exc:
            self._state._request_error(exc)
            raise
        # Transport streaming returns headers before body consumption or decoding.
        self._state.accept_responses()
        raw = _AutoRawResponse(response, context)
        try:
            parsed = response.parse()
            if inspect.isawaitable(parsed):
                parsed = await parsed
            if callable(getattr(parsed, "__aiter__", None)):
                parsed = _AutoParsedStream(parsed, raw)
            else:
                await raw.close()
        except BaseException:
            await raw.close()
            raise
        raw._parsed = parsed
        return raw if raw_mode else parsed

    async def _await_response(self, value: Any) -> Any:
        try:
            result = await value
        except Exception as exc:
            self._state._request_error(exc)
            raise
        self._state.accept_responses()
        return result


class _AutoStreamingResource:
    def __init__(self, resource: Any, state: AutoModelAPIState) -> None:
        self._resource = resource
        self._state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return _AutoResponseContext(self._resource.create(*args, **kwargs), self._state)


class AutoModelRuntime:
    """Own one native runtime at a time and cache the first model API decision.

    Native runtimes keep their resource entry/exit on lifecycle-owner tasks.
    Negotiation is serialized until an API is chosen. Once any Responses request
    is accepted, even a later tool/model failure cannot trigger a different API.
    """

    def __init__(
        self, factory: Callable[[ModelAPI, AutoModelAPIState | None], RuntimeSession],
    ) -> None:
        self._factory = factory
        self.state = AutoModelAPIState()
        self._stack = AsyncExitStack()
        self._lifecycle = AsyncExitStackLifecycle(self._stack)
        self._scope: AsyncExitStack | None = None
        self._runtime: RuntimeSession | None = None
        self._negotiation_lock = asyncio.Lock()
        self._failure: AgentRunError | None = None

    async def _activate(self, api: ModelAPI) -> None:
        scope = AsyncExitStack()
        self._stack.push_async_callback(scope.aclose)
        runtime = self._factory(api, self.state if api == "responses" else None)
        scope.push_async_exit(runtime)
        try:
            await runtime.__aenter__()
        except BaseException:
            await scope.aclose()
            raise
        self._scope = scope
        self._runtime = runtime

    async def __aenter__(self) -> RuntimeSession:
        async def start() -> RuntimeSession:
            await self._activate("responses")
            return self

        return await self._lifecycle.enter(start)

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return await self._lifecycle.exit(exc_type, exc, tb)

    async def discard_session(self, session_id: str) -> None:
        discard = getattr(self._runtime, "discard_session", None)
        if callable(discard):
            result = discard(session_id)
            if inspect.isawaitable(result):
                await result

    async def run(self, request: RunRequest) -> RunResult:
        if self._failure is not None:
            raise self._failure
        if self.state.selected is not None:
            assert self._runtime is not None
            return await self._runtime.run(request)
        async with self._negotiation_lock:
            if self._failure is not None:
                raise self._failure
            assert self._runtime is not None
            try:
                return await self._runtime.run(request)
            except ModelAPIFallback:
                if self.state.selected is not None:
                    raise
                assert self._scope is not None
                try:
                    await self._scope.aclose()
                    await self._activate("chat_completions")
                except BaseException as exc:
                    self._failure = AgentRunError(
                        "runtime failed to start", code="RuntimeStartFailed", fatal=True,
                    )
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    raise self._failure from exc
                self.state.select_chat()
                return await self._runtime.run(request)

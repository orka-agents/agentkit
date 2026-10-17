"""Agent Client Protocol stdio adapter for Orka ACP supervisors.

The Orka supervisor owns process isolation, prompt leases, provider and MCP
brokers, workspace governance, and descendant cleanup. This module is only the
provider child. It translates the newline-delimited ACP JSON-RPC protocol into
the framework-neutral ``RuntimeFactory`` / ``RuntimeSession`` seam.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import ipaddress
import json
import os
import re
import secrets
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, BinaryIO, Callable, Iterator, Mapping
from urllib.parse import urlsplit

from .adapter_support import _attach_secondary_error, _wait_for_owner_task
from .config import AgentSpec, load_with_bytes
from .conversation import ConversationTurn, RunRequest, ToolCallEvent
from .runtime import AgentRunError, RuntimeFactory, RuntimeSession
from .skills import (
    SkillCatalog,
    SkillConfigurationError,
    validate_packaged_skill_providers,
)

ACP_PROTOCOL_VERSION = 1
ACP_AGENT_CONFIGURATION_DIGEST_ENV = "AGENTKIT_ACP_AGENT_CONFIGURATION_DIGEST"
ACP_MODEL_ENV = "AGENTKIT_ACP_MODEL"
ACP_PROVIDER_BASE_URL_ENV = "AGENTKIT_ACP_PROVIDER_BASE_URL"
ACP_PROVIDER_TOKEN_ENV = "AGENTKIT_ACP_PROVIDER_TOKEN"

_JSONRPC_VERSION = "2.0"
_METHOD_INITIALIZE = "initialize"
_METHOD_SESSION_NEW = "session/new"
_METHOD_SESSION_PROMPT = "session/prompt"
_METHOD_SESSION_CANCEL = "session/cancel"
_METHOD_SESSION_UPDATE = "session/update"
_METHOD_CANCEL_REQUEST = "$/cancel_request"
_MAX_MESSAGE_BYTES = 8 << 20
# orka.harness.v2 limits assistant-message chunks to 4 KiB of UTF-8 text.
# ACP v1 does not negotiate that limit, so the child must apply it before the
# supervisor maps notifications into harness events. Even worst-case JSON
# escaping remains well below harness v2's 512 KiB default event-line limit.
_MAX_ASSISTANT_MESSAGE_CHUNK_BYTES = 4 << 10

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_REQUEST_CANCELLED = -32800

_MessageSender = Callable[[Mapping[str, Any]], Awaitable[None]]


class ACPProtocolError(Exception):
    """A JSON-RPC error safe to return to the ACP client."""

    def __init__(self, code: int, message: str, data: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = dict(data) if data is not None else None


class ACPConfigurationError(ValueError):
    """An ACP startup binding or strict-mode configuration failure."""


@dataclass
class _PendingStdioWrite:
    frame: bytes
    completed: asyncio.Future[None]


def _retrieve_future_exception(future: asyncio.Future[None]) -> None:
    if not future.cancelled():
        future.exception()


class _ACPStdioWriter:
    """Serialize protocol frames without blocking the asyncio event loop."""

    def __init__(self, output_stream: BinaryIO) -> None:
        self._output_stream = output_stream
        self._queue: asyncio.Queue[_PendingStdioWrite | None] = asyncio.Queue()
        self._failure: BaseException | None = None
        self._closed = False
        self._worker = asyncio.create_task(self._run(), name="agentkit-acp-stdio-writer")

    async def send(self, message: Mapping[str, Any]) -> None:
        frame = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(frame) > _MAX_MESSAGE_BYTES:
            raise RuntimeError("ACP response exceeds the 8 MiB limit")
        if self._failure is not None:
            raise RuntimeError("ACP stdio writer failed") from self._failure
        if self._closed:
            raise RuntimeError("ACP stdio writer is closed")

        completed = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_PendingStdioWrite(frame=frame, completed=completed))
        try:
            await asyncio.shield(completed)
        except asyncio.CancelledError:
            # The worker owns an enqueued frame until the physical write finishes.
            # Keep its completion observable without letting caller cancellation
            # release serialization or produce an unhandled future exception.
            completed.add_done_callback(_retrieve_future_exception)
            raise

    async def close(self, *, preserve: BaseException | None = None) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)
        await _wait_for_owner_task(self._worker, preserve=preserve)

    async def _run(self) -> None:
        pending: _PendingStdioWrite | None = None
        try:
            while True:
                pending = await self._queue.get()
                if pending is None:
                    return
                await asyncio.to_thread(self._write, pending.frame)
                pending.completed.set_result(None)
                pending = None
        except BaseException as exc:  # noqa: BLE001 - wake every sender on terminal worker failure.
            self._failure = exc
            self._closed = True
            if pending is not None and not pending.completed.done():
                pending.completed.set_exception(exc)
            self._fail_queued_writes(exc)
            raise

    def _write(self, frame: bytes) -> None:
        self._output_stream.write(frame)
        self._output_stream.flush()

    def _fail_queued_writes(self, error: BaseException) -> None:
        while True:
            try:
                pending = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if pending is not None and not pending.completed.done():
                pending.completed.set_exception(error)


@dataclass
class _SessionState:
    session_id: str
    context: RuntimeSession
    runtime: RuntimeSession
    previous_environment: dict[str, str | None]
    history: list[ConversationTurn] = field(default_factory=list)
    active_request_key: str | None = None
    active_run: asyncio.Task[Any] | None = None
    cancel_requested: bool = False


class _ACPToolObserver:
    """Serialize payload-free tool updates within one prompt's lifetime."""

    def __init__(self, session_id: str, send: _MessageSender) -> None:
        self.session_id = session_id
        self.send = send
        self.pending: dict[str, dict[str, str]] = {}
        self.closed = False
        self.lock = asyncio.Lock()

    async def observe(self, event: ToolCallEvent) -> None:
        async with self.lock:
            if self.closed:
                return
            if not event.tool_call_id or event.status not in {"in_progress", "completed", "failed"}:
                raise AgentRunError("runtime emitted an invalid tool lifecycle event")
            if event.status == "in_progress":
                if event.tool_call_id in self.pending:
                    raise AgentRunError("runtime repeated an active tool call ID")
                # Never forward provider call IDs, arguments, results, or exception
                # text. A new ACP ID also avoids collisions across prompt turns.
                title = "Tool call"
                if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", event.tool_name):
                    title = event.tool_name
                tool = {"toolCallId": "tool-" + secrets.token_hex(16), "title": title, "kind": "other"}
                self.pending[event.tool_call_id] = tool
                update = {"sessionUpdate": "tool_call", **tool, "status": "in_progress"}
            else:
                tool = self.pending.pop(event.tool_call_id, None)
                if tool is None:
                    raise AgentRunError("runtime finished an unknown tool call")
                # Remove before awaiting output, since a cancelled write remains
                # owned by the stdio writer and must not get a second terminal.
                update = {"sessionUpdate": "tool_call_update", **tool, "status": event.status}
            await self._send(update)

    async def close(self) -> bool:
        async with self.lock:
            self.closed = True
            unfinished = bool(self.pending)
            while self.pending:
                _, tool = self.pending.popitem()
                await self._send({"sessionUpdate": "tool_call_update", **tool, "status": "failed"})
            return unfinished

    async def _send(self, update: dict[str, str]) -> None:
        await self.send(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "method": _METHOD_SESSION_UPDATE,
                "params": {"sessionId": self.session_id, "update": update},
            }
        )


def _factory_supports_http_mcp(factory: RuntimeFactory) -> bool:
    capability = getattr(factory, "supports_acp_http_mcp", None)
    return bool(capability()) if callable(capability) else False


def _factory_supports_packaged_skills(factory: RuntimeFactory) -> bool:
    capability = getattr(factory, "supports_acp_packaged_skills", None)
    return bool(capability()) if callable(capability) else False


def _only_packaged_skill_providers(spec: AgentSpec) -> bool:
    try:
        validate_packaged_skill_providers(spec)
    except SkillConfigurationError:
        return False
    return True


def _request_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ACPProtocolError(_INVALID_REQUEST, "JSON-RPC id must be a string or integer")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _required_object(value: Any, *, name: str = "params") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must be an object")
    return value


def _required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must be a non-empty string")
    if value != value.strip():
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must not contain surrounding whitespace")
    return value


def _required_prompt_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must be a non-empty string")
    return value


async def _discard_runtime_session(runtime: RuntimeSession, session_id: str) -> None:
    discard = getattr(runtime, "discard_session", None)
    if not callable(discard):
        return
    result = discard(session_id)
    if inspect.isawaitable(result):
        await result


def _utf8_chunks(value: str, max_bytes: int) -> Iterator[str]:
    encoded = value.encode("utf-8")
    offset = 0
    while offset < len(encoded):
        end = min(offset + max_bytes, len(encoded))
        while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
            end -= 1
        yield encoded[offset:end].decode("utf-8")
        offset = end


def _loopback_http_url(value: str, *, name: str) -> str:
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must be a valid loopback HTTP URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must be an absolute loopback HTTP URL")
    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ACPProtocolError(
                _INVALID_PARAMS,
                f"{name} must use a loopback IP address",
            ) from exc
        if not address.is_loopback:
            raise ACPProtocolError(_INVALID_PARAMS, f"{name} must use a loopback IP address")
    return value


def _safe_environment_value(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} must be a non-empty string")
    if "\r" in value or "\n" in value or "\x00" in value:
        raise ACPProtocolError(_INVALID_PARAMS, f"{name} contains forbidden control characters")
    return value


def _required_environment(name: str) -> str:
    try:
        return _safe_environment_value(os.environ.get(name), name=name)
    except ACPProtocolError as exc:
        raise ACPConfigurationError(exc.message) from exc


def load_verified_acp_runtime_binding(config_path: str | Path) -> AgentSpec:
    """Read, parse, and verify one immutable ACP agent configuration buffer."""
    spec, config_bytes = load_with_bytes(config_path)
    validate_acp_runtime_binding(config_bytes, spec)
    return spec


def validate_acp_runtime_binding(config_bytes: bytes, spec: AgentSpec) -> None:
    """Fail closed unless Orka's immutable profile matches the parsed config bytes."""

    if spec.tools:
        raise ACPConfigurationError("ACP strict mode rejects baked direct tools")
    if spec.brokered_tools:
        raise ACPConfigurationError("ACP strict mode rejects baked brokeredTools")
    try:
        validate_packaged_skill_providers(spec)
    except SkillConfigurationError as exc:
        raise ACPConfigurationError(str(exc)) from exc

    expected_digest = _required_environment(ACP_AGENT_CONFIGURATION_DIGEST_ENV)
    prefix = "sha256:"
    encoded_digest = expected_digest.removeprefix(prefix)
    if (
        not expected_digest.startswith(prefix)
        or len(encoded_digest) != 64
        or encoded_digest.lower() != encoded_digest
    ):
        raise ACPConfigurationError(
            f"{ACP_AGENT_CONFIGURATION_DIGEST_ENV} must be a lowercase sha256 digest"
        )
    try:
        bytes.fromhex(encoded_digest)
    except ValueError as exc:
        raise ACPConfigurationError(
            f"{ACP_AGENT_CONFIGURATION_DIGEST_ENV} must be a lowercase sha256 digest"
        ) from exc

    actual_digest = prefix + hashlib.sha256(config_bytes).hexdigest()
    if not secrets.compare_digest(expected_digest, actual_digest):
        raise ACPConfigurationError(
            f"{ACP_AGENT_CONFIGURATION_DIGEST_ENV} does not match the exact agent config bytes"
        )

    expected_model = _required_environment(ACP_MODEL_ENV)
    if expected_model != spec.model.name:
        raise ACPConfigurationError(f"{ACP_MODEL_ENV} does not match model.name in the agent config")

    provider_base_url = _required_environment(ACP_PROVIDER_BASE_URL_ENV)
    try:
        _loopback_http_url(provider_base_url, name=ACP_PROVIDER_BASE_URL_ENV)
    except ACPProtocolError as exc:
        raise ACPConfigurationError(exc.message) from exc
    _required_environment(ACP_PROVIDER_TOKEN_ENV)
    try:
        spec._packaged_skill_catalog = SkillCatalog.from_spec(spec)
    except SkillConfigurationError as exc:
        raise ACPConfigurationError(str(exc)) from exc


class ACPStdioServer:
    """Concurrent ACP JSON-RPC dispatcher over an injected message sender."""

    def __init__(self, spec: AgentSpec, factory: RuntimeFactory, send: _MessageSender) -> None:
        self.spec = spec
        self.factory = factory
        self.send = send
        self.initialized = False
        self.http_mcp = (
            _factory_supports_http_mcp(factory)
            and not spec.tools
            and not spec.brokered_tools
            and (
                not spec.context.providers
                or (_factory_supports_packaged_skills(factory) and _only_packaged_skill_providers(spec))
            )
        )
        self.sessions: dict[str, _SessionState] = {}
        self.requests: dict[str, asyncio.Task[None]] = {}
        self.active_handlers: dict[str, asyncio.Task[None]] = {}
        self.started_handlers: set[str] = set()
        self.handler_cancellations: set[str] = set()
        self.prompt_requests: dict[str, _SessionState] = {}
        self.session_creation_active = False
        self.closed = False

    async def accept_line(self, line: bytes) -> None:
        """Decode and schedule one newline-delimited JSON-RPC message."""

        if len(line) > _MAX_MESSAGE_BYTES:
            await self._send_error(None, _INVALID_REQUEST, "ACP message exceeds the 8 MiB limit")
            return
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            await self._send_error(None, _PARSE_ERROR, "invalid JSON")
            return
        await self.accept(message)

    async def accept(self, message: Any) -> None:
        if self.closed:
            return
        if not isinstance(message, dict) or message.get("jsonrpc") != _JSONRPC_VERSION:
            response_id = message.get("id") if isinstance(message, dict) else None
            await self._send_error(response_id, _INVALID_REQUEST, "invalid JSON-RPC request")
            return
        method = message.get("method")
        if not isinstance(method, str) or not method:
            await self._send_error(message.get("id"), _INVALID_REQUEST, "JSON-RPC method is required")
            return
        if "id" not in message:
            await self._handle_notification(method, message.get("params"))
            return
        response_id = message["id"]
        try:
            key = _request_key(response_id)
        except ACPProtocolError as exc:
            await self._send_protocol_error(response_id, exc)
            return
        if key in self.requests:
            await self._send_error(response_id, _INVALID_REQUEST, "duplicate active JSON-RPC id")
            return
        task = asyncio.create_task(
            self._dispatch_request(response_id, key, method, message.get("params")),
            name=f"agentkit-acp-{method}",
        )
        self.requests[key] = task
        if method != _METHOD_SESSION_PROMPT:
            self.active_handlers[key] = task
        task.add_done_callback(lambda completed, request_key=key: self._request_done(request_key, completed))
        # Give prompt dispatch a chance to register its cancellation target before
        # the reader accepts an immediately following session/cancel notification.
        await asyncio.sleep(0)

    async def wait_idle(self) -> None:
        while self.requests:
            active = tuple(self.requests.items())
            await asyncio.gather(*(task for _, task in active), return_exceptions=True)
            for key, task in active:
                if task.done() and self.requests.get(key) is task:
                    self.requests.pop(key, None)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for state in self.sessions.values():
            state.cancel_requested = True
            if state.active_run is not None and not state.active_run.done():
                state.active_run.cancel()
        for key in tuple(self.active_handlers):
            self._cancel_handler(key)
        await self.wait_idle()
        errors: list[BaseException] = []
        for state in reversed(tuple(self.sessions.values())):
            try:
                await state.context.__aexit__(None, None, None)
            except BaseException as exc:  # noqa: BLE001 - close every runtime before surfacing failure.
                errors.append(exc)
            finally:
                self._restore_environment(state.previous_environment)
        self.sessions.clear()
        if errors:
            raise errors[0]

    def _request_done(self, key: str, task: asyncio.Task[None]) -> None:
        if self.requests.get(key) is task:
            self.requests.pop(key, None)
        if self.active_handlers.get(key) is task:
            self.active_handlers.pop(key, None)
        self.started_handlers.discard(key)
        self.handler_cancellations.discard(key)
        # Retrieve exceptions even if stdout failed after the caller disconnected.
        if not task.cancelled():
            task.exception()

    async def _dispatch_request(self, response_id: Any, key: str, method: str, params: Any) -> None:
        handler = asyncio.current_task()
        if handler is None:
            raise RuntimeError("ACP request dispatch requires an asyncio task")
        if method != _METHOD_SESSION_PROMPT:
            self.started_handlers.add(key)
        result: Any = None
        error: tuple[int, str, Mapping[str, Any] | None] | None = None
        try:
            try:
                if key in self.handler_cancellations:
                    raise asyncio.CancelledError
                if method == _METHOD_INITIALIZE:
                    result = self._initialize(params)
                elif method == _METHOD_SESSION_NEW:
                    result = await self._new_session(params)
                elif method == _METHOD_SESSION_PROMPT:
                    result = await self._prompt(key, params)
                else:
                    raise ACPProtocolError(_METHOD_NOT_FOUND, f"unsupported ACP method {method!r}")
            except asyncio.CancelledError:
                error = (_REQUEST_CANCELLED, "ACP request cancelled", None)
            except ACPProtocolError as exc:
                error = (exc.code, exc.message, exc.data)
            except AgentRunError as exc:
                data = {"code": exc.code or exc.__class__.__name__}
                error = (_INTERNAL_ERROR, "AgentKit runtime prompt failed", data)
            except BaseException as exc:  # noqa: BLE001 - keep provider details off stdout.
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                error = (
                    _INTERNAL_ERROR,
                    "AgentKit ACP request failed",
                    {"code": exc.__class__.__name__},
                )
        finally:
            if self.active_handlers.get(key) is handler:
                self.active_handlers.pop(key, None)
            self.started_handlers.discard(key)
            self.handler_cancellations.discard(key)
        if error is not None:
            await self._send_error(response_id, *error)
            return
        await self.send({"jsonrpc": _JSONRPC_VERSION, "id": response_id, "result": result})

    async def _handle_notification(self, method: str, params: Any) -> None:
        if method == _METHOD_SESSION_CANCEL:
            if not isinstance(params, dict) or not isinstance(params.get("sessionId"), str):
                return
            state = self.sessions.get(params["sessionId"])
            self._cancel_state(state)
            return
        if method == _METHOD_CANCEL_REQUEST:
            if not isinstance(params, dict) or "requestId" not in params:
                return
            try:
                key = _request_key(params["requestId"])
            except ACPProtocolError:
                return
            state = self.prompt_requests.get(key)
            if state is not None:
                self._cancel_state(state)
                return
            self._cancel_handler(key)

    def _initialize(self, params: Any) -> dict[str, Any]:
        request = _required_object(params)
        version = request.get("protocolVersion")
        # Answer with the version this agent implements. The client decides
        # whether that is new enough. Valid ACP version numbers are 0 through 65535.
        if type(version) is not int or version < 0 or version > 65535:
            raise ACPProtocolError(
                _INVALID_PARAMS,
                "protocolVersion must be an integer from 0 to 65535",
            )
        self.initialized = True
        mcp_capabilities: dict[str, bool] = {"http": True} if self.http_mcp else {}
        return {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": False,
                },
                "mcpCapabilities": mcp_capabilities,
                "sessionCapabilities": {},
                "auth": {},
            },
            "agentInfo": {
                "name": "agentkit",
                "title": "AgentKit ACP runtime",
                "version": "0.0.0",
            },
        }

    async def _new_session(self, params: Any) -> dict[str, Any]:
        if not self.initialized:
            raise ACPProtocolError(_INVALID_REQUEST, "initialize must complete before session/new")
        if self.sessions or self.session_creation_active:
            raise ACPProtocolError(_INVALID_REQUEST, "ACP child already owns a session")
        self.session_creation_active = True
        try:
            return await self._create_session(params)
        finally:
            self.session_creation_active = False

    async def _create_session(self, params: Any) -> dict[str, Any]:
        request = _required_object(params)
        cwd = _required_string(request.get("cwd"), name="cwd")
        if not os.path.isabs(cwd) or os.path.realpath(cwd) != os.path.realpath(os.getcwd()):
            raise ACPProtocolError(_INVALID_PARAMS, "cwd must match the ACP child working directory")
        additional = request.get("additionalDirectories", [])
        if not isinstance(additional, list) or additional:
            raise ACPProtocolError(_INVALID_PARAMS, "additionalDirectories are not supported")
        if self.spec.tools:
            raise ACPProtocolError(_INVALID_PARAMS, "ACP strict mode rejects baked direct tools")
        if self.spec.brokered_tools:
            raise ACPProtocolError(_INVALID_PARAMS, "ACP strict mode rejects baked brokeredTools")
        try:
            validate_packaged_skill_providers(self.spec)
            if self.spec.context.providers:
                if not _factory_supports_packaged_skills(self.factory):
                    raise SkillConfigurationError("this runtime does not support ACP packaged skills")
                if self.spec._packaged_skill_catalog is None:
                    self.spec._packaged_skill_catalog = SkillCatalog.from_spec(self.spec)
        except SkillConfigurationError as exc:
            raise ACPProtocolError(_INVALID_PARAMS, str(exc)) from exc

        mcp_servers = request.get("mcpServers", [])
        if not isinstance(mcp_servers, list):
            raise ACPProtocolError(_INVALID_PARAMS, "mcpServers must be an array")
        if len(mcp_servers) > 1:
            raise ACPProtocolError(_INVALID_PARAMS, "ACP strict mode accepts at most one MCP server")
        if mcp_servers and not self.http_mcp:
            raise ACPProtocolError(_INVALID_PARAMS, "this runtime cannot consume ACP HTTP MCP servers")

        session_id = "agentkit-" + secrets.token_hex(16)
        projected, environment = self._project_spec(session_id, mcp_servers)
        previous_environment = self._install_environment(environment)
        context: RuntimeSession | None = None
        try:
            context = self.factory.build_runtime(projected)
            runtime = await context.__aenter__()
        except BaseException:
            error_info = sys.exc_info()
            try:
                if context is not None:
                    try:
                        await context.__aexit__(*error_info)
                    except BaseException:
                        pass
            finally:
                self._restore_environment(previous_environment)
            raise
        self.sessions[session_id] = _SessionState(
            session_id=session_id,
            context=context,
            runtime=runtime,
            previous_environment=previous_environment,
        )
        return {"sessionId": session_id}

    def _project_spec(
        self,
        session_id: str,
        mcp_servers: list[Any],
    ) -> tuple[AgentSpec, dict[str, str]]:
        provider_base_url = os.environ.get(ACP_PROVIDER_BASE_URL_ENV, "")
        provider_auth_value = os.environ.get(ACP_PROVIDER_TOKEN_ENV, "")
        provider_base_url = _loopback_http_url(provider_base_url, name=ACP_PROVIDER_BASE_URL_ENV)
        _safe_environment_value(provider_auth_value, name=ACP_PROVIDER_TOKEN_ENV)

        data = self.spec.model_dump(by_alias=True)
        data["model"]["baseURL"] = provider_base_url
        data["model"]["apiKeyEnv"] = ACP_PROVIDER_TOKEN_ENV
        data["model"]["auth"] = None
        data["tools"] = []
        environment: dict[str, str] = {}
        prefix = "AGENTKIT_ACP_SESSION_" + session_id.removeprefix("agentkit-").upper()
        seen_names: set[str] = set()
        for index, raw_server in enumerate(mcp_servers):
            server = _required_object(raw_server, name=f"mcpServers[{index}]")
            if server.get("type") != "http":
                raise ACPProtocolError(_INVALID_PARAMS, "ACP mode supports only HTTP MCP servers")
            if server.get("command") or server.get("args") or server.get("env"):
                raise ACPProtocolError(_INVALID_PARAMS, "HTTP MCP servers must not carry process fields")
            name = _required_string(server.get("name"), name=f"mcpServers[{index}].name")
            if name in seen_names:
                raise ACPProtocolError(_INVALID_PARAMS, f"duplicate MCP server name {name!r}")
            seen_names.add(name)
            url = _required_string(server.get("url"), name=f"mcpServers[{index}].url")
            url = _loopback_http_url(url, name=f"mcpServers[{index}].url")
            url_env = f"{prefix}_MCP_{index}_URL"
            environment[url_env] = url

            headers = server.get("headers", [])
            if not isinstance(headers, list):
                raise ACPProtocolError(_INVALID_PARAMS, f"mcpServers[{index}].headers must be an array")
            projected_headers: list[dict[str, str]] = []
            seen_headers: set[str] = set()
            for header_index, raw_header in enumerate(headers):
                header = _required_object(
                    raw_header,
                    name=f"mcpServers[{index}].headers[{header_index}]",
                )
                header_name = _required_string(
                    header.get("name"),
                    name=f"mcpServers[{index}].headers[{header_index}].name",
                )
                canonical_header = header_name.lower()
                if canonical_header in seen_headers:
                    raise ACPProtocolError(_INVALID_PARAMS, f"duplicate MCP header {header_name!r}")
                seen_headers.add(canonical_header)
                header_value = _safe_environment_value(
                    header.get("value"),
                    name=f"mcpServers[{index}].headers[{header_index}].value",
                )
                value_env = f"{prefix}_MCP_{index}_HEADER_{header_index}"
                environment[value_env] = header_value
                projected_headers.append({"name": header_name, "valueEnv": value_env})
            authorization = next(
                (
                    header
                    for header in headers
                    if isinstance(header, dict)
                    and isinstance(header.get("name"), str)
                    and header["name"].lower() == "authorization"
                ),
                None,
            )
            authorization_value = "" if authorization is None else str(authorization.get("value", ""))
            if not authorization_value.startswith("Bearer ") or not authorization_value[7:]:
                raise ACPProtocolError(
                    _INVALID_PARAMS,
                    "ACP HTTP MCP server must include a bearer Authorization header",
                )
            data["tools"].append(
                {
                    "name": name,
                    "type": "mcp",
                    "transport": "streamable-http",
                    "urlEnv": url_env,
                    "headers": projected_headers,
                }
            )
        try:
            projected = AgentSpec.model_validate(data)
        except ValueError as exc:
            raise ACPProtocolError(_INVALID_PARAMS, "ACP MCP server configuration is invalid") from exc
        projected._packaged_skill_catalog = self.spec._packaged_skill_catalog
        return projected, environment

    async def _prompt(self, request_key: str, params: Any) -> dict[str, str]:
        request = _required_object(params)
        session_id = _required_string(request.get("sessionId"), name="sessionId")
        state = self.sessions.get(session_id)
        if state is None:
            raise ACPProtocolError(_INVALID_PARAMS, "unknown ACP sessionId")
        if state.active_request_key is not None:
            raise ACPProtocolError(_REQUEST_CANCELLED, "ACP session already has an active prompt")
        prompt = request.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ACPProtocolError(_INVALID_PARAMS, "prompt must be a non-empty array")
        text_blocks: list[str] = []
        for index, raw_block in enumerate(prompt):
            block = _required_object(raw_block, name=f"prompt[{index}]")
            block_type = block.get("type")
            if block_type == "text":
                text_blocks.append(
                    _required_prompt_text(block.get("text"), name=f"prompt[{index}].text")
                )
                continue
            if block_type == "resource_link":
                name = _required_string(block.get("name"), name=f"prompt[{index}].name")
                uri = _required_string(block.get("uri"), name=f"prompt[{index}].uri")
                projected = [f"Resource link: {name}", f"URI: {uri}"]
                if block.get("mimeType") is not None:
                    mime_type = _required_string(
                        block.get("mimeType"),
                        name=f"prompt[{index}].mimeType",
                    )
                    projected.append(f"MIME type: {mime_type}")
                text_blocks.append("\n".join(projected))
                continue
            raise ACPProtocolError(
                _INVALID_PARAMS,
                "ACP mode accepts only text and resource_link prompt blocks",
            )

        tool_observer = _ACPToolObserver(session_id, self.send)
        run_request = RunRequest(
            prompt="\n".join(text_blocks),
            history=tuple(state.history),
            session_id=session_id,
            on_tool_event=tool_observer.observe,
        )
        state.cancel_requested = False
        state.active_request_key = request_key
        state.active_run = asyncio.create_task(
            state.runtime.run(run_request),
            name="agentkit-acp-runtime-prompt",
        )
        self.prompt_requests[request_key] = state
        try:
            cancelled = False
            runtime_error: BaseException | None = None
            try:
                result = await state.active_run
            except asyncio.CancelledError:
                cancelled = True
                result = None
            except BaseException as exc:  # noqa: BLE001 - cancellation wins over its fallout.
                result = None
                runtime_error = exc

            try:
                unfinished_tools = await tool_observer.close()
                if unfinished_tools and runtime_error is None and not cancelled:
                    runtime_error = AgentRunError("runtime returned before its tool calls settled")
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as exc:  # noqa: BLE001 - retain the original runtime failure.
                if runtime_error is None:
                    runtime_error = exc
                else:
                    _attach_secondary_error(runtime_error, exc, label="tool event completion failed")

            if (
                cancelled
                or state.cancel_requested
                or runtime_error is not None
                or result is None
            ):
                await _discard_runtime_session(state.runtime, session_id)
            if cancelled or state.cancel_requested:
                return {"stopReason": "cancelled"}
            if runtime_error is not None:
                raise runtime_error
            if result is None:
                raise ACPProtocolError(_INTERNAL_ERROR, "runtime returned no prompt result")
            try:
                for chunk in _utf8_chunks(result.text, _MAX_ASSISTANT_MESSAGE_CHUNK_BYTES):
                    if state.cancel_requested:
                        break
                    await self.send(
                        {
                            "jsonrpc": _JSONRPC_VERSION,
                            "method": _METHOD_SESSION_UPDATE,
                            "params": {
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text", "text": chunk},
                                },
                            },
                        }
                    )
            except BaseException:  # noqa: BLE001 - failed output must roll back state.
                await _discard_runtime_session(state.runtime, session_id)
                raise
            if state.cancel_requested:
                await _discard_runtime_session(state.runtime, session_id)
                return {"stopReason": "cancelled"}
            state.history.append(ConversationTurn(role="user", text=run_request.prompt))
            state.history.append(ConversationTurn(role="assistant", text=result.text))
            return {"stopReason": "end_turn"}
        finally:
            tool_observer.closed = True
            self.prompt_requests.pop(request_key, None)
            state.active_request_key = None
            state.active_run = None

    def _cancel_state(self, state: _SessionState | None) -> None:
        if state is None:
            return
        state.cancel_requested = True
        if state.active_run is not None and not state.active_run.done():
            state.active_run.cancel()

    def _cancel_handler(self, key: str) -> None:
        handler = self.active_handlers.get(key)
        if handler is None or handler.done():
            return
        if key in self.handler_cancellations:
            return
        self.handler_cancellations.add(key)
        if key in self.started_handlers:
            handler.cancel()

    @staticmethod
    def _install_environment(environment: Mapping[str, str]) -> dict[str, str | None]:
        previous = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        return previous

    @staticmethod
    def _restore_environment(environment: Mapping[str, str | None]) -> None:
        for name, previous in environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    async def _send_protocol_error(self, response_id: Any, error: ACPProtocolError) -> None:
        await self._send_error(response_id, error.code, error.message, error.data)

    async def _send_error(
        self,
        response_id: Any,
        code: int,
        message: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        # Invalid request IDs cannot be correlated by the JSON-RPC client.
        if isinstance(response_id, bool) or not isinstance(response_id, (int, str)):
            response_id = None
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = dict(data)
        await self.send({"jsonrpc": _JSONRPC_VERSION, "id": response_id, "error": error})


async def serve_acp_stdio(
    spec: AgentSpec,
    factory: RuntimeFactory,
    *,
    reader: BinaryIO | None = None,
    writer: BinaryIO | None = None,
) -> None:
    """Serve ACP until stdin closes, then close every runtime session."""

    input_stream = reader or sys.stdin.buffer
    output_stream = writer or sys.stdout.buffer
    stdio_writer = _ACPStdioWriter(output_stream)
    server: ACPStdioServer | None = None
    try:
        server = ACPStdioServer(spec, factory, stdio_writer.send)
        while True:
            line = await asyncio.to_thread(input_stream.readline, _MAX_MESSAGE_BYTES + 2)
            if not line:
                break
            complete = line.endswith(b"\n")
            frame = line.rstrip(b"\r\n") if complete else line
            if len(frame) > _MAX_MESSAGE_BYTES:
                await server.accept_line(frame)
                while not complete:
                    line = await asyncio.to_thread(input_stream.readline, _MAX_MESSAGE_BYTES + 2)
                    if not line:
                        break
                    complete = line.endswith(b"\n")
                continue
            await server.accept_line(frame)
    finally:
        primary_error = sys.exc_info()[1]
        try:
            if server is not None:
                await server.close()
        except BaseException as close_error:
            if primary_error is None:
                primary_error = close_error
                raise
            _attach_secondary_error(
                primary_error,
                close_error,
                label="ACP server cleanup also failed",
            )
        finally:
            await stdio_writer.close(preserve=primary_error)


def run_acp_stdio(spec: AgentSpec, factory: RuntimeFactory) -> None:
    """Synchronous console entrypoint for ACP stdio mode."""

    protocol_output = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        asyncio.run(serve_acp_stdio(spec, factory, writer=protocol_output))

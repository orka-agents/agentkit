"""Framework-neutral helpers shared by runtime adapter modules.

The adapter modules should spend their complexity budget on translating the
frozen ``agent.yaml`` ABI into their framework's concrete agent/client/tool
objects. Cross-runtime invariants live here instead: cancellation-safe lifecycle
cleanup, model API-key resolution, secret-safe tool env projection, MCP timeout
parsing, command validation, and normalizing framework/model failures into the
common run error.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
from collections.abc import Awaitable, Callable, Collection
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Literal, Mapping, TypeVar, cast

from .config import AgentSpec, ToolSpec
from .conversation import FORWARDED_ROLES
from .runtime import AgentRunError

# Placeholder API key for OpenAI-compatible endpoints that need no auth (many
# local servers reject an EMPTY string but accept any non-empty token). Used only
# when the spec declares no apiKeyEnv. Never a real secret.
NO_AUTH_API_KEY = "not-needed"


MODEL_API_ENV = "AGENTKIT_MODEL_API"
MODEL_APIS = ("chat_completions", "responses")
ModelAPI = Literal["chat_completions", "responses"]


MCP_TIMEOUT_ENV = "AGENTKIT_MCP_TIMEOUT"
WORKLOAD_TOKEN_ENV = "AGENTKIT_WORKLOAD_IDENTITY_TOKEN"
WORKLOAD_TOKEN_COMMAND_ENV = "AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND"
_BRACED_ENV_REF_RE = re.compile(r"\$\{([^}]+)\}")
_T = TypeVar("_T")


class AgentBuildError(Exception):
    """Raised when an adapter cannot construct its concrete agent."""


def resolve_model_api(
    *, supported: Collection[str] | None = None, runtime: str = "this runtime",
) -> ModelAPI:
    """Resolve the startup-only model protocol, never per-turn environment data."""
    value = os.environ.get(MODEL_API_ENV, "chat_completions")
    if value not in MODEL_APIS:
        raise AgentBuildError(f"{MODEL_API_ENV} must be chat_completions or responses")
    if supported is not None and value not in supported:
        raise AgentBuildError(f"{runtime} does not support {MODEL_API_ENV}={value}")
    return cast(ModelAPI, value)


def _attach_secondary_error(
    primary: BaseException,
    secondary: BaseException,
    *,
    label: str,
) -> None:
    if secondary is primary:
        return
    primary.add_note(f"{label} with {secondary.__class__.__name__}")
    for note in getattr(secondary, "__notes__", ()):
        primary.add_note(note)
    if primary.__cause__ is None:
        primary.__cause__ = secondary


async def _wait_for_owner_task(
    task: asyncio.Task[_T],
    *,
    preserve: BaseException | None = None,
) -> _T | None:
    """Wait for a lifecycle-owner task without forwarding caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            if not task.done():
                continue
            try:
                result = task.result()
            except BaseException as owner_error:
                if preserve is not None:
                    _attach_secondary_error(
                        preserve,
                        owner_error,
                        label="lifecycle owner also failed",
                    )
                    return None
                _attach_secondary_error(
                    cancellation,
                    owner_error,
                    label="cleanup also failed",
                )
                raise cancellation from owner_error
            if preserve is not None:
                preserve.add_note("cleanup wait was also cancelled")
                return None
            raise cancellation
        except BaseException as owner_error:
            if preserve is not None:
                _attach_secondary_error(
                    preserve,
                    owner_error,
                    label="lifecycle owner also failed",
                )
                return None
            if cancellation is not None:
                _attach_secondary_error(
                    cancellation,
                    owner_error,
                    label="cleanup also failed",
                )
                raise cancellation from owner_error
            raise
        else:
            if cancellation is not None:
                if preserve is not None:
                    preserve.add_note("cleanup wait was also cancelled")
                    return None
                raise cancellation
            return result


async def _close_exit_stack_inline(
    stack: AsyncExitStack,
    exc_type: type[BaseException] | None = None,
    exc: BaseException | None = None,
    tb: TracebackType | None = None,
    *,
    preserve: BaseException | None = None,
) -> bool | None:
    """Close a stack in its owner task without replacing a primary exception."""
    try:
        return await stack.__aexit__(exc_type, exc, tb)
    except BaseException as cleanup_error:
        if preserve is None:
            raise
        _attach_secondary_error(
            preserve,
            cleanup_error,
            label="cleanup also failed",
        )
        return None


class AsyncExitStackLifecycle:
    """Keep async resource entry and exit on one cancellation-isolated task."""

    def __init__(self, stack: AsyncExitStack) -> None:
        self.stack = stack
        self._owner_task: asyncio.Task[bool | None] | None = None
        self._close_request: asyncio.Future[
            tuple[
                type[BaseException] | None,
                BaseException | None,
                TracebackType | None,
            ]
        ] | None = None
        self._closing = False

    async def enter(self, start: Callable[[], Awaitable[_T]]) -> _T:
        """Run startup in the owner task and return its initialized runtime."""
        if self._owner_task is not None:
            raise RuntimeError("async lifecycle is already entered")

        loop = asyncio.get_running_loop()
        ready: asyncio.Future[_T] = loop.create_future()
        close_request = loop.create_future()
        self._close_request = close_request
        owner_task = asyncio.create_task(self._run(start, ready, close_request))
        self._owner_task = owner_task

        try:
            return await asyncio.shield(ready)
        except BaseException as exc:
            if not owner_task.done() and not self._closing:
                owner_task.cancel()
            await _wait_for_owner_task(owner_task, preserve=exc)
            if ready.done() and not ready.cancelled():
                owner_error = ready.exception()
                if owner_error is not None:
                    _attach_secondary_error(
                        exc,
                        owner_error,
                        label="lifecycle owner also failed",
                    )
            self._clear_if_done()
            raise

    async def exit(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Request owner-task cleanup and wait for every exit callback to finish."""
        owner_task = self._owner_task
        close_request = self._close_request
        if owner_task is None or close_request is None:
            return None
        if not close_request.done():
            close_request.set_result((exc_type, exc, tb))
        try:
            return await _wait_for_owner_task(owner_task, preserve=exc)
        finally:
            self._clear_if_done()

    async def _run(
        self,
        start: Callable[[], Awaitable[_T]],
        ready: asyncio.Future[_T],
        close_request: asyncio.Future[
            tuple[
                type[BaseException] | None,
                BaseException | None,
                TracebackType | None,
            ]
        ],
    ) -> bool | None:
        try:
            result = await start()
            ready.set_result(result)
            exc_type, exc, tb = await close_request
        except BaseException as error:
            self._closing = True
            await _close_exit_stack_inline(
                self.stack,
                type(error),
                error,
                error.__traceback__,
                preserve=error,
            )
            if not ready.done():
                ready.set_exception(error)
                return None
            raise

        self._closing = True
        return await _close_exit_stack_inline(
            self.stack,
            exc_type,
            exc,
            tb,
            preserve=exc,
        )

    def _clear_if_done(self) -> None:
        if self._owner_task is not None and self._owner_task.done():
            self._owner_task = None
            self._close_request = None
            self._closing = False


def _env_get(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """Resolve one env var with per-run values taking precedence over process env."""
    if env is not None and name in env:
        return env[name]
    return os.environ.get(name)


def resolve_api_key(spec: AgentSpec, env: Mapping[str, str] | None = None) -> str:
    """Resolve the model API key from the env var NAMED in the spec.

    Per the ABI, ``model.apiKeyEnv`` is the NAME of an env var; the value is
    injected at runtime (``docker run -e``). If the name is declared but the var
    is absent, fail fast with a clear, SECRET-FREE message.
    """
    name = spec.model.api_key_env
    if not name:
        # No auth declared — e.g. a co-located local model over baseURL.
        return NO_AUTH_API_KEY
    value = _env_get(name, env)
    if value is None or value == "":
        raise AgentBuildError(
            f"model.apiKeyEnv {name!r} is declared in agent.yaml but env var "
            f"{name!r} is not set; inject it at runtime, e.g. "
            f"`docker run -e {name}=...`"
        )
    return value


def declared_tool_env(tool: ToolSpec, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Select ONLY the env vars NAMED in ``tool.env``.

    The MCP subprocess must never inherit the full container environment — that
    would bleed the model API key (and every other secret) into every tool. We
    pass through exactly the declared names that are actually present.

    Some MCP stdio transports expand braced references like ``${VAR}`` inside env
    values against the parent process environment. Reject references to undeclared
    vars so a declared tool env such as ``TOOL_CONFIG=${OPENAI_API_KEY}`` cannot
    smuggle the model key into a subprocess unless that key was explicitly listed
    in the tool's own ``env`` allowlist.
    """
    allowed = set(tool.env)
    out = {name: value for name in tool.env if (value := _env_get(name, env)) is not None}
    for name, value in out.items():
        undeclared = sorted(ref for ref in _BRACED_ENV_REF_RE.findall(value) if ref not in allowed)
        if undeclared:
            raise AgentBuildError(
                f"tool {tool.name!r} env var {name!r} references undeclared env var(s) "
                f"{', '.join(undeclared)}; list every referenced env var in that tool's "
                "env allowlist or remove the ${...} reference"
            )
    return out


def resolve_tool_url(tool: ToolSpec, env: Mapping[str, str] | None = None) -> str:
    """Resolve a remote MCP URL from the env var named by ``tool.urlEnv``."""
    name = tool.url_env
    if not name:
        raise AgentBuildError(f"tool {tool.name!r} has no urlEnv")
    value = _env_get(name, env)
    if value is None or value == "":
        raise AgentBuildError(
            f"tool {tool.name!r} urlEnv {name!r} is declared in agent.yaml but env var "
            f"{name!r} is not set; inject it at runtime, e.g. `docker run -e {name}=...`"
        )
    return value


def resolve_tool_headers(
    tool: ToolSpec,
    *,
    include_workload_identity: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve static/env/auth headers for a remote MCP tool without logging values."""
    headers: dict[str, str] = {}
    for header in tool.headers:
        if header.value is not None and header.value != "":
            headers[header.name] = header.value
            continue
        name = header.value_env
        if not name:
            raise AgentBuildError(f"tool {tool.name!r} header {header.name!r} has no value")
        value = _env_get(name, env)
        if value is None or value == "":
            raise AgentBuildError(
                f"tool {tool.name!r} header {header.name!r} uses valueEnv {name!r} but "
                f"env var {name!r} is not set; inject it at runtime"
            )
        headers[header.name] = value

    if tool.auth is None:
        return headers

    if any(name.lower() == "authorization" for name in headers):
        raise AgentBuildError(
            f"tool {tool.name!r} sets both Authorization header and auth; use one auth path"
        )

    if tool.auth.type == "bearer":
        token_env = tool.auth.token_env
        token = _env_get(token_env or "", env)
        if not token_env or token is None or token == "":
            raise AgentBuildError(
                f"tool {tool.name!r} bearer auth tokenEnv {token_env!r} is declared in "
                "agent.yaml but the env var is not set; inject it at runtime"
            )
        headers["Authorization"] = f"Bearer {token}"
        return headers

    if tool.auth.type == "workload-identity-token":
        if not include_workload_identity:
            return headers
        token = resolve_workload_identity_token(tool.auth.audience or "", env=env)
        headers["Authorization"] = f"Bearer {token}"
        return headers

    raise AgentBuildError(f"tool {tool.name!r} auth type {tool.auth.type!r} is not supported")


def resolve_workload_identity_token(audience: str, env: Mapping[str, str] | None = None) -> str:
    """Resolve a workload identity token through a provider-neutral runtime hook.

    Deployment profiles may either inject ``AGENTKIT_WORKLOAD_IDENTITY_TOKEN`` for
    simple smoke tests or set ``AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND`` to a
    command that accepts the opaque audience as argv[1] and prints a token. If
    neither hook is configured, fall back to optional ``azure-identity`` when it
    is installed in a runtime image. Values are returned to callers only; they are
    never logged by this helper.
    """
    if not audience:
        raise AgentBuildError("workload identity auth requires a non-empty audience")

    token = _env_get(WORKLOAD_TOKEN_ENV, env)
    if token:
        return token

    command = _env_get(WORKLOAD_TOKEN_COMMAND_ENV, env)
    if command:
        argv = shlex.split(command) + [audience]
        try:
            completed = subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AgentBuildError(
                f"workload identity token command failed for audience {audience!r}: {exc}"
            ) from exc
        token = completed.stdout.strip()
        if not token:
            raise AgentBuildError(
                f"workload identity token command for audience {audience!r} returned no token"
            )
        return token

    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AgentBuildError(
            "workload identity auth is configured but no token hook is available; set "
            f"{WORKLOAD_TOKEN_COMMAND_ENV}, set {WORKLOAD_TOKEN_ENV}, or install azure-identity"
        ) from exc

    identity_client = DefaultAzureCredential()
    try:
        value = identity_client.get_token(audience).token
    except BaseException as exc:
        error: BaseException
        if isinstance(exc, Exception):
            error = AgentBuildError(
                f"workload identity token acquisition failed for audience {audience!r}: {exc}"
            )
        else:
            error = exc
        try:
            identity_client.close()
        except BaseException as cleanup_error:
            error.add_note(
                f"credential cleanup also failed with {cleanup_error.__class__.__name__}"
            )
        if error is exc:
            raise
        raise error from exc

    try:
        identity_client.close()
    except Exception as exc:  # noqa: BLE001 - normalize provider failures.
        raise AgentBuildError(
            f"workload identity credential cleanup failed for audience {audience!r}: {exc}"
        ) from exc
    return value


def same_origin_mcp_httpx_client_factory(tool: ToolSpec, url: str, *, timeout: float | int | None):
    """Return an httpx AsyncClient factory that injects tool headers same-origin only.

    FastMCP/LangChain factories may otherwise follow redirects with caller-supplied
    headers. This factory disables redirects and resolves headers lazily for each
    request so workload tokens can refresh and credentials are not replayed to a
    redirected origin.
    """
    from httpx import AsyncClient, URL

    target_url = URL(url)
    target_origin = (target_url.scheme, target_url.host, target_url.port)

    async def inject_headers(request):  # noqa: ANN001
        request_origin = (request.url.scheme, request.url.host, request.url.port)
        if request_origin != target_origin:
            return
        for key, value in (await asyncio.to_thread(resolve_tool_headers, tool)).items():
            request.headers[key] = value

    def factory(**kwargs):  # noqa: ANN003
        base_headers = dict(kwargs.get("headers") or {})
        return AsyncClient(
            headers=base_headers,
            auth=kwargs.get("auth"),
            event_hooks={"request": [inject_headers]},
            follow_redirects=False,
            timeout=timeout if timeout is not None else kwargs.get("timeout"),
        )

    return factory


def split_tool_command(tool: ToolSpec, *, example: str) -> tuple[str, list[str]]:
    """Return ``(command, args)`` for a stdio MCP tool, or fail clearly."""
    if not tool.command:
        raise AgentBuildError(
            f"tool {tool.name!r} has an empty command; a stdio MCP server needs "
            f"at least the executable, e.g. command: {example}"
        )
    return tool.command[0], list(tool.command[1:])


def positive_float_env(name: str = MCP_TIMEOUT_ENV, *, default: float, env: Mapping[str, str] | None = None) -> float:
    """Read a positive float env var, falling back to ``default``."""
    raw = _env_get(name, env)
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def positive_int_env(name: str = MCP_TIMEOUT_ENV, *, default: int | None, env: Mapping[str, str] | None = None) -> int | None:
    """Read a positive integer env var, accepting float strings, else ``default``."""
    raw = _env_get(name, env)
    if not raw:
        return default
    try:
        val = int(float(raw))
    except ValueError:
        return default
    return val if val > 0 else default


def upstream_status_code(exc: BaseException, *, default: int = 502) -> int:
    """Best-effort upstream HTTP status from a framework/model exception.

    Duck-typed on ``status_code`` (the OpenAI SDK's ``APIStatusError`` exposes it)
    so adapters can map real upstream 4xx/5xx responses through without importing
    SDK-specific exception classes. Some frameworks wrap the SDK error in their
    own exception and store the original as ``inner_exception`` or as the normal
    exception cause/context, so this walks the bounded exception chain.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    for _ in range(10):  # bounded walk; guards against pathological cycles
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        code = getattr(cur, "status_code", None)
        if isinstance(code, int) and 400 <= code <= 599:
            return code
        cur = getattr(cur, "inner_exception", None) or cur.__cause__ or cur.__context__
    return default


def normalize_agent_run_error(exc: Exception) -> AgentRunError:
    """Convert an adapter/framework exception into the common façade error."""
    return AgentRunError(
        f"agent run failed: {exc}",
        status=upstream_status_code(exc),
        code=exc.__class__.__name__,
    )

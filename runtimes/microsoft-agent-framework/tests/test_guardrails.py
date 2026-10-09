"""Guardrail regressions for the MAF adapter's non-negotiable invariants.

Two things in this adapter must never silently regress (plan §10, §12):

1. The LOCK-IN BOUNDARY — ``agent_factory.py`` may import Agent Framework core,
   OpenAI, and the narrow Foundry/Azure identity surface needed for generic
   model workload identity. It must not pull CopilotStudio/Purview/etc.
2. The SECRET-BLEED RULE — a tool subprocess receives ONLY the env var NAMES the
   tool declares, never the full container environment (which holds the model key).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest import mock

from agentkit_serve import agent_factory
from agentkit_serve_common.config import ToolSpec

# Forbidden import SEGMENTS. A module is forbidden if ANY dotted segment matches
# one of these. This catches BOTH spellings of MAF's cloud integrations:
#   * separate distributions:  agent_framework_azure, agent_framework_foundry, ...
#     (their import name has the root segment "agent_framework_azure", etc.)
#   * FIRST-PARTY SUBMODULES:   agent_framework.azure / .foundry / .microsoft /
#     .copilotstudio — these live INSIDE agent_framework_core and re-export the
#     cloud surface (e.g. agent_framework.microsoft.CopilotStudioAgent). Matching
#     on segments (not just the root) is what catches these; a root-only check
#     would wave them through because their root is the allowed "agent_framework".
_FORBIDDEN_IMPORT_SEGMENTS = (
    "azure",
    "copilotstudio",
    "copilot_studio",
    "microsoft",  # agent_framework.microsoft re-exports CopilotStudioAgent + Purview
    "agent_framework_azure",
    "agent_framework_copilotstudio",
    "agent_framework_purview",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def _is_forbidden(mod: str) -> bool:
    """A module is forbidden if any of its dotted segments is a forbidden token."""
    if mod == "azure.identity":
        return False
    if mod == "agent_framework.foundry" or mod.startswith("agent_framework.foundry."):
        return False
    return any(seg in _FORBIDDEN_IMPORT_SEGMENTS for seg in mod.split("."))


def _is_allowed_framework_module(mod: str) -> bool:
    """An agent_framework* import is allowed ONLY if it is the bare core package or
    the OpenAI client — matched EXACTLY, never as a blanket ``agent_framework.*``
    prefix (which would admit agent_framework.azure/.foundry/.microsoft).
    """
    return (
        mod == "agent_framework"
        or mod == "agent_framework.openai"
        or mod.startswith("agent_framework.openai.")
        or mod == "agent_framework.foundry"
        or mod.startswith("agent_framework.foundry.")
        or mod == "azure.identity"
    )


def test_agent_factory_import_boundary():
    """agent_factory imports must not cross into Azure/Foundry/CopilotStudio.

    Covers both the separate-distribution spelling (agent_framework_azure) AND the
    first-party submodule spelling (agent_framework.azure / .microsoft), which a
    future edit is most likely to reach for.
    """
    mods = _imported_modules(Path(agent_factory.__file__))
    for mod in mods:
        assert not _is_forbidden(mod), f"agent_factory.py imports forbidden module {mod!r} (plan §12 lock-in boundary)"


def test_only_minimal_agent_framework_surface():
    """Any agent_framework import must be core or the openai client — not a cloud pkg."""
    mods = _imported_modules(Path(agent_factory.__file__))
    for mod in mods:
        if mod.startswith("agent_framework"):
            assert _is_allowed_framework_module(
                mod
            ), f"agent_factory.py imports non-minimal agent_framework module {mod!r}"


def test_guardrail_actually_rejects_cloud_submodules():
    """Meta-test: the boundary check must REJECT the cloud submodule import forms.

    Without this, the guardrail can silently rot into a false-green (the bare
    'agent_framework' prefix used to admit every agent_framework.* submodule).
    These names are the idiomatic way to pull MAF's cloud surface, so they must
    fail BOTH gates.
    """
    must_reject = [
        "agent_framework.azure",
        "agent_framework.microsoft",  # re-exports CopilotStudioAgent
        "agent_framework_azure",
        "agent_framework_copilotstudio",
        "azure.ai.projects",
    ]
    for mod in must_reject:
        assert _is_forbidden(mod), f"boundary should forbid {mod!r} but did not"
        assert not _is_allowed_framework_module(mod), f"boundary should NOT allow {mod!r}"
    # …and the legitimate minimal surface must still pass both gates.
    for mod in ["agent_framework", "agent_framework.openai", "agent_framework.foundry", "azure.identity"]:
        assert not _is_forbidden(mod), f"{mod!r} should be allowed"
        assert _is_allowed_framework_module(mod), f"{mod!r} should be allowed"


def test_tool_env_passes_only_declared_names():
    """_tool_env selects ONLY declared names that are present — never the full env."""
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=["FETCH_TOKEN", "ABSENT_VAR"])
    with mock.patch.dict(
        os.environ,
        {"FETCH_TOKEN": "tok", "OPENAI_API_KEY": "sk-should-not-leak", "OTHER": "x"},
        clear=True,
    ):
        env = agent_factory._tool_env(tool)
    assert env == {"FETCH_TOKEN": "tok"}  # declared+present only
    assert "OPENAI_API_KEY" not in env  # the model secret never bleeds into a tool
    assert "ABSENT_VAR" not in env  # declared-but-absent is omitted, not empty


def test_build_tool_threads_command_and_args():
    """build_tool maps command[0]→executable, command[1:]→args, declared env only."""
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch", "--flag"], env=[])
    with mock.patch.dict(os.environ, {}, clear=True):
        mcp_tool = agent_factory.build_tool(tool)
    assert mcp_tool.command == "uvx"
    assert list(mcp_tool.args) == ["mcp-server-fetch", "--flag"]


def test_build_tool_sets_tool_name_prefix():
    """build_tool MUST namespace exposed tool names via tool_name_prefix.

    In MAF the per-server ``name`` does NOT prefix the tool functions a server
    exposes — only ``tool_name_prefix`` does. Without it, two servers exposing a
    same-named sub-tool make the agent raise "Duplicate tool name" at run time.
    This mirrors the pydantic-ai adapter's ``tool_prefix=tool.name`` guard.
    """
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=[])
    with mock.patch.dict(os.environ, {}, clear=True):
        mcp_tool = agent_factory.build_tool(tool)
    assert mcp_tool.tool_name_prefix == "fetch"


def test_build_tool_rejects_empty_command():
    tool = ToolSpec.model_construct(name="broken", command=[], env=[])
    try:
        agent_factory.build_tool(tool)
    except agent_factory.AgentBuildError as exc:
        assert "empty command" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected AgentBuildError for empty command")


def test_missing_api_key_env_fails_fast():
    """A declared apiKeyEnv whose var is absent fails with a secret-free message."""
    spec_data = {
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://api.openai.com/v1",
            "name": "gpt-4o-mini",
            "apiKeyEnv": "DEFINITELY_UNSET_KEY_42",
        },
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    }
    from agentkit_serve_common.config import AgentSpec

    spec = AgentSpec.model_validate(spec_data)
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            agent_factory.build_client(spec)
        except agent_factory.AgentBuildError as exc:
            assert "DEFINITELY_UNSET_KEY_42" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected AgentBuildError for missing key env")


def test_status_of_unwraps_maf_wrapped_error():
    """_status_of walks the exception chain so a real upstream 4xx passes through.

    MAF wraps the OpenAI SDK error (which carries .status_code) inside its own
    ChatClientException (which does not). Without unwrapping, every upstream 4xx
    would flatten to 502 — diverging from the pydantic-ai adapter. Verified live
    against a 400 model_not_supported; this locks it offline.
    """

    class _BadRequest(Exception):
        status_code = 400

    class _ChatClientExc(Exception):
        def __init__(self, msg, inner=None):
            super().__init__(msg)
            self.inner_exception = inner

    # direct status (the simple SDK case)
    assert agent_factory._status_of(_BadRequest()) == 400
    # MAF-style: wrapped via inner_exception AND `from` (__cause__)
    try:
        try:
            raise _BadRequest("upstream 400")
        except _BadRequest as ex:
            raise _ChatClientExc("wrapped", inner=ex) from ex
    except _ChatClientExc as e:
        assert agent_factory._status_of(e) == 400
    # no status anywhere → 502; out-of-range ignored → 502
    assert agent_factory._status_of(ValueError("x")) == 502


def test_build_client_uses_foundry_for_model_workload_identity(monkeypatch):
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")
    from agentkit_serve_common.config import AgentSpec

    calls = {}

    class FakeCredential:
        pass

    class FakeFoundryClient:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("azure.identity.DefaultAzureCredential", FakeCredential)
    monkeypatch.setattr("agent_framework.foundry.FoundryChatClient", FakeFoundryClient)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
            "name": "gpt-4.1-mini",
            "auth": {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"},
        },
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })

    client = agent_factory.build_client(spec)

    assert isinstance(client, FakeFoundryClient)
    assert calls["project_endpoint"] == "https://example.services.ai.azure.com/api/projects/proj"
    assert calls["model"] == "gpt-4.1-mini"
    assert isinstance(calls["credential"], FakeCredential)


def test_build_agent_adds_filesystem_skills_provider():
    from agentkit_serve_common.config import AgentSpec

    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://api.openai.com/v1",
            "name": "gpt-4o-mini",
        },
        "instructions": "hi",
        "tools": [],
        "context": {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]},
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)

    async def build():
        providers = await runtime._build_context_providers()
        await runtime.stack.aclose()
        return providers

    import asyncio
    providers = asyncio.run(build())
    assert providers is not None
    assert providers[0].__class__.__name__ == "SkillsProvider"


def test_build_client_uses_env_token_for_model_workload_identity_standalone(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    calls = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", FakeOpenAIClient)
    monkeypatch.setenv("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN", "token")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
            "name": "gpt-4.1-mini",
            "auth": {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"},
        },
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })

    client = agent_factory.build_client(spec)

    assert isinstance(client, FakeOpenAIClient)
    import asyncio
    assert asyncio.run(calls["api_key"]()) == "token"
    assert calls["base_url"].endswith("/openai/v1")


def test_build_agent_adds_search_context_provider(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    calls = {}

    class FakeSearchProvider:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    class FakeModule:
        AzureAISearchContextProvider = FakeSearchProvider

    def fake_import(name):
        assert name == "agent_framework.azure"
        return FakeModule

    monkeypatch.setattr(agent_factory.importlib, "import_module", fake_import)
    monkeypatch.setattr(agent_factory, "_credential_for_context", lambda provider, default_audience, async_credential=False: "credential")
    monkeypatch.setenv("SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("SEARCH_INDEX", "kb")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://api.openai.com/v1",
            "name": "gpt-4o-mini",
        },
        "instructions": "hi",
        "tools": [],
        "context": {"providers": [{
            "name": "knowledge",
            "type": "search",
            "endpointEnv": "SEARCH_ENDPOINT",
            "indexEnv": "SEARCH_INDEX",
        }]},
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)

    import asyncio
    providers = asyncio.run(runtime._build_context_providers())

    assert providers is not None
    assert providers[0].__class__.__name__ == "FakeSearchProvider"
    assert "source_id" not in calls
    assert calls["endpoint"] == "https://example.search.windows.net"
    assert calls["index_name"] == "kb"
    assert calls["credential"] == "credential"


def test_build_agent_adds_memory_context_provider(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    calls = {}

    class FakeMemoryProvider:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    class FakeModule:
        FoundryMemoryProvider = FakeMemoryProvider

    def fake_import(name):
        assert name == "agent_framework.foundry"
        return FakeModule

    monkeypatch.setattr(agent_factory.importlib, "import_module", fake_import)
    monkeypatch.setattr(agent_factory, "_credential_for_context", lambda provider, default_audience, async_credential=False: "credential")
    monkeypatch.setenv("MEMORY_ENDPOINT", "https://example.services.ai.azure.com/api/projects/proj")
    monkeypatch.setenv("MEMORY_STORE_NAME", "agentkit-memory")
    monkeypatch.setenv("AGENTKIT_MEMORY_SCOPE", "scope-1")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://api.openai.com/v1",
            "name": "gpt-4o-mini",
        },
        "instructions": "hi",
        "tools": [],
        "context": {"providers": [{
            "name": "memory",
            "type": "memory",
            "endpointEnv": "MEMORY_ENDPOINT",
            "storeNameEnv": "MEMORY_STORE_NAME",
            "auth": {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"},
        }]},
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)

    import asyncio
    providers = asyncio.run(runtime._build_context_providers())

    assert providers is not None
    assert providers[0].__class__.__name__ == "FakeMemoryProvider"
    assert calls["source_id"] == "memory"
    assert calls["project_endpoint"] == "https://example.services.ai.azure.com/api/projects/proj"
    assert calls["memory_store_name"] == "agentkit-memory"
    assert calls["credential"] == "credential"
    assert calls["scope"] == "scope-1"
    assert calls["update_delay"] == 0


def test_runtime_reuses_sessions_by_request_session_id(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import RunRequest
    from agentkit_serve_common.runtime import RunResult

    seen_sessions = []

    async def fake_run_agent(agent, request, *, session=None):
        seen_sessions.append(session)
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()

    import asyncio
    asyncio.run(runtime.run(RunRequest(prompt="one", session_id="s1")))
    asyncio.run(runtime.run(RunRequest(prompt="two", session_id="s1")))
    asyncio.run(runtime.run(RunRequest(prompt="three", session_id="s2")))

    assert seen_sessions[0] is seen_sessions[1]
    assert seen_sessions[0] is not seen_sessions[2]
    assert seen_sessions[0].session_id == "s1"
    assert seen_sessions[2].session_id == "s2"


def test_runtime_session_cache_is_bounded(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import RunRequest
    from agentkit_serve_common.runtime import RunResult

    async def fake_run_agent(agent, request, *, session=None):
        return RunResult(text=session.session_id if session else "none")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    monkeypatch.setenv("AGENTKIT_SESSION_CACHE_MAX", "2")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()

    import asyncio
    asyncio.run(runtime.run(RunRequest(prompt="one", session_id="s1")))
    asyncio.run(runtime.run(RunRequest(prompt="two", session_id="s2")))
    asyncio.run(runtime.run(RunRequest(prompt="three", session_id="s3")))

    assert list(runtime.sessions) == ["s2", "s3"]


def test_session_cache_evicts_idle_entry_when_new_session_acquires(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import RunRequest
    from agentkit_serve_common.runtime import RunResult

    import asyncio

    b_started = asyncio.Event()
    release_b = asyncio.Event()
    seen_sessions = {}

    async def fake_run_agent(agent, request, *, session=None):
        seen_sessions[session.session_id] = session
        if session.session_id == "b":
            b_started.set()
            await release_b.wait()
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    monkeypatch.setenv("AGENTKIT_SESSION_CACHE_MAX", "1")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()

    async def exercise():
        await runtime.run(RunRequest(prompt="one", session_id="a"))
        b_task = asyncio.create_task(runtime.run(RunRequest(prompt="two", session_id="b")))
        await asyncio.wait_for(b_started.wait(), timeout=1)
        state_while_b_runs = (
            list(runtime.sessions),
            runtime.sessions.get("b") is seen_sessions["b"],
            runtime.session_claims.get("b"),
            runtime.session_locks["b"].locked(),
        )
        release_b.set()
        await b_task
        return state_while_b_runs

    assert asyncio.run(exercise()) == (["b"], True, 1, True)


def test_session_cache_overflow_keeps_current_session_serialized(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import RunRequest
    from agentkit_serve_common.runtime import RunResult

    import asyncio

    busy_started = asyncio.Event()
    release_busy = asyncio.Event()
    first_current_started = asyncio.Event()
    release_first_current = asyncio.Event()
    second_current_started = asyncio.Event()
    current_sessions = []

    async def fake_run_agent(agent, request, *, session=None):
        if session.session_id == "busy":
            busy_started.set()
            await release_busy.wait()
        elif session.session_id == "current":
            current_sessions.append(session)
            if len(current_sessions) == 1:
                first_current_started.set()
                await release_first_current.wait()
            else:
                second_current_started.set()
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    monkeypatch.setenv("AGENTKIT_SESSION_CACHE_MAX", "1")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()

    async def exercise():
        busy = asyncio.create_task(runtime.run(RunRequest(prompt="hold", session_id="busy")))
        await asyncio.wait_for(busy_started.wait(), timeout=1)

        first_current = asyncio.create_task(runtime.run(RunRequest(prompt="one", session_id="current")))
        await asyncio.wait_for(first_current_started.wait(), timeout=1)
        cached_during_overflow = list(runtime.sessions)

        second_current = asyncio.create_task(runtime.run(RunRequest(prompt="two", session_id="current")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        serialized_while_first_running = not second_current_started.is_set()

        release_first_current.set()
        await asyncio.wait_for(second_current_started.wait(), timeout=1)
        release_busy.set()
        await asyncio.gather(busy, first_current, second_current)
        cached_after_saturation = list(runtime.sessions)
        await runtime.run(RunRequest(prompt="three", session_id="current"))
        return cached_during_overflow, serialized_while_first_running, cached_after_saturation

    cached_during_overflow, serialized_while_first_running, cached_after_saturation = asyncio.run(exercise())
    same_session_entry = len(current_sessions) == 3 and all(session is current_sessions[0] for session in current_sessions)

    assert (cached_during_overflow, serialized_while_first_running, cached_after_saturation, same_session_entry) == (
        ["busy", "current"],
        True,
        ["current"],
        True,
    )


def test_context_credential_uses_async_default_for_search(monkeypatch):
    from agentkit_serve_common.config import ContextProviderSpec

    class FakeAsyncCredential:
        pass

    class FakeSyncCredential:
        pass

    class FakeAioIdentity:
        DefaultAzureCredential = FakeAsyncCredential

    class FakeSyncIdentity:
        DefaultAzureCredential = FakeSyncCredential

    def fake_import(name):
        if name == "azure.identity.aio":
            return FakeAioIdentity
        if name == "azure.identity":
            return FakeSyncIdentity
        raise ImportError(name)

    monkeypatch.setattr(agent_factory.importlib, "import_module", fake_import)
    monkeypatch.delenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN", raising=False)
    monkeypatch.delenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND", raising=False)
    provider = ContextProviderSpec.model_validate({
        "type": "search",
        "endpointEnv": "SEARCH_ENDPOINT",
        "indexEnv": "SEARCH_INDEX",
    })

    credential = agent_factory._credential_for_context(
        provider,
        default_audience="https://search.azure.com/.default",
        async_credential=True,
    )

    assert isinstance(credential, FakeAsyncCredential)


def test_build_client_uses_generic_workload_identity_hook_for_model(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    calls = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(agent_factory, "OpenAIChatCompletionClient", FakeOpenAIClient)
    monkeypatch.setattr(agent_factory, "resolve_workload_identity_token", lambda audience: f"token-for-{audience}")
    monkeypatch.delenv("AGENTKIT_MODEL_WORKLOAD_IDENTITY_TOKEN", raising=False)
    monkeypatch.setenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN_COMMAND", "/bin/token")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://example.services.ai.azure.com/api/projects/proj/openai/v1",
            "name": "gpt-4.1-mini",
            "auth": {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"},
        },
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })

    client = agent_factory.build_client(spec)

    assert isinstance(client, FakeOpenAIClient)
    import asyncio
    assert asyncio.run(calls["api_key"]()) == "token-for-https://ai.azure.com/.default"


def test_memory_context_requires_explicit_scope(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    class FakeMemoryProvider:
        def __init__(self, **kwargs):
            pass

    class FakeModule:
        FoundryMemoryProvider = FakeMemoryProvider

    monkeypatch.setattr(agent_factory.importlib, "import_module", lambda name: FakeModule)
    monkeypatch.setattr(agent_factory, "_credential_for_context", lambda provider, default_audience, async_credential=False: "credential")
    monkeypatch.setenv("MEMORY_ENDPOINT", "https://example.services.ai.azure.com/api/projects/proj")
    monkeypatch.setenv("MEMORY_STORE_NAME", "agentkit-memory")
    monkeypatch.delenv("AGENTKIT_MEMORY_SCOPE", raising=False)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "context": {"providers": [{"type": "memory", "endpointEnv": "MEMORY_ENDPOINT", "storeNameEnv": "MEMORY_STORE_NAME"}]},
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)

    import asyncio
    import pytest
    with pytest.raises(agent_factory.AgentBuildError, match="AGENTKIT_MEMORY_SCOPE"):
        asyncio.run(runtime._build_context_providers())


def test_reused_session_runs_with_each_request_history(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import ConversationTurn, RunRequest
    from agentkit_serve_common.runtime import RunResult

    seen = []

    async def fake_run_agent(agent, request, *, session=None):
        seen.append((request.history, session))
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()
    request = RunRequest(
        prompt="current",
        history=(ConversationTurn(role="user", text="old"),),
        session_id="s1",
    )

    import asyncio
    asyncio.run(runtime.run(request))
    asyncio.run(runtime.run(request))

    assert [history for history, _ in seen] == [request.history, request.history]
    assert seen[0][1] is seen[1][1]


def test_discarded_failed_first_turn_replaces_session(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import ConversationTurn, RunRequest
    from agentkit_serve_common.runtime import RunResult

    seen_sessions = []

    async def fake_run_agent(agent, request, *, session=None):
        seen_sessions.append(session)
        if len(seen_sessions) == 1:
            raise RuntimeError("provider unavailable")
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()
    request = RunRequest(
        prompt="current",
        history=(ConversationTurn(role="user", text="old"),),
        session_id="s1",
    )

    import asyncio
    import pytest

    async def exercise():
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await runtime.run(request)
        await runtime.discard_session("s1")
        await runtime.run(request)
        await runtime.run(request)

    asyncio.run(exercise())

    assert seen_sessions[0] is not seen_sessions[1]
    assert seen_sessions[1] is seen_sessions[2]


def test_failed_reused_session_keeps_session_state(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import RunRequest
    from agentkit_serve_common.runtime import RunResult

    seen_sessions = []

    async def fake_run_agent(agent, request, *, session=None):
        seen_sessions.append(session)
        if len(seen_sessions) == 2:
            raise RuntimeError("provider unavailable")
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()
    request = RunRequest(prompt="current", session_id="s1")

    import asyncio
    import pytest

    async def exercise():
        await runtime.run(request)
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await runtime.run(request)
        await runtime.run(request)

    asyncio.run(exercise())

    assert seen_sessions[0] is seen_sessions[1] is seen_sessions[2]


def test_remote_mcp_disables_ping(monkeypatch):
    tool = ToolSpec.model_validate({
        "name": "toolbox",
        "type": "mcp",
        "transport": "streamable-http",
        "urlEnv": "TOOLBOX_ENDPOINT",
    })
    monkeypatch.setenv("TOOLBOX_ENDPOINT", "https://example.test/mcp")

    mcp_tool = agent_factory.build_tool(tool)

    assert getattr(mcp_tool, "_ping_available") is False


def test_build_mcp_skills_provider_uses_pinned_streamable_http_factory(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    calls = {}

    class FakeTransport:
        async def __aenter__(self):
            calls["transport_entered"] = True
            return "read", "write", lambda: None

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def fake_streamablehttp_client(**kwargs):
        calls.update(kwargs)
        return FakeTransport()

    class FakeClientSession:
        def __init__(self, read, write, *, read_timeout_seconds):
            calls["session_args"] = (read, write)
            calls["session_timeout"] = read_timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            calls["initialized"] = True

    import mcp.client.session as session_mod
    import mcp.client.streamable_http as streamable_mod

    monkeypatch.setattr(streamable_mod, "streamablehttp_client", fake_streamablehttp_client)
    monkeypatch.setattr(session_mod, "ClientSession", FakeClientSession)
    monkeypatch.setenv("TOOLBOX_ENDPOINT", "https://example.test/toolboxes/t/mcp")
    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "7")
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [{"name": "toolbox", "type": "mcp", "transport": "streamable-http", "urlEnv": "TOOLBOX_ENDPOINT"}],
        "context": {"providers": [{"type": "skills", "source": "mcp", "toolRef": "toolbox"}]},
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)

    import asyncio
    provider = asyncio.run(runtime._build_mcp_skills_provider(spec.context.providers[0]))
    asyncio.run(runtime.stack.aclose())

    assert provider.__class__.__name__ == "SkillsProvider"
    assert calls["url"] == "https://example.test/toolboxes/t/mcp"
    assert "httpx_client_factory" in calls
    assert "http_client" not in calls
    assert calls["initialized"] is True
    assert calls["session_timeout"].total_seconds() == 7



def test_result_usage_reads_usage_details_attributes():
    class Usage:
        input_token_count = 3
        output_token_count = 4
        total_token_count = 7

    class Result:
        usage_details = Usage()

    assert agent_factory._result_usage(Result()) == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_result_usage_reads_usage_details_to_dict():
    class Usage:
        def to_dict(self):
            return {"input_token_count": 5, "output_token_count": 6, "total_token_count": 11}

    class Result:
        usage_details = Usage()

    assert agent_factory._result_usage(Result()) == {
        "prompt_tokens": 5,
        "completion_tokens": 6,
        "total_tokens": 11,
    }



def test_concurrent_same_session_runs_are_serialized(monkeypatch):
    from agentkit_serve_common.config import AgentSpec
    from agentkit_serve_common.conversation import ConversationTurn, RunRequest
    from agentkit_serve_common.runtime import RunResult

    started: list[bool] = []

    import asyncio
    first_can_finish = asyncio.Event()

    async def fake_run_agent(agent, request, *, session=None):
        started.append(True)
        if len(started) == 1:
            await first_can_finish.wait()
        return RunResult(text="ok")

    monkeypatch.setattr(agent_factory, "run_agent", fake_run_agent)
    spec = AgentSpec.model_validate({
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    })
    runtime = agent_factory.MAFRuntime(spec)
    runtime.agent = object()
    request = RunRequest(
        prompt="current",
        history=(ConversationTurn(role="user", text="old"),),
        session_id="s1",
    )

    async def run_two():
        first = asyncio.create_task(runtime.run(request))
        await asyncio.sleep(0)
        second = asyncio.create_task(runtime.run(request))
        await asyncio.sleep(0.05)
        assert len(started) == 1
        first_can_finish.set()
        await asyncio.gather(first, second)

    asyncio.run(run_two())

    assert len(started) == 2


def test_maf_orka_offline_echo_bypasses_provider_runtime(monkeypatch):
    from agentkit_serve_common.config import AgentSpec

    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "x"},
            "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
            "instructions": "hi",
            "tools": [],
            "expose": {"openai": True, "port": 8080},
        }
    )
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "orka")
    monkeypatch.setenv("AGENTKIT_ORKA_OFFLINE_ECHO", "1")

    runtime = agent_factory.build_runtime(spec)

    assert runtime.__class__.__name__ == "OfflineEchoRuntime"
    monkeypatch.setenv("AGENTKIT_PROTOCOL", "acp")
    assert agent_factory.supports_acp_http_mcp() is True

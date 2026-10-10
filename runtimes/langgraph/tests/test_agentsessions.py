"""Real LangGraph/OpenAI SDK -> authenticated loopback -> native Harness.Connect."""
from __future__ import annotations

import asyncio
import logging
import hashlib
import os
from contextlib import asynccontextmanager

import grpc
import httpx
import openai
import pytest
import yaml
from langchain_core.caches import BaseCache
from langchain_core.globals import get_debug, get_llm_cache, get_verbose, set_debug, set_llm_cache, set_verbose
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import RunnableLambda
from langsmith import Client, tracing_context

from agentkit_serve import agent_factory
from agentkit_serve_common.agentsessions import (
    AgentsessionsConfigurationError,
    create_server,
    load_verified_agentsessions_binding,
)
from agentkit_serve_common.agentsessions._generated import common_pb2 as c
from agentkit_serve_common.agentsessions._generated import harness_pb2 as h
from agentkit_serve_common.agentsessions._generated import harness_pb2_grpc as g


def text(role, value):
    return c.Message(role=role, parts=[c.Part(text=c.TextPart(text=value))])


@pytest.fixture
def binding_file(tmp_path, monkeypatch):
    data = {
        "abiVersion": "v0",
        "metadata": {"name": "host-bound"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "http://127.0.0.1:1/v1",
            "name": "host-model",
            "apiKeyEnv": "LANGGRAPH_PROVIDER_KEY",
            "auth": {"type": "workload-identity-token", "audience": "https://identity.invalid"},
        },
        "instructions": "Only baked rules.",
        "env": [{"name": "LANGGRAPH_PROVIDER_KEY", "required": True}],
        "expose": {"openai": True, "port": 8080},
    }
    path = tmp_path / "agent.yaml"
    monkeypatch.delenv("LANGGRAPH_PROVIDER_KEY", raising=False)
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST", "sha256:" + "c" * 64)

    def load(*, model=None, **changes):
        data.update(changes)
        data["model"].update(model or {})
        path.write_bytes(yaml.safe_dump(data).encode())
        monkeypatch.setenv(
            "AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST",
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return load_verified_agentsessions_binding(path)

    return load


@pytest.fixture
def binding(binding_file):
    return binding_file()


def hook():
    runner = getattr(agent_factory, "run_agentsessions", None)
    assert callable(runner), "LangGraph agentsessions runner missing"
    return runner


@asynccontextmanager
async def live(binding, runner=None):
    server = create_server(
        binding, runner=runner or getattr(agent_factory, "run_agentsessions", None),
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield g.HarnessStub(channel)
    finally:
        await server.stop(0)


async def turn(stub, *, inputs, history=(), config=b"\xffopaque\x00", reply="host answer", control=None, wait_admission=False):
    while True:
        call = stub.Connect()
        await call.write(h.ControllerFrame(
            execution_id="exec-1",
            session="same-session",
            start=h.Start(inputs=[text("user", value) for value in inputs], history=history, config=config),
        ))
        event = await asyncio.wait_for(call.read(), 5)
        if wait_admission and event.kind == c.EVENT_END and event.end.error.code == 8:
            assert await call.read() is grpc.aio.EOF
            continue
        break
    assert event.kind == c.EVENT_MODEL_CALL, str(event)
    if control == "cancel":
        await call.write(h.ControllerFrame(execution_id="exec-1", session="same-session", cancel=h.Cancel()))
    elif control == "eof":
        await call.done_writing()
    elif control == "disconnect":
        call.cancel()
        return event, None
    else:
        result = h.ModelResult(model_call_id=event.model.id, message=text("assistant", reply))
        if control == "malformed":
            result.message.parts[0].CopyFrom(c.Part(data=c.DataPart()))
        await call.write(h.ControllerFrame(execution_id="exec-1", session="same-session", model=result))
    end = await asyncio.wait_for(call.read(), 5)
    assert end.kind == c.EVENT_END  # The host owns OUTPUT/usage; no duplicate events.
    assert await call.read() is grpc.aio.EOF
    return event, end


@pytest.mark.parametrize("instructions,inputs,prior,want", [
    ("Only baked rules.", ["first", "latest"], [("user", "past"), ("assistant", "answer")], [("system", "Only baked rules."), ("user", "past"), ("assistant", "answer"), ("user", "first"), ("user", "latest")]),
    ("Only baked rules.", [""], [], [("system", "Only baked rules."), ("user", "")]),
    ("Only baked rules.", [], [], [("system", "Only baked rules.")]),
    ("Only baked rules.", [], [("user", "old"), ("assistant", "reply")], [("system", "Only baked rules."), ("user", "old"), ("assistant", "reply")]),
    ("Only baked rules.", ["", "last"], [("user", ""), ("assistant", "")], [("system", "Only baked rules."), ("user", ""), ("assistant", ""), ("user", ""), ("user", "last")]),
    ("", [], [], []),
    ("", [""], [], [("user", "")]),
])
def test_actual_sdk_preserves_ordered_empty_turns_and_opaque_config(binding_file, instructions, inputs, prior, want):
    # Dropping empty turns, inventing an inputless user, or injecting Config breaks this.
    binding = binding_file(instructions=instructions)

    async def check():
        seen = []

        async def runner(binding, request, exchange):
            seen.append(request.config)
            result = await hook()(binding, request, exchange)
            assert result is None

        history = [
            c.Event(kind=c.EVENT_INPUT if role == "user" else c.EVENT_OUTPUT, message=text(role, value))
            for role, value in prior
        ]
        async with live(binding, runner) as stub:
            event, end = await turn(stub, inputs=inputs, history=history)
            assert [(m.role, "".join(p.text.text for p in m.parts)) for m in event.model.messages] == want
            assert end.end.state == "COMPLETED"
            assert event.model.model == "host-model"
            assert dict(event.model.params) == {} and event.model.input_hash == ""
        assert seen == [b"\xffopaque\x00"]

    asyncio.run(check())


@pytest.mark.parametrize("model_name", ["o1", "o1-mini", "o1-preview", "o3", "host-model"])
def test_actual_sdk_preserves_baked_model_without_implicit_options(binding_file, model_name):
    # o1 defaults temperature to 1 and o-series rewrites system to developer;
    # neither extra options nor developer roles belong to the strict text profile.
    binding = binding_file(model={"name": model_name})

    async def check():
        async with live(binding) as stub:
            event, end = await turn(stub, inputs=["hello"])
            assert event.model.model == model_name
            assert dict(event.model.params) == {}
            assert [(m.role, m.parts[0].text.text) for m in event.model.messages] == [
                ("system", "Only baked rules."), ("user", "hello"),
            ]
            assert end.end.state == "COMPLETED"

    asyncio.run(check())


def test_empty_host_completion_is_not_retried_or_duplicated(binding):
    async def check():
        async with live(binding) as stub:
            _, end = await turn(stub, inputs=["hello"], reply="")
            assert end.end.state == "COMPLETED"

    asyncio.run(check())


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_global_console_settings_stay_enabled_without_exporting_agentsessions(binding_file, capsys, debug, nested):
    binding = binding_file(instructions="private-baked-instructions-marker")
    previous_debug, previous_verbose = get_debug(), get_verbose()
    set_debug(debug)
    set_verbose(True)

    async def check():
        history = [c.Event(kind=c.EVENT_INPUT, message=text("user", "private-history-marker"))]
        async def runner(binding, request, exchange):
            if nested:
                async def invoke(_):
                    return await hook()(binding, request, exchange)
                return await RunnableLambda(invoke).ainvoke("ordinary-parent-input")
            return await hook()(binding, request, exchange)

        async with live(binding, runner) as stub:
            _, end = await turn(
                stub, inputs=["private-input-marker"], history=history,
                reply="private-host-output-marker",
            )
            assert end.end.state == "COMPLETED"
            assert get_debug() is debug
            assert get_verbose() is True
        captured = capsys.readouterr()
        assert not any(marker in captured.out + captured.err for marker in [
            "private-baked-instructions-marker", "private-history-marker",
            "private-input-marker", "private-host-output-marker",
        ])
        # Other protocols and tasks must retain the ambient debug behavior.
        if debug:
            await RunnableLambda(lambda value: value).ainvoke("ordinary-debug-control")
            assert "ordinary-debug-control" in capsys.readouterr().out
        assert get_debug() is debug
        assert get_verbose() is True

    try:
        asyncio.run(check())
    finally:
        set_debug(previous_debug)
        set_verbose(previous_verbose)


def test_parent_graph_checkpoints_do_not_persist_agentsessions(binding_file):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph

    binding = binding_file(instructions="private-baked-instructions-marker")
    saver = InMemorySaver()

    async def runner(binding, request, exchange):
        async def invoke(_):
            await hook()(binding, request, exchange)
            return {"ordinary": "parent-finished"}

        graph = StateGraph(dict)
        graph.add_node("invoke", invoke)
        graph.set_entry_point("invoke")
        graph.set_finish_point("invoke")
        parent = graph.compile(checkpointer=saver)
        await parent.ainvoke(
            {"ordinary": "parent-input"},
            config={"configurable": {"thread_id": "ordinary-parent-thread"}},
        )

    async def check():
        async with live(binding, runner) as stub:
            _, end = await turn(stub, inputs=["private-input-marker"], reply="private-host-output-marker")
            assert end.end.state == "COMPLETED"
        checkpoints = list(saver.list(None))
        assert checkpoints
        payload = repr([item.checkpoint for item in checkpoints])
        assert "parent-input" in payload and "parent-finished" in payload
        assert not any(marker in payload for marker in [
            "private-baked-instructions-marker", "private-input-marker", "private-host-output-marker",
        ])

    asyncio.run(check())


class PoisonCache(BaseCache):
    """A real global cache whose use would skip the journaled model effect."""

    def __init__(self):
        self.effects = []

    def lookup(self, prompt, llm_string):
        self.effects.append("lookup")
        return [ChatGeneration(message=AIMessage(content="unhosted cache answer"))]

    def update(self, prompt, llm_string, return_val):
        self.effects.append("update")

    def clear(self, **kwargs):
        self.effects.append("clear")


@pytest.mark.parametrize("ambient_headers", [
    "Authorization: Bearer unrelated-ambient-token",
    "authorization: Bearer unrelated-ambient-token\naUtHoRiZaTiOn: Bearer another-ambient-token",
])
def test_fresh_resources_ignore_ambient_provider_proxy_auth_tracing_and_cache(binding_file, monkeypatch, ambient_headers):
    async def check():
        hits, clients, models, graphs, traces, requests = [], [], [], [], [], []

        async def trap(reader, writer):
            hits.append(await reader.read(4096))
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(trap, "127.0.0.1", 0)
        url = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]) + "/v1"
        binding = binding_file(model={"baseURL": url})
        for name in ["OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"]:
            monkeypatch.setenv(name, url)
        for name in ["OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID", "LANGSMITH_API_KEY"]:
            monkeypatch.setenv(name, "unused-ambient-value")
        monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", ambient_headers)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        real_client, real_model, real_graph = openai.AsyncOpenAI, agent_factory.ChatOpenAI, agent_factory.create_agent
        real_send = httpx.AsyncHTTPTransport.handle_async_request

        def client(**kwargs):
            value = real_client(**kwargs)
            clients.append(value)
            return value

        def model(*args, **kwargs):
            value = real_model(*args, **kwargs)
            models.append(value)
            return value

        def graph(*args, **kwargs):
            value = real_graph(*args, **kwargs)
            graphs.append(value)
            return value

        async def send(transport, request):
            requests.append(request)
            return await real_send(transport, request)

        def trace(*args, **kwargs):
            traces.append((args, kwargs))

        def forbidden(*args, **kwargs):
            raise AssertionError("normal provider or unowned sync client used")

        monkeypatch.setattr(openai, "AsyncOpenAI", client)
        monkeypatch.setattr(openai, "OpenAI", forbidden)
        monkeypatch.setattr(agent_factory, "ChatOpenAI", model)
        monkeypatch.setattr(agent_factory, "create_agent", graph)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
        monkeypatch.setattr(Client, "create_run", trace)
        monkeypatch.setattr(Client, "update_run", trace)
        monkeypatch.setattr(agent_factory, "build_runtime", forbidden)
        monkeypatch.setattr(agent_factory, "build_model", forbidden)
        monkeypatch.setattr(agent_factory, "resolve_api_key", forbidden)
        monkeypatch.setattr(agent_factory, "_to_messages", forbidden)
        previous_cache, cache = get_llm_cache(), PoisonCache()
        set_llm_cache(cache)
        before = dict(os.environ)
        try:
            with tracing_context(enabled=True, tags=["ambient"], metadata={"opaque": "not invocation config"}):
                async with live(binding) as stub:
                    for _ in range(2):
                        event, end = await turn(stub, inputs=["identical"])
                        assert end.end.state == "COMPLETED"
                        assert [(m.role, m.parts[0].text.text) for m in event.model.messages] == [
                            ("system", "Only baked rules."), ("user", "identical"),
                        ]
            assert hits == traces == cache.effects == []
            assert len(clients) == len(models) == len(graphs) == len(requests) == 2
            assert clients[0] is not clients[1] and models[0] is not models[1] and graphs[0] is not graphs[1]
            assert all(client.is_closed() for client in clients)
            assert clients[0].api_key != clients[1].api_key
            for client, request in zip(clients, requests):
                assert str(request.url).startswith(str(client.base_url))
                assert str(client.base_url).rstrip("/") != url
                assert request.headers.get_list("authorization") == ["Bearer " + client.api_key]
                assert request.headers.get("openai-organization") == ""
                assert request.headers.get("openai-project") == ""
            assert dict(os.environ) == before
            assert get_llm_cache() is cache
        finally:
            set_llm_cache(previous_cache)
            server.close()
            await server.wait_closed()

    asyncio.run(check())


def test_bridge_failure_has_no_sdk_retry_or_provider_fallback(binding):
    async def check():
        async with live(binding) as stub:
            # Valid protobuf text expands beyond the bridge JSON bound (HTTP 502).
            # An SDK retry would produce model-2 rather than the required END.
            _, end = await turn(stub, inputs=["hello"], reply="\x00" * (200 * 1024))
            assert end.end.state == "FAILED"
            assert end.end.error.code == 13
            assert end.end.error.description == "execution failed"
            _, end = await turn(stub, inputs=["next"])
            assert end.end.state == "COMPLETED"

    asyncio.run(check())


def test_sdk_does_not_follow_redirect_away_from_owned_loopback(binding, monkeypatch):
    async def check():
        hits, requests = [], []

        async def trap(reader, writer):
            hits.append(await reader.read(4096))
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(trap, "127.0.0.1", 0)
        destination = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]) + "/outside"
        real_send = httpx.AsyncHTTPTransport.handle_async_request

        async def redirect(transport, request):
            requests.append(str(request.url))
            response = await real_send(transport, request)
            if len(requests) == 1:
                # Host completes the real bridge effect before simulating a
                # redirect at the external HTTP response boundary.
                await response.aclose()
                return httpx.Response(307, headers={"location": destination}, request=request)
            return response

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", redirect)
        try:
            async with live(binding) as stub:
                _, end = await turn(stub, inputs=["first"])
                assert end.end.state == "FAILED" and end.end.error.code == 13
                assert len(requests) == 1 and hits == []
                _, end = await turn(stub, inputs=["next"])
                assert end.end.state == "COMPLETED"
                assert len(requests) == 2 and hits == []
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(check())


@pytest.mark.parametrize("control", ["reply", "cancel"])
def test_host_wait_has_connect_bound_but_no_transport_read_deadline(binding, monkeypatch, control):
    async def check():
        phases = []
        expire, checked = asyncio.Event(), asyncio.Event()
        real_send = httpx.AsyncHTTPTransport.handle_async_request

        async def delayed_send(transport, request):
            timeout = request.extensions["timeout"]
            phases.append(dict(timeout))
            response = asyncio.create_task(real_send(transport, request))
            try:
                # Accelerate only a finite read deadline, keeping the actual HTTP/SDK.
                await expire.wait()
                checked.set()
                if timeout["read"] is not None:
                    raise httpx.ReadTimeout("controlled read deadline", request=request)
                return await response
            finally:
                if not response.done():
                    response.cancel()
                await asyncio.gather(response, return_exceptions=True)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", delayed_send)
        async with live(binding) as stub:
            call, pending = stub.Connect(), None
            try:
                await call.write(h.ControllerFrame(execution_id="slow", start=h.Start(inputs=[text("user", "wait")])))
                event = await asyncio.wait_for(call.read(), 5)
                assert event.kind == c.EVENT_MODEL_CALL
                pending = asyncio.create_task(call.read())
                expire.set()
                await asyncio.wait_for(checked.wait(), 2)
                assert phases == [{"connect": 5, "read": None, "write": None, "pool": None}]
                assert not pending.done()
                if control == "cancel":
                    await call.write(h.ControllerFrame(execution_id="slow", cancel=h.Cancel()))
                else:
                    await call.write(h.ControllerFrame(execution_id="slow", model=h.ModelResult(
                        model_call_id=event.model.id, message=text("assistant", "delayed host answer"),
                    )))
                end = await asyncio.wait_for(pending, 5)
                assert end.kind == c.EVENT_END
                assert end.end.state == ("CANCELED" if control == "cancel" else "COMPLETED")
                assert await call.read() is grpc.aio.EOF
                _, end = await turn(stub, inputs=["next"])
                assert end.end.state == "COMPLETED"
            finally:
                expire.set()
                call.cancel()
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.parametrize("control", ["cancel", "eof", "disconnect", "malformed"])
def test_blocked_effect_closes_clients_and_listener_before_next_admission(binding, monkeypatch, control):
    async def check():
        clients, urls, unhandled = [], [], []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        real_client = openai.AsyncOpenAI

        def client(**kwargs):
            value = real_client(**kwargs)
            clients.append(value)
            urls.append(httpx.URL(str(value.base_url)))
            return value

        monkeypatch.setattr(openai, "AsyncOpenAI", client)
        async with live(binding) as stub:
            _, end = await turn(stub, inputs=["first"], control=control)
            if end is not None:
                assert end.end.state == ("FAILED" if control == "malformed" else "CANCELED")
                assert clients[0].is_closed()
                with pytest.raises(OSError):
                    await asyncio.open_connection(urls[0].host, urls[0].port)
            _, end = await asyncio.wait_for(turn(stub, inputs=["next"], wait_admission=control == "disconnect"), 5)
            assert end.end.state == "COMPLETED"
            assert all(client.is_closed() for client in clients)
            for url in urls:
                with pytest.raises(OSError):
                    await asyncio.open_connection(url.host, url.port)
        assert unhandled == []

    asyncio.run(check())


def test_repeated_cancel_during_sdk_close_keeps_admission_reserved(binding, monkeypatch):
    async def check():
        closing, resume = asyncio.Event(), asyncio.Event()
        clients, transports, urls, unhandled = [], [], [], []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        real_client, real_close = openai.AsyncOpenAI, httpx.AsyncClient.aclose

        def client(**kwargs):
            value = real_client(**kwargs)
            clients.append(value)
            transports.append(kwargs["http_client"])
            urls.append(httpx.URL(str(value.base_url)))
            return value

        async def runner(binding, request, exchange):
            if clients:
                assert clients[0].is_closed() and transports[0].is_closed
                with pytest.raises(OSError):
                    await asyncio.open_connection(urls[0].host, urls[0].port)
            await hook()(binding, request, exchange)

        async def close(http):
            if transports and http is transports[0]:
                # Pause the actual owned transport teardown, not graph execution.
                closing.set()
                await resume.wait()
            await real_close(http)

        monkeypatch.setattr(openai, "AsyncOpenAI", client)
        monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
        async with live(binding, runner) as stub:
            call = stub.Connect()
            try:
                await call.write(h.ControllerFrame(execution_id="cancel-close", start=h.Start(inputs=[text("user", "wait")])))
                assert (await asyncio.wait_for(call.read(), 5)).kind == c.EVENT_MODEL_CALL
                await call.write(h.ControllerFrame(execution_id="cancel-close", cancel=h.Cancel()))
                await asyncio.wait_for(closing.wait(), 5)
                call.cancel()  # Disconnect while the cancellation teardown is awaiting close.
                call.cancel()
                denied = stub.Connect()
                await denied.write(h.ControllerFrame(execution_id="denied", start=h.Start(inputs=[text("user", "not admitted")])))
                end = await asyncio.wait_for(denied.read(), 5)
                assert end.kind == c.EVENT_END and end.end.error.code == 8
                assert await denied.read() is grpc.aio.EOF
                assert len(clients) == 1 and not clients[0].is_closed()
                resume.set()
                # Check resources at runner entry: admission cannot precede teardown.
                _, end = await asyncio.wait_for(turn(stub, inputs=["next"], wait_admission=True), 5)
                assert end.end.state == "COMPLETED"
                assert len(clients) == 2 and all(value.is_closed() for value in clients)
                for url in urls:
                    with pytest.raises(OSError):
                        await asyncio.open_connection(url.host, url.port)
            finally:
                resume.set()
                call.cancel()
        assert unhandled == []

    asyncio.run(check())


@pytest.mark.parametrize("changes,expected_message", [
    ({"tools": [{"name": "direct", "command": ["not-executed"]}]}, "agentsessions rejects baked direct tools"),
    ({"brokeredTools": [{"name": "brokered", "description": "Read records.", "brokeredClass": "read", "parameters": {"type": "object"}}]}, "agentsessions rejects baked brokeredTools"),
    ({"context": {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}}, "agentsessions rejects baked context providers"),
])
def test_unsupported_baked_tools_and_context_are_refused(binding_file, changes, expected_message):
    with pytest.raises(AgentsessionsConfigurationError, match=expected_message):
        binding_file(**changes)


@pytest.mark.parametrize("start", [
    h.Start(inputs=[c.Message(role="user", parts=[c.Part(data=c.DataPart())])]),
    h.Start(inputs=[text("system", "request instructions")]),
    h.Start(history=[c.Event(kind=c.EVENT_TOOL_CALL)]),
    h.Start(history=[c.Event(kind=c.EVENT_MODEL_CALL, model=c.ModelCall(params={"temperature": "0"}))]),
])
def test_unsupported_start_content_and_options_emit_no_model_effect(binding, start):
    async def check():
        async with live(binding) as stub:
            call = stub.Connect()
            await call.write(h.ControllerFrame(execution_id="unsupported", start=start))
            end = await asyncio.wait_for(call.read(), 5)
            assert end.kind == c.EVENT_END and end.end.state == "FAILED"
            assert end.end.error.code == 12
            assert await call.read() is grpc.aio.EOF
            _, end = await turn(stub, inputs=["valid next"])
            assert end.end.state == "COMPLETED"

    asyncio.run(check())


def test_sdk_debug_diagnostics_do_not_export_execution_content(binding_file, caplog):
    logger = logging.getLogger("openai._base_client")
    caplog.set_level(logging.DEBUG, logger="openai")
    filters = list(logger.filters)
    binding = binding_file(instructions="private-sdk-instructions-marker")

    async def check():
        history = [c.Event(kind=c.EVENT_INPUT, message=text("user", "private-sdk-history-marker"))]
        async with live(binding) as stub:
            _, end = await turn(
                stub, inputs=["private-sdk-input-marker"], history=history,
                reply="private-sdk-output-marker",
            )
            assert end.end.state == "COMPLETED"

    asyncio.run(check())
    assert not any(marker in "\n".join(caplog.messages) for marker in [
        "private-sdk-instructions-marker", "private-sdk-history-marker",
        "private-sdk-input-marker", "private-sdk-output-marker",
    ])
    assert logger.filters == filters
    assert logger.getEffectiveLevel() == logging.DEBUG
    logger.debug("ordinary-sdk-debug-control")
    assert "ordinary-sdk-debug-control" in caplog.messages

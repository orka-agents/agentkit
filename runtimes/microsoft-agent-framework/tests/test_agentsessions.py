"""Real MAF/OpenAI SDK -> loopback HTTP -> native Harness.Connect (offline)."""
from __future__ import annotations

import asyncio
import logging
import hashlib
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager

import grpc
import httpx
import pytest
import yaml

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
def bake(tmp_path, monkeypatch):
    def build(*, model=None, **extra):
        data = {
            "abiVersion": "v0",
            "metadata": {"name": "host-bound"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "http://127.0.0.1:1/v1",
                "name": "host-model",
                "apiKeyEnv": "MAF_PROVIDER_KEY",
                "auth": {"type": "workload-identity-token", "audience": "https://identity.invalid"},
            },
            "instructions": "Only baked rules.",
            "env": [{"name": "MAF_PROVIDER_KEY", "required": True}],
            "expose": {"openai": True, "port": 8080},
            **extra,
        }
        data["model"].update(model or {})
        path = tmp_path / "agent.yaml"
        path.write_bytes(yaml.safe_dump(data).encode())
        monkeypatch.setenv(
            "AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST",
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST", "sha256:" + "c" * 64)
        monkeypatch.delenv("MAF_PROVIDER_KEY", raising=False)
        return load_verified_agentsessions_binding(path)
    return build


@pytest.fixture
def binding(bake):
    return bake()


@asynccontextmanager
async def live(binding, runner=None):
    # With the hook missing, the real native service fails closed (RED).
    server = create_server(binding, runner=runner or getattr(agent_factory, "run_agentsessions", None))
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
            execution_id="exec-1", session="same-session",
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
        return event, []
    else:
        result = h.ModelResult(model_call_id=event.model.id, message=text("assistant", reply))
        if control == "malformed":
            result.message.parts[0].CopyFrom(c.Part(data=c.DataPart()))
        await call.write(h.ControllerFrame(execution_id="exec-1", session="same-session", model=result))
    end = await asyncio.wait_for(call.read(), 5)
    assert end.kind == c.EVENT_END  # Host owns OUTPUT/usage; neither may be duplicated.
    assert await call.read() is grpc.aio.EOF
    return event, [end]


def projected(event):
    return [(message.role, "".join(part.text.text for part in message.parts)) for message in event.model.messages]


def test_native_rpc_actual_sdk_mediates_without_duplicate_output(binding):
    async def check():
        async with live(binding) as stub:
            event, ends = await turn(stub, inputs=["hello"])
            assert projected(event) == [("system", "Only baked rules."), ("user", "hello")]
            assert event.model.model == "host-model"
            assert ends[0].end.state == "COMPLETED"
    asyncio.run(check())


def _assert_ambient_telemetry_is_protocol_local(binding):
    """Run in a child process: OTel globals can only be installed once publicly."""
    from agent_framework import Agent, Message
    from agent_framework.openai import OpenAIChatCompletionClient
    from agent_framework.observability import OBSERVABILITY_SETTINGS
    from openai import AsyncOpenAI
    from opentelemetry import metrics, trace
    from opentelemetry._logs import get_logger_provider, set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from agentkit_serve_common.agentsessions.bridge import loopback_bridge

    spans, logs, reader = InMemorySpanExporter(), InMemoryLogRecordExporter(), InMemoryMetricReader()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    meter_provider = MeterProvider(metric_readers=[reader])
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    set_logger_provider(logger_provider)
    before_env = dict(os.environ)
    before_settings = vars(OBSERVABILITY_SETTINGS).copy()

    async def normal_runner(binding, request, exchange):
        # Positive control: real default-enabled Agent + client, same native/SDK
        # path. A normal runtime must remain instrumented after mediated runs.
        async with loopback_bridge(exchange) as local:
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as http:
                async with AsyncOpenAI(
                    base_url=local.base_url, api_key=local.token, http_client=http,
                    max_retries=0, timeout=httpx.Timeout(None, connect=5),
                ) as sdk:
                    client = OpenAIChatCompletionClient(
                        model=binding.spec.model.name, async_client=sdk,
                        api_key=local.token, base_url=local.base_url, org_id="",
                    )
                    async with Agent(client=client, instructions=binding.spec.instructions) as agent:
                        await agent.run([Message(role="user", contents=[request.prompt])], stream=False)

    def framework_metrics():
        data = reader.get_metrics_data()
        return [] if data is None else [
            metric for resource in data.resource_metrics for scope in resource.scope_metrics
            if scope.scope.name == "agent_framework" for metric in scope.metrics
        ]

    async def check():
        async with live(binding) as stub:
            event, ends = await turn(
                stub, inputs=["private-input-marker"], config=b"private-config-marker",
                reply="private-output-marker",
            )
            assert projected(event) == [
                ("system", "private-instructions-marker"), ("user", "private-input-marker"),
            ]
            assert ends[0].end.state == "COMPLETED"
        exported = spans.get_finished_spans()
        payload = repr([(span.name, dict(span.attributes)) for span in exported])
        assert all(marker not in payload for marker in [
            "private-input-marker", "private-config-marker", "private-output-marker",
            "private-instructions-marker", "private-provider-secret-marker",
        ]), payload
        assert not any(span.instrumentation_scope.name == "agent_framework" for span in exported), payload
        assert not any(log.instrumentation_scope.name == "agent_framework" for log in logs.get_finished_logs())
        assert framework_metrics() == []
        assert trace.get_tracer_provider() is tracer_provider
        assert metrics.get_meter_provider() is meter_provider
        assert get_logger_provider() is logger_provider
        assert dict(os.environ) == before_env
        assert vars(OBSERVABILITY_SETTINGS) == before_settings

        async with live(binding, normal_runner) as stub:
            _, ends = await turn(stub, inputs=["normal-input-marker"], reply="normal-output-marker")
            assert ends[0].end.state == "COMPLETED"
        exported = [
            span for span in spans.get_finished_spans()
            if span.instrumentation_scope.name == "agent_framework"
        ]
        assert sorted(span.attributes["gen_ai.operation.name"] for span in exported) == [
            "chat", "invoke_agent",
        ]
        payload = repr([dict(span.attributes) for span in exported])
        assert all(marker in payload for marker in [
            "normal-input-marker", "normal-output-marker", "private-instructions-marker",
        ]), payload
        assert framework_metrics()
        normal_logs = [
            log for log in logs.get_finished_logs()
            if log.instrumentation_scope.name == "agent_framework"
        ]
        # Older supported SDKs export spans/metrics but lack message-event logs.
        # Keep that capability's positive control strict when it is available.
        if getattr(OBSERVABILITY_SETTINGS, "enable_message_events", False):
            assert normal_logs
            payload = repr([(log.log_record.body, log.log_record.attributes) for log in normal_logs])
            assert all(marker in payload for marker in [
                "normal-input-marker", "normal-output-marker", "private-instructions-marker",
            ]), payload

    try:
        asyncio.run(check())
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()
        logger_provider.shutdown()


def test_actual_sdk_does_not_export_to_ambient_telemetry(bake, tmp_path):
    # Removing either protocol-specific telemetry suppression leaks real spans.
    bake(instructions="private-instructions-marker")
    child_env = {
        **os.environ,
        "ENABLE_INSTRUMENTATION": "true",
        "ENABLE_SENSITIVE_DATA": "true",
        "ENABLE_MESSAGE_EVENTS": "true",
        "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
        "OPENAI_API_KEY": "private-provider-secret-marker",
    }
    result = subprocess.run(
        [sys.executable, "-c", (
            "import runpy, sys; ns = runpy.run_path(sys.argv[1]); "
            "binding = ns['load_verified_agentsessions_binding'](sys.argv[2]); "
            "ns['_assert_ambient_telemetry_is_protocol_local'](binding)"
        ), __file__, str(tmp_path / "agent.yaml")],
        env=child_env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("instructions,inputs,prior,want", [
    ("Only baked rules.", ["first", "latest"], [("user", "past"), ("assistant", "answer")], [("system", "Only baked rules."), ("user", "past"), ("assistant", "answer"), ("user", "first"), ("user", "latest")]),
    ("Only baked rules.", [""], [], [("system", "Only baked rules."), ("user", "")]),
    ("Only baked rules.", [], [], [("system", "Only baked rules.")]),
    ("Only baked rules.", [], [("user", "old"), ("assistant", "reply")], [("system", "Only baked rules."), ("user", "old"), ("assistant", "reply")]),
    ("Only baked rules.", ["", "last"], [("user", ""), ("assistant", "")], [("system", "Only baked rules."), ("user", ""), ("assistant", ""), ("user", ""), ("user", "last")]),
    ("", [], [], []),
    ("", [""], [], [("user", "")]),
    ("", [], [("assistant", "")], [("assistant", "")]),
])
def test_actual_sdk_preserves_history_empty_input_boundaries_and_opaque_config(bake, instructions, inputs, prior, want):
    binding = bake(instructions=instructions)
    async def check():
        seen = []
        async def runner(binding, request, exchange):
            seen.append(request.config)
            return await agent_factory.run_agentsessions(binding, request, exchange)
        history = [c.Event(kind=c.EVENT_INPUT if role == "user" else c.EVENT_OUTPUT, message=text(role, value)) for role, value in prior]
        async with live(binding, runner) as stub:
            event, ends = await turn(stub, inputs=inputs, history=history)
            assert projected(event) == want
            assert dict(event.model.params) == {} and event.model.input_hash == ""
            assert ends[0].end.state == "COMPLETED"
        assert seen == [b"\xffopaque\x00"]
    asyncio.run(check())


@pytest.mark.parametrize("ambient_headers", [
    "Authorization: Bearer unrelated-ambient-token",
    "authorization: Bearer unrelated-ambient-token\naUtHoRiZaTiOn: Bearer another-ambient-token",
])
def test_fresh_local_sdk_per_start_ignores_provider_proxy_and_ambient_auth(bake, monkeypatch, ambient_headers):
    async def check():
        hits, clients, agents, requests = [], [], [], []
        async def trap(reader, writer):
            hits.append(await reader.read(4096))
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(trap, "127.0.0.1", 0)
        url = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]) + "/v1"
        binding = bake(model={"baseURL": url})
        for key in ["OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            monkeypatch.setenv(key, url)
        monkeypatch.setenv("NO_PROXY", "")
        for key in ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_ORG", "OPENAI_PROJECT_ID", "OPENAI_MODEL", "OPENAI_CHAT_COMPLETION_MODEL"]:
            monkeypatch.setenv(key, "unused-ambient-value")
        monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", ambient_headers)
        import openai
        real_client, real_agent = openai.AsyncOpenAI, agent_factory.RawAgent
        real_send = httpx.AsyncHTTPTransport.handle_async_request
        def client(**kwargs):
            value = real_client(**kwargs)
            clients.append(value)
            return value
        def agent(*args, **kwargs):
            value = real_agent(*args, **kwargs)
            agents.append(value)
            return value
        async def send(transport, request):
            requests.append(request)
            return await real_send(transport, request)
        def forbidden(*args, **kwargs):
            raise AssertionError("normal provider builder used")
        monkeypatch.setattr(openai, "AsyncOpenAI", client)
        monkeypatch.setattr(agent_factory, "RawAgent", agent)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
        for name in ["build_runtime", "build_client", "build_agent", "resolve_api_key", "resolve_workload_identity_token"]:
            monkeypatch.setattr(agent_factory, name, forbidden)
        before = dict(os.environ)
        try:
            async with live(binding) as stub:
                for value in ["one", "two"]:
                    event, ends = await turn(stub, inputs=[value])
                    assert projected(event) == [("system", "Only baked rules."), ("user", value)]
                    assert ends[0].end.state == "COMPLETED"
            assert hits == []
            assert len(clients) == len(agents) == len(requests) == 2
            assert clients[0] is not clients[1] and agents[0] is not agents[1]
            assert all(value.is_closed() for value in clients)
            assert clients[0].api_key != clients[1].api_key
            for sdk, request in zip(clients, requests, strict=True):
                assert sdk.max_retries == 0
                assert sdk.api_key != "unused-ambient-value"
                assert str(sdk.base_url).startswith("http://127.0.0.1:")
                assert str(sdk.base_url).rstrip("/") != url
                assert request.headers.get_list("authorization") == ["Bearer " + sdk.api_key]
                assert request.headers.get("openai-organization", "") == ""
                assert request.headers.get("openai-project", "") == ""
                body = json.loads(request.content)
                assert set(body) <= {"model", "messages", "stream"}
                assert body["model"] == "host-model" and not body.get("stream", False)
            assert dict(os.environ) == before
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(check())


def test_sdk_bridge_failure_has_no_retry_or_provider_fallback(binding):
    async def check():
        async with live(binding) as stub:
            # Bounded protobuf expands past the JSON limit. A retry would emit
            # model-2 rather than the required single terminal failure.
            _, ends = await turn(stub, inputs=["hello"], reply="\x00" * (200 * 1024))
            assert ends[0].end.state == "FAILED"
            assert ends[0].end.error.code == 13
            assert ends[0].end.error.description == "execution failed"
            _, ends = await turn(stub, inputs=["next"])
            assert ends[0].end.state == "COMPLETED"
    asyncio.run(check())


@pytest.mark.parametrize("control", ["reply", "cancel"])
def test_actual_sdk_host_wait_has_no_transport_read_deadline(binding, monkeypatch, control):
    async def check():
        phases = []
        expire, checked = asyncio.Event(), asyncio.Event()
        real_send = httpx.AsyncHTTPTransport.handle_async_request
        async def delayed_send(transport, request):
            timeout = request.extensions["timeout"]
            phases.append(dict(timeout))
            response = asyncio.create_task(real_send(transport, request))
            try:
                # Accelerate only finite read deadlines; HTTP and SDK stay real.
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
            call = stub.Connect()
            pending = None
            try:
                await call.write(h.ControllerFrame(execution_id="slow", start=h.Start(inputs=[text("user", "wait for host")])))
                model = await asyncio.wait_for(call.read(), 5)
                assert model.kind == c.EVENT_MODEL_CALL
                pending = asyncio.create_task(call.read())
                expire.set()
                await asyncio.wait_for(checked.wait(), 2)
                assert phases == [{"connect": 5, "read": None, "write": None, "pool": None}]
                assert not pending.done()
                if control == "cancel":
                    await call.write(h.ControllerFrame(execution_id="slow", cancel=h.Cancel()))
                else:
                    await call.write(h.ControllerFrame(execution_id="slow", model=h.ModelResult(model_call_id=model.model.id, message=text("assistant", "delayed host answer"))))
                end = await asyncio.wait_for(pending, 5)
                assert end.kind == c.EVENT_END
                assert end.end.state == ("CANCELED" if control == "cancel" else "COMPLETED")
                assert await call.read() is grpc.aio.EOF
                _, ends = await turn(stub, inputs=["next"])
                assert ends[0].end.state == "COMPLETED"
            finally:
                expire.set()
                call.cancel()
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(check())


def test_empty_host_completion_finishes_without_retry_or_duplicate(binding):
    async def check():
        async with live(binding) as stub:
            _, ends = await turn(stub, inputs=["hello"], reply="")
            assert ends[0].end.state == "COMPLETED"
    asyncio.run(check())


@pytest.mark.parametrize("control", ["cancel", "eof", "disconnect", "malformed"])
def test_actual_sdk_blocked_effect_cleanup_allows_next_start(binding, control):
    async def check():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unhandled.append(ctx))
        async with live(binding) as stub:
            _, ends = await turn(stub, inputs=["first"], control=control)
            if ends:
                assert ends[0].end.state == ("FAILED" if control == "malformed" else "CANCELED")
            _, ends = await asyncio.wait_for(turn(stub, inputs=["next"], wait_admission=control == "disconnect"), 5)
            assert ends[0].end.state == "COMPLETED"
        assert unhandled == []
    asyncio.run(check())


@pytest.mark.parametrize("control", ["repeat-cancel", "cancel-disconnect", "eof"])
def test_native_cancel_waits_for_owned_sdk_cleanup_before_new_admission(binding, monkeypatch, control):
    async def check():
        closing, release, cleaned, interrupted = (asyncio.Event() for _ in range(4))
        real_close = httpx.AsyncClient.aclose
        calls = 0
        async def delayed_close(client):
            nonlocal calls
            calls += 1
            if calls == 1:
                closing.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
                await real_close(client)
                cleaned.set()
            else:
                await real_close(client)
        monkeypatch.setattr(httpx.AsyncClient, "aclose", delayed_close)
        async with live(binding) as stub:
            call = stub.Connect()
            pending = None
            try:
                await call.write(h.ControllerFrame(execution_id="first", start=h.Start(inputs=[text("user", "hold")])))
                event = await asyncio.wait_for(call.read(), 5)
                assert event.kind == c.EVENT_MODEL_CALL
                pending = asyncio.create_task(call.read())
                if control == "eof":
                    await call.done_writing()
                else:
                    await call.write(h.ControllerFrame(execution_id="first", cancel=h.Cancel()))
                await asyncio.wait_for(closing.wait(), 2)
                assert not pending.done()
                if control == "repeat-cancel":
                    for _ in range(2):
                        await call.write(h.ControllerFrame(execution_id="first", cancel=h.Cancel()))
                elif control == "cancel-disconnect":
                    call.cancel()
                blocked = stub.Connect()
                await blocked.write(h.ControllerFrame(execution_id="blocked", start=h.Start()))
                end = await asyncio.wait_for(blocked.read(), 5)
                assert end.kind == c.EVENT_END and end.end.error.code == 8
                assert await blocked.read() is grpc.aio.EOF
                assert not interrupted.is_set() and not cleaned.is_set()
                release.set()
                if control == "cancel-disconnect":
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    # Local read cancellation is not a server teardown barrier.
                    await asyncio.wait_for(cleaned.wait(), 2)
                else:
                    end = await asyncio.wait_for(pending, 5)
                    assert end.kind == c.EVENT_END and end.end.state == "CANCELED"
                    assert await call.read() is grpc.aio.EOF
                assert cleaned.is_set() and not interrupted.is_set()
                _, ends = await turn(stub, inputs=["fresh"], wait_admission=control == "cancel-disconnect")
                assert ends[0].end.state == "COMPLETED"
            finally:
                release.set()
                call.cancel()
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(check())


@pytest.mark.parametrize("bad", ["tool-role", "file", "data", "reasoning", "tool-history", "model-options"])
def test_unsupported_start_is_refused_before_any_sdk_effect(binding, monkeypatch, bad):
    async def check():
        effects = []
        real_send = httpx.AsyncHTTPTransport.handle_async_request
        async def send(transport, request):
            effects.append(request)
            return await real_send(transport, request)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
        start = h.Start(inputs=[text("user", "hello")])
        if bad == "tool-role":
            start.inputs[0].role = "tool"
        elif bad == "file":
            start.inputs[0].parts[0].CopyFrom(c.Part(file=c.FilePart(uri="file:///private")))
        elif bad == "data":
            start.inputs[0].parts[0].CopyFrom(c.Part(data=c.DataPart()))
        elif bad == "reasoning":
            start.inputs[0].parts[0].CopyFrom(c.Part(reasoning=c.ReasoningPart(opaque_bytes=b"private")))
        elif bad == "tool-history":
            start.history.append(c.Event(kind=c.EVENT_TOOL_CALL, tool=c.ToolCall(id="tool")))
        else:
            start.history.append(c.Event(kind=c.EVENT_MODEL_CALL, model=c.ModelCall(params={"temperature": "0.5"})))
        async with live(binding) as stub:
            call = stub.Connect()
            await call.write(h.ControllerFrame(execution_id="unsupported", start=start))
            end = await asyncio.wait_for(call.read(), 5)
            assert end.kind == c.EVENT_END and end.end.error.code == 12
            assert await call.read() is grpc.aio.EOF
        assert effects == []
    asyncio.run(check())


@pytest.mark.parametrize("extra", [
    {"tools": [{"name": "fetch", "command": ["false"]}]},
    {"brokeredTools": [{"name": "lookup", "description": "Read records.", "brokeredClass": "read", "parameters": {"type": "object"}}]},
    {"context": {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}},
])
def test_baked_tools_and_context_are_refused_before_adapter_build(bake, extra):
    with pytest.raises(AgentsessionsConfigurationError):
        bake(**extra)


def test_sdk_debug_diagnostics_do_not_export_execution_content(bake, caplog):
    logger = logging.getLogger("openai._base_client")
    caplog.set_level(logging.DEBUG, logger="openai")
    filters = list(logger.filters)
    binding = bake(instructions="private-sdk-instructions-marker")

    async def check():
        history = [c.Event(kind=c.EVENT_INPUT, message=text("user", "private-sdk-history-marker"))]
        async with live(binding) as stub:
            _, end = await turn(
                stub, inputs=["private-sdk-input-marker"], history=history,
                reply="private-sdk-output-marker",
            )
            assert end[0].end.state == "COMPLETED"

    asyncio.run(check())
    assert not any(marker in "\n".join(caplog.messages) for marker in [
        "private-sdk-instructions-marker", "private-sdk-history-marker",
        "private-sdk-input-marker", "private-sdk-output-marker",
    ])
    assert logger.filters == filters
    assert logger.getEffectiveLevel() == logging.DEBUG
    logger.debug("ordinary-sdk-debug-control")
    assert "ordinary-sdk-debug-control" in caplog.messages

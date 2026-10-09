"""Real pydantic/OpenAI SDK -> loopback HTTP -> native Harness.Connect (offline)."""
from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager

import grpc
import pytest
import yaml

from agentkit_serve import agent_factory
from agentkit_serve_common.agentsessions import load_verified_agentsessions_binding, create_server
from agentkit_serve_common.agentsessions._generated import common_pb2 as c, harness_pb2 as h, harness_pb2_grpc as g


def text(role, value):
    return c.Message(role=role, parts=[c.Part(text=c.TextPart(text=value))])


@pytest.fixture
def binding(tmp_path, monkeypatch):
    data = {"abiVersion": "v0", "metadata": {"name": "host-bound"}, "model": {"provider": "openai-compatible", "baseURL": "http://127.0.0.1:1/v1", "name": "host-model", "apiKeyEnv": "P2_PROVIDER_KEY", "auth": {"type": "workload-identity-token", "audience": "https://identity.invalid"}}, "instructions": "Only baked rules.", "env": [{"name": "P2_PROVIDER_KEY", "required": True}], "expose": {"openai": True, "port": 8080}}
    path = tmp_path / "agent.yaml"
    path.write_bytes(yaml.safe_dump(data).encode())
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST", "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST", "sha256:" + "c" * 64)
    monkeypatch.delenv("P2_PROVIDER_KEY", raising=False)
    return load_verified_agentsessions_binding(path)


def hook():
    result = getattr(agent_factory, "run_agentsessions", None)
    assert callable(result), "pydantic agentsessions runner missing"
    return result


@asynccontextmanager
async def live(binding, runner=None):
    server = create_server(binding, runner=runner or hook())
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
        await call.write(h.ControllerFrame(execution_id="exec-1", start=h.Start(inputs=[text("user", x) for x in inputs], history=history, config=config)))
        event = await asyncio.wait_for(call.read(), 5)
        if wait_admission and event.kind == c.EVENT_END and event.end.error.code == 8:
            assert await call.read() is grpc.aio.EOF
            continue
        break
    assert event.kind == c.EVENT_MODEL_CALL, str(event)
    if control == "cancel": await call.write(h.ControllerFrame(execution_id="exec-1", cancel=h.Cancel()))
    elif control == "eof": await call.done_writing()
    elif control == "disconnect": call.cancel()
    else:
        result = h.ModelResult(model_call_id=event.model.id, message=text("assistant", reply))
        if control == "malformed": result.message.parts[0].CopyFrom(c.Part(data=c.DataPart()))
        await call.write(h.ControllerFrame(execution_id="exec-1", model=result))
    if control == "disconnect": return event, []
    end = await asyncio.wait_for(call.read(), 5)
    assert end.kind == c.EVENT_END  # no duplicate OUTPUT / fabricated USAGE
    assert await call.read() is grpc.aio.EOF
    return event, [end]


@pytest.mark.parametrize("inputs,prior,want", [
    (["first", "latest"], [("user", "past"), ("assistant", "answer")], [("system", "Only baked rules."), ("user", "past"), ("assistant", "answer"), ("user", "first"), ("user", "latest")]),
    ([""], [], [("system", "Only baked rules."), ("user", "")]),
    ([], [], [("system", "Only baked rules.")]),
    ([], [("user", "old"), ("assistant", "reply")], [("system", "Only baked rules."), ("user", "old"), ("assistant", "reply")]),
    (["", "last"], [("user", ""), ("assistant", "")], [("system", "Only baked rules."), ("user", ""), ("assistant", ""), ("user", ""), ("user", "last")]),
])
def test_actual_sdk_preserves_history_input_boundaries_and_config(binding, inputs, prior, want):
    async def check():
        seen = []
        async def runner(binding, request, exchange):
            seen.append(request.config)
            return await hook()(binding, request, exchange)
        history = [c.Event(kind=c.EVENT_INPUT if role == "user" else c.EVENT_OUTPUT, message=text(role, value)) for role, value in prior]
        async with live(binding, runner) as stub:
            event, ends = await turn(stub, inputs=inputs, history=history)
            assert [(m.role, "".join(p.text.text for p in m.parts)) for m in event.model.messages] == want
            assert ends[0].end.state == "COMPLETED"
            assert dict(event.model.params) == {} and event.model.input_hash == ""
        assert seen == [b"\xffopaque\x00"]
    asyncio.run(check())


@pytest.mark.parametrize("ambient_headers", [
    "Authorization: Bearer unrelated-ambient-token",
    "authorization: Bearer unrelated-ambient-token\naUtHoRiZaTiOn: Bearer another-ambient-token",
])
def test_fresh_sdk_clients_agents_local_tokens_cleanup_no_provider_or_proxy(binding, monkeypatch, ambient_headers):
    async def check():
        hits, clients, agents = [], [], []
        async def trap(reader, writer):
            hits.append(await reader.read(4096))
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(trap, "127.0.0.1", 0)
        url = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]) + "/v1"
        binding.spec.model.base_url = url
        monkeypatch.setenv("OPENAI_BASE_URL", url)
        monkeypatch.setenv("HTTP_PROXY", url)
        monkeypatch.setenv("ALL_PROXY", url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "unused-ambient-value")
        monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", ambient_headers)
        import openai
        real_client = openai.AsyncOpenAI
        real_agent = agent_factory.Agent
        def client(**kwargs):
            value = real_client(**kwargs)
            clients.append(value)
            return value
        def agent(*args, **kwargs):
            value = real_agent(*args, **kwargs)
            agents.append(value)
            return value
        monkeypatch.setattr(openai, "AsyncOpenAI", client)
        monkeypatch.setattr(agent_factory, "Agent", agent)
        def forbidden(*args, **kwargs): raise AssertionError("normal provider builder used")
        monkeypatch.setattr(agent_factory, "build_runtime", forbidden)
        monkeypatch.setattr(agent_factory, "build_model", forbidden)
        monkeypatch.setattr(agent_factory, "resolve_api_key", forbidden)
        before = dict(os.environ)
        try:
            async with live(binding) as stub:
                for value in ["one", "two"]:
                    _, ends = await turn(stub, inputs=[value])
                    assert ends[0].end.state == "COMPLETED"
            assert hits == []
            assert len(clients) == len(agents) == 2
            assert clients[0] is not clients[1] and agents[0] is not agents[1]
            assert agents[0].model is not agents[1].model
            assert all(client.is_closed() for client in clients)
            assert clients[0].api_key != clients[1].api_key
            assert all(client.api_key != "unused-ambient-value" and client.max_retries == 0 for client in clients)
            assert all(str(client.base_url).startswith("http://127.0.0.1:") and str(client.base_url).rstrip("/") != url for client in clients)
            assert dict(os.environ) == before
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(check())


def test_sdk_bridge_failure_has_no_retry_or_fallback(binding):
    async def check():
        async with live(binding) as stub:
            # A valid bounded protobuf result expands beyond the JSON bound.
            # If SDK retries are enabled, the next frame would be model-2, not END.
            _, ends = await turn(stub, inputs=["hello"], reply="\x00" * (200 * 1024))
            assert ends[0].end.state == "FAILED"
            assert ends[0].end.error.code == 13
            assert ends[0].end.error.description == "execution failed"
            _, ends = await turn(stub, inputs=["next"])
            assert ends[0].end.state == "COMPLETED"
    asyncio.run(check())


@pytest.mark.parametrize("control", ["reply", "cancel"])
def test_actual_sdk_host_wait_has_no_transport_read_deadline(
    binding, monkeypatch, control,
):
    async def check():
        import httpx

        phases = []
        expire = asyncio.Event()
        checked = asyncio.Event()
        real_send = httpx.AsyncHTTPTransport.handle_async_request

        async def delayed_send(transport, request):
            timeout = request.extensions["timeout"]
            phases.append(dict(timeout))
            response = asyncio.create_task(real_send(transport, request))
            try:
                # Accelerate only the transport's finite read-deadline branch;
                # keep the real SDK, HTTP request and host bridge in the loop.
                await expire.wait()
                checked.set()
                if timeout["read"] is not None:
                    raise httpx.ReadTimeout("controlled read deadline", request=request)
                return await response
            finally:
                if not response.done():
                    response.cancel()
                await asyncio.gather(response, return_exceptions=True)

        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", delayed_send,
        )
        async with live(binding) as stub:
            call = stub.Connect()
            pending = None
            try:
                await call.write(h.ControllerFrame(
                    execution_id="slow",
                    start=h.Start(inputs=[text("user", "wait for host")]),
                ))
                model = await asyncio.wait_for(call.read(), 5)
                assert model.kind == c.EVENT_MODEL_CALL
                pending = asyncio.create_task(call.read())
                expire.set()
                await asyncio.wait_for(checked.wait(), 2)
                assert phases == [{"connect": 5, "read": None, "write": None, "pool": None}]
                assert not pending.done()
                if control == "cancel":
                    await call.write(h.ControllerFrame(
                        execution_id="slow", cancel=h.Cancel(),
                    ))
                else:
                    await call.write(h.ControllerFrame(
                        execution_id="slow",
                        model=h.ModelResult(
                            model_call_id=model.model.id,
                            message=text("assistant", "delayed host answer"),
                        ),
                    ))
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


def test_empty_host_completion_is_not_retried_or_duplicated(binding):
    async def check():
        async with live(binding) as stub:
            _, ends = await turn(stub, inputs=["hello"], reply="")
            assert ends[0].end.state == "COMPLETED"
    asyncio.run(check())


@pytest.mark.parametrize("control", ["cancel", "eof", "disconnect", "malformed"])
def test_actual_sdk_blocked_effect_cleanup_and_next_execution(binding, control):
    async def check():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unhandled.append(ctx))
        async with live(binding) as stub:
            _, ends = await turn(stub, inputs=["first"], control=control)
            if ends:
                assert ends[0].end.state == ("FAILED" if control == "malformed" else "CANCELED")
            # Disconnect has no writable cleanup barrier. Poll admission only;
            # rejected admissions emit no model effect and are not model retries.
            _, ends = await asyncio.wait_for(
                turn(stub, inputs=["next"], wait_admission=control == "disconnect"), 5,
            )
            assert ends[0].end.state == "COMPLETED"
        assert unhandled == []
    asyncio.run(check())

from __future__ import annotations

import asyncio
import hashlib
import importlib
from contextlib import asynccontextmanager

import pytest
import yaml


def protocol():
    # A missing protocol is a feature failure, not a collection/import error.
    try:
        return importlib.import_module("agentkit_serve_common.agentsessions")
    except ModuleNotFoundError:
        pytest.fail("agentsessions native gRPC protocol has not been implemented")


@pytest.fixture
def binding_file(tmp_path, monkeypatch):
    data = {
        "abiVersion": "v0",
        "metadata": {"name": "wire-test"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://provider.invalid/v1",
            "name": "host-model",
            "apiKeyEnv": "WIRE_MODEL_KEY",
        },
        "instructions": "Be helpful.",
        "env": [{"name": "WIRE_MODEL_KEY", "required": True}],
        "expose": {"openai": True, "port": 8080},
    }
    path = tmp_path / "agent.yaml"
    path.write_bytes(yaml.safe_dump(data).encode())
    monkeypatch.setenv(
        "AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST",
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.delenv("WIRE_MODEL_KEY", raising=False)
    return path, data


def test_verified_binding_reads_exact_bytes_and_never_resolves_provider(binding_file, monkeypatch):
    path, _ = binding_file
    p = protocol()
    binding = p.load_verified_agentsessions_binding(path)
    assert binding.spec.model.name == "host-model"
    assert binding.configuration_digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding.implementation_digest == "sha256:" + "a" * 64
    assert binding.configuration_digest in binding.descriptor_id
    assert binding.implementation_digest in binding.descriptor_id
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(p.AgentsessionsConfigurationError, match="exact agent config bytes"):
        p.load_verified_agentsessions_binding(path)


def test_verified_binding_returned_spec_cannot_change_identity_or_next_spec(binding_file):
    binding = protocol().load_verified_agentsessions_binding(binding_file[0])
    digest = binding.configuration_digest
    descriptor_id = binding.descriptor_id
    returned = binding.spec
    returned.instructions = "changed instructions"
    returned.model.name = "changed-model"
    returned.expose.port = 9090
    returned.env.clear()
    returned.tools.append("changed tool")

    assert binding.configuration_digest == digest
    assert binding.descriptor_id == descriptor_id
    assert binding.spec.instructions == "Be helpful."
    assert binding.spec.model.name == "host-model"
    assert binding.spec.expose.port == 8080
    assert len(binding.spec.env) == 1
    assert binding.spec.tools == []


def test_verified_binding_owns_constructor_bytes(binding_file):
    from agentkit_serve_common.agentsessions.binding import VerifiedAgentsessionsBinding

    raw = bytearray(binding_file[0].read_bytes())
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    binding = VerifiedAgentsessionsBinding(raw, "sha256:" + "a" * 64)
    raw[:] = b"changed caller buffer"
    assert binding.configuration_digest == digest
    assert binding.spec.model.name == "host-model"
    assert binding.spec.instructions == "Be helpful."


def test_verified_binding_retains_exact_bytes_not_reserialized_yaml(binding_file, monkeypatch):
    path, _ = binding_file
    p = protocol()
    first = p.load_verified_agentsessions_binding(path)
    raw = path.read_bytes() + b"\n# same configuration, distinct exact-byte identity\n"
    path.write_bytes(raw)
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST", "sha256:" + hashlib.sha256(raw).hexdigest())
    second = p.load_verified_agentsessions_binding(path)
    assert first.spec == second.spec
    assert first.configuration_digest != second.configuration_digest
    assert second.configuration_digest == "sha256:" + hashlib.sha256(raw).hexdigest()
    path.write_bytes(b"changed after verification")
    assert second.configuration_digest == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert second.spec.model.name == "host-model"


@pytest.mark.parametrize("digest", [None, "", "sha256:" + "0" * 64, "sha256:" + "A" * 64, "not-a-digest"])
def test_digest_missing_invalid_or_mismatch_fails(binding_file, monkeypatch, digest):
    path, _ = binding_file
    p = protocol()
    if digest is None:
        monkeypatch.delenv("AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST")
    else:
        monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST", digest)
    with pytest.raises(p.AgentsessionsConfigurationError):
        p.load_verified_agentsessions_binding(path)


def test_implementation_identity_is_required(binding_file, monkeypatch):
    p = protocol()
    monkeypatch.delenv("AGENTKIT_AGENTSESSIONS_IMPLEMENTATION_DIGEST")
    with pytest.raises(p.AgentsessionsConfigurationError, match="IMPLEMENTATION_DIGEST"):
        p.load_verified_agentsessions_binding(binding_file[0])


def test_supplied_provider_credential_rejected_without_echo(binding_file, monkeypatch):
    p = protocol()
    monkeypatch.setenv("WIRE_MODEL_KEY", "private-test-value")
    with pytest.raises(p.AgentsessionsConfigurationError) as failure:
        p.load_verified_agentsessions_binding(binding_file[0])
    assert "credential" in str(failure.value)
    assert "private-test-value" not in str(failure.value)


@pytest.mark.parametrize("extra", [
    {"tools": [{"name": "fetch", "command": ["false"]}]},
    {"brokeredTools": [{"name": "lookup", "description": "Read records.", "brokeredClass": "read", "parameters": {"type": "object"}}]},
    {"context": {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}},
])
def test_baked_tools_and_context_rejected(binding_file, monkeypatch, extra):
    path, data = binding_file
    p = protocol()
    path.write_bytes(yaml.safe_dump({**data, **extra}).encode())
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST", "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(p.AgentsessionsConfigurationError):
        p.load_verified_agentsessions_binding(path)


@asynccontextmanager
async def live(binding_file, runner=None, token=None):
    p = protocol()
    import grpc
    from agentkit_serve_common.agentsessions._generated import common_pb2 as c, harness_pb2 as h, harness_pb2_grpc as g
    binding = p.load_verified_agentsessions_binding(binding_file[0])
    server = p.create_server(binding, runner=runner, auth_token=token)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield p, c, h, g.HarnessStub(channel), binding
    finally:
        await server.stop(0)


def text(c, role, value):
    return c.Message(role=role, parts=[c.Part(text=c.TextPart(text=value))])


def start(h, c, **kwargs):
    return h.ControllerFrame(execution_id="exec-1", session="session-1", start=h.Start(inputs=[text(c, "user", "hello")], **kwargs))


async def collect(call):
    return [event async for event in call]


def terminal(events, c, state, code=0):
    ends = [event for event in events if event.kind == c.EVENT_END]
    assert len(ends) == 1
    assert events[-1] == ends[0]
    assert ends[0].execution_id == "exec-1"
    assert ends[0].end.state == state
    assert ends[0].end.error.code == code
    assert all(event.execution_id == "exec-1" for event in events)


def test_describe_and_skeleton_unimplemented(binding_file):
    async def check():
        async with live(binding_file) as (_, c, h, stub, binding):
            descriptor = await stub.Describe(h.DescribeRequest())
            assert descriptor.id == binding.descriptor_id
            assert list(descriptor.models) == ["host-model"]
            assert list(descriptor.tools) == []
            assert descriptor.capabilities.resumability == h.RESUMABILITY_STATELESS_REPLAY
            assert descriptor.capabilities.fork_safe
            assert not descriptor.capabilities.requires_gpu
            assert not descriptor.capabilities.streaming
            assert not descriptor.capabilities.reasoning_replay
            call = stub.Connect()
            await call.write(start(h, c))
            terminal(await collect(call), c, "FAILED", 12)
    asyncio.run(check())


def test_describe_and_next_execution_ignore_returned_spec_mutation(binding_file):
    async def check():
        from agentkit_serve_common.runtime import RunResult

        async def runner(binding, request, exchange):
            assert binding.spec.model.name == "host-model"
            assert binding.spec.instructions == "Be helpful."
            return RunResult(text="original configuration")

        async with live(binding_file, runner) as (_, c, h, stub, binding):
            before = await stub.Describe(h.DescribeRequest())
            returned = binding.spec
            returned.model.name = "changed-model"
            returned.instructions = "changed instructions"
            after = await stub.Describe(h.DescribeRequest())
            assert after == before
            assert list(after.models) == ["host-model"]
            call = stub.Connect()
            await call.write(start(h, c))
            terminal(await collect(call), c, "COMPLETED")
    asyncio.run(check())


def test_runner_receives_history_multiple_inputs_and_opaque_config(binding_file):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        seen = []
        async def runner(binding, request, exchange):
            seen.append((binding, request))
            return RunResult(text="neutral reply")
        async with live(binding_file, runner) as (_, c, h, stub, binding):
            history = [
                c.Event(kind=c.EVENT_EXECUTION_START, execution_start=c.ExecutionStart(config=b"old config", input_count=1)),
                c.Event(kind=c.EVENT_INPUT, message=text(c, "user", "previous")),
                c.Event(kind=c.EVENT_MODEL_CALL, model=c.ModelCall(model="host-model", id="previous-call")),
                c.Event(kind=c.EVENT_OUTPUT, message=text(c, "assistant", "answer")),
                c.Event(kind=c.EVENT_USAGE, usage=c.Usage(input_tokens=1)),
                c.Event(kind=c.EVENT_END, end=c.HarnessEnd(state="COMPLETED")),
            ]
            frame = h.ControllerFrame(execution_id="exec-1", start=h.Start(config=b"\x00opaque\xff", history=history, inputs=[text(c, "user", "first"), text(c, "user", "latest")]))
            call = stub.Connect()
            await call.write(frame)
            events = await collect(call)
            terminal(events, c, "COMPLETED")
            assert [event.message.parts[0].text.text for event in events if event.kind == c.EVENT_OUTPUT] == ["neutral reply"]
            assert seen[0][0] is binding
            request = seen[0][1]
            assert request.config == b"\x00opaque\xff"
            assert "opaque" not in repr(request)
            assert request.prompt == "latest"
            assert [(turn.role, turn.text) for turn in request.history] == [("user", "previous"), ("assistant", "answer"), ("user", "first")]
            assert request.turn_id == "exec-1"
            assert request.session_id is None
    asyncio.run(check())


def test_inputless_start_is_not_invented_input(binding_file):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        async def runner(binding, request, exchange):
            assert request.prompt == ""
            assert request.history == ()
            assert request.config == b""
            return RunResult(text="")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            await call.write(h.ControllerFrame(execution_id="exec-1", start=h.Start()))
            terminal(await collect(call), c, "COMPLETED")
    asyncio.run(check())


def test_runner_failure_is_sanitized_and_ends_once(binding_file):
    async def runner(binding, request, exchange):
        raise RuntimeError("do not leak private request/config/provider material")
    async def check():
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            await call.write(start(h, c))
            events = await collect(call)
            terminal(events, c, "FAILED", 13)
            assert "private" not in str(events)
    asyncio.run(check())


@pytest.mark.parametrize("frame_type", ["cancel", "empty", "missing-id", "oversized-id"])
def test_start_first_and_execution_id_required(binding_file, frame_type):
    async def check():
        import grpc
        async with live(binding_file) as (_, c, h, stub, _):
            frame = {"cancel": h.ControllerFrame(execution_id="exec-1", cancel=h.Cancel()), "empty": h.ControllerFrame(), "missing-id": h.ControllerFrame(start=h.Start()), "oversized-id": h.ControllerFrame(execution_id="x" * (4 * 1024 * 1024 - 32), start=h.Start())}[frame_type]
            call = stub.Connect()
            await call.write(frame)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await collect(call)
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    asyncio.run(check())


@pytest.mark.parametrize("bad", ["tool-role", "file", "data", "reasoning", "empty-part", "tool-history", "unknown-history"])
def test_unsupported_start_content_fails_closed(binding_file, bad):
    async def check():
        async with live(binding_file) as (_, c, h, stub, _):
            frame = start(h, c)
            if bad == "tool-role":
                frame.start.inputs[0].role = "tool"
            elif bad == "file":
                frame.start.inputs[0].parts[0].CopyFrom(c.Part(file=c.FilePart(uri="file:///private")))
            elif bad == "data":
                frame.start.inputs[0].parts[0].CopyFrom(c.Part(data=c.DataPart()))
            elif bad == "reasoning":
                frame.start.inputs[0].parts[0].CopyFrom(c.Part(reasoning=c.ReasoningPart(opaque_bytes=b"private")))
            elif bad == "empty-part":
                frame.start.inputs[0].parts[0].Clear()
            elif bad == "tool-history":
                frame.start.history.append(c.Event(kind=c.EVENT_TOOL_CALL, tool=c.ToolCall(id="tool")))
            else:
                frame.start.history.append(c.Event(kind=999))
            call = stub.Connect()
            await call.write(frame)
            terminal(await collect(call), c, "FAILED", 12)
    asyncio.run(check())


@pytest.mark.parametrize("control", ["cancel", "wrong-id", "wrong-session", "model", "approval", "tool", "start", "unknown", "half-close", "disconnect"])
def test_reader_controls_blocked_runner_and_releases_admission(binding_file, control):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        calls = 0
        async def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cleaned.set()
            return RunResult(text="next run")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            await call.write(start(h, c))
            await asyncio.wait_for(entered.wait(), 2)
            if control == "disconnect":
                call.cancel()
            elif control == "half-close":
                await call.done_writing()
            else:
                fields = {"cancel": {"cancel": h.Cancel(reason="USER")}, "wrong-id": {"cancel": h.Cancel()}, "wrong-session": {"cancel": h.Cancel()}, "model": {"model": h.ModelResult(model_call_id="unexpected")}, "approval": {"approval": c.ApprovalResult()}, "tool": {"tool": c.ToolResult()}, "start": {"start": h.Start()}, "unknown": {}}[control]
                frame = h.ControllerFrame(execution_id="wrong" if control == "wrong-id" else "exec-1", session="wrong" if control == "wrong-session" else "session-1", **fields)
                await call.write(frame)
            if control != "disconnect":
                events = await asyncio.wait_for(collect(call), 2)
                expected = ("CANCELED", 1) if control in {"cancel", "half-close"} else ("FAILED", 3 if control in {"wrong-id", "wrong-session", "start", "unknown", "model"} else 12)
                terminal(events, c, *expected)
            await asyncio.wait_for(cleaned.wait(), 2)
            next_call = stub.Connect()
            await next_call.write(start(h, c))
            terminal(await asyncio.wait_for(collect(next_call), 2), c, "COMPLETED")
    asyncio.run(check())


def test_concurrent_execution_rejected_without_poisoning_active_run(binding_file):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        entered = asyncio.Event()
        release = asyncio.Event()
        async def runner(binding, request, exchange):
            entered.set()
            await release.wait()
            return RunResult(text="one")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            active = stub.Connect()
            await active.write(start(h, c))
            await asyncio.wait_for(entered.wait(), 2)
            other = stub.Connect()
            await other.write(start(h, c))
            terminal(await collect(other), c, "FAILED", 8)
            release.set()
            terminal(await collect(active), c, "COMPLETED")
    asyncio.run(check())


def test_metadata_bearer_auth_on_both_rpcs(binding_file):
    async def check():
        import grpc
        async with live(binding_file, token="local-test-token") as (_, c, h, stub, _):
            for metadata in [(), (("authorization", "Bearer wrong"),), (("authorization", "Bearer local-test-token"), ("authorization", "Bearer local-test-token"))]:
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await stub.Describe(h.DescribeRequest(), metadata=metadata)
                assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
                call = stub.Connect(metadata=metadata)
                await call.write(start(h, c))
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await collect(call)
                assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
            metadata = (("authorization", "Bearer local-test-token"),)
            assert (await stub.Describe(h.DescribeRequest(), metadata=metadata)).models[0] == "host-model"
            call = stub.Connect(metadata=metadata)
            await call.write(start(h, c))
            terminal(await collect(call), c, "FAILED", 12)
    asyncio.run(check())


def test_cancel_then_disconnect_cannot_interrupt_async_cleanup(binding_file):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        entered = asyncio.Event()
        closing = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleaned = asyncio.Event()
        calls = 0
        async def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    closing.set()
                    await release_cleanup.wait()
                    cleaned.set()
            return RunResult(text="fresh")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            await call.write(start(h, c))
            await asyncio.wait_for(entered.wait(), 2)
            await call.write(h.ControllerFrame(execution_id="exec-1", session="session-1", cancel=h.Cancel()))
            await asyncio.wait_for(closing.wait(), 2)
            call.cancel()
            # Disconnect while the runner is already closing must not free the
            # admission slot or cancel the resource owner's awaited teardown.
            while_closing = stub.Connect()
            await while_closing.write(start(h, c))
            terminal(await collect(while_closing), c, "FAILED", 8)
            release_cleanup.set()
            await asyncio.wait_for(cleaned.wait(), 2)
            next_call = stub.Connect()
            await next_call.write(start(h, c))
            terminal(await collect(next_call), c, "COMPLETED")
    asyncio.run(check())


@pytest.mark.parametrize("url", ["https://user:private-test-value@provider.invalid/v1", "https://provider.invalid/v1?api_key=private-test-value"])
def test_credentials_baked_in_model_url_are_rejected(binding_file, monkeypatch, url):
    path, data = binding_file
    p = protocol()
    data["model"]["baseURL"] = url
    path.write_bytes(yaml.safe_dump(data).encode())
    monkeypatch.setenv("AGENTKIT_AGENTSESSIONS_AGENT_CONFIGURATION_DIGEST", "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(p.AgentsessionsConfigurationError) as error:
        p.load_verified_agentsessions_binding(path)
    assert "private-test-value" not in str(error.value)


def test_direct_serve_nonloopback_without_auth_is_rejected(binding_file):
    async def check():
        p = protocol()
        binding = p.load_verified_agentsessions_binding(binding_file[0])
        with pytest.raises(ValueError, match="authentication"):
            await p.serve(binding, bind="0.0.0.0", port=0)
    asyncio.run(check())


def test_wire_receive_and_send_bounds(binding_file):
    async def check():
        import grpc
        async def runner(binding, request, exchange):
            from agentkit_serve_common.runtime import RunResult
            return RunResult(text="x" * (4 * 1024 * 1024))
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            await call.write(start(h, c, config=b"x" * (4 * 1024 * 1024)))
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await collect(call)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            bounded = stub.Connect()
            await bounded.write(start(h, c))
            terminal(await collect(bounded), c, "FAILED", 8)
    asyncio.run(check())


def test_surrogate_output_fails_safely_and_next_execution_is_admitted(binding_file, caplog):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        calls = 0
        async def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            return RunResult(text="private-output-prefix-\ud800" if calls == 1 else "healthy")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            await call.write(start(h, c, config=b"private-config"))
            events = await collect(call)
            terminal(events, c, "FAILED", 13)
            assert [event.kind for event in events] == [c.EVENT_END]
            assert events[-1].end.error.description == "execution failed"
            next_call = stub.Connect()
            await next_call.write(start(h, c))
            terminal(await collect(next_call), c, "COMPLETED")
            assert "private-output-prefix" not in caplog.text
            assert "private-config" not in caplog.text
    asyncio.run(check())


@pytest.mark.parametrize("control", ["cancel", "half-close"])
def test_control_does_not_second_cancel_runner_already_closing(binding_file, monkeypatch, control):
    async def check():
        from agentkit_serve_common.agentsessions.service import HarnessService
        from agentkit_serve_common.runtime import RunResult
        closing = asyncio.Event()
        control_read = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleaned = asyncio.Event()
        interrupted = asyncio.Event()
        calls = 0
        original_reader = HarnessService._read_controls
        async def observed_reader(self, frames, first, exchange):
            result = await original_reader(self, frames, first, exchange)
            control_read.set()
            return result
        monkeypatch.setattr(HarnessService, "_read_controls", observed_reader)
        async def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            if calls == 1:
                asyncio.current_task().cancel()
                try:
                    await asyncio.Future()
                finally:
                    closing.set()
                    try:
                        await release_cleanup.wait()
                        cleaned.set()
                    except asyncio.CancelledError:
                        interrupted.set()
                        raise
            return RunResult(text="healthy")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            try:
                call = stub.Connect()
                await call.write(start(h, c))
                await asyncio.wait_for(closing.wait(), 2)
                if control == "cancel":
                    await call.write(h.ControllerFrame(execution_id="exec-1", session="session-1", cancel=h.Cancel()))
                else:
                    await call.done_writing()
                # Observe the real reader processing the native control/EOF;
                # the spy changes no frame or execution behavior.
                await asyncio.wait_for(control_read.wait(), 2)
                blocked = stub.Connect()
                await blocked.write(start(h, c))
                terminal(await collect(blocked), c, "FAILED", 8)
                assert not interrupted.is_set()
                release_cleanup.set()
                terminal(await collect(call), c, "CANCELED", 1)
                await asyncio.wait_for(cleaned.wait(), 2)
                next_call = stub.Connect()
                await next_call.write(start(h, c))
                terminal(await collect(next_call), c, "COMPLETED")
            finally:
                release_cleanup.set()
    asyncio.run(check())


@pytest.mark.parametrize("control", ["cancel", "half-close", "disconnect"])
@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_canceled_cleanup_failure_has_safe_diagnosis_and_releases_admission(binding_file, caplog, control, cleanup_raises):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        calls = 0
        async def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cleaned.set()
                    if cleanup_raises:
                        raise RuntimeError(f"private-token: {request.config!r} {request.prompt}")
            return RunResult(text="healthy")
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            frame = start(h, c, config=b"private-config")
            frame.start.inputs[0].parts[0].text.text = "private-prompt"
            await call.write(frame)
            await asyncio.wait_for(entered.wait(), 2)
            if control == "cancel":
                await call.write(h.ControllerFrame(execution_id="exec-1", session="session-1", cancel=h.Cancel()))
            elif control == "half-close":
                await call.done_writing()
            else:
                call.cancel()
            if control != "disconnect":
                events = await collect(call)
                terminal(events, c, "FAILED" if cleanup_raises else "CANCELED", 13 if cleanup_raises else 1)
                if cleanup_raises:
                    assert events[-1].end.error.description == "execution cleanup failed"
                assert [event.kind for event in events] == [c.EVENT_END]
            await asyncio.wait_for(cleaned.wait(), 2)
            # A disconnected RPC has no writable END/EOF barrier. Admission may
            # reject until cleanup settlement completes, but must then recover.
            async def admitted_next():
                while True:
                    next_call = stub.Connect()
                    await next_call.write(start(h, c))
                    events = await collect(next_call)
                    if events[-1].end.error.code == 8:
                        terminal(events, c, "FAILED", 8)
                        continue
                    terminal(events, c, "COMPLETED")
                    return
            await asyncio.wait_for(admitted_next(), 2)
            messages = [record.getMessage() for record in caplog.records]
            assert messages.count("agentsessions execution cleanup failed") == int(cleanup_raises)
            assert not any(record.exc_info for record in caplog.records)
            assert "private-token" not in caplog.text
            assert "private-config" not in caplog.text
            assert "private-prompt" not in caplog.text
    asyncio.run(check())


@pytest.mark.parametrize("failed_task", ["execution", "reader"])
def test_connect_cancellation_retrieves_already_failed_tasks(binding_file, caplog, failed_task):
    async def check():
        import gc
        from agentkit_serve_common.agentsessions._generated import common_pb2 as c, harness_pb2 as h
        from agentkit_serve_common.agentsessions.service import HarnessService
        from agentkit_serve_common.runtime import RunResult
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        calls = 0
        async def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            if calls == 1:
                if failed_task == "execution":
                    loop.call_soon(consumer.cancel)
                    raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
                await asyncio.Future()
            return RunResult(text="healthy")
        async def frames():
            yield start(h, c)
            if failed_task == "reader":
                loop.call_soon(consumer.cancel)
                raise RuntimeError("PRIVATE-PROMPT-CONFIG-TOKEN")
            await asyncio.Future()
        async def next_frames():
            yield start(h, c)
            await asyncio.Future()
        binding = protocol().load_verified_agentsessions_binding(binding_file[0])
        service = HarnessService(binding, runner=runner)
        try:
            # Drive the real Connect async-generator lifecycle with native frames.
            # No auth token means this valid path never uses the RPC context.
            consumer = asyncio.create_task(collect(service.Connect(frames(), None)))
            with pytest.raises(asyncio.CancelledError):
                await consumer
            del consumer
            gc.collect()
            await asyncio.sleep(0)  # Drain scheduled exception-handler callbacks.
            events = await asyncio.wait_for(collect(service.Connect(next_frames(), None)), 2)
            terminal(events, c, "COMPLETED")
            assert unhandled == []
            assert "agentsessions execution cleanup failed" not in caplog.text
            assert "PRIVATE-PROMPT-CONFIG-TOKEN" not in caplog.text
        finally:
            loop.set_exception_handler(previous_handler)
    asyncio.run(check())


@pytest.mark.parametrize("kind", ["future", "custom-awaitable", "synchronous-failure"])
def test_runner_accepts_awaitables_and_sanitizes_synchronous_failure(binding_file, caplog, kind):
    async def check():
        from agentkit_serve_common.runtime import RunResult
        calls = 0
        class Response:
            def __init__(self, future):
                self.future = future
            def __await__(self):
                return self.future.__await__()
        def runner(binding, request, exchange):
            nonlocal calls
            calls += 1
            if kind == "synchronous-failure" and calls == 1:
                raise RuntimeError(f"private-token: {request.config!r} {request.prompt}")
            future = asyncio.get_running_loop().create_future()
            future.set_result(RunResult(text="healthy"))
            return Response(future) if kind == "custom-awaitable" else future
        async with live(binding_file, runner) as (_, c, h, stub, _):
            call = stub.Connect()
            frame = start(h, c, config=b"private-config")
            frame.start.inputs[0].parts[0].text.text = "private-prompt"
            await call.write(frame)
            events = await collect(call)
            terminal(events, c, "FAILED" if kind == "synchronous-failure" else "COMPLETED", 13 if kind == "synchronous-failure" else 0)
            if kind == "synchronous-failure":
                assert [event.kind for event in events] == [c.EVENT_END]
                assert events[-1].end.error.description == "execution failed"
            else:
                assert [event.kind for event in events] == [c.EVENT_OUTPUT, c.EVENT_END]
                assert events[0].message.parts[0].text.text == "healthy"
            next_call = stub.Connect()
            await next_call.write(start(h, c))
            terminal(await collect(next_call), c, "COMPLETED")
            assert "private-token" not in caplog.text
            assert "private-config" not in caplog.text
            assert "private-prompt" not in caplog.text
    asyncio.run(check())

from __future__ import annotations

import asyncio
import importlib
import json
from contextlib import asynccontextmanager, suppress

import httpx
import pytest

from agentkit_serve_common.agentsessions import ExecutionExchange
from agentkit_serve_common.agentsessions._generated import common_pb2 as c, harness_pb2 as h


def bridge_module():
    try:
        return importlib.import_module("agentkit_serve_common.agentsessions.bridge")
    except ModuleNotFoundError:
        pytest.fail("agentsessions loopback bridge is missing")


def message(value="answer"):
    return c.Message(role="assistant", parts=[c.Part(text=c.TextPart(text=value))])


def payload(**kwargs):
    return {"model": "host-model", "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": [{"type": "text", "text": "latest"}]}], "stream": False, **kwargs}


@asynccontextmanager
async def live():
    exchange = ExecutionExchange("exec-1", "host-model", 1)
    async with bridge_module().loopback_bridge(exchange) as binding:
        async with httpx.AsyncClient(base_url=binding.base_url, headers={"Authorization": "Bearer " + binding.token}, trust_env=False, follow_redirects=False) as client:
            yield exchange, client, binding
    exchange.close()


def test_bridge_roundtrip_is_authenticated_text_only_and_no_usage():
    async def check():
        async with live() as (exchange, client, binding):
            assert binding.base_url.startswith("http://127.0.0.1:")
            assert binding.token and "token" not in repr(binding)
            task = asyncio.create_task(client.post("chat/completions", json=payload()))
            event = await asyncio.wait_for(exchange.events.get(), 3)
            assert [(m.role, "".join(p.text.text for p in m.parts)) for m in event.model.messages] == [("system", "rules"), ("user", ""), ("assistant", ""), ("user", "latest")]
            exchange.mark_emitted(event.model.id)
            exchange.accept(h.ModelResult(model_call_id=event.model.id, message=message(), usage=c.Usage(model="host-model", input_tokens=123)))
            response = await task
            assert response.status_code == 200
            body = response.json()
            assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]
            assert body["model"] == "host-model" and "usage" not in body
        async with httpx.AsyncClient(trust_env=False) as probe:
            with pytest.raises(httpx.ConnectError):
                await probe.get(binding.base_url)
    asyncio.run(check())


@pytest.mark.parametrize("headers", [{"Authorization": ""}, {"Authorization": "Bearer wrong"}, None])
def test_bridge_authentication_refuses_before_effect(headers):
    async def check():
        async with live() as (exchange, client, binding):
            auth = headers if headers is not None else [("Authorization", "Bearer " + binding.token)] * 2
            response = await client.post("chat/completions", json=payload(), headers=auth)
            assert response.status_code == 401
            assert exchange.events.empty()
    asyncio.run(check())


@pytest.mark.parametrize("extra", [
    {"model": "other"}, {"tools": []}, {"tool_choice": "none"}, {"parallel_tool_calls": False},
    {"functions": []}, {"response_format": {"type": "json_object"}}, {"temperature": 0},
    {"n": 2}, {"stream": "true"}, {"stream_options": {"include_usage": True}},
    {"reasoning_effort": "low"}, {"user": "private"}, {"store": True}, {"max_tokens": 5},
    {"messages": [{"role": "tool", "content": "no"}]},
    {"messages": [{"role": "developer", "content": "no"}]},
    {"messages": [{"role": "assistant", "content": "no", "tool_calls": []}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://trap.invalid"}}]}]},
    {"messages": [{"role": "user", "content": None}]},
])
def test_bridge_refuses_unsupported_options_before_any_effect(extra):
    async def check():
        async with live() as (exchange, client, _):
            response = await client.post("chat/completions", json=payload(**extra))
            assert response.status_code == 400
            assert exchange.events.empty()
            assert "private" not in response.text and "trap.invalid" not in response.text
    asyncio.run(check())


@pytest.mark.parametrize("raw", [b"not-json", b"[]", b'{"model":"host-model","model":"other"}', b'{"temperature":NaN}', b'{"model":"host-model","messages":[{"role":"user","content":"\\ud800"}]}', b"x" * (1024 * 1024 + 1)])
def test_bridge_malformed_or_oversized_body_has_safe_error(raw):
    async def check():
        async with live() as (exchange, client, _):
            response = await client.post("chat/completions", content=raw)
            assert response.status_code in {400, 413}
            assert exchange.events.empty()
    asyncio.run(check())


def test_stream_true_buffers_full_result_then_terminal_sse_without_usage():
    async def check():
        async with live() as (exchange, client, _):
            task = asyncio.create_task(client.post("chat/completions", json=payload(stream=True)))
            event = await asyncio.wait_for(exchange.events.get(), 3)
            assert not task.done()
            exchange.mark_emitted(event.model.id)
            exchange.accept(h.ModelResult(model_call_id=event.model.id, message=message("full text")))
            response = await task
            assert response.headers["content-type"].startswith("text/event-stream")
            frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert frames[-1] == "[DONE]"
            chunk = json.loads(frames[0])
            assert chunk["choices"] == [{"index": 0, "delta": {"role": "assistant", "content": "full text"}, "finish_reason": "stop"}]
            assert "usage" not in chunk
    asyncio.run(check())


def test_pending_bridge_call_cancel_closes_without_task_exception_or_retry():
    async def check():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unhandled.append(ctx))
        async with live() as (exchange, client, _):
            task = asyncio.create_task(client.post("chat/completions", json=payload()))
            await asyncio.wait_for(exchange.events.get(), 3)
            exchange.close()
            response = await asyncio.wait_for(task, 3)
            assert response.status_code == 502
            assert exchange.events.empty()
        assert unhandled == []
    asyncio.run(check())


@pytest.mark.parametrize("stream", [False, True])
def test_encoded_response_expansion_is_bounded(stream):
    async def check():
        async with live() as (exchange, client, _):
            task = asyncio.create_task(client.post("chat/completions", json=payload(stream=stream)))
            event = await asyncio.wait_for(exchange.events.get(), 3)
            exchange.mark_emitted(event.model.id)
            exchange.accept(h.ModelResult(model_call_id=event.model.id, message=message("\x00" * (200 * 1024))))
            response = await task
            assert response.status_code == 502
            assert len(response.content) < 1024
    asyncio.run(check())


def test_bridge_does_not_change_other_protocols_logging(caplog):
    async def check():
        import logging
        logger = logging.getLogger("uvicorn.error")
        with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
            async with live():
                assert logger.level == logging.WARNING
            assert logger.level == logging.WARNING
    asyncio.run(check())


def test_bridge_startup_constructor_failure_closes_reserved_socket(monkeypatch):
    async def check():
        module = bridge_module()
        sockets = []
        real_socket = module.socket.socket
        def allocate(*args, **kwargs):
            sock = real_socket(*args, **kwargs)
            sockets.append(sock)
            return sock
        def fail(*args, **kwargs):
            raise RuntimeError("startup failure")
        monkeypatch.setattr(module.socket, "socket", allocate)
        monkeypatch.setattr(module, "_ExecutionServer", fail)
        with pytest.raises(RuntimeError, match="startup failure"):
            async with module.loopback_bridge(ExecutionExchange("exec", "model", 0)):
                pytest.fail("startup unexpectedly succeeded")
        assert sockets and all(sock.fileno() == -1 for sock in sockets)
    asyncio.run(check())


def test_bridge_cancel_during_startup_closes_listener_before_next_execution(monkeypatch):
    async def check():
        module = bridge_module()
        loop = asyncio.get_running_loop()
        registered, resume_startup = asyncio.Event(), asyncio.Event()
        closing, resume_close = asyncio.Event(), asyncio.Event()
        listeners, servers = [], []
        real_create_server = loop.create_server
        real_server = module._ExecutionServer

        def record_server(*args, **kwargs):
            server = real_server(*args, **kwargs)
            servers.append(server)
            return server

        async def pause_registration(*args, **kwargs):
            listener = await real_create_server(*args, **kwargs)
            listeners.append(listener)
            if len(listeners) == 1:
                real_wait_closed = listener.wait_closed

                async def pause_close():
                    closing.set()
                    await resume_close.wait()
                    await real_wait_closed()

                monkeypatch.setattr(listener, "wait_closed", pause_close)
                registered.set()
                await resume_startup.wait()
            return listener

        monkeypatch.setattr(module, "_ExecutionServer", record_server)
        monkeypatch.setattr(loop, "create_server", pause_registration)

        async def enter():
            async with module.loopback_bridge(ExecutionExchange("first", "host-model", 0)):
                pytest.fail("canceled startup unexpectedly yielded")

        task = asyncio.create_task(enter())
        close_waiter = asyncio.create_task(closing.wait())
        try:
            await asyncio.wait_for(registered.wait(), 3)
            task.cancel()
            async with asyncio.timeout(3):
                while not servers[0].should_exit:
                    await asyncio.sleep(0)
            task.cancel()  # Repeated caller cancellation must not cancel serve.
            await asyncio.sleep(0)
            assert not task.done()
            resume_startup.set()
            # On 0.29 serve returns here without shutdown. The bridge must still
            # close/await the actual asyncio listener, not just its raw socket.
            done, _ = await asyncio.wait((task, close_waiter), timeout=3, return_when=asyncio.FIRST_COMPLETED)
            assert close_waiter in done, "serve completed without awaiting listener cleanup"
            assert not listeners[0].is_serving()
            task.cancel()  # Also cancel while the listener cleanup is awaited.
            await asyncio.sleep(0)
            assert not task.done()
            resume_close.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert listeners[0].sockets == ()
            async with live() as (exchange, client, _):
                response = await client.post("chat/completions", json=payload(), headers={"Authorization": "Bearer wrong"})
                assert response.status_code == 401
                assert exchange.events.empty()
        finally:
            resume_startup.set()
            resume_close.set()
            if not task.done():
                task.cancel()
            close_waiter.cancel()
            await asyncio.gather(task, close_waiter, return_exceptions=True)
            # Broken production may already have closed the raw FD; the test's
            # asyncio.run closes the loop/selector even if listener.close fails.
            for listener in listeners:
                with suppress(ValueError):
                    listener.close()
                    await listener.wait_closed()
    asyncio.run(check())


@pytest.mark.parametrize("phase", ["startup", "main_loop"])
def test_bridge_server_exception_closes_registered_listener(monkeypatch, phase):
    async def check():
        module = bridge_module()
        servers = []
        real_server = module._ExecutionServer

        class FailingServer(real_server):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                servers.append(self)

        original = getattr(real_server, phase)

        async def fail(self, *args, **kwargs):
            if phase == "startup":
                await original(self, *args, **kwargs)
            raise RuntimeError("registered listener failure")

        monkeypatch.setattr(FailingServer, phase, fail)
        monkeypatch.setattr(module, "_ExecutionServer", FailingServer)
        try:
            with pytest.raises(RuntimeError, match="registered listener failure"):
                async with module.loopback_bridge(ExecutionExchange("failed", "host-model", 0)):
                    pass
            assert servers and servers[0].servers
            assert all(not listener.is_serving() and listener.sockets == () for listener in servers[0].servers)
        finally:
            for server in servers:
                for listener in getattr(server, "servers", []):
                    with suppress(ValueError):
                        listener.close()
                        await listener.wait_closed()
        monkeypatch.setattr(module, "_ExecutionServer", real_server)
        async with live() as (exchange, client, _):
            response = await client.post("chat/completions", json=payload(), headers={"Authorization": "Bearer wrong"})
            assert response.status_code == 401
            assert exchange.events.empty()
    asyncio.run(check())


def test_concurrent_requests_refused_instead_of_unbounded_effect_queue():
    async def check():
        async with live() as (exchange, client, _):
            first = asyncio.create_task(client.post("chat/completions", json=payload()))
            event = await asyncio.wait_for(exchange.events.get(), 3)
            second = await client.post("chat/completions", json=payload())
            assert second.status_code == 409
            assert exchange.events.empty()
            exchange.mark_emitted(event.model.id)
            exchange.accept(h.ModelResult(model_call_id=event.model.id, message=message()))
            assert (await first).status_code == 200
    asyncio.run(check())

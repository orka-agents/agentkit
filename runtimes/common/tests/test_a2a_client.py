"""SDK defaults qualified over real TCP/HTTP, including streaming SSE."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import contextmanager
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("a2a")
import uvicorn
from a2a.client import ClientConfig, ClientFactory
from a2a.types import Message, Part, TaskQueryParams, TaskState, TextPart
from test_a2a_protocol import Factory, app


@contextmanager
def live_server(application):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        # The card's endpoint is operator supplied, never synthesized from Host.
        application.state.a2a_card.url = f"http://127.0.0.1:{port}/"
        server = uvicorn.Server(
            uvicorn.Config(application, log_level="critical", lifespan="on")
        )
        thread = threading.Thread(
            target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "HTTP server did not start"
        try:
            yield f"http://127.0.0.1:{port}/"
        finally:
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive(), "HTTP server did not drain"


def message(text="hello", **kwargs):
    return Message(
        message_id=str(uuid4()),
        role="user",
        parts=[Part(root=TextPart(text=text))],
        **kwargs,
    )


async def connect(url, http, **config):
    return await ClientFactory.connect(
        url, client_config=ClientConfig(httpx_client=http, **config)
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_sdk_default_configuration_send_stream_get_and_context(streaming):
    factory = Factory()
    with live_server(app(factory, auth_token="sdk-token")) as url:

        async def exercise():
            async with httpx.AsyncClient(
                headers={"authorization": "Bearer sdk-token"}, timeout=5
            ) as http:
                client = await connect(url, http, streaming=streaming)
                events, states = [], []
                async for event in client.send_message(message()):
                    events.append(event)
                    # SDK client trackers mutate their task object on later events.
                    states.append(event[0].status.state)
                task = events[-1][0]
                assert task.status.state == TaskState.completed
                assert task.artifacts[0].parts[0].root.text == "answer: hello"
                assert (
                    await client.get_task(TaskQueryParams(id=task.id))
                ).status.state == TaskState.completed
                following = [
                    event
                    async for event in client.send_message(
                        message("next", context_id=task.context_id)
                    )
                ]
                assert following[-1][0].context_id == task.context_id
                assert following[-1][0].id != task.id
                if streaming:
                    assert TaskState.working in states
                    assert events[-1][1].final is True
                assert [turn.text for turn in factory.runtime.requests[-1].history] == [
                    "hello",
                    "answer: hello",
                ]

        asyncio.run(exercise())

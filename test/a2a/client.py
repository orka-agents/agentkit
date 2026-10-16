"""Use the pinned SDK client against a real built-image HTTP server."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, Task, TaskQueryParams, TaskState, TextPart


async def exercise() -> None:
    url = "http://agent:8080/"
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {os.environ['AGENTKIT_AUTH_TOKEN']}"},
        timeout=20,
    ) as http:
        async with asyncio.timeout(60):
            while True:
                try:
                    response = await http.get(url + ".well-known/agent-card.json")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        previous = None
        for streaming in (False, True):
            client = await ClientFactory.connect(
                url, client_config=ClientConfig(httpx_client=http, streaming=streaming)
            )
            message = Message(
                message_id=str(uuid4()),
                role=Role.user,
                parts=[Part(root=TextPart(text="smoke turn"))],
                context_id=previous.context_id if previous else None,
            )
            task = None
            async for result in client.send_message(message):
                assert isinstance(result, tuple)
                task, _ = result
            assert isinstance(task, Task) and task.status.state == TaskState.completed
            assert (
                task.artifacts[0].parts[0].root.text
                == f"answer with {2 if previous else 1} user turns"
            )
            persisted = await client.get_task(TaskQueryParams(id=task.id))
            assert persisted.status.state == TaskState.completed
            if previous:
                assert task.context_id == previous.context_id and task.id != previous.id
            previous = task
    print("Built-image A2A SDK send/stream/get and context smoke passed")


if __name__ == "__main__":
    asyncio.run(exercise())

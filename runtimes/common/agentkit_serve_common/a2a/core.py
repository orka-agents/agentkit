"""Single-runtime, process-local text A2A skin for the pinned 0.3 SDK."""

from __future__ import annotations

import asyncio
import hmac
import logging
import sys
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlsplit
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.events import EventConsumer, EventQueue, InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    HTTPAuthSecurityScheme,
    InternalError,
    InvalidParamsError,
    JSONRPCError,
    Message,
    Part,
    Role,
    SecurityScheme,
    Task,
    TaskArtifactUpdateEvent,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from fastapi import FastAPI
from starlette.responses import JSONResponse

from ..config import AgentSpec
from ..conversation import ConversationTurn, RunRequest
from ..runtime import RuntimeFactory, RuntimeHealth

MAX_TASKS = 128
MAX_CONTEXTS = 64
MAX_HISTORY_TURNS = 32
MAX_HISTORY_BYTES = 512 * 1024
MAX_TEXT_BYTES = 128 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_SUBSCRIBERS = 32
TERMINAL = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}
CARDS = {"/.well-known/agent-card.json", "/.well-known/agent.json"}
_LOG = logging.getLogger(__name__)


class _PayloadFilter(logging.Filter):
    def filter(self, record):
        # SDK validation tracebacks and request debug logs contain client input.
        record.exc_info = None
        record.exc_text = None
        return not record.msg.startswith("Request body:")


class _Application(A2AFastAPIApplication):
    def _generate_error_response(self, request_id, error):
        # SDK validation data includes the rejected input; never echo it publicly.
        root = getattr(error, "root", error)
        safe = root.model_copy(update={"data": None})
        if isinstance(safe, InternalError):
            safe.message = "Internal server error"
        elif safe.code in (-32600, -32602, -32700):
            safe.message = "Invalid request parameters"
        return super()._generate_error_response(
            request_id, JSONRPCError(code=safe.code, message=safe.message)
        )


class _Queue(EventQueue):
    def __init__(self):
        super().__init__(max_queue_size=16)

    def tap(self):
        # Reserve one bounded child for the single shared cancellation consumer.
        if len(self._children) >= MAX_SUBSCRIBERS + 1:
            raise ServerError(
                UnsupportedOperationError(message="Subscriber limit reached")
            )
        child = _Queue()
        self._children.append(child)
        return child

    async def close(self, immediate=False):
        # SDK EventConsumer calls close(True) before yielding a final event.
        # Clearing here loses sibling fanout; joining hangs on disconnected taps.
        # Retain buffered events, marking the queue closed without waiting for readers.
        async with self._lock:
            self._is_closed = True
            if hasattr(self.queue, "shutdown"):
                self.queue.shutdown(False)
        for child in self._children:
            await child.close()


class _Queues(InMemoryQueueManager):
    async def create_or_tap(self, task_id):
        async with self._lock:
            if task_id not in self._task_queue:
                self._task_queue[task_id] = _Queue()
                return self._task_queue[task_id]
            return self._task_queue[task_id].tap()

    async def close(self, task_id):
        # Settlement and SDK cleanup may both retire the same queue.
        async with self._lock:
            queue = self._task_queue.pop(task_id, None)
        if queue:
            await queue.close()


class _Store(TaskStore):
    def __init__(self):
        self.tasks = OrderedDict()

    async def publish(self, task):
        self.tasks[task.id] = task.model_copy(deep=True)
        while len(self.tasks) > MAX_TASKS:
            expired = next(
                (
                    key
                    for key, value in self.tasks.items()
                    if value.status.state in TERMINAL
                ),
                None,
            )
            if expired is None:
                break  # Admission guarantees at most one live task; never evict it.
            del self.tasks[expired]

    async def save(self, task, context=None):
        # Execution publishes authoritative snapshots before fanout. SDK consumers
        # aggregate independently; stale/reordered subscriber writes must not roll
        # back terminal state, resurrect evicted tasks, or duplicate artifacts.
        pass

    async def get(self, task_id, context=None):
        task = self.tasks.get(task_id)
        return task.model_copy(deep=True) if task else None

    async def delete(self, task_id, context=None):
        self.tasks.pop(task_id, None)


class _Executor(AgentExecutor):
    def __init__(self, store, queues):
        self.store, self.queues = store, queues
        self.runtime = None
        self.ready = False
        self.busy = False
        self.lock = asyncio.Lock()
        self.health = RuntimeHealth()
        self.contexts = OrderedDict()
        self.active_id = None
        self.owned = None
        self.run_task = None
        self.cancel_requested = False

    def validate(self, params):
        message = params.message
        if message.task_id is not None:
            raise ServerError(
                InvalidParamsError(message="Named tasks cannot be continued")
            )
        if message.context_id is not None and message.context_id not in self.contexts:
            raise ServerError(InvalidParamsError(message="Unknown or expired context"))
        if (
            message.role != Role.user
            or not message.parts
            or any(not isinstance(part.root, TextPart) for part in message.parts)
        ):
            raise ServerError(
                InvalidParamsError(message="Only user text messages are supported")
            )
        text = "\n".join(part.root.text for part in message.parts)
        try:
            text_bytes = len(text.encode("utf-8"))
        except UnicodeError:
            raise ServerError(
                InvalidParamsError(message="Text must be valid UTF-8")
            ) from None
        if not text.strip() or text_bytes > MAX_TEXT_BYTES:
            raise ServerError(
                InvalidParamsError(message="Text is empty or exceeds the input limit")
            )
        if (
            params.metadata
            or message.metadata
            or message.reference_task_ids
            or message.extensions
            or any(part.root.metadata for part in message.parts)
        ):
            raise ServerError(
                InvalidParamsError(
                    message="Client metadata and task references are unsupported"
                )
            )
        config = params.configuration
        if config:
            if config.push_notification_config:
                raise ServerError(UnsupportedOperationError())
            if config.accepted_output_modes and any(
                mode != "text/plain" for mode in config.accepted_output_modes
            ):
                raise ServerError(
                    InvalidParamsError(message="Only text/plain output is supported")
                )
        if not self.ready or not self.health.healthy:
            raise ServerError(InternalError(message="Runtime unavailable"))
        if self.busy:
            raise ServerError(InvalidParamsError(message="Runtime is busy"))

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        self.active_id = context.task_id
        self.cancel_requested = not self.ready
        self.owned = asyncio.create_task(self._execute_owned(context, event_queue))
        # The SDK owns this outer producer, not runtime execution or its cleanup.
        await asyncio.shield(self.owned)

    async def _execute_owned(self, context, queue):
        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
            history=[context.message],
        )
        context_id = task.context_id
        self.contexts.setdefault(context_id, ())
        self.contexts.move_to_end(context_id)
        while len(self.contexts) > MAX_CONTEXTS:
            self.contexts.popitem(last=False)
        await self.store.publish(task)
        await queue.enqueue_event(task.model_copy(deep=True))
        request = RunRequest(
            prompt=context.get_user_input(),
            history=self.contexts[context_id],
            session_id=context_id,
            turn_id=task.id,
        )
        self.run_task = asyncio.create_task(self.runtime.run(request))
        if self.cancel_requested:
            self.run_task.cancel()  # Shutdown may precede owned execution startup.
        state, text, failed = TaskState.failed, None, False
        try:
            result = await asyncio.shield(self.run_task)
            if len(result.text.encode("utf-8")) <= MAX_TEXT_BYTES:
                state, text = TaskState.completed, result.text
        except asyncio.CancelledError:
            state = TaskState.canceled
        except Exception as exc:  # noqa: BLE001 - runtime boundary; never expose exception payloads.
            failed = True
            self.health.record(exc)
            _LOG.warning(
                "A2A execution failed task_id=%s error_type=%s",
                task.id,
                type(exc).__name__,
            )
        async with self.lock:
            if self.cancel_requested and not failed:
                state, text = TaskState.canceled, None
            if state == TaskState.completed:
                history = self.contexts[context_id] + (
                    ConversationTurn("user", request.prompt),
                    ConversationTurn("assistant", text),
                )
                history = history[-MAX_HISTORY_TURNS:]
                while (
                    history
                    and sum(len(turn.text.encode("utf-8")) for turn in history)
                    > MAX_HISTORY_BYTES
                ):
                    history = history[2:]
                self.contexts[context_id] = history
                artifact = Artifact(
                    artifact_id=str(uuid4()), parts=[Part(root=TextPart(text=text))]
                )
                task.artifacts = [artifact]
                await queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=task.id,
                        context_id=context_id,
                        artifact=artifact,
                        last_chunk=True,
                    )
                )
            task.status = TaskStatus(state=state)
            if state == TaskState.failed:
                task.status.message = Message(
                    message_id=str(uuid4()),
                    role=Role.agent,
                    parts=[Part(root=TextPart(text="Agent execution failed"))],
                )
            await self.store.publish(task)
            await queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=context_id,
                    status=task.status,
                    final=True,
                )
            )
            await self.queues.close(task.id)
            # Admission and terminal publication are one locked settlement.
            self.busy = False
            self.active_id = None
            self.run_task = None

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        async with self.lock:
            if context.task_id != self.active_id or not self.owned or self.owned.done():
                raise ServerError(TaskNotCancelableError())
            if not self.cancel_requested:
                self.cancel_requested = True
                if self.run_task:
                    self.run_task.cancel()  # Once only: repeated cancel must not interrupt cleanup.
            owned = self.owned
        # SDK passes a child queue; _execute_owned publishes on the original queue.
        await asyncio.shield(owned)


class _Handler(DefaultRequestHandler):
    def __init__(self, executor: _Executor, store: _Store, *, queue_manager: _Queues):
        super().__init__(executor, store, queue_manager=queue_manager)
        self._cancellations: dict[str, asyncio.Task[Task | None]] = {}

    async def on_cancel_task(self, params, context=None):
        # Repeated/disconnected cancellation requests share one SDK child consumer,
        # rather than exhausting the subscriber bound or interrupting cleanup.
        pending = self._cancellations.get(params.id)
        if pending is None:
            pending = asyncio.create_task(super().on_cancel_task(params, context))
            self._cancellations[params.id] = pending

            def settled(task):
                self._cancellations.pop(params.id, None)
                if not task.cancelled():
                    task.exception()  # Retrieve errors even if the caller disconnected.

            pending.add_done_callback(settled)
        return await asyncio.shield(pending)

    async def _setup_message_execution(self, params, context=None):
        executor = self.agent_executor
        async with executor.lock:
            # Guard the client-supplied IDs before RequestContext mutates them and
            # before create_or_tap can share an existing task with another producer.
            executor.validate(params)
            executor.busy = True
            try:
                return await super()._setup_message_execution(params, context)
            except BaseException:
                executor.busy = False
                raise

    async def _cleanup_producer(self, producer_task, task_id):
        with suppress(asyncio.CancelledError):
            await asyncio.shield(producer_task)
        # Outer cancellation is not retirement of the owned runtime/cleanup.
        owned = self.agent_executor.owned
        if owned and self.agent_executor.active_id == task_id:
            await asyncio.shield(owned)
        await self._queue_manager.close(task_id)
        async with self._running_agents_lock:
            if self._running_agents.get(task_id) is producer_task:
                self._running_agents.pop(task_id, None)

    async def on_resubscribe_to_task(self, params, context=None):
        async with self.agent_executor.lock:
            task = await self.task_store.get(params.id, context)
            if not task:
                raise ServerError(TaskNotFoundError())
            if task.status.state in TERMINAL:
                raise ServerError(InvalidParamsError(message="Task is terminal"))
            original = await self._queue_manager.get(params.id)
            if not original:
                raise ServerError(TaskNotFoundError())
            if len(original._children) >= MAX_SUBSCRIBERS:
                raise ServerError(
                    UnsupportedOperationError(message="Subscriber limit reached")
                )
            queue = original.tap()
        # A snapshot makes active resubscription usable by default SDK clients.
        yield task
        async for event in EventConsumer(queue).consume_all():
            yield event

    async def drain(self):
        # Withdrawn readiness prevents admission. Cancel active work once, including
        # the setup/startup gap, then await owned cleanup before runtime retirement.
        executor = self.agent_executor
        async with executor.lock:
            if executor.busy and not executor.cancel_requested:
                executor.cancel_requested = True
                if executor.run_task:
                    executor.run_task.cancel()
        producers = list(self._running_agents.values())
        if producers:
            await asyncio.gather(
                *(asyncio.shield(task) for task in producers), return_exceptions=True
            )
        if self.agent_executor.owned:
            await asyncio.shield(self.agent_executor.owned)
        if self._cancellations:
            await asyncio.gather(
                *(asyncio.shield(task) for task in list(self._cancellations.values())),
                return_exceptions=True,
            )
        while self._background_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in list(self._background_tasks)),
                return_exceptions=True,
            )


class _Boundary:
    def __init__(self, app, executor, token):
        self.app, self.executor = app, executor
        self.token = token.encode("utf-8") if token is not None else None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == "/" and self.token is not None:
            headers = dict(scope["headers"])
            authorization = headers.get(b"authorization", b"")
            scheme, _, credential = authorization.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(
                credential, self.token
            ):
                return await JSONResponse(
                    {"detail": "Unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
        if scope["path"] in CARDS | {"/"} and (
            not self.executor.ready
            or (scope["path"] in CARDS and not self.executor.health.healthy)
        ):
            return await JSONResponse(
                {"detail": "Runtime unavailable"}, status_code=503
            )(scope, receive, send)
        if scope["path"] == "/" and scope["method"] == "POST":
            body = bytearray()
            while True:
                chunk = await receive()
                if chunk["type"] == "http.disconnect":
                    return
                body.extend(chunk.get("body", b""))
                if len(body) > MAX_REQUEST_BYTES:
                    return await JSONResponse(
                        {"detail": "Request too large"}, status_code=413
                    )(scope, receive, send)
                if not chunk.get("more_body", False):
                    break
            replayed = False

            async def bounded_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            return await self.app(scope, bounded_receive, send)
        await self.app(scope, receive, send)


def create_a2a_app(
    spec: AgentSpec,
    factory: RuntimeFactory,
    auth_token: str | None = None,
    *,
    advertised_url: str = "http://localhost:8080/",
) -> FastAPI:
    """Create the optional text-only A2A application with one configured principal."""
    supports = getattr(factory, "supports_a2a", None)
    if not callable(supports) or supports() is not True:
        raise ValueError("Runtime factory does not support A2A")
    try:
        url = urlsplit(advertised_url)
        valid = (
            url.scheme in {"http", "https"}
            and url.hostname
            and not url.username
            and not url.password
            and not url.query
            and not url.fragment
            and url.port != 0
            and not any(char.isspace() for char in advertised_url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Advertised URL must be an absolute HTTP(S) URL without credentials, query or fragment"
        )
    if spec.brokered_tools:
        raise ValueError("A2A does not support brokered tools")
    if auth_token is not None and not auth_token:
        raise ValueError("A2A auth token must not be empty")
    logger = logging.getLogger("a2a.server.apps.jsonrpc.jsonrpc_app")
    if not any(isinstance(filter_, _PayloadFilter) for filter_ in logger.filters):
        logger.addFilter(_PayloadFilter())
    store, queues = _Store(), _Queues()
    executor = _Executor(store, queues)
    handler = _Handler(executor, store, queue_manager=queues)
    security = (
        {"bearer": SecurityScheme(root=HTTPAuthSecurityScheme(scheme="bearer"))}
        if auth_token is not None
        else None
    )
    card = AgentCard(
        name=spec.metadata.name,
        description="AgentKit text agent",
        version="0.0.0",
        protocol_version="0.3.0",
        url=advertised_url,
        preferred_transport="JSONRPC",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="chat", name="Chat", description="Text conversation", tags=["chat"]
            )
        ],
        security_schemes=security,
        security=[{"bearer": []}] if security else None,
    )

    @asynccontextmanager
    async def lifespan(application):
        session = factory.build_runtime(spec)
        executor.runtime = await session.__aenter__()
        executor.ready = True
        try:
            yield
        finally:
            executor.ready = False
            exception = sys.exc_info()

            async def retire():
                try:
                    await handler.drain()
                finally:
                    executor.runtime = None
                    await session.__aexit__(*exception)

            # Forced/repeated shutdown cancellation must not exit the long-lived
            # runtime while an execution still owns tool/process cleanup.
            cleanup = asyncio.create_task(retire())
            canceled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    canceled = True
            cleanup.result()
            if canceled:
                raise asyncio.CancelledError

    application = _Application(
        card, handler, max_content_length=MAX_REQUEST_BYTES
    ).build(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.add_middleware(_Boundary, executor=executor, token=auth_token)
    application.state.a2a_handler = handler
    application.state.a2a_card = card
    return application

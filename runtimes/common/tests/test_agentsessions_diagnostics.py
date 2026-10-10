from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from threading import Event

import pytest

from agentkit_serve_common.agentsessions.diagnostics import suppress_sdk_diagnostics


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _RejectMessage(logging.Filter):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def filter(self, record):
        return record.getMessage() != self.message


@pytest.fixture
def sdk_logger(monkeypatch):
    logger = logging.getLogger("openai._base_client")
    handler = _CaptureHandler()
    previous_level = logger.level
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "filters", [])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "disabled", False)
    logger.setLevel(logging.DEBUG)
    try:
        yield logger, handler
    finally:
        logger.setLevel(previous_level)
        handler.close()


@pytest.mark.parametrize("secret", ["baked-instructions", "private-history", "execution-input"])
def test_suppression_is_scoped_to_the_sdk_logger(sdk_logger, caplog, secret):
    logger, handler = sdk_logger
    other_logger = logging.getLogger("openai._response")
    environment = dict(os.environ)
    handlers = logger.handlers
    options = {"json_data": {"messages": [{"role": "user", "content": secret}]}}

    logger.debug("Request options: %s", options)
    with caplog.at_level(logging.DEBUG, logger=other_logger.name):
        with suppress_sdk_diagnostics():
            logger.debug("Request options: %s", options)
            other_logger.debug("ordinary SDK diagnostic")
            assert logger.level == logging.DEBUG
            assert logger.handlers is handlers
            assert not logger.disabled
            assert not logger.propagate
            assert dict(os.environ) == environment
    logger.debug("Request options: %s", options)

    assert len(handler.messages) == 2
    assert all(secret in message for message in handler.messages)
    assert (other_logger.name, logging.DEBUG, "ordinary SDK diagnostic") in caplog.record_tuples
    assert logger.filters == []
    assert logger.handlers is handlers
    assert logger.level == logging.DEBUG
    assert dict(os.environ) == environment


@pytest.mark.parametrize("error", [ValueError, asyncio.CancelledError])
def test_exception_restores_context_and_filter(sdk_logger, error):
    logger, handler = sdk_logger

    with pytest.raises(error):
        with suppress_sdk_diagnostics():
            active_filter = logger.filters[-1]
            logger.debug("sensitive diagnostic")
            raise error()
    assert active_filter.filter(logging.makeLogRecord({}))
    logger.debug("ordinary after exception")
    # A later scope must restore its own previous context, not a stale True.
    with suppress_sdk_diagnostics():
        logger.debug("sensitive later diagnostic")
    logger.debug("ordinary after later scope")

    assert handler.messages == ["ordinary after exception", "ordinary after later scope"]
    assert logger.filters == []


def test_nested_scopes_share_one_filter_and_restore_outer_context(sdk_logger):
    logger, handler = sdk_logger

    with suppress_sdk_diagnostics():
        outer_filters = logger.filters.copy()
        assert len(outer_filters) == 1
        logger.debug("sensitive outer diagnostic")
        with pytest.raises(ValueError):
            with suppress_sdk_diagnostics():
                assert logger.filters == outer_filters
                logger.debug("sensitive inner diagnostic")
                raise ValueError()
        assert logger.filters == outer_filters
        logger.debug("sensitive outer diagnostic after inner exit")
    logger.debug("ordinary after outer exit")

    assert outer_filters[0].filter(logging.makeLogRecord({}))
    assert logger.filters == []
    assert handler.messages == ["ordinary after outer exit"]


def test_overlapping_asyncio_scopes_do_not_suppress_ordinary_tasks(sdk_logger):
    logger, handler = sdk_logger

    async def check():
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release_first = asyncio.Event()
        first_exited = asyncio.Event()
        second_checked = asyncio.Event()
        release_second = asyncio.Event()

        async def first():
            with suppress_sdk_diagnostics():
                first_entered.set()
                await release_first.wait()
                logger.debug("sensitive first execution")
            logger.debug("ordinary first task after exit")
            first_exited.set()

        async def second():
            await first_entered.wait()
            with suppress_sdk_diagnostics():
                second_entered.set()
                await first_exited.wait()
                logger.debug("sensitive second execution after first exit")
                second_checked.set()
                await release_second.wait()
            logger.debug("ordinary second task after exit")

        tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
        try:
            await asyncio.wait_for(second_entered.wait(), 3)
            assert len(logger.filters) == 1
            logger.debug("ordinary task during both executions")
            release_first.set()
            await asyncio.wait_for(second_checked.wait(), 3)
            assert len(logger.filters) == 1
            logger.debug("ordinary task during second execution")
            release_second.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(check())

    assert logger.filters == []
    assert handler.messages == [
        "ordinary task during both executions",
        "ordinary first task after exit",
        "ordinary task during second execution",
        "ordinary second task after exit",
    ]


def test_task_cancellation_removes_filter_and_restores_context(sdk_logger):
    logger, handler = sdk_logger

    async def check():
        entered = asyncio.Event()

        async def execution():
            try:
                with suppress_sdk_diagnostics():
                    active_filter = logger.filters[-1]
                    logger.debug("sensitive before cancellation")
                    entered.set()
                    await asyncio.Future()
            finally:
                assert active_filter.filter(logging.makeLogRecord({}))
                logger.debug("ordinary task after cancellation")

        task = asyncio.create_task(execution())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            logger.debug("ordinary concurrent task")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())

    assert logger.filters == []
    assert handler.messages == ["ordinary concurrent task", "ordinary task after cancellation"]


def test_existing_logger_and_handler_filters_remain_owned_by_their_callers(sdk_logger):
    logger, handler = sdk_logger
    first = _RejectMessage("logger-filtered")
    second = logging.Filter()
    added = _RejectMessage("later-filtered")
    handler_filter = _RejectMessage("handler-filtered")
    logger.addFilter(first)
    logger.addFilter(second)
    handler.addFilter(handler_filter)
    logger_filters = logger.filters
    handler_filters = handler.filters
    handlers = logger.handlers

    with suppress_sdk_diagnostics():
        active_filters = logger.filters
        assert active_filters is not logger_filters
        assert logger_filters == [first, second]
        assert logger.filters[1:] == [first, second]
        assert len(logger.filters) == 3
        logger.addFilter(added)
        logger.debug("sensitive diagnostic")
        assert logger.filters is active_filters
        assert handler.filters is handler_filters
        assert handler.filters == [handler_filter]
        assert logger.handlers is handlers
        assert handler.level == logging.NOTSET
    assert logger.filters is not active_filters
    assert logger_filters == [first, second]
    assert active_filters[1:] == [first, second, added]
    assert logger.filters == [first, second, added]
    assert handler.filters is handler_filters
    assert handler.filters == [handler_filter]
    logger.debug("logger-filtered")
    logger.debug("handler-filtered")
    logger.debug("later-filtered")
    logger.debug("ordinary accepted diagnostic")

    assert handler.messages == ["ordinary accepted diagnostic"]


def test_sensitive_records_never_reach_a_queue_drained_after_scope_exit(sdk_logger, monkeypatch):
    logger, handler = sdk_logger
    parent = logging.getLogger("openai")
    records = Queue()
    queue_handler = QueueHandler(records)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(parent, "handlers", [queue_handler])
    monkeypatch.setattr(parent, "propagate", False)

    logger.debug("ordinary before scope")
    with suppress_sdk_diagnostics():
        logger.debug("Request options: %s", {"json_data": {"messages": "private-input"}})
    logger.debug("ordinary after scope")
    assert records.qsize() == 2
    assert handler.messages == []
    assert logger.filters == []

    listener = QueueListener(records, handler)
    listener.start()
    listener.stop()
    queue_handler.close()

    assert handler.messages == ["ordinary before scope", "ordinary after scope"]


def test_context_propagating_workers_are_suppressed_but_ordinary_threads_are_not(sdk_logger):
    logger, handler = sdk_logger

    async def check():
        with ThreadPoolExecutor(max_workers=1) as workers:
            with suppress_sdk_diagnostics():
                await asyncio.to_thread(logger.debug, "sensitive context-propagating worker")
                workers.submit(logger.debug, "ordinary worker").result(timeout=3)
            await asyncio.to_thread(logger.debug, "ordinary context-propagating worker after exit")

    asyncio.run(check())

    assert logger.filters == []
    assert handler.messages == ["ordinary worker", "ordinary context-propagating worker after exit"]


def test_overlapping_thread_scopes_keep_filter_until_the_last_exit(sdk_logger):
    logger, handler = sdk_logger
    entered = Event()
    release = Event()

    def execution_worker():
        with suppress_sdk_diagnostics():
            entered.set()
            assert release.wait(3)
            logger.debug("sensitive worker after main scope exit")

    with ThreadPoolExecutor(max_workers=1) as workers:
        try:
            with suppress_sdk_diagnostics():
                worker = workers.submit(execution_worker)
                assert entered.wait(3)
                assert len(logger.filters) == 1
                logger.debug("sensitive main execution")
            assert len(logger.filters) == 1
            logger.debug("ordinary main thread during worker execution")
        finally:
            release.set()
        worker.result(timeout=3)

    assert logger.filters == []
    assert handler.messages == ["ordinary main thread during worker execution"]


def test_side_effecting_filters_never_observe_scoped_request_options(sdk_logger):
    logger, handler = sdk_logger
    observed = []

    class RecordingFilter(logging.Filter):
        def filter(self, record):
            observed.append((self.name, record.getMessage()))
            return True

    first = RecordingFilter("first")
    second = RecordingFilter("second")
    logger.addFilter(first)
    logger.addFilter(second)
    original_filters = logger.filters
    original_order = original_filters.copy()
    logger.debug("ordinary before scope")

    async def check():
        async def ordinary():
            logger.debug("ordinary concurrent task")

        # Created outside suppression, but emits while the guard is installed.
        ordinary_task = asyncio.create_task(ordinary())
        with suppress_sdk_diagnostics():
            assert logger.filters[1:] == original_order
            active_order = logger.filters.copy()
            logger.debug("Request options: %s", {"json_data": {"messages": "private-input"}})
            with suppress_sdk_diagnostics():
                assert logger.filters == active_order
                logger.debug("Request options: %s", {"json_data": {"messages": "private-history"}})
            assert logger.filters == active_order
            await ordinary_task
        logger.debug("ordinary after scope")

    asyncio.run(check())

    expected_messages = ["ordinary before scope", "ordinary concurrent task", "ordinary after scope"]
    assert observed == [
        (name, message) for message in expected_messages for name in ("first", "second")
    ]
    assert handler.messages == expected_messages
    assert original_filters == original_order
    assert logger.filters == original_order


@pytest.mark.parametrize("transition", ["enter", "exit"])
def test_inflight_ordinary_filters_run_once_across_scope_changes(sdk_logger, transition):
    logger, handler = sdk_logger
    blocked = Event()
    release = Event()
    observed = []

    class RecordingFilter(logging.Filter):
        def filter(self, record):
            observed.append(self.name)
            if self.name == "first":
                blocked.set()
                assert release.wait(3)
            return True

    first = RecordingFilter("first")
    second = RecordingFilter("second")
    logger.addFilter(first)
    logger.addFilter(second)
    with ThreadPoolExecutor(max_workers=1) as workers, ExitStack() as scopes:
        if transition == "exit":
            scopes.enter_context(suppress_sdk_diagnostics())
        worker = workers.submit(logger.debug, "ordinary in-flight diagnostic")
        try:
            assert blocked.wait(3)
            if transition == "enter":
                scopes.enter_context(suppress_sdk_diagnostics())
            else:
                scopes.close()
        finally:
            release.set()
        worker.result(timeout=3)

    assert observed == ["first", "second"]
    assert handler.messages == ["ordinary in-flight diagnostic"]
    assert logger.filters == [first, second]

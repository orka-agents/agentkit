from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

import agentkit_serve_common.foundry as foundry_module
from agentkit_serve_common import foundry_model_loop as foundry_model_loop_module
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.conversation import RunRequest
from agentkit_serve_common.foundry import create_foundry_app
from agentkit_serve_common.foundry_model_loop import BrokeredChatModelLoop
from agentkit_serve_common.runtime import AgentRunError, RunResult, RuntimeSession


@pytest.fixture(autouse=True)
def responses_model_api(monkeypatch):
    """These existing fixtures exercise the explicitly selected Responses transport."""
    monkeypatch.setenv("AGENTKIT_MODEL_API", "responses")


CONTINUATION_PROOF = "test-orka-continuation-proof"
CONTINUATION_AUTH = {"x-agentkit-brokered-continuation-proof": CONTINUATION_PROOF}
CONTINUATION_PROOF_BODY_FIELD = "brokered_continuation_proof"


def _app(spec: AgentSpec | None = None, factory: NoDirectRunFactory | None = None, **kwargs: Any):
    return create_foundry_app(
        spec or _spec(),
        factory or NoDirectRunFactory(),
        brokered_continuation_proof=CONTINUATION_PROOF,
        **kwargs,
    )


def _spec(*, tool_name: str = "conformance_read", brokered_class: str = "read") -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "foundry-brokered-test"},
            "model": {
                "provider": "openai-compatible",
                "baseURL": "https://api.openai.com/v1",
                "name": "gpt-4o-mini",
            },
            "instructions": "Be helpful.",
            "tools": [],
            "brokeredTools": [
                {
                    "name": tool_name,
                    "description": "Safe deterministic conformance tool.",
                    "brokeredClass": brokered_class,
                    "parameters": {"type": "object", "properties": {"probe": {"type": "boolean"}}},
                }
            ],
            "expose": {"openai": True, "port": 8080},
        }
    )


def _multi_tool_spec() -> AgentSpec:
    data = _spec().model_dump(by_alias=True)
    data["brokeredTools"] = [
        {
            "name": "check-network-telemetry",
            "description": "Read telemetry.",
            "brokeredClass": "read",
            "parameters": {"type": "object", "properties": {"site": {"type": "string"}}, "required": ["site"]},
        },
        {
            "name": "get-active-incidents",
            "description": "Read active incidents.",
            "brokeredClass": "read",
            "parameters": {"type": "object"},
        },
    ]
    return AgentSpec.model_validate(data)


class NoDirectRunRuntime:
    def __init__(self) -> None:
        self.run_requests: list[RunRequest] = []

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None

    async def run(self, request: RunRequest) -> RunResult:
        self.run_requests.append(request)
        raise AssertionError("Foundry brokered /responses must not execute direct AgentKit-owned tools")


class NoDirectRunFactory:
    def __init__(self) -> None:
        self.runtime = NoDirectRunRuntime()

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        return self.runtime


def _start(client: TestClient, prompt: str = "please read telemetry") -> dict[str, Any]:
    resp = client.post("/responses", json={"input": prompt})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _continuation(response_id: str, call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_response_id": response_id,
        "input": [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(payload, separators=(",", ":"), sort_keys=True),
                "status": "completed",
            }
        ],
    }


def _call(body: dict[str, Any]) -> dict[str, Any]:
    assert body["status"] == "completed"
    assert body["id"].startswith("caresp_")
    output = body["output"]
    assert len(output) == 1
    assert output[0]["type"] == "function_call"
    return output[0]


def _message_text(body: dict[str, Any]) -> str:
    assert body["status"] == "completed"
    message = body["output"][0]
    assert message["type"] == "message"
    return message["content"][0]["text"]


def test_foundry_brokered_requires_continuation_proof_for_readiness_and_initial_call():
    app = create_foundry_app(_spec(), NoDirectRunFactory())

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        initial = client.post("/responses", json={"input": "please read telemetry"})

    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False
    assert readiness.json()["foundryResponses"]["continuationAuth"] == "missing"
    assert initial.status_code == 503
    assert initial.json()["error"]["code"] == "brokered_continuation_auth_required"


def test_foundry_brokered_initial_response_emits_static_function_call_without_direct_execution():
    factory = NoDirectRunFactory()
    app = _app(_spec(), factory)

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        body = _start(client)

    assert readiness.status_code == 200
    assert readiness.json()["foundryResponses"] == {
        "brokeredTools": 1,
        "ownedToolsDisabled": 0,
        "stateBackend": "memory",
        "stateTtlSeconds": 900.0,
        "stateMaxPending": 128,
        "continuationAuth": "configured",
        "runtime": "deterministic",
        "scaling": "single-replica-or-sticky-routing-required",
    }
    call = _call(body)
    assert call["name"] == "conformance_read"
    assert call["call_id"] == f"call_{body['id']}_1"
    assert json.loads(call["arguments"]) == {"probe": True}
    assert factory.runtime.run_requests == []


def test_foundry_brokered_deterministic_arguments_reject_unsafe_prompt_text():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string"}},
        "required": ["site"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry https://internal.example"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsafeBrokeredArguments"


def test_foundry_brokered_synthesizes_arguments_from_required_schema_fields():
    spec = _spec(tool_name="check-network-telemetry")
    tool = spec.brokered_tools[0]
    tool.parameters["required"] = ["site"]
    app = _app(spec)

    with TestClient(app) as client:
        body = _start(client, "please call check-network-telemetry")

    call = _call(body)
    assert call["name"] == "check-network-telemetry"
    assert json.loads(call["arguments"]) == {"site": "please call check-network-telemetry"}


def test_foundry_brokered_normal_followup_with_previous_response_id_is_not_treated_as_tool_output():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/responses", json={"previous_response_id": "caresp_completed_elsewhere", "input": "next question"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["previous_response_id"] == "caresp_completed_elsewhere"
    call = _call(body)
    assert call["type"] == "function_call"


def test_foundry_brokered_rejects_normal_followup_while_previous_response_is_pending_tool_output():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        resp = client.post("/responses", json={"previous_response_id": initial["id"], "input": "next question"})

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "response_pending_function_call_output"


def test_foundry_brokered_pending_state_store_is_bounded():
    app = _app(max_pending_responses=1)

    with TestClient(app) as client:
        first = _start(client)
        second = client.post("/responses", json={"input": "another pending request"})

    assert _call(first)
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "brokered_response_state_full"


def test_foundry_brokered_failed_initial_state_persist_is_retryable_without_consuming_capacity(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    original_replace = Path.replace
    failed_once = False

    def fail_first_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            failed_once = True
            raise OSError("simulated state storage failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_first_replace)
    app = _app(response_state_file=state_file, max_pending_responses=1)

    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.post("/responses", json={"input": "please read telemetry"})
        retried = client.post("/responses", json={"input": "please read telemetry"})

    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "message": "brokered response state storage unavailable",
        "code": "brokered_response_state_storage_error",
    }
    assert retried.status_code == 200, retried.text
    assert _call(retried.json())


def test_foundry_brokered_state_transactions_do_not_deepcopy_existing_state_graph(monkeypatch):
    app = _app(max_pending_responses=3)
    original_deepcopy = foundry_module.deepcopy

    def reject_full_store_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        if isinstance(value, dict) and value and all(type(entry).__name__ == "_HostedResponseState" for entry in value.values()):
            raise AssertionError("state transactions must not deepcopy the entire state store")
        return original_deepcopy(value, memo) if memo is not None else original_deepcopy(value)

    with TestClient(app, raise_server_exceptions=False) as client:
        first = _start(client)
        first_call = _call(first)
        monkeypatch.setattr(foundry_module, "deepcopy", reject_full_store_deepcopy)
        second = client.post("/responses", json={"input": "please read telemetry again"})
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(first["id"], first_call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert second.status_code == 200, second.text
    assert completed.status_code == 200, completed.text


def test_foundry_bounded_json_walk_rejects_wide_mapping_without_bulk_child_copy():
    class WideMapping(dict[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.item_yields = 0

        def keys(self):
            raise AssertionError("bounded traversal must not bulk-copy mapping keys")

        def values(self):
            raise AssertionError("bounded traversal must not bulk-copy mapping values")

        def items(self):
            for index in range(10_000):
                self.item_yields += 1
                yield str(index), index

    value = WideMapping()
    try:
        foundry_module._bounded_json_bytes(value, max_bytes=128)
    except foundry_module._SerializedPayloadTooLarge:
        pass
    else:
        raise AssertionError("wide mapping must exceed the bounded JSON budget")

    assert value.item_yields <= 65


def test_foundry_bounded_json_rejects_escaped_string_before_encoder(monkeypatch):
    def reject_encoder(*_args: Any, **_kwargs: Any):
        raise AssertionError("oversized escaped strings must be rejected before JSONEncoder")

    monkeypatch.setattr(json.JSONEncoder, "iterencode", reject_encoder)
    try:
        foundry_module._bounded_json_bytes("é" * 20, max_bytes=64)
    except foundry_module._SerializedPayloadTooLarge:
        pass
    else:
        raise AssertionError("escaped JSON string must exceed the byte budget")


def test_foundry_state_byte_eviction_serializes_aggregate_once():
    tool = foundry_module.brokered_tool_definitions(_spec())[0]

    def state(response_id: str, *, completed: bool) -> Any:
        call_id = f"call_{response_id}"
        call = foundry_module._PendingCall(
            call_id=call_id,
            item_id=f"item_{response_id}",
            tool=tool,
            arguments={},
        )
        return foundry_module._HostedResponseState(
            response_id=response_id,
            session_id=None,
            pending_calls={call_id: call},
            expires_at=time.time() + 60,
            status="completed" if completed else "pending",
            accepted_output_digests={call_id: "sha256:" + "0" * 64} if completed else {},
            final_payload={"result": "x" * 200} if completed else None,
            model_messages=[{"role": "user", "content": "y" * 500}] if not completed else None,
        )

    candidate = state("candidate", completed=False)
    states = {
        "completed-1": state("completed-1", completed=True),
        "completed-2": state("completed-2", completed=True),
        "completed-3": state("completed-3", completed=True),
        candidate.response_id: candidate,
    }
    store = foundry_module._FoundryResponseStateStore(
        ttl_seconds=60,
        max_entries=10,
        max_bytes=1_000_000,
    )
    store.max_bytes = max(
        len(store._serialize({response_id: states[response_id], candidate.response_id: candidate}))
        for response_id in ("completed-1", "completed-2", "completed-3")
    )
    original_serialize = store._serialize
    aggregate_serializations = 0

    def count_aggregate(current_states: dict[str, Any]) -> bytes:
        nonlocal aggregate_serializations
        if len(current_states) > 1:
            aggregate_serializations += 1
        return original_serialize(current_states)

    store._serialize = count_aggregate  # type: ignore[method-assign]
    store._serialize_candidate(
        states,
        response_id=candidate.response_id,
        excluded_response_ids=set(),
    )

    assert aggregate_serializations <= 1
    assert candidate.response_id in states
    assert len(states) == 2


def test_foundry_state_cache_fallback_contains_capacity_failure(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app()
    with TestClient(app) as client:
        initial = _start(client)

    state = stores[0].get(initial["id"])

    def reject_cache(*_args: Any, **_kwargs: Any):
        raise foundry_module._StateStoreFull("simulated concurrent capacity pressure")

    monkeypatch.setattr(stores[0], "_serialize_candidate", reject_cache)
    assert stores[0].cache_in_memory(state) is False


def test_foundry_brokered_completed_state_is_evicted_before_rejecting_new_pending_state():
    app = _app(max_pending_responses=1)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )
        next_initial = client.post("/responses", json={"input": "another pending request"})

    assert completed.status_code == 200, completed.text
    assert next_initial.status_code == 200, next_initial.text
    assert _call(next_initial.json())


def test_foundry_brokered_arguments_honor_typeless_const_and_enum_values():
    spec = _spec(tool_name="typeless-constraints")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "probe": {"const": True},
            "mode": {"enum": ["safe"]},
        },
        "required": ["probe", "mode"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "typeless-constraints"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"probe": True, "mode": "safe"}


def test_foundry_brokered_min_properties_runs_dependent_required_closure():
    spec = _spec(tool_name="minprops-dependent")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "const": "a"},
            "b": {"type": "string", "const": "b"},
        },
        "minProperties": 1,
        "dependentRequired": {"a": ["b"]},
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "minprops-dependent"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"a": "a", "b": "b"}


def test_foundry_brokered_arguments_honor_dependent_required_and_null_type():
    spec = _spec(tool_name="dependent-null")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "region": {"type": "string", "const": "west"},
            "site": {"type": "string", "const": "sfo"},
            "marker": {"type": "null"},
        },
        "required": ["region", "marker"],
        "dependentRequired": {"region": ["site"]},
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "dependent-null"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"region": "west", "site": "sfo", "marker": None}


def test_foundry_brokered_integer_arguments_honor_float_bounds():
    spec = _spec(tool_name="bounded-integer")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1.0},
            "after": {"type": "integer", "exclusiveMinimum": 0.5},
            "below": {"type": "integer", "exclusiveMaximum": 4.5},
        },
        "required": ["count", "after", "below"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "bounded-integer"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"count": 1, "after": 1, "below": 0}


def test_foundry_brokered_synthesizes_arguments_that_honor_basic_constraints():
    spec = _spec(tool_name="bounded-lookup")
    tool = spec.brokered_tools[0]
    tool.parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1},
            "mode": {"type": "string", "enum": ["safe"]},
            "label": {"type": "string", "minLength": 5, "maxLength": 7},
        },
        "required": ["count", "mode", "label"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = _start(client, "please call bounded-lookup")

    assert json.loads(_call(body)["arguments"]) == {"count": 1, "mode": "safe", "label": "label"}


def test_foundry_brokered_rejects_schema_constraints_it_cannot_synthesize():
    spec = _spec(tool_name="pattern-lookup")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "pattern": "^[A-Z]+$"}},
        "required": ["site"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "pattern-lookup sfo"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_numeric_argument_synthesis_respects_upper_bounds():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "maximum": -1},
            "ratio": {"type": "number", "exclusiveMaximum": 0},
        },
        "required": ["count", "ratio"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"count": -1, "ratio": -1}


def test_foundry_brokered_optional_prompt_argument_is_synthesized_through_schema():
    spec = _spec(tool_name="prompt-tool")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"prompt": {"type": "string", "const": "fixed"}},
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "please call prompt-tool with arbitrary text"}).json()

    assert json.loads(_call(body)["arguments"]) == {"prompt": "fixed"}


def test_foundry_brokered_deterministic_arguments_honor_root_const_object_schema():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {"type": "object", "const": {"site": "iad"}}
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"site": "iad"}


def test_foundry_brokered_validates_deterministic_arguments_before_emitting_call():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {"type": "object", "required": ["site"], "additionalProperties": False}
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_argument_synthesis_honors_dependent_required_and_min_properties():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "site": {"type": "string"},
            "region": {"type": "string", "default": "west"},
            "extra": {"type": "boolean"},
        },
        "required": ["site"],
        "dependentRequired": {"site": ["region"]},
        "minProperties": 3,
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"site": "check-network-telemetry", "region": "west", "extra": True}


def test_foundry_brokered_integer_synthesis_rejects_multiple_of():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1, "multipleOf": 2}},
        "required": ["n"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"



def test_foundry_brokered_number_synthesis_uses_midpoint_for_fractional_exclusive_range():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}},
        "required": ["ratio"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "check-network-telemetry"}).json()

    assert json.loads(_call(body)["arguments"]) == {"ratio": 0.5}


def test_foundry_brokered_rejects_unbounded_schema_synthesis_before_allocating():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "minLength": 1_000_000_000}},
        "required": ["site"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "brokered_arguments_too_large"


def test_foundry_brokered_tool_selection_requires_token_boundary_match():
    data = _multi_tool_spec().model_dump(by_alias=True)
    data["brokeredTools"] = [
        {"name": "read", "description": "read", "brokeredClass": "read", "parameters": {"type": "object"}},
        {"name": "read_telemetry", "description": "read telemetry", "brokeredClass": "read", "parameters": {"type": "object"}},
    ]
    app = _app(AgentSpec.model_validate(data))

    with TestClient(app) as client:
        telemetry = client.post("/responses", json={"input": "please call read_telemetry"})
        unrelated = client.post("/responses", json={"input": "already done"})

    assert telemetry.status_code == 200, telemetry.text
    assert _call(telemetry.json())["name"] == "read_telemetry"
    assert unrelated.status_code == 400
    assert unrelated.json()["error"]["code"] == "brokered_tool_selection_required"


def test_foundry_brokered_rejects_schema_that_would_generate_huge_arguments_before_allocating():
    spec = _spec(tool_name="huge-args")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "minItems": 1000000000, "items": {"type": "string"}},
            "label": {"type": "string", "minLength": 1000000000},
        },
        "required": ["items", "label"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "huge-args"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "brokered_arguments_too_large"


def test_foundry_brokered_rejects_arguments_that_exceed_state_budget():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters["required"] = ["site"]
    app = _app(spec, max_brokered_argument_bytes=8)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "check-network-telemetry: a prompt that is too large for the site argument"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "brokered_arguments_too_large"


def test_foundry_brokered_single_read_tool_requires_explicit_name_except_conformance():
    app = _app(_spec(tool_name="check-network-telemetry"))

    with TestClient(app) as client:
        unrelated = client.post("/responses", json={"input": "my password should stay in chat"})
        explicit = client.post("/responses", json={"input": "please call check-network-telemetry"})

    assert unrelated.status_code == 400
    assert unrelated.json()["error"]["code"] == "brokered_tool_selection_required"
    assert explicit.status_code == 200, explicit.text
    assert _call(explicit.json())["name"] == "check-network-telemetry"


def test_foundry_brokered_single_write_tool_requires_explicit_tool_name():
    app = _app(_spec(tool_name="dispatch-work-order", brokered_class="write"))

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hello"})
        explicit = client.post("/responses", json={"input": "please call dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "brokered_tool_selection_required"
    assert explicit.status_code == 200, explicit.text
    assert _call(explicit.json())["name"] == "dispatch-work-order"


def test_foundry_brokered_selects_named_tool_when_multiple_schemas_are_configured():
    app = _app(_multi_tool_spec())

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "please call get-active-incidents"})

    assert resp.status_code == 200, resp.text
    assert _call(resp.json())["name"] == "get-active-incidents"


def test_foundry_brokered_rejects_ambiguous_multi_tool_prompt():
    app = _app(_multi_tool_spec())

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "please inspect the network"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "brokered_tool_selection_required"


def test_foundry_brokered_rejects_nonfinite_function_call_output_values():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"value":NaN}}',
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"


def test_foundry_nonfinite_validation_traverses_wide_lists_lazily():
    class ExplodingTailList(list):
        def __iter__(self):
            yield float("nan")
            raise AssertionError("non-finite validation eagerly consumed the full list")

    with pytest.raises(ValueError, match="must be finite"):
        foundry_module._reject_nonfinite_json_values(
            {"result": ExplodingTailList([0, 1])},
            path="function_call_output.output",
        )


def test_foundry_nonfinite_validation_binds_sibling_paths():
    with pytest.raises(ValueError, match=r"function_call_output\.output\.items\[1\] must be finite"):
        foundry_module._reject_nonfinite_json_values(
            {"items": [0.0, float("inf")]},
            path="function_call_output.output",
        )


def test_foundry_request_ceiling_ignores_expired_replay_sizes():
    store = foundry_module._FoundryResponseStateStore(
        ttl_seconds=60,
        max_entries=2,
        max_bytes=4 * 1024,
    )
    store._states["expired"] = foundry_module._HostedResponseState(
        response_id="expired",
        session_id=None,
        pending_calls={},
        expires_at=time.time() - 1,
        status="completed",
        accepted_output_digests={"call-1": "sha256:" + "0" * 64},
        accepted_output_sizes={"call-1": 500_000},
        final_payload={"status": "completed"},
    )

    assert store.max_accepted_output_bytes() == 0


def test_foundry_brokered_rejects_function_output_with_lone_surrogate():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {}})
        payload["input"][0]["output"] = r'{"approved":true,"output":{"value":"\ud800"}}'
        rejected = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_function_call_output"
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_rejects_oversized_function_call_output_before_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file, max_brokered_output_bytes=128)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        oversized = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial["id"],
                call["call_id"],
                {"approved": True, "output": {"blob": "x" * 256}},
            ),
        )
        persisted_after_rejection = state_file.read_bytes()
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert oversized.status_code == 413
    assert oversized.json()["error"] == {
        "message": "brokered function_call_output is too large",
        "code": "brokered_output_too_large",
    }
    assert persisted_after_rejection == persisted_before
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_replay_ceiling_is_independent_from_non_brokered_request_limit():
    app = _app(
        max_request_body_bytes=64,
        max_brokered_output_bytes=512,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        continuation = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"blob": "x" * 128}},
        )
        assert len(json.dumps(continuation).encode("utf-8")) > 64
        response = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)

    assert response.status_code == 200, response.text


def test_foundry_brokered_rejects_oversized_request_body_without_truncation_or_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(
        response_state_file=state_file,
        max_brokered_output_bytes=128,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        oversized_payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"ok": True}},
        )
        oversized_payload["ignored_padding"] = "x" * 100_000
        rejected = client.post(
            "/responses",
            headers={**CONTINUATION_AUTH, "content-type": "application/json"},
            content=json.dumps(oversized_payload, separators=(",", ":")),
        )
        persisted_after_rejection = state_file.read_bytes()
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 413
    assert rejected.json()["error"] == {
        "message": "brokered Responses request body is too large",
        "code": "brokered_request_too_large",
    }
    assert persisted_after_rejection == persisted_before
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_bounded_json_value_error_returns_400():
    app = _app(max_response_state_bytes=32 * 1024)
    oversized_integer = '{"input":' + '9' * 5_000 + '}'

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/responses",
            headers={"content-type": "application/json"},
            content=oversized_integer,
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Request body must be JSON",
        "code": "invalid_json",
    }


def test_foundry_brokered_request_limit_allows_valid_reescaped_model_output():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Large tool output accepted."}),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        max_brokered_output_bytes=64 * 1024,
        max_response_state_bytes=96 * 1024,
    )
    message = "\x7f" * 10_800
    raw_output = '{"approved":false,"error":{"message":"' + message + '"}}'
    raw_output += "\n" * (64 * 1024 - len(raw_output) - 100)
    parsed_output = json.loads(raw_output)
    canonical_output = json.dumps(parsed_output, separators=(",", ":"), sort_keys=True)
    assert len(raw_output.encode("utf-8")) < 64 * 1024
    assert len(canonical_output.encode("utf-8")) < 64 * 1024

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": False})
        payload["input"][0]["output"] = raw_output
        encoded_request = json.dumps(payload, separators=(",", ":"))
        assert len(encoded_request.encode("utf-8")) > 2 * 64 * 1024 + 16 * 1024
        final = client.post("/responses", headers=CONTINUATION_AUTH, content=encoded_request)

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Large tool output accepted."


def test_foundry_brokered_rejects_object_valued_function_call_output():
    app = _app(max_brokered_output_bytes=64)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": {"approved": True, "output": {"blob": "x" * 256}},
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"



def test_foundry_brokered_rejects_continuation_that_would_exceed_total_state_bytes(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(
        response_state_file=state_file,
        max_brokered_output_bytes=8 * 1024,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        oversized_state = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial["id"],
                call["call_id"],
                {"approved": True, "output": {"blob": "x" * 5_000}},
            ),
        )
        persisted_after_rejection = state_file.read_bytes()
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert oversized_state.status_code == 413
    assert oversized_state.json()["error"] == {
        "message": "brokered response state exceeds the configured byte limit",
        "code": "brokered_response_state_too_large",
    }
    assert persisted_after_rejection == persisted_before
    assert accepted.status_code == 200, accepted.text


def test_foundry_brokered_model_loop_counts_output_limit_in_utf8_bytes():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Bounded resume worked."}),
        ]
    )
    app = _model_loop_app(spec, fake, max_brokered_output_bytes=80)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        oversized_payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {}})
        oversized_output = json.dumps(
            {"approved": True, "output": {"value": "é" * 21}},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        assert len(oversized_output) < 80 < len(oversized_output.encode("utf-8"))
        oversized_payload["input"][0]["output"] = oversized_output
        rejected = client.post("/responses", headers=CONTINUATION_AUTH, json=oversized_payload)
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "brokered_output_too_large"
    assert accepted.status_code == 200, accepted.text
    assert _message_text(accepted.json()) == "Bounded resume worked."
    assert len(fake.requests) == 2


def test_foundry_brokered_continuation_accepts_matching_tool_output_and_completes():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        cont = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert cont.status_code == 200, cont.text
    final = cont.json()
    assert final["previous_response_id"] == initial["id"]
    assert final["id"].startswith("caresp_")
    assert final["id"] != initial["id"]
    assert _message_text(final) == 'Brokered tool conformance_read completed with output: {"success":true}'


def test_foundry_brokered_accepts_body_continuation_proof_without_header():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        response = client.post("/responses", json=payload)

    assert response.status_code == 200, response.text
    assert _message_text(response.json()) == 'Brokered tool conformance_read completed with output: {"success":true}'


def test_foundry_brokered_accepts_either_matching_proof_candidate():
    app = _app()

    with TestClient(app) as client:
        first_initial = _start(client)
        first_call = _call(first_initial)
        first_payload = _continuation(first_initial["id"], first_call["call_id"], {"approved": True, "output": {"success": True}})
        first_payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        correct_body = client.post(
            "/responses",
            headers={"x-agentkit-brokered-continuation-proof": "wrong-header"},
            json=first_payload,
        )

        second_initial = _start(client)
        second_call = _call(second_initial)
        second_payload = _continuation(second_initial["id"], second_call["call_id"], {"approved": True, "output": {"success": True}})
        second_payload[CONTINUATION_PROOF_BODY_FIELD] = "wrong-body"
        correct_header = client.post("/responses", headers=CONTINUATION_AUTH, json=second_payload)

    assert correct_body.status_code == 200, correct_body.text
    assert correct_header.status_code == 200, correct_header.text


def test_foundry_brokered_rejects_wrong_or_non_string_body_continuation_proof():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        payload[CONTINUATION_PROOF_BODY_FIELD] = "wrong-proof"
        wrong = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = 123
        non_string = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = ""
        empty = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = "wrong-💥"
        non_ascii = client.post("/responses", json=payload)
        payload[CONTINUATION_PROOF_BODY_FIELD] = "\ud800"
        unpaired_surrogate = client.post(
            "/responses",
            content=json.dumps(payload, ensure_ascii=True),
            headers={"content-type": "application/json"},
        )

    assert wrong.status_code == 403
    assert wrong.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert non_string.status_code == 403
    assert non_string.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert empty.status_code == 403
    assert empty.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert non_ascii.status_code == 403
    assert non_ascii.json()["error"]["code"] == "brokered_continuation_forbidden"
    assert unpaired_surrogate.status_code == 403
    assert unpaired_surrogate.json()["error"]["code"] == "brokered_continuation_forbidden"


def test_foundry_brokered_rejects_function_call_output_without_orka_continuation_auth():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        resp = client.post(
            "/responses",
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "brokered_continuation_forbidden"


def test_foundry_brokered_rejects_orphan_function_call_output_without_previous_response_id():
    app = _app()

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_missing",
                        "output": '{"approved":true,"output":{}}',
                    }
                ]
            },
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_previous_response_id"


def test_foundry_brokered_rejects_unknown_previous_response_id():
    app = _app()

    with TestClient(app) as client:
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation("caresp_unknown", "call_unknown", {"approved": True, "output": {}}),
        )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_previous_response_id"


def test_foundry_brokered_rejects_unknown_call_id():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], "call_other", {"approved": True, "output": {}}),
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unknown_call_id"


def test_foundry_brokered_duplicate_continuation_is_idempotent_but_conflicts_are_rejected():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        first = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        duplicate = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)
        conflicting_payload = deepcopy(payload)
        conflicting_payload["input"][0]["output"] = '{"approved":true,"output":{"success":false}}'
        conflict = client.post("/responses", json=conflicting_payload, headers=CONTINUATION_AUTH)

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "conflicting_duplicate_continuation"


def test_foundry_brokered_persists_one_output_copy_while_preserving_idempotency(tmp_path):
    state_file = tmp_path / "foundry-state.json"
    marker = "unique-brokered-output-marker"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"value": marker}},
        )
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        conflicting = deepcopy(payload)
        conflicting["input"][0]["output"] = '{"approved":true,"output":{"value":"different"}}'
        conflict = client.post("/responses", headers=CONTINUATION_AUTH, json=conflicting)

    assert completed.status_code == 200, completed.text
    assert duplicate.json() == completed.json()
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "conflicting_duplicate_continuation"
    assert state_file.read_text(encoding="utf-8").count(marker) == 1


def test_foundry_brokered_file_state_survives_restart_for_deterministic_continuation(tmp_path):
    state_file = tmp_path / "foundry-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)
        readiness = client.get("/readiness")

    assert readiness.json()["foundryResponses"]["stateBackend"] == "file"
    assert state_file.exists()

    with TestClient(_app(response_state_file=state_file)) as client:
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == 'Brokered tool conformance_read completed with output: {"success":true}'

    with TestClient(_app(response_state_file=state_file)) as client:
        duplicate = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}}),
        )

    assert duplicate.status_code == 200
    assert duplicate.json() == final.json()


def test_foundry_brokered_loads_legacy_full_output_state_for_idempotent_replay(tmp_path):
    state_file = tmp_path / "foundry-legacy-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = persisted["states"][initial["id"]]
    state.pop("acceptedOutputDigests")
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state_file.write_text(json.dumps(persisted, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    with TestClient(_app(response_state_file=state_file)) as client:
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == completed.json()


def test_foundry_brokered_replays_legacy_output_larger_than_current_limit(tmp_path):
    state_file = tmp_path / "foundry-legacy-large-output-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    large_output = {"approved": True, "output": {"blob": "x" * 80_000}}
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Persisted large output replay."}),
        ]
    )
    with TestClient(
        _model_loop_app(
            spec,
            fake,
            response_state_file=state_file,
            max_brokered_output_bytes=128 * 1024,
        )
    ) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], large_output)
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = persisted["states"][initial.json()["id"]]
    state.pop("acceptedOutputDigests")
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state_file.write_text(json.dumps(persisted, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    with TestClient(
        _app(
            spec,
            response_state_file=state_file,
            max_brokered_output_bytes=64 * 1024,
        )
    ) as client:
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == completed.json()


def test_foundry_brokered_replays_digest_output_larger_than_current_and_state_limits(tmp_path):
    state_file = tmp_path / "foundry-digest-large-output-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    message = "\x7f" * 10_800
    raw_output = '{"approved":false,"error":{"message":"' + message + '"}}'
    raw_output += "\n" * (400 * 1024 - len(raw_output) - 100)
    parsed_output = json.loads(raw_output)
    canonical_output = json.dumps(parsed_output, separators=(",", ":"), sort_keys=True)
    assert len(raw_output.encode("utf-8")) < 512 * 1024
    assert len(canonical_output.encode("utf-8")) < 128 * 1024
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Persisted digest replay."}),
        ]
    )
    with TestClient(
        _model_loop_app(
            spec,
            fake,
            response_state_file=state_file,
            max_brokered_output_bytes=512 * 1024,
            max_response_state_bytes=128 * 1024,
        )
    ) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": False})
        payload["input"][0]["output"] = raw_output
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    stored = persisted["states"][initial.json()["id"]]
    assert stored["acceptedOutputSizes"][call["call_id"]] == len(raw_output.encode("utf-8"))
    assert state_file.stat().st_size < 128 * 1024

    with TestClient(
        _app(
            spec,
            response_state_file=state_file,
            max_brokered_output_bytes=64 * 1024,
            max_response_state_bytes=128 * 1024,
        )
    ) as client:
        encoded_request = json.dumps(payload, separators=(",", ":"))
        assert len(encoded_request.encode("utf-8")) > 6 * 128 * 1024 + 16 * 1024
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, content=encoded_request)

    assert completed.status_code == 200, completed.text
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json() == completed.json()


def test_foundry_brokered_rejects_legacy_state_that_expands_past_byte_limit(tmp_path):
    state_file = tmp_path / "foundry-legacy-state.json"

    with TestClient(_app(response_state_file=state_file)) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": False, "error": {"code": "denied", "message": "denied"}},
        )
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert completed.status_code == 200, completed.text
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = persisted["states"][initial["id"]]
    state.pop("acceptedOutputDigests")
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    raw_state = json.dumps(persisted, separators=(",", ":"), sort_keys=True)
    state_file.write_text(raw_state, encoding="utf-8")
    legacy_bytes = state_file.stat().st_size

    with pytest.raises(RuntimeError, match="exceeds configured byte limit"):
        _app(response_state_file=state_file, max_response_state_bytes=legacy_bytes)

    assert state_file.read_text(encoding="utf-8") == raw_state



def test_foundry_brokered_rejects_caller_session_conflicts_with_trusted_gateway_header(monkeypatch):
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)
    app = _app()

    with TestClient(app) as client:
        body_conflict = client.post(
            "/responses",
            headers={"x-agent-session-id": "trusted-session"},
            json={"input": "please read telemetry", "agent_session_id": "caller-session"},
        )
        query_conflict = client.post(
            "/responses?session_id=caller-session",
            headers={"x-agent-session-id": "trusted-session"},
            json={"input": "please read telemetry"},
        )

    for response in (body_conflict, query_conflict):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "response_session_mismatch"


def test_foundry_brokered_rejects_session_conflicts_with_hosted_environment(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "hosted-session")
    app = _app()

    with TestClient(app) as client:
        gateway_conflict = client.post(
            "/responses",
            headers={"x-agent-session-id": "other-session"},
            json={"input": "please read telemetry"},
        )
        caller_conflict = client.post(
            "/responses",
            json={"input": "please read telemetry", "agent_session_id": "other-session"},
        )
        matching = client.post(
            "/responses?session_id=hosted-session",
            headers={"x-agent-session-id": "hosted-session"},
            json={"input": "please read telemetry", "agent_session_id": "hosted-session"},
        )

    for response in (gateway_conflict, caller_conflict):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "response_session_mismatch"
    assert matching.status_code == 200, matching.text


def test_foundry_brokered_trusted_gateway_session_accepts_matching_local_compatibility(monkeypatch, tmp_path):
    state_file = tmp_path / "foundry-trusted-session-state.json"
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)

    with TestClient(_app(response_state_file=state_file)) as client:
        response = client.post(
            "/responses?session_id=gateway-session",
            headers={
                "x-agent-session-id": "gateway-session",
                "x-agentkit-session-id": "gateway-session",
            },
            json={"input": "please read telemetry", "agent_session_id": "gateway-session"},
        )

    assert response.status_code == 200, response.text
    stored = json.loads(state_file.read_text(encoding="utf-8"))["states"][response.json()["id"]]
    assert stored["sessionID"] == "gateway-session"


def test_foundry_brokered_preserves_local_session_fallback_precedence_without_trusted_identity(monkeypatch, tmp_path):
    state_file = tmp_path / "foundry-local-session-state.json"
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)

    with TestClient(_app(response_state_file=state_file)) as client:
        response = client.post(
            "/responses?agent_session_id=query-session",
            headers={"x-agentkit-session-id": "legacy-header-session"},
            json={
                "input": "please read telemetry",
                "agent_session_id": "body-session",
                "session_id": "legacy-body-session",
            },
        )

    assert response.status_code == 200, response.text
    stored = json.loads(state_file.read_text(encoding="utf-8"))["states"][response.json()["id"]]
    assert stored["sessionID"] == "body-session"


def test_foundry_brokered_file_state_tracks_session_and_rejects_cross_session_continuation(monkeypatch, tmp_path):
    state_file = tmp_path / "foundry-session-state.json"
    monkeypatch.delenv("FOUNDRY_AGENT_SESSION_ID", raising=False)

    with TestClient(_app(response_state_file=state_file)) as client:
        initial_response = client.post(
            "/responses",
            json={"input": "please read telemetry", "agent_session_id": "session-a"},
        )
        assert initial_response.status_code == 200, initial_response.text
        initial = initial_response.json()
        call = _call(initial)

    stored = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial["id"]]
    assert stored["sessionID"] == "session-a"

    payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
    payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
    with TestClient(_app(response_state_file=state_file)) as client:
        missing = client.post("/responses", json=payload)

    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "response_session_mismatch"

    payload["agent_session_id"] = "session-b"
    with TestClient(_app(response_state_file=state_file)) as client:
        mismatch = client.post("/responses", json=payload)

    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "response_session_mismatch"

    payload["agent_session_id"] = "session-a"
    with TestClient(_app(response_state_file=state_file)) as client:
        completed = client.post("/responses", json=payload)

    assert completed.status_code == 200, completed.text


def test_foundry_brokered_body_proof_is_not_persisted_or_echoed(tmp_path):
    state_file = tmp_path / "foundry-proof-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial_response = client.post(
            "/responses",
            json={"input": "please read telemetry", CONTINUATION_PROOF_BODY_FIELD: CONTINUATION_PROOF},
        )
        assert initial_response.status_code == 200, initial_response.text
        initial = initial_response.json()
        call = _call(initial)
        payload = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"success": True}})
        payload[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        final_response = client.post("/responses", json=payload)

    assert final_response.status_code == 200, final_response.text
    assert CONTINUATION_PROOF not in json.dumps(initial, sort_keys=True)
    assert CONTINUATION_PROOF not in json.dumps(final_response.json(), sort_keys=True)
    assert CONTINUATION_PROOF not in state_file.read_text(encoding="utf-8")


def test_foundry_brokered_file_state_survives_restart_for_model_loop_continuation(tmp_path):
    state_file = tmp_path / "foundry-model-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters["required"] = ["site"]
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
                        }
                    ],
                }
            )
        ]
    )

    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "call check-network-telemetry"}).json()
        call = _call(initial)

    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Restart resume worked."})])
    with TestClient(_model_loop_app(spec, second_fake, response_state_file=state_file)) as client:
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"status": "ok"}}),
        )

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Restart resume worked."
    assert second_fake.requests[0]["input"][-1]["call_id"] == call["call_id"]


def test_foundry_brokered_model_loop_continuation_requires_model_loop_after_restart(tmp_path):
    state_file = tmp_path / "foundry-model-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )

    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"}).json()
        call = _call(initial)

    before = state_file.read_bytes()
    with TestClient(_app(spec, response_state_file=state_file, brokered_model_loop_enabled=False)) as client:
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"status": "ok"}}),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "brokered_model_loop_unavailable"
    assert state_file.read_bytes() == before



def test_foundry_brokered_rejects_expired_response_state():
    app = _app(state_ttl_seconds=0)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        time.sleep(0.01)
        resp = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {}}),
        )

    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "response_state_expired"


def test_foundry_brokered_refuses_to_synthesize_nonliteral_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"incident": {"type": "string"}},
        "required": ["incident"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_refuses_multi_value_enum_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"operation": {"type": "string", "enum": ["delete", "create"]}},
        "required": ["operation"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_allows_single_value_enum_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"operation": {"type": "string", "enum": ["create"]}},
        "required": ["operation"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 200
    assert json.loads(_call(resp.json())["arguments"]) == {"operation": "create"}


def test_foundry_brokered_allows_literal_write_arguments_for_conformance():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"incident": {"type": "string", "const": "INC-1"}},
        "required": ["incident"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        body = client.post("/responses", json={"input": "dispatch-work-order"})

    assert body.status_code == 200, body.text
    assert json.loads(_call(body.json())["arguments"]) == {"incident": "INC-1"}


def test_foundry_brokered_decline_policy_rejection_and_execution_error_are_truthful_final_answers():
    cases = [
        (
            {"approved": False, "error": {"code": "approval_declined", "message": "Human declined dispatch-work-order"}},
            "Brokered tool dispatch-work-order: approval_declined: The tool call was declined.",
        ),
        (
            {"approved": False, "error": {"code": "tool_policy_rejected", "message": "writes are disabled"}},
            "Brokered tool dispatch-work-order was not performed: tool_policy_rejected: writes are disabled",
        ),
        (
            {"approved": False, "error": {"code": "tool_execution_failed", "message": "downstream timed out"}},
            "Brokered tool dispatch-work-order: tool_execution_failed: MCP tool execution failed.",
        ),
    ]

    for payload, expected in cases:
        app = _app(_spec(tool_name="dispatch-work-order", brokered_class="write"))
        with TestClient(app) as client:
            initial = _start(client, "please call dispatch-work-order")
            call = _call(initial)
            resp = client.post("/responses", json=_continuation(initial["id"], call["call_id"], payload), headers=CONTINUATION_AUTH)
        assert resp.status_code == 200, resp.text
        assert _message_text(resp.json()) == expected


def test_foundry_brokered_rejects_multiple_tool_outputs_deterministically():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        request = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {}})
        request["input"].append(dict(request["input"][0]))
        resp = client.post("/responses", json=request, headers=CONTINUATION_AUTH)

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "multiple_tool_outputs_unsupported"



def test_foundry_brokered_disables_invocations_direct_runtime_bypass():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/invocations", json={"message": "bypass"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invocations_disabled_in_brokered_mode"

class _FakeChatTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.requests.append(json.loads(request.content.decode("utf-8")))
        if not self.responses:
            return httpx.Response(500, json={"error": "unexpected extra model call"})
        body = json.dumps(self.responses.pop(0), separators=(",", ":")).encode("utf-8")
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})


def _chat_response(message: dict[str, Any], *, prompt_tokens: int = 1, completion_tokens: int = 1) -> dict[str, Any]:
    """Build the Responses API payload a model returns for a chat-shaped assistant message."""
    output: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str):
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content, "annotations": []}]})
    elif content is not None:
        # Keep malformed content malformed so response validation still sees it.
        output.append({"type": "message", "role": "assistant", "content": content})
    if "refusal" in message:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": message["refusal"]}]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "call_id": call.get("id"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return {
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _model_loop_app(spec: AgentSpec, fake: _FakeChatTransport, **kwargs: Any):
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return _app(spec, brokered_model_loop_enabled=True, brokered_model_http_client=client, **kwargs)


def test_foundry_brokered_model_loop_bounds_streamed_upstream_body_before_json_materialization():
    class TrackingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.chunks_read = 0
            self.tail_read = False

        async def __aiter__(self):
            chunks = [
                b'{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"',
                b"x" * 300,
                b'"}]}],"usage":{}}',
            ]
            for index, chunk in enumerate(chunks):
                self.chunks_read += 1
                if index == 2:
                    self.tail_read = True
                yield chunk

        async def aclose(self) -> None:
            return None

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.stream = TrackingStream()
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(200, request=request, stream=self.stream)
            return httpx.Response(
                200,
                request=request,
                json=_chat_response({"role": "assistant", "content": "Retry stayed available."}),
            )

    transport = Transport()
    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=transport),
        max_pending_responses=1,
        max_response_state_bytes=256,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        oversized = client.post("/responses", json={"input": "Say hello"})
        retry = client.post("/responses", json={"input": "Say hello again"})

    assert oversized.status_code == 502
    assert oversized.json()["error"] == {
        "message": "model response is too large to retain safely",
        "code": "ModelResponseTooLarge",
    }
    assert transport.stream.tail_read is False
    assert retry.status_code == 200, retry.text
    assert _message_text(retry.json()) == "Retry stayed available."


def test_foundry_brokered_max_pending_reserves_capacity_before_initial_model_work():
    spec = _spec(tool_name="check-network-telemetry")

    class BlockingInitialTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content.decode("utf-8")))
            if len(self.requests) == 1:
                self.first_started.set()
                while not self.release_first.is_set():
                    await asyncio.sleep(0.001)
            return httpx.Response(
                200,
                request=request,
                json=_chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"model_call_{len(self.requests)}",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": "{}"},
                            }
                        ],
                    }
                ),
            )

    transport = BlockingInitialTransport()
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=transport),
        max_pending_responses=1,
    )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(client.post, "/responses", json={"input": "check-network-telemetry"})
        try:
            assert transport.first_started.wait(timeout=2)
            excess = client.post("/responses", json={"input": "check-network-telemetry again"})
        finally:
            transport.release_first.set()
        first = first_future.result(timeout=2)

    assert first.status_code == 200, first.text
    assert excess.status_code == 429
    assert excess.json()["error"]["code"] == "brokered_response_state_full"
    assert len(transport.requests) == 1


def test_foundry_brokered_initial_model_reservation_releases_for_final_and_error_paths():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response({"role": "assistant", "content": "No tool needed."}),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "unknown_call",
                            "type": "function",
                            "function": {"name": "unknown-tool", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "valid_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
        ]
    )
    app = _model_loop_app(spec, fake, max_pending_responses=1)

    with TestClient(app) as client:
        final = client.post("/responses", json={"input": "say hello"})
        invalid = client.post("/responses", json={"input": "request an unknown tool"})
        pending = client.post("/responses", json={"input": "check-network-telemetry"})

    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "No tool needed."
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "unknown_brokered_tool"
    assert pending.status_code == 200, pending.text
    assert _call(pending.json())["name"] == "check-network-telemetry"
    assert len(fake.requests) == 3


def test_foundry_brokered_final_model_result_does_not_evict_completed_replay_state():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "initial_tool_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Stored completion."}),
            _chat_response({"role": "assistant", "content": "No pending state needed."}),
        ]
    )
    app = _model_loop_app(spec, fake, max_pending_responses=1)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        continuation = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        final_only = client.post("/responses", json={"input": "say hello without a tool"})
        replay = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)

    assert completed.status_code == 200, completed.text
    assert final_only.status_code == 200, final_only.text
    assert _message_text(final_only.json()) == "No pending state needed."
    assert replay.status_code == 200, replay.text
    assert replay.json() == completed.json()


def test_foundry_brokered_initial_model_reservation_releases_after_state_persist_failure(monkeypatch, tmp_path):
    state_file = tmp_path / "model-loop-responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "first_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "retry_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_pending_responses=1,
    )
    original_replace = Path.replace
    failed_once = False

    def fail_first_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            failed_once = True
            raise OSError("simulated model-loop state storage failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_first_replace)
    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.post("/responses", json={"input": "check-network-telemetry"})
        retried = client.post("/responses", json={"input": "check-network-telemetry"})

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "brokered_response_state_storage_error"
    assert retried.status_code == 200, retried.text
    assert _call(retried.json())["name"] == "check-network-telemetry"
    assert len(fake.requests) == 2


def test_foundry_brokered_cancelled_initial_model_work_releases_reserved_capacity():
    spec = _spec(tool_name="check-network-telemetry")

    class CancellingInitialTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError
            return httpx.Response(
                200,
                request=request,
                json=_chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "model_call_after_cancellation",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": "{}"},
                            }
                        ],
                    }
                ),
            )

    transport = CancellingInitialTransport()
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=transport),
        max_pending_responses=1,
    )

    async def exercise() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            try:
                await client.post("/responses", json={"input": "cancel this model call"})
            except asyncio.CancelledError:
                pass
            return await client.post("/responses", json={"input": "check-network-telemetry"})

    response = asyncio.run(exercise())

    assert response.status_code == 200, response.text
    assert _call(response.json())["name"] == "check-network-telemetry"
    assert transport.calls == 2


def test_foundry_brokered_active_resume_survives_ttl_and_completed_state_retains_from_completion():
    spec = _spec(tool_name="check-network-telemetry")

    class BlockingResumeTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.resume_started = threading.Event()
            self.release_resume = threading.Event()
            self.initial_calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            self.requests.append(payload)
            if not any(item["type"] == "function_call_output" for item in payload["input"]):
                self.initial_calls += 1
                return httpx.Response(
                    200,
                    request=request,
                    json=_chat_response(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"model_call_{self.initial_calls}",
                                    "type": "function",
                                    "function": {"name": "check-network-telemetry", "arguments": "{}"},
                                }
                            ],
                        }
                    ),
                )
            self.resume_started.set()
            while not self.release_resume.is_set():
                await asyncio.sleep(0.001)
            return httpx.Response(
                200,
                request=request,
                json=_chat_response({"role": "assistant", "content": "Resume completed after the lease window."}),
            )

    transport = BlockingResumeTransport()
    model_client = httpx.AsyncClient(transport=transport)
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
        state_ttl_seconds=0.05,
    )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        future = executor.submit(client.post, "/responses", headers=CONTINUATION_AUTH, json=payload)
        try:
            assert transport.resume_started.wait(timeout=2)
            time.sleep(0.08)
            another_pending = client.post("/responses", json={"input": "check-network-telemetry again"})
            active_duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        finally:
            transport.release_resume.set()
        completed = future.result(timeout=2)
        immediate_duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert another_pending.status_code == 200, another_pending.text
    assert active_duplicate.status_code == 409
    assert active_duplicate.json()["error"]["code"] == "duplicate_continuation_in_progress"
    assert completed.status_code == 200, completed.text
    assert _message_text(completed.json()) == "Resume completed after the lease window."
    assert immediate_duplicate.status_code == 200, immediate_duplicate.text
    assert immediate_duplicate.json() == completed.json()


def test_foundry_brokered_abandoned_resuming_state_expires_and_releases_capacity(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app(max_pending_responses=1)

    with TestClient(app) as client:
        initial = _start(client)
        abandoned = stores[0].get(initial["id"])
        abandoned.status = "resuming"
        abandoned.expires_at = time.time() - 1
        stores[0].save(abandoned)
        replacement = client.post("/responses", json={"input": "please read telemetry again"})

    assert replacement.status_code == 200, replacement.text
    assert _call(replacement.json())


def test_foundry_brokered_model_loop_emits_model_requested_tool_and_resumes_to_final_answer():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters["required"] = ["site"]
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
                        }
                    ],
                },
                prompt_tokens=2,
                completion_tokens=3,
            ),
            _chat_response({"role": "assistant", "content": "Telemetry is healthy."}, prompt_tokens=5, completion_tokens=7),
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "Check SFO telemetry"})
        call = _call(initial.json())
        continuation = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"status": "healthy"}})
        continuation[CONTINUATION_PROOF_BODY_FIELD] = CONTINUATION_PROOF
        final = client.post(
            "/responses",
            json=continuation,
        )

    assert initial.status_code == 200, initial.text
    assert call["name"] == "check-network-telemetry"
    assert call["call_id"] == f"call_{initial.json()['id']}_1"
    assert json.loads(call["arguments"]) == {"site": "sfo"}
    assert initial.json()["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Telemetry is healthy."
    assert final.json()["usage"] == {"input_tokens": 7, "output_tokens": 10, "total_tokens": 17}
    assert fake.urls == ["https://api.openai.com/v1/responses"] * 2
    tool = fake.requests[0]["tools"][0]
    assert tool["type"] == "function"
    assert tool["name"] == "check-network-telemetry"
    assert tool["description"].startswith("Brokered class: read.")
    assert tool["strict"] is False
    assert fake.requests[0]["input"] == [
        {"type": "message", "role": "system", "content": "Be helpful."},
        {"type": "message", "role": "user", "content": "Check SFO telemetry"},
    ]
    assert fake.requests[1]["input"] == [
        *fake.requests[0]["input"],
        {"type": "function_call", "call_id": call["call_id"], "name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
        {"type": "function_call_output", "call_id": call["call_id"], "output": '{"approved":true,"output":{"status":"healthy"}}'},
    ]
    assert fake.requests[1]["tools"] == fake.requests[0]["tools"]
    assert all(request["store"] is False for request in fake.requests)
    assert fake.requests[1]["parallel_tool_calls"] is False
    assert CONTINUATION_PROOF not in json.dumps(fake.requests, sort_keys=True)


def test_foundry_brokered_model_loop_missing_api_key_is_not_ready_or_caller_error(monkeypatch):
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    spec_data["model"]["apiKeyEnv"] = "MISSING_FOUNDRY_MODEL_KEY"
    spec = AgentSpec.model_validate(spec_data)
    monkeypatch.delenv("MISSING_FOUNDRY_MODEL_KEY", raising=False)
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        readiness = client.get("/readiness")
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False
    assert readiness.json()["foundryResponses"]["modelAuth"] == "missing"
    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "model authentication is not configured",
        "code": "ModelAuthMissing",
    }
    assert "MISSING_FOUNDRY_MODEL_KEY" not in readiness.text + response.text
    assert fake.requests == []


def test_foundry_brokered_model_loop_missing_key_precedes_capacity_errors(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    spec_data["model"]["apiKeyEnv"] = "FOUNDRY_MODEL_KEY"
    spec = AgentSpec.model_validate(spec_data)
    monkeypatch.setenv("FOUNDRY_MODEL_KEY", "mock-token")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(
        _model_loop_app(
            spec,
            first_fake,
            response_state_file=state_file,
            max_pending_responses=1,
        )
    ) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
    assert initial.status_code == 200, initial.text

    monkeypatch.delenv("FOUNDRY_MODEL_KEY")
    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    with TestClient(
        _model_loop_app(
            spec,
            second_fake,
            response_state_file=state_file,
            max_pending_responses=1,
        )
    ) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry again"})

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "model authentication is not configured",
        "code": "ModelAuthMissing",
    }
    assert second_fake.requests == []


def test_foundry_brokered_model_loop_missing_key_precedes_continuation_state_errors(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    spec_data["model"]["apiKeyEnv"] = "FOUNDRY_MODEL_KEY"
    spec = AgentSpec.model_validate(spec_data)
    monkeypatch.setenv("FOUNDRY_MODEL_KEY", "mock-token")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
    persisted_before = state_file.read_bytes()

    monkeypatch.delenv("FOUNDRY_MODEL_KEY")
    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    with TestClient(_model_loop_app(spec, second_fake, response_state_file=state_file)) as client:
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "model authentication is not configured",
        "code": "ModelAuthMissing",
    }
    assert state_file.read_bytes() == persisted_before
    assert second_fake.requests == []


def test_foundry_brokered_model_loop_sanitizes_upstream_auth_failures():
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    internal_url = "https://private-model.internal.example/v1"
    spec_data["model"]["baseURL"] = internal_url
    spec = AgentSpec.model_validate(spec_data)

    for upstream_status in (401, 403):
        def reject(request: httpx.Request, *, status: int = upstream_status) -> httpx.Response:
            return httpx.Response(status, request=request, json={"error": "credential rejected"})

        model_client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        app = _app(
            spec,
            brokered_model_loop_enabled=True,
            brokered_model_http_client=model_client,
        )
        with TestClient(app) as client:
            response = client.post("/responses", json={"input": "check-network-telemetry"})

        assert response.status_code == 503
        assert response.json()["error"] == {
            "message": "model service rejected configured credentials",
            "code": "ModelAuthRejected",
            "upstream_status": upstream_status,
        }
        assert internal_url not in response.text
        assert "/chat/completions" not in response.text


def test_foundry_brokered_model_loop_sanitizes_transport_failure_urls():
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    internal_url = "https://private-model.internal.example/v1"
    spec_data["model"]["baseURL"] = internal_url
    spec = AgentSpec.model_validate(spec_data)

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {internal_url}", request=request)

    model_client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
    )

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service request failed",
        "code": "ModelUpstreamError",
    }
    assert internal_url not in response.text
    assert "/chat/completions" not in response.text


def test_foundry_brokered_model_loop_normalizes_non_object_json_response():
    non_object_json = b"[0]"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=non_object_json,
            headers={"content-type": "application/json"},
        )

    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service returned an invalid response",
        "code": "InvalidModelResponse",
    }


def test_foundry_model_loop_rejects_oversized_output_before_encoding():
    class ExplodingEncodeString(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("oversized output must be rejected before encoding")

    loop = BrokeredChatModelLoop(
        _spec(tool_name="check-network-telemetry"),
        [],
        max_output_bytes=128,
    )

    with pytest.raises(AgentRunError) as exc_info:
        asyncio.run(
            loop.resume(
                [{"role": "assistant", "content": None}],
                call_id="call-1",
                output=ExplodingEncodeString("x" * 129),
            )
        )

    assert exc_info.value.status == 413
    assert exc_info.value.code == "brokered_output_too_large"


def test_foundry_model_loop_rejects_lone_surrogate_tool_output_before_encoding():
    loop = BrokeredChatModelLoop(
        _spec(tool_name="check-network-telemetry"),
        [],
    )

    with pytest.raises(AgentRunError) as exc_info:
        asyncio.run(
            loop.resume(
                [{"role": "assistant", "content": None}],
                call_id="call-1",
                output='{"value":"\ud800"}',
            )
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code == "InvalidToolOutput"
    assert "valid Unicode" in str(exc_info.value)


def test_foundry_brokered_model_loop_rejects_lone_surrogate_in_decoded_tool_arguments():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json=_chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"value":"\\ud800"}',
                            },
                        }
                    ],
                }
            ),
        )

    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"



def test_foundry_brokered_model_loop_rejects_decoded_lone_surrogate():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=(
                b'{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"\\ud800"}]}],'
                b'"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}'
            ),
            headers={"content-type": "application/json"},
        )

    model_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    app = _app(
        _spec(tool_name="check-network-telemetry"),
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model service returned an invalid response",
        "code": "InvalidModelResponse",
    }


def test_foundry_brokered_model_loop_sanitizes_upstream_auth_failure_on_resume():
    spec_data = _spec(tool_name="check-network-telemetry").model_dump(by_alias=True)
    internal_url = "https://private-model.internal.example/v1"
    spec_data["model"]["baseURL"] = internal_url
    spec = AgentSpec.model_validate(spec_data)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                request=request,
                json=_chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "model_generated_call_id",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": "{}"},
                            }
                        ],
                    }
                ),
            )
        if calls == 2:
            return httpx.Response(401, request=request, json={"error": "credential rejected"})
        return httpx.Response(
            200,
            request=request,
            json=_chat_response({"role": "assistant", "content": "Retry after auth recovery."}),
        )

    model_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    app = _app(
        spec,
        brokered_model_loop_enabled=True,
        brokered_model_http_client=model_client,
    )

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "message": "model service rejected configured credentials",
        "code": "ModelAuthRejected",
        "upstream_status": 401,
    }
    assert internal_url not in failed.text
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry after auth recovery."


def test_foundry_brokered_malformed_file_state_fails_startup_without_overwriting(tmp_path):
    state_file = tmp_path / "responses-state.json"
    raw_state = "{not valid json"
    state_file.write_text(raw_state, encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid Foundry response state file"):
        _app(response_state_file=state_file)

    assert state_file.read_text(encoding="utf-8") == raw_state



def test_foundry_brokered_state_file_larger_than_limit_fails_startup_without_overwriting(tmp_path):
    state_file = tmp_path / "responses-state.json"
    raw_state = json.dumps({"states": {}, "padding": "x" * 8192})
    state_file.write_text(raw_state, encoding="utf-8")

    with pytest.raises(RuntimeError, match="exceeds configured byte limit"):
        _app(response_state_file=state_file, max_response_state_bytes=4 * 1024)

    assert state_file.read_text(encoding="utf-8") == raw_state



def test_foundry_brokered_file_state_is_written_with_private_permissions(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        _start(client)

    assert state_file.exists()
    assert state_file.stat().st_mode & 0o777 == 0o600


def test_foundry_brokered_failed_continuation_state_persist_is_retryable_without_stranding_progress(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Recovered after storage retry."}),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file)
    original_replace = Path.replace
    failed_once = False

    def fail_first_resuming_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            stored_state = next(iter(persisted["states"].values()))
            if stored_state["status"] == "resuming":
                failed_once = True
                raise OSError("simulated continuation state storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_first_resuming_replace)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "message": "brokered response state storage unavailable",
        "code": "brokered_response_state_storage_error",
    }
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Recovered after storage retry."
    assert len(fake.requests) == 2


def test_foundry_brokered_failed_final_state_persist_retries_cached_completion_without_second_resume(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Persist this exact completion."}),
            _chat_response({"role": "assistant", "content": "A duplicate resume incorrectly ran."}),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file)
    original_replace = Path.replace
    failed_once = False

    def fail_first_completed_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            stored_state = next(iter(persisted["states"].values()))
            if stored_state["status"] == "completed":
                failed_once = True
                raise OSError("simulated completed state storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_first_completed_replace)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "brokered_response_state_storage_error"
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Persist this exact completion."
    assert len(fake.requests) == 2
    persisted_state = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
    assert persisted_state["status"] == "completed"
    assert persisted_state["finalPayload"] == retried.json()


def test_foundry_brokered_unrelated_full_map_persist_marks_cached_completion_durable(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")

    def tool_request(call_id: str) -> dict[str, Any]:
        return _chat_response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "check-network-telemetry", "arguments": "{}"},
                    }
                ],
            }
        )

    fake = _FakeChatTransport(
        [
            tool_request("model_generated_call_id"),
            _chat_response({"role": "assistant", "content": "Persisted by an unrelated transaction."}),
            tool_request("unrelated_call_1"),
            tool_request("unrelated_call_2"),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file, max_pending_responses=2)
    original_replace = Path.replace
    first_completed_write_failed = False
    fail_next_write = False
    second_failure_triggered = False

    def fail_selected_replaces(path: Path, target: Path) -> Path:
        nonlocal first_completed_write_failed, second_failure_triggered
        if path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if not first_completed_write_failed and any(state["status"] == "completed" for state in persisted["states"].values()):
                first_completed_write_failed = True
                raise OSError("simulated completed state storage failure")
            if fail_next_write:
                second_failure_triggered = True
                raise OSError("simulated later storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_selected_replaces)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        unrelated = client.post("/responses", json={"input": "check-network-telemetry unrelated"})
        durable_state = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
        fail_next_write = True
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        fail_next_write = False
        replacement = client.post("/responses", json={"input": "check-network-telemetry replacement"})

    assert failed.status_code == 503
    assert unrelated.status_code == 200, unrelated.text
    assert durable_state["status"] == "completed"
    assert durable_state["finalPayload"] is not None
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Persisted by an unrelated transaction."
    assert second_failure_triggered is False
    assert replacement.status_code == 200, replacement.text
    assert _call(replacement.json())
    assert len(fake.requests) == 4


def test_foundry_brokered_recovered_storage_persists_and_evicts_cached_completion_at_capacity(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "Completion cached while storage failed."}),
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "replacement_model_call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file, max_pending_responses=1)
    original_replace = Path.replace
    failed_once = False

    def fail_first_completed_replace(path: Path, target: Path) -> Path:
        nonlocal failed_once
        if not failed_once and path.name == f".{state_file.name}.tmp":
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if any(state["status"] == "completed" for state in persisted["states"].values()):
                failed_once = True
                raise OSError("simulated completed state storage failure")
        return original_replace(path, target)

    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        monkeypatch.setattr(Path, "replace", fail_first_completed_replace)
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        replacement = client.post("/responses", json={"input": "check-network-telemetry replacement"})

    assert failed.status_code == 503
    assert replacement.status_code == 200, replacement.text
    replacement_body = replacement.json()
    assert _call(replacement_body)
    persisted_states = json.loads(state_file.read_text(encoding="utf-8"))["states"]
    assert initial.json()["id"] not in persisted_states
    assert replacement_body["id"] in persisted_states
    assert len(fake.requests) == 3


def test_foundry_brokered_file_state_recovers_unfinalized_accepted_continuation_after_restart(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake_first = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{}'},
                        }
                    ],
                }
            )
        ]
    )
    first_app = _app(spec, brokered_model_loop_enabled=True, brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_first.handler)), response_state_file=state_file)

    with TestClient(first_app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())

    payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    state = state_data["states"][initial.json()["id"]]
    state["acceptedOutputs"] = {call["call_id"]: payload["input"][0]["output"]}
    state["status"] = "resuming"
    state["finalPayload"] = None
    state_file.write_text(json.dumps(state_data, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    fake_second = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Recovered."})])
    second_app = _app(spec, brokered_model_loop_enabled=True, brokered_model_http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_second.handler)), response_state_file=state_file)
    with TestClient(second_app) as client:
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Recovered."


def test_foundry_brokered_model_loop_accepts_integer_arguments_encoded_as_integral_float():
    spec = _spec(tool_name="retry-tool")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"retries": {"type": "integer"}},
        "required": ["retries"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "retry-tool", "arguments": '{"retries":1.0}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call retry-tool"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {"retries": 1.0}


def test_foundry_brokered_model_loop_unexpected_resume_failure_can_be_retried():
    spec = _spec(tool_name="check-network-telemetry")

    class FlakyResumeTransport:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.resume_attempts = 0

        def handler(self, request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            self.requests.append(payload)
            if len(self.requests) == 1:
                return httpx.Response(
                    200,
                    json=_chat_response(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "model_generated_call_id",
                                    "type": "function",
                                    "function": {"name": "check-network-telemetry", "arguments": '{}'},
                                }
                            ],
                        }
                    ),
                )
            self.resume_attempts += 1
            if self.resume_attempts == 1:
                raise RuntimeError("transient resume failure")
            return httpx.Response(200, json=_chat_response({"role": "assistant", "content": "Recovered after retry."}))

    fake = FlakyResumeTransport()
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert failed.json()["error"] == {"message": "model service request failed", "code": "ModelUpstreamError"}
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Recovered after retry."


def test_foundry_brokered_model_loop_failed_resume_can_be_retried():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{}'},
                        }
                    ],
                }
            ),
            {"output": None},
            _chat_response({"role": "assistant", "content": "Retry worked."}),
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry worked."


def test_foundry_brokered_model_loop_bounds_resumed_final_text_without_installing_completion(tmp_path):
    state_file = tmp_path / "responses-state.json"
    oversized_text = "é" * 80
    assert len(oversized_text) < 128 < len(oversized_text.encode("utf-8"))
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": oversized_text}),
            _chat_response({"role": "assistant", "content": "Retry stayed bounded."}),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_brokered_output_bytes=128,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        state_after_failure = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert failed.json()["error"] == {
        "message": "model response is too large to retain safely",
        "code": "ModelResponseTooLarge",
    }
    assert state_after_failure["status"] == "pending"
    assert state_after_failure["finalPayload"] is None
    assert state_after_failure["acceptedOutputDigests"] == {}
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry stayed bounded."
    assert len(fake.requests) == 3


def test_foundry_brokered_model_loop_oversized_completion_can_be_retried():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "x" * 5_000}),
            _chat_response({"role": "assistant", "content": "Retry stayed bounded."}),
        ]
    )
    app = _model_loop_app(spec, fake, max_response_state_bytes=4 * 1024)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        payload = _continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        retried = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert failed.status_code == 502
    assert failed.json()["error"] == {
        "message": "model response is too large to retain safely",
        "code": "ModelResponseTooLarge",
    }
    assert retried.status_code == 200, retried.text
    assert _message_text(retried.json()) == "Retry stayed bounded."
    assert len(fake.requests) == 3


def test_foundry_brokered_model_loop_discards_resume_transcript_before_final_state_sizing(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    first_fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model-generated-call",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(_model_loop_app(spec, first_fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry " + "x" * 2_000})
        call = _call(initial.json())
    pending_state_bytes = state_file.stat().st_size

    second_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "Small final answer."})])
    with TestClient(
        _model_loop_app(
            spec,
            second_fake,
            response_state_file=state_file,
            max_response_state_bytes=pending_state_bytes + 256,
        )
    ) as client:
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert completed.status_code == 200, completed.text
    assert _message_text(completed.json()) == "Small final answer."
    persisted = json.loads(state_file.read_text(encoding="utf-8"))["states"][initial.json()["id"]]
    assert persisted["modelMessages"] is None
    assert persisted["initialUsage"] == {}


def test_foundry_brokered_model_loop_aggregate_state_pressure_is_terminal_for_duplicate(tmp_path):
    state_file = tmp_path / "responses-state.json"
    # Keep the same pressure point after adding the API marker to both pending states.
    state_budget = 2_500 + 2 * len(',"modelAPI":"responses"')
    spec = _spec(tool_name="check-network-telemetry")

    def tool_request(call_id: str) -> dict[str, Any]:
        return _chat_response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "check-network-telemetry", "arguments": "{}"},
                    }
                ],
            }
        )

    fake = _FakeChatTransport(
        [
            tool_request("first-model-call"),
            tool_request("second-model-call"),
            _chat_response({"role": "assistant", "content": "Computed once under capacity pressure."}),
            _chat_response({"role": "assistant", "content": "must not be called for duplicate"}),
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_pending_responses=3,
        max_response_state_bytes=state_budget,
    )

    with TestClient(app) as client:
        first = client.post("/responses", json={"input": "check-network-telemetry first"})
        first_call = _call(first.json())
        second = client.post("/responses", json={"input": "check-network-telemetry second"})
        _call(second.json())
        payload = _continuation(first.json()["id"], first_call["call_id"], {"approved": True, "output": {"ok": True}})
        failed = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)
        duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    restart_fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run after restart"})])
    with TestClient(
        _model_loop_app(
            spec,
            restart_fake,
            response_state_file=state_file,
            max_pending_responses=3,
            max_response_state_bytes=state_budget,
        )
    ) as client:
        restarted_duplicate = client.post("/responses", headers=CONTINUATION_AUTH, json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert failed.status_code == 429
    assert failed.json()["error"]["code"] == "brokered_response_state_full"
    assert duplicate.status_code == 429
    assert duplicate.json() == failed.json()
    assert len(fake.requests) == 3
    assert restarted_duplicate.status_code == 429
    assert restarted_duplicate.json() == failed.json()
    assert restart_fake.requests == []


def test_foundry_brokered_bounds_initial_model_state_before_copying(monkeypatch):
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": "x" * 5_000,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake, max_response_state_bytes=4 * 1024)
    original_deepcopy = foundry_module.deepcopy

    def reject_oversized_state_copy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        if type(value).__name__ == "_HostedResponseState" and value.model_messages:
            if any(len(message.get("content") or "") > 4 * 1024 for message in value.model_messages):
                raise AssertionError("oversized state must be rejected before deepcopy")
        return original_deepcopy(value, memo) if memo is not None else original_deepcopy(value)

    monkeypatch.setattr(foundry_module, "deepcopy", reject_oversized_state_copy)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "model response is too large to retain safely",
        "code": "ModelResponseTooLarge",
    }


def test_foundry_brokered_model_loop_bounds_immediate_final_text_in_utf8_bytes_without_state_install():
    oversized_text = "é" * 80
    assert len(oversized_text) < 128 < len(oversized_text.encode("utf-8"))
    fake = _FakeChatTransport(
        [
            _chat_response({"role": "assistant", "content": oversized_text}),
            _chat_response({"role": "assistant", "content": "Retry stayed available."}),
        ]
    )
    app = _model_loop_app(
        _spec(tool_name="check-network-telemetry"),
        fake,
        max_pending_responses=1,
        max_brokered_output_bytes=128,
        max_response_state_bytes=4 * 1024,
    )

    with TestClient(app) as client:
        oversized = client.post("/responses", json={"input": "Say hello"})
        retry = client.post("/responses", json={"input": "Say hello again"})

    assert oversized.status_code == 502
    assert oversized.json()["error"] == {
        "message": "model response is too large to retain safely",
        "code": "ModelResponseTooLarge",
    }
    assert retry.status_code == 200, retry.text
    assert _message_text(retry.json()) == "Retry stayed available."
    assert len(fake.requests) == 2


def test_foundry_brokered_model_loop_can_return_final_message_without_tool_call():
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "No tool needed."})])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Say hello"})

    assert response.status_code == 200, response.text
    assert _message_text(response.json()) == "No tool needed."
    assert fake.requests[0]["tool_choice"] == "auto"


@pytest.mark.parametrize("phase", [None, "commentary"])
@pytest.mark.parametrize("persisted", [False, True])
def test_foundry_brokered_model_loop_preserves_assistant_phase_on_tool_resume(tmp_path, phase, persisted):
    assistant = {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "Checking SFO."}],
    }
    if phase is not None:
        assistant["phase"] = phase
    fake = _FakeChatTransport([
        {
            "status": "completed", "output": [
                {"type": "reasoning", "summary": [], "encrypted_content": "opaque-reasoning"},
                assistant,
                {"type": "reasoning", "summary": [], "encrypted_content": "opaque-second-reasoning"},
                {"type": "function_call", "call_id": "call_model",
                 "name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
            ],
        },
        _chat_response({"role": "assistant", "content": "Telemetry is healthy."}),
    ])
    spec = _spec(tool_name="check-network-telemetry")
    kwargs = {"response_state_file": tmp_path / "phase-tool-resume.json"} if persisted else {}
    app = _model_loop_app(spec, fake, **kwargs)
    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "Check SFO telemetry"})
        assert initial.status_code == 200, initial.text
        call = _call(initial.json())
    if persisted:
        app = _model_loop_app(spec, fake, **kwargs)
    with TestClient(app) as client:
        final = client.post(
            "/responses", headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )
    assert final.status_code == 200, final.text
    expected = {"type": "message", "role": "assistant", "content": "Checking SFO."}
    if phase is not None:
        expected["phase"] = phase
    assert fake.requests[1]["input"][-5:] == [
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque-reasoning"},
        expected,
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque-second-reasoning"},
        {"type": "function_call", "call_id": call["call_id"],
         "name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
        {"type": "function_call_output", "call_id": call["call_id"], "output": '{"approved":true,"output":{"ok":true}}'},
    ]


@pytest.mark.parametrize("phase", [None, "final_answer"])
@pytest.mark.parametrize("content_type", ["output_text", "refusal"])
@pytest.mark.parametrize("commentary", [False, True])
def test_foundry_brokered_model_loop_preserves_message_phases_in_persisted_session(
    tmp_path, phase, content_type, commentary,
):
    state_file = tmp_path / "phase-session.json"
    answer = "The answer is 42." if content_type == "output_text" else "I cannot help with that."
    part = {"type": "output_text", "text": answer} if content_type == "output_text" else {"type": "refusal", "refusal": answer}
    output = [
        {"type": "message", "role": "assistant", "phase": "commentary",
         "content": [{"type": "output_text", "text": "Checking the answer."}]},
        {"type": "message", "role": "assistant",
         "content": [part]},
    ]
    if phase is not None:
        output[1]["phase"] = phase
    if not commentary:
        output = output[1:]
    fake = _FakeChatTransport([
        {"status": "completed", "output": output},
        _chat_response({"role": "assistant", "content": "Still 42."}),
    ])
    spec = _spec()
    headers = {"x-agent-session-id": "phase-session"}
    with TestClient(_model_loop_app(spec, fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", headers=headers, json={"input": "What is the answer?"})
    assert initial.status_code == 200, initial.text
    assert _message_text(initial.json()) == ("Checking the answer.\n\n" if commentary else "") + answer
    # Reload from disk, rather than relying on in-memory metadata surviving.
    with TestClient(_model_loop_app(spec, fake, response_state_file=state_file)) as client:
        followup = client.post(
            "/responses", headers=headers,
            json={"previous_response_id": initial.json()["id"], "input": "What was the answer?"},
        )
    assert followup.status_code == 200, followup.text
    expected_answer = {"type": "message", "role": "assistant", "content": answer}
    if phase is not None:
        expected_answer["phase"] = phase
    expected_history = [{"type": "message", "role": "user", "content": "What is the answer?"}]
    if commentary:
        expected_history.append({
            "type": "message", "role": "assistant", "content": "Checking the answer.", "phase": "commentary",
        })
    expected_history.extend([
        expected_answer,
        {"type": "message", "role": "user", "content": "What was the answer?"},
    ])
    assert fake.requests[1]["input"][-len(expected_history):] == expected_history


@pytest.mark.parametrize("phase", ["unexpected", 1, [], {}])
def test_foundry_brokered_model_loop_rejects_invalid_assistant_phase(phase):
    fake = _FakeChatTransport([{
        "status": "completed", "output": [
            {"type": "message", "role": "assistant", "phase": phase,
             "content": [{"type": "output_text", "text": "Do not retain this."}]},
        ],
    }])
    with TestClient(_model_loop_app(_spec(), fake)) as client:
        response = client.post("/responses", json={"input": "Say hello"})
    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "InvalidModelResponse"


@pytest.mark.parametrize("phase", [None, "commentary", "final_answer"])
def test_foundry_brokered_model_loop_preserves_assistant_phase_in_input_history(phase):
    assistant = {"role": "assistant", "content": "A previous answer.", "phase": phase}
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "A new answer."})])
    with TestClient(_model_loop_app(_spec(), fake)) as client:
        response = client.post("/responses", json={"input": [
            assistant, {"role": "user", "content": "Continue."},
        ]})
    assert response.status_code == 200, response.text
    expected = {"type": "message", "role": "assistant", "content": "A previous answer."}
    if phase is not None:
        expected["phase"] = phase
    assert fake.requests[0]["input"][-2] == expected


@pytest.mark.parametrize("phase", ["unexpected", 1, [], {}])
@pytest.mark.parametrize("content", ["", "A previous answer."])
def test_foundry_brokered_model_loop_rejects_invalid_input_history_phase(phase, content):
    fake = _FakeChatTransport([])
    with TestClient(_model_loop_app(_spec(), fake)) as client:
        response = client.post("/responses", json={"input": [
            {"role": "assistant", "content": content, "phase": phase},
            {"role": "user", "content": "Continue."},
        ]})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_input"
    assert fake.requests == []


def test_foundry_brokered_model_loop_reads_responses_output_around_reasoning_items():
    spec = _spec(tool_name="check-network-telemetry")
    reasoning = {"type": "reasoning", "id": "rs_model", "summary": []}
    encrypted = {
        "type": "reasoning",
        "id": "rs_encrypted",
        "summary": [{"type": "summary_text", "text": "Need telemetry."}],
        "encrypted_content": "opaque-reasoning",
        "status": "completed",
    }
    fake = _FakeChatTransport(
        [
            {
                "object": "response",
                "status": "completed",
                "output": [
                    reasoning,
                    encrypted,
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking SFO."}]},
                    {
                        "type": "function_call",
                        "id": "fc_model",
                        "call_id": "call_model",
                        "name": "check-network-telemetry",
                        "arguments": '{"site":"sfo"}',
                    },
                ],
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            },
            {
                "object": "response",
                "status": "completed",
                "output": [
                    reasoning,
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Telemetry is healthy."}]},
                ],
                "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            },
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "Check SFO telemetry"})
        call = _call(initial.json())
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"status": "healthy"}}),
        )

    assert initial.status_code == 200, initial.text
    assert json.loads(call["arguments"]) == {"site": "sfo"}
    assert final.status_code == 200, final.text
    assert _message_text(final.json()) == "Telemetry is healthy."
    assert final.json()["usage"] == {"input_tokens": 7, "output_tokens": 10, "total_tokens": 17}
    # Only reasoning with encrypted content can be replayed statelessly; it precedes its call.
    assert fake.requests[1]["input"][-4:] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Need telemetry."}],
            "encrypted_content": "opaque-reasoning",
            "id": "rs_encrypted",
        },
        {"type": "message", "role": "assistant", "content": "Checking SFO."},
        {"type": "function_call", "call_id": call["call_id"], "name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
        {"type": "function_call_output", "call_id": call["call_id"], "output": '{"approved":true,"output":{"status":"healthy"}}'},
    ]


def _function_call_item(name: str = "check-network-telemetry") -> dict[str, Any]:
    return {"type": "function_call", "call_id": f"call_{name}", "name": name, "arguments": "{}"}


@pytest.mark.parametrize(
    ("output", "status", "code"),
    [
        ([{"type": "web_search_call", "id": "ws_model", "status": "completed"}], 400, "unsupported_tool_call"),
        ([_function_call_item(), _function_call_item()], 400, "multiple_tool_calls_unsupported"),
        ([{"type": "reasoning", "id": "rs_model", "summary": []}], 502, "InvalidModelResponse"),
        ([{"type": "message", "role": "user", "content": [{"type": "output_text", "text": "hi"}]}], 502, "InvalidModelResponse"),
        ([{"type": "message", "role": "assistant", "content": [{"type": "output_audio"}]}], 502, "InvalidModelResponse"),
        (["not an item"], 502, "InvalidModelResponse"),
    ],
)
def test_foundry_brokered_model_loop_rejects_unusable_responses_output(output: list[Any], status: int, code: str):
    fake = _FakeChatTransport([{"object": "response", "status": "completed", "output": output, "usage": {}}])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Check telemetry"})

    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", "in_progress", None])
@pytest.mark.parametrize("item", [_function_call_item(), {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Trunc"}]}])
def test_foundry_brokered_model_loop_rejects_responses_that_did_not_complete(status: str | None, item: dict[str, Any]):
    payload = {"object": "response", "status": status, "output": [item], "usage": {}}
    if status is None:
        del payload["status"]
    fake = _FakeChatTransport([payload])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Check telemetry"})

    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "InvalidModelResponse"


@pytest.mark.parametrize("status", ["incomplete", "in_progress", "failed", "cancelled"])
@pytest.mark.parametrize("item_type", ["message", "function_call", "reasoning"])
def test_foundry_brokered_model_loop_rejects_unfinished_output_items(status, item_type):
    if item_type == "function_call":
        item = _function_call_item()
    elif item_type == "reasoning":
        item = {"type": "reasoning", "summary": [], "encrypted_content": "opaque-test-state"}
    else:
        item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Partial"}]}
    item["status"] = status
    output = [item]
    if item_type == "reasoning":
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]})
    fake = _FakeChatTransport([{"object": "response", "status": "completed", "output": output, "usage": {}}])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Check telemetry"})

    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "InvalidModelResponse"
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.test/openai/v1",
        "https://example.test/openai/v1/",
        "https://example.test/openai/v1/responses",
        "https://example.test/openai/v1/chat/completions",
    ],
)
def test_foundry_model_loop_calls_responses_endpoint(base_url: str):
    assert foundry_model_loop_module._responses_url(base_url) == "https://example.test/openai/v1/responses"


def test_foundry_brokered_model_loop_rejects_unsupported_pattern_deterministically():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string", "pattern": "\\p{L}+"}},
        "required": ["site"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"


def test_foundry_brokered_model_loop_validates_additional_properties_schema():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string"}},
        "additionalProperties": {"type": "string"},
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo","count":1}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_unknown_model_tool_request():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "unknown", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call unknown"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_brokered_tool"


def test_foundry_brokered_model_loop_validates_schema_valued_additional_properties():
    spec = _spec(tool_name="flex-tool")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "flex-tool", "arguments": '{"safe":"not-int"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call flex-tool"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_nonfinite_model_arguments():
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"value":NaN}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_uses_type_strict_const_and_enum_matching():
    for property_schema, arguments in [
        ({"const": 1}, '{"value":true}'),
        ({"enum": [0]}, '{"value":false}'),
    ]:
        spec = _spec(tool_name="check-network-telemetry")
        spec.brokered_tools[0].parameters = {
            "type": "object",
            "properties": {"value": property_schema},
            "required": ["value"],
        }
        fake = _FakeChatTransport(
            [
                _chat_response(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_model",
                                "type": "function",
                                "function": {"name": "check-network-telemetry", "arguments": arguments},
                            }
                        ],
                    }
                )
            ]
        )
        app = _model_loop_app(spec, fake)

        with TestClient(app) as client:
            response = client.post("/responses", json={"input": "call check-network-telemetry"})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_arguments_that_violate_declared_schema():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"site": {"type": "string"}},
        "required": ["site"],
        "additionalProperties": False,
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":123,"extra":"nope"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_validates_large_integer_bounds_exactly():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer", "maximum": 9007199254740992}},
        "required": ["count"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"count":9007199254740993}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_float_arguments_that_cannot_round_trip_exactly():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer", "maximum": 9007199254740992}},
        "required": ["count"],
    }
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"count":9007199254740993.0}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"


def test_foundry_brokered_model_loop_rejects_unsafe_model_generated_arguments():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": '{"site":"sfo","tokenValue":"ghp_not_real"}'},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "call check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsafeBrokeredArguments"


def test_foundry_brokered_rejects_tool_choice_none_instead_of_ignoring_it():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "tool_choice": "none"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tool_choice_unsupported"


def test_foundry_brokered_rejects_request_supplied_tools_even_when_static_tools_exist():
    app = _app()

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "hi", "tools": [{"type": "function"}]})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tools_unsupported"


def _fixture(name: str) -> dict[str, Any]:
    path = __import__("pathlib").Path(__file__).parent / "fixtures" / "foundry_brokered" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_initial_response(body: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(body)
    actual_response_id = normalized["id"]
    normalized["id"] = "caresp_test"
    normalized["created_at"] = 0
    item = normalized["output"][0]
    item["id"] = "fc_test"
    item["call_id"] = item["call_id"].replace(actual_response_id, "caresp_test")
    item["response_id"] = "caresp_test"
    return normalized


def _normalize_final_response(body: dict[str, Any], *, previous_response_id: str) -> dict[str, Any]:
    normalized = deepcopy(body)
    actual_response_id = normalized["id"]
    normalized["id"] = "caresp_final"
    normalized["created_at"] = 0
    normalized["previous_response_id"] = "caresp_test"
    item = normalized["output"][0]
    item["id"] = "msg_final"
    item["response_id"] = "caresp_final"
    assert previous_response_id
    assert actual_response_id
    return normalized


def _materialize_continuation(fixture: dict[str, Any], *, response_id: str, call_id: str) -> dict[str, Any]:
    materialized = deepcopy(fixture)
    materialized["previous_response_id"] = response_id
    materialized["input"][0]["call_id"] = call_id
    return materialized


def test_foundry_brokered_golden_fixtures_pin_function_call_loop_and_errors():
    app = _app()

    with TestClient(app) as client:
        initial = client.post("/responses", json=_fixture("initial_request.json"))
        assert initial.status_code == 200, initial.text
        initial_body = initial.json()
        call = _call(initial_body)
        assert _normalize_initial_response(initial_body) == _fixture("function_call_response.json")

        continuation = _materialize_continuation(
            _fixture("continuation_request.json"),
            response_id=initial_body["id"],
            call_id=call["call_id"],
        )
        final = client.post("/responses", json=continuation, headers=CONTINUATION_AUTH)
        assert final.status_code == 200, final.text
        assert _normalize_final_response(final.json(), previous_response_id=initial_body["id"]) == _fixture("final_message_response.json")

        unknown_prev = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_materialize_continuation(
                _fixture("continuation_request.json"),
                response_id="caresp_unknown",
                call_id=call["call_id"],
            ),
        )
        assert unknown_prev.status_code == 404
        assert unknown_prev.json() == _fixture("unknown_previous_response_id_error.json")

    app_for_errors = _app()
    with TestClient(app_for_errors) as client:
        initial_body = _start(client)
        call = _call(initial_body)
        unknown_call = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_materialize_continuation(
                _fixture("continuation_request.json"),
                response_id=initial_body["id"],
                call_id="call_unknown",
            ),
        )
        assert unknown_call.status_code == 400
        assert unknown_call.json() == _fixture("unknown_call_id_error.json")

        multiple = _materialize_continuation(
            _fixture("continuation_request.json"),
            response_id=initial_body["id"],
            call_id=call["call_id"],
        )
        multiple["input"].append(dict(multiple["input"][0]))
        multiple_resp = client.post("/responses", json=multiple, headers=CONTINUATION_AUTH)
        assert multiple_resp.status_code == 400
        assert multiple_resp.json() == _fixture("multiple_function_calls_unsupported_error.json")


# Regression coverage retained from the merged brokered-continuation stack.

@pytest.mark.parametrize("ttl_seconds", [float("nan"), float("inf"), float("-inf")])
def test_foundry_brokered_nonfinite_explicit_state_ttl_uses_default(ttl_seconds: float):
    app = _app(state_ttl_seconds=ttl_seconds)

    with TestClient(app) as client:
        readiness = client.get("/readiness")

    assert readiness.status_code == 200
    assert readiness.json()["foundryResponses"]["stateTtlSeconds"] == 900.0

@pytest.mark.parametrize("raw_ttl", ["nan", "inf", "-inf"])
def test_foundry_brokered_nonfinite_state_ttl_env_uses_default(monkeypatch, raw_ttl: str):
    monkeypatch.setenv("AGENTKIT_FOUNDRY_RESPONSE_STATE_TTL_SECONDS", raw_ttl)
    app = _app()

    with TestClient(app) as client:
        readiness = client.get("/readiness")

    assert readiness.status_code == 200
    assert readiness.json()["foundryResponses"]["stateTtlSeconds"] == 900.0

def test_foundry_brokered_rejects_normal_followup_while_previous_response_is_resuming(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        state = stores[0].get(initial["id"])
        state.status = "resuming"
        stores[0].save(state)
        resp = client.post("/responses", json={"previous_response_id": initial["id"], "input": "next question"})

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "response_pending_function_call_output"

def test_foundry_brokered_integer_arguments_preserve_large_integer_bounds():
    huge = 10**100
    spec = _spec(tool_name="large-integer-bounds")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "minimum": {"type": "integer", "minimum": huge},
            "exclusiveMinimum": {"type": "integer", "exclusiveMinimum": huge},
            "exclusiveMaximum": {"type": "integer", "exclusiveMaximum": -huge},
        },
        "required": ["minimum", "exclusiveMinimum", "exclusiveMaximum"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "large-integer-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {
        "minimum": huge,
        "exclusiveMinimum": huge + 1,
        "exclusiveMaximum": -huge - 1,
    }

def test_foundry_brokered_integer_synthesis_rejects_unsupported_multiple_of():
    spec = _spec(tool_name="check-network-telemetry")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1, "multipleOf": 2}},
        "required": ["n"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"

def test_foundry_brokered_number_synthesis_handles_large_finite_bounds_without_overflow():
    spec = _spec(tool_name="large-number-bounds")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "midpoint": {"type": "number", "minimum": 1e308, "maximum": 1.1e308},
            "above": {"type": "number", "exclusiveMinimum": 1e308},
            "below": {"type": "number", "exclusiveMaximum": -1e308},
            "exactInteger": {"type": "number", "minimum": 9007199254740993, "maximum": 9007199254740994},
            "roundedMidpoint": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 0.9999999999999999},
            "narrowValue": {"type": "number", "exclusiveMinimum": 1.0, "maximum": 1.0000000000000002},
        },
        "required": ["midpoint", "above", "below", "exactInteger", "roundedMidpoint", "narrowValue"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "large-number-bounds"})

    assert response.status_code == 200, response.text
    arguments = json.loads(_call(response.json())["arguments"])
    assert arguments["midpoint"] == 1e308
    assert arguments["above"] == math.nextafter(1e308, math.inf)
    assert arguments["below"] == math.nextafter(-1e308, -math.inf)
    assert arguments["exactInteger"] == 9007199254740993
    assert 0 < arguments["roundedMidpoint"] < 0.9999999999999999
    assert arguments["narrowValue"] == 1.0000000000000002

def test_foundry_brokered_number_synthesis_rejects_unsupported_multiple_of():
    spec = _spec(tool_name="number-multiple")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"value": {"type": "number", "minimum": 0, "multipleOf": 0.5}},
        "required": ["value"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "number-multiple"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"

def test_foundry_brokered_number_synthesis_combines_all_declared_bounds():
    spec = _spec(tool_name="combined-number-bounds")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "minimum": 0, "exclusiveMinimum": 0, "maximum": 1},
        },
        "required": ["value"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "combined-number-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {"value": 1}

@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")])
def test_foundry_brokered_number_synthesis_rejects_nonfinite_bounds(bound: float):
    spec = _spec(tool_name="nonfinite-number-bound")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"value": {"type": "number", "minimum": bound}},
        "required": ["value"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "nonfinite-number-bound"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedBrokeredSchema"

def test_foundry_brokered_number_synthesis_prefers_compact_zero_within_wide_bounds():
    spec = _spec(tool_name="compact-number-bounds")
    properties = {
        f"value{index}": {"type": "number", "minimum": -1e308, "maximum": 1e308}
        for index in range(27)
    }
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "compact-number-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {name: 0 for name in properties}

def test_foundry_brokered_number_synthesis_prefers_compact_float_boundary_over_large_integer():
    spec = _spec(tool_name="compact-large-number-bounds")
    properties = {
        f"value{index}": {"type": "number", "minimum": 1e308, "maximum": 1.1e308}
        for index in range(27)
    }
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "compact-large-number-bounds"})

    assert response.status_code == 200, response.text
    assert json.loads(_call(response.json())["arguments"]) == {name: 1e308 for name in properties}

def test_foundry_brokered_bounds_nested_array_synthesis_before_recursive_allocation():
    spec = _spec(tool_name="nested-array")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 32,
                "items": {
                    "type": "array",
                    "minItems": 32,
                    "items": {"type": "string"},
                },
            }
        },
        "required": ["values"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "nested-array"})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_arguments_too_large"

def test_foundry_brokered_counts_wide_object_members_against_synthesis_budget():
    item_properties = {f"field{index}": {"type": "string"} for index in range(32)}
    spec = _spec(tool_name="wide-object-array")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 32,
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": list(item_properties),
                },
            }
        },
        "required": ["values"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "wide-object-array"})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_arguments_too_large"

def test_foundry_brokered_rejects_overly_deep_deterministic_synthesis():
    child: dict[str, Any] = {"type": "string", "default": "leaf"}
    for _ in range(129):
        child = {
            "type": "object",
            "properties": {"child": child},
            "required": ["child"],
        }
    spec = _spec(tool_name="deep-synthesis")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"root": child},
        "required": ["root"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "deep-synthesis"})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_arguments_too_large"

def test_foundry_brokered_rejects_overly_deep_root_literal_synthesis():
    literal: dict[str, Any] = {}
    for _ in range(129):
        literal = {"nested": literal}
    spec = _spec(tool_name="deep-root-literal")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "const": literal,
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "deep-root-literal"})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_arguments_too_large"

@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ({"const": 2.0}, 2.0),
        ({"default": 1.0}, 1.0),
        ({"enum": [3.0]}, 3.0),
    ],
)
def test_foundry_brokered_honors_integral_float_integer_literals(
    literal: dict[str, Any], expected: float
):
    spec = _spec(tool_name="integer-literal")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer", **literal}},
        "required": ["count"],
    }
    app = _app(spec)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "integer-literal"})

    assert response.status_code == 200, response.text
    count = json.loads(_call(response.json())["arguments"])["count"]
    assert count == expected
    assert isinstance(count, float)

def test_foundry_brokered_rejects_lossy_function_call_output_float_before_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        lossy = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"id":9007199254740993.0}}',
                    }
                ],
            },
        )
        persisted_after_rejection = state_file.read_bytes()
        exact = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"id":9007199254740992.0}}',
                    }
                ],
            },
        )

    assert lossy.status_code == 400
    assert lossy.json()["error"]["code"] == "invalid_function_call_output"
    assert persisted_after_rejection == persisted_before
    assert exact.status_code == 200, exact.text
    assert _message_text(exact.json()).endswith('{"id":9007199254740992.0}')

def test_foundry_brokered_rejects_duplicate_keys_in_function_call_output(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":false,"approved":true,"output":{}}',
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"
    assert state_file.read_bytes() == persisted_before

def test_foundry_brokered_rejects_object_valued_function_call_output_before_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    app = _app(response_state_file=state_file)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        persisted_before = state_file.read_bytes()
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": {"approved": True, "output": {"id": 9007199254740993.0}},
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"
    assert state_file.read_bytes() == persisted_before

def test_foundry_brokered_rejects_oversized_raw_output_before_json_parsing(monkeypatch):
    app = _app(max_brokered_output_bytes=64)

    def fail_if_parsed(_output: Any) -> dict[str, Any]:
        raise AssertionError("oversized raw output must be rejected before JSON parsing")

    monkeypatch.setattr(foundry_module, "_json_object_from_output", fail_if_parsed)
    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json={
                "previous_response_id": initial["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": '{"approved":true,"output":{"blob":"' + ("x" * 128) + '"}}',
                    }
                ],
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_output_too_large"

def test_foundry_brokered_accepting_continuation_refreshes_state_expiry(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app(state_ttl_seconds=60)

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        state = stores[0].get(initial["id"])
        state.expires_at = time.time() + 1
        stores[0].save(state)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert response.status_code == 200, response.text
    refreshed = stores[0].get(initial["id"])
    assert refreshed.expires_at > time.time() + 50


def test_foundry_brokered_replays_persisted_output_after_limit_is_lowered(tmp_path):
    state_file = tmp_path / "responses-state.json"

    with TestClient(_app(response_state_file=state_file, max_brokered_output_bytes=1024)) as client:
        initial = _start(client)
        call = _call(initial)
        payload = _continuation(
            initial["id"],
            call["call_id"],
            {"approved": True, "output": {"blob": "x" * 256}},
        )
        completed = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)

    with TestClient(_app(response_state_file=state_file, max_brokered_output_bytes=128)) as client:
        replay = client.post("/responses", json=payload, headers=CONTINUATION_AUTH)

    assert completed.status_code == 200, completed.text
    assert replay.status_code == 200, replay.text
    assert replay.json() == completed.json()


def test_foundry_brokered_rejects_persisted_model_loop_continuation_when_loop_is_disabled(tmp_path):
    state_file = tmp_path / "foundry-model-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    with TestClient(_model_loop_app(spec, fake, response_state_file=state_file)) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"}).json()
        call = _call(initial)

    persisted_before = state_file.read_bytes()
    with TestClient(_app(spec, response_state_file=state_file, brokered_model_loop_enabled=False)) as client:
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "brokered_model_loop_unavailable"
    assert state_file.read_bytes() == persisted_before

def test_foundry_brokered_rejects_normal_followup_to_expired_pending_response():
    app = _app(state_ttl_seconds=0)

    with TestClient(app) as client:
        initial = _start(client)
        time.sleep(0.01)
        response = client.post(
            "/responses",
            json={"previous_response_id": initial["id"], "input": "next question"},
        )

    assert response.status_code == 410
    assert response.json()["error"]["code"] == "response_state_expired"

def test_foundry_brokered_allows_normal_followup_after_completed_state_expires(monkeypatch):
    stores: list[Any] = []
    original_store = foundry_module._FoundryResponseStateStore

    def capture_store(*args: Any, **kwargs: Any) -> Any:
        store = original_store(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(foundry_module, "_FoundryResponseStateStore", capture_store)
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        completed = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )
        state = stores[0].get(initial["id"])
        state.expires_at = time.time() - 1
        stores[0].save(state)
        followup = client.post(
            "/responses",
            json={"previous_response_id": initial["id"], "input": "next question"},
        )

    assert completed.status_code == 200, completed.text
    assert followup.status_code == 200, followup.text
    assert _call(followup.json())

@pytest.mark.parametrize("brokered_class", ["write", "coordination"])
def test_foundry_brokered_refuses_multi_value_root_enum_side_effecting_arguments(brokered_class: str):
    spec = _spec(tool_name="dispatch-work-order", brokered_class=brokered_class)
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "enum": [
            {"operation": "delete"},
            {"operation": "create"},
        ],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"

def test_foundry_brokered_allows_single_value_root_enum_write_arguments():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "enum": [{"operation": "create"}],
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order"})

    assert resp.status_code == 200
    assert json.loads(_call(resp.json())["arguments"]) == {"operation": "create"}

@pytest.mark.parametrize("brokered_class", ["write", "coordination"])
def test_foundry_brokered_refuses_nonliteral_optional_prompt_for_side_effecting_tools(brokered_class: str):
    spec = _spec(tool_name="dispatch-work-order", brokered_class=brokered_class)
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order with arbitrary payload"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UnsupportedBrokeredSchema"

def test_foundry_brokered_allows_literal_optional_prompt_for_write_tools():
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    spec.brokered_tools[0].parameters = {
        "type": "object",
        "properties": {"prompt": {"type": "string", "const": "fixed"}},
    }
    app = _app(spec)

    with TestClient(app) as client:
        resp = client.post("/responses", json={"input": "dispatch-work-order with arbitrary payload"})

    assert resp.status_code == 200
    assert json.loads(_call(resp.json())["arguments"]) == {"prompt": "fixed"}

def test_foundry_brokered_rejects_continuation_with_extra_non_output_item():
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        request = _continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}})
        request["input"].append({"type": "message", "role": "user", "content": "ignored"})
        rejected = client.post("/responses", json=request, headers=CONTINUATION_AUTH)
        accepted = client.post(
            "/responses",
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
            headers=CONTINUATION_AUTH,
        )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "multiple_tool_outputs_unsupported"
    assert accepted.status_code == 200, accepted.text

def test_foundry_brokered_model_loop_omits_authorization_when_auth_is_omitted(monkeypatch):
    captured_headers: dict[str, str] = {}

    class FakeStream:
        async def __aenter__(self) -> httpx.Response:
            return httpx.Response(
                200,
                request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
                content=json.dumps(_chat_response({"role": "assistant", "content": "done"})).encode("utf-8"),
                headers={"content-type": "application/json"},
            )

        async def __aexit__(self, *args: Any) -> None:
            return None

    class FakeClient:
        def __init__(self, *, headers: dict[str, str] | None = None, timeout: int) -> None:
            assert timeout == 60
            captured_headers.update(headers or {})

        def stream(self, method: str, url: str, **kwargs: Any) -> FakeStream:
            assert method == "POST"
            assert url.endswith("/responses")
            captured_headers.update(kwargs.get("headers") or {})
            return FakeStream()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(foundry_model_loop_module.httpx, "AsyncClient", FakeClient)
    app = _app(_spec(tool_name="check-network-telemetry"), brokered_model_loop_enabled=True)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 200, response.text
    assert "Authorization" not in captured_headers


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": {"unexpected": 1}},
        {"output_tokens": "not-a-number"},
        {"input_tokens": "12"},
        {"total_tokens": 1.5},
    ],
)
def test_foundry_brokered_model_loop_normalizes_malformed_usage(usage: dict[str, Any]):
    fake = _FakeChatTransport(
        [
            {
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}],
                "usage": usage,
            }
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "InvalidModelResponse"

def test_foundry_brokered_model_loop_preserves_total_only_usage_across_resume():
    spec = _spec(tool_name="check-network-telemetry")
    initial_model_response = _chat_response(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "model_generated_call_id",
                    "type": "function",
                    "function": {"name": "check-network-telemetry", "arguments": "{}"},
                }
            ],
        }
    )
    initial_model_response["usage"] = {"total_tokens": 5}
    final_model_response = _chat_response({"role": "assistant", "content": "done"})
    final_model_response["usage"] = {"total_tokens": 7}
    app = _model_loop_app(spec, _FakeChatTransport([initial_model_response, final_model_response]))

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        final = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial.json()["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert initial.json()["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 5}
    assert final.status_code == 200, final.text
    assert final.json()["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 12}

def test_foundry_brokered_model_loop_rejects_noncanonical_denied_output_before_resume(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="dispatch-work-order", brokered_class="write")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "dispatch-work-order", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(spec, fake, response_state_file=state_file)

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "dispatch-work-order"})
        call = _call(initial.json())
        persisted_before = state_file.read_bytes()
        denied = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial.json()["id"],
                call["call_id"],
                {
                    "approved": False,
                    "error": {"code": "approval_declined", "message": "denied"},
                    "output": {"sensitiveDiagnostic": "must-not-reach-model"},
                },
            ),
        )

    assert denied.status_code == 400
    assert denied.json()["error"]["code"] == "invalid_function_call_output"
    assert state_file.read_bytes() == persisted_before
    assert len(fake.requests) == 1

@pytest.mark.parametrize("payload", [{"approved": True}, {"approved": True, "output": None}])
def test_foundry_brokered_rejects_approved_continuation_without_object_output(payload: dict[str, Any]):
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], payload),
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"

@pytest.mark.parametrize("payload", [{"approved": False}, {"approved": False, "error": None}])
def test_foundry_brokered_rejects_denied_continuation_without_error_object(payload: dict[str, Any]):
    app = _app()

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        response = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], payload),
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_function_call_output"

def test_foundry_brokered_rejects_deep_continuation_output_without_consuming_state():
    app = _app()
    nested: dict[str, Any] = {}
    for _ in range(200):
        nested = {"nested": nested}

    with TestClient(app) as client:
        initial = _start(client)
        call = _call(initial)
        rejected = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": nested}),
        )
        accepted = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(initial["id"], call["call_id"], {"approved": True, "output": {"ok": True}}),
        )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_function_call_output"
    assert accepted.status_code == 200, accepted.text

def test_foundry_brokered_bounds_request_body_before_model_call():
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": "must not run"})])
    app = _model_loop_app(
        _spec(tool_name="check-network-telemetry"),
        fake,
        max_request_body_bytes=64,
    )

    with TestClient(app) as client:
        response = client.post(
            "/responses",
            content=json.dumps({"input": "x" * 128}),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
    assert fake.requests == []

def test_foundry_brokered_model_message_size_matches_persistence_encoding():
    messages = [{"role": "user", "content": "😀" * 8}]

    measured = foundry_module._persistence_json_bytes(messages)
    persisted = json.dumps(
        messages,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    utf8_compact = json.dumps(
        messages,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert measured == persisted
    assert len(measured) > len(utf8_compact)


def test_foundry_brokered_model_loop_accepts_exact_compact_message_limit():
    spec = _spec().model_copy(update={"instructions": ""})
    fake = _FakeChatTransport([
        _chat_response({"role": "assistant", "content": "Within the limit."}),
    ])
    app = _model_loop_app(spec, fake, max_model_messages_bytes=512)
    # A compact user message record adds 30 bytes around its ASCII content.
    with TestClient(app) as client:
        accepted = client.post("/responses", json={"input": "x" * 482})
        rejected = client.post("/responses", json={"input": "x" * 483})

    assert accepted.status_code == 200, accepted.text
    assert _message_text(accepted.json()) == "Within the limit."
    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "brokered_model_messages_too_large"
    assert len(fake.requests) == 1


def test_foundry_brokered_bounds_persisted_model_messages_and_releases_reservation():
    tool_response = _chat_response(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_model",
                    "type": "function",
                    "function": {"name": "check-network-telemetry", "arguments": "{}"},
                }
            ],
        }
    )
    fake = _FakeChatTransport([tool_response, deepcopy(tool_response)])
    app = _model_loop_app(
        _spec(tool_name="check-network-telemetry"),
        fake,
        max_request_body_bytes=4096,
        max_model_messages_bytes=512,
        max_pending_responses=1,
    )

    with TestClient(app) as client:
        oversized = client.post(
            "/responses",
            json={"input": "check-network-telemetry " + ("x" * 1024)},
        )
        accepted = client.post("/responses", json={"input": "check-network-telemetry"})

    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "brokered_model_messages_too_large"
    assert accepted.status_code == 200, accepted.text
    assert _call(accepted.json())
    assert len(fake.requests) == 1  # Oversized model input is rejected before inference.

def test_foundry_brokered_model_loop_reserves_capacity_before_model_call():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "must not be called"}),
        ]
    )
    app = _model_loop_app(
        _spec(tool_name="check-network-telemetry"),
        fake,
        max_pending_responses=1,
    )

    with TestClient(app) as client:
        first = client.post("/responses", json={"input": "check-network-telemetry"})
        second = client.post("/responses", json={"input": "check-network-telemetry"})

    assert first.status_code == 200, first.text
    assert _call(first.json())
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "brokered_response_state_full"
    assert len(fake.requests) == 1

def test_foundry_brokered_model_loop_unused_reservation_preserves_completed_replay():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chat_response({"role": "assistant", "content": "First response completed."}),
            _chat_response({"role": "assistant", "content": "No tool needed."}),
        ]
    )
    app = _model_loop_app(
        _spec(tool_name="check-network-telemetry"),
        fake,
        max_pending_responses=1,
    )

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        continuation = _continuation(
            initial.json()["id"],
            call["call_id"],
            {"approved": True, "output": {"ok": True}},
        )
        completed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)
        direct = client.post("/responses", json={"input": "answer without a tool"})
        replayed = client.post("/responses", headers=CONTINUATION_AUTH, json=continuation)

    assert completed.status_code == 200, completed.text
    assert direct.status_code == 200, direct.text
    assert replayed.status_code == 200, replayed.text
    assert replayed.json() == completed.json()
    assert len(fake.requests) == 3

def test_foundry_brokered_model_loop_rejects_oversized_output_before_resume_or_state_change(tmp_path):
    state_file = tmp_path / "responses-state.json"
    spec = _spec(tool_name="check-network-telemetry")
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "model_generated_call_id",
                            "type": "function",
                            "function": {"name": "check-network-telemetry", "arguments": "{}"},
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(
        spec,
        fake,
        response_state_file=state_file,
        max_brokered_output_bytes=128,
    )

    with TestClient(app) as client:
        initial = client.post("/responses", json={"input": "check-network-telemetry"})
        call = _call(initial.json())
        persisted_before = state_file.read_bytes()
        oversized = client.post(
            "/responses",
            headers=CONTINUATION_AUTH,
            json=_continuation(
                initial.json()["id"],
                call["call_id"],
                {"approved": True, "output": {"blob": "x" * 256}},
            ),
        )

    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "brokered_output_too_large"
    assert state_file.read_bytes() == persisted_before
    assert len(fake.requests) == 1

@pytest.mark.parametrize("raw_state", ["{not valid json", "[]"])
def test_foundry_brokered_invalid_file_state_fails_startup_without_overwriting(tmp_path, raw_state: str):
    state_file = tmp_path / "responses-state.json"
    state_file.write_text(raw_state, encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid Foundry response state file"):
        _app(response_state_file=state_file)

    assert state_file.read_text(encoding="utf-8") == raw_state

@pytest.mark.parametrize("expires_at", [float("nan"), float("inf"), float("-inf")])
def test_foundry_brokered_rejects_nonfinite_persisted_expiry(tmp_path, expires_at: float):
    state_file = tmp_path / "responses-state.json"
    with TestClient(_app(response_state_file=state_file)) as client:
        _start(client)
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    state = next(iter(persisted["states"].values()))
    state["expiresAt"] = expires_at
    raw_state = json.dumps(persisted, separators=(",", ":"))
    state_file.write_text(raw_state, encoding="utf-8")

    with pytest.raises(RuntimeError, match="stored expiresAt must be finite"):
        _app(response_state_file=state_file)

    assert state_file.read_text(encoding="utf-8") == raw_state

def test_foundry_brokered_model_loop_returns_assistant_refusal_without_tool_call():
    fake = _FakeChatTransport(
        [_chat_response({"role": "assistant", "content": None, "refusal": "I cannot help with that."})]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Do something unsafe"})

    assert response.status_code == 200, response.text
    assert _message_text(response.json()) == "I cannot help with that."

@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "\ud800"},
        {"role": "assistant", "content": None, "refusal": "\ud800"},
    ],
)
def test_foundry_brokered_model_loop_rejects_surrogate_final_text(message: dict[str, Any]):
    fake = _FakeChatTransport([_chat_response(message)])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Say hello"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "InvalidModelResponse"

@pytest.mark.parametrize("content", [None, {"text": "not supported"}, ["not supported"]])
def test_foundry_brokered_model_loop_rejects_non_string_final_content(content: Any):
    fake = _FakeChatTransport([_chat_response({"role": "assistant", "content": content})])
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "Say hello"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "InvalidModelResponse"

def test_foundry_brokered_model_loop_rejects_object_valued_tool_arguments():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": {"id": 9007199254740993.0},
                            },
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"

def test_foundry_brokered_model_loop_rejects_duplicate_tool_argument_keys():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"site":"sfo","site":"sea"}',
                            },
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"

def test_foundry_brokered_model_loop_rejects_decoded_surrogate_arguments():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"site":"\\ud800"}',
                            },
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"

def test_foundry_brokered_model_loop_rejects_excessively_nested_arguments():
    nested = '"leaf"'
    for _ in range(200):
        nested = f"[{nested}]"
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"site":' + nested + "}",
                            },
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"

def test_foundry_brokered_model_loop_bounds_raw_arguments_before_parsing(monkeypatch):
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"blob":"' + ("x" * 128) + '"}',
                            },
                        }
                    ],
                }
            )
        ]
    )

    def fail_if_parsed(_raw: Any) -> dict[str, Any]:
        raise AssertionError("oversized model arguments must be rejected before parsing")

    monkeypatch.setattr(foundry_model_loop_module, "_parse_arguments", fail_if_parsed)
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake, max_brokered_argument_bytes=64)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "brokered_arguments_too_large"

def test_foundry_brokered_model_loop_normalizes_json_parser_failures():
    fake = _FakeChatTransport(
        [
            _chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_model",
                            "type": "function",
                            "function": {
                                "name": "check-network-telemetry",
                                "arguments": '{"value":' + ("9" * 5000) + "}",
                            },
                        }
                    ],
                }
            )
        ]
    )
    app = _model_loop_app(_spec(tool_name="check-network-telemetry"), fake)

    with TestClient(app) as client:
        response = client.post("/responses", json={"input": "check-network-telemetry"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidToolArguments"

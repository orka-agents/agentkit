"""Reusable conformance suite for AgentKit runtime adapters.

These are the v0 HARD invariants of the OpenAI ``/v1`` facade (plan §6, §10) that
EVERY adapter must honor identically — the contract as executable tests. An
adapter inherits the whole suite by re-exporting it from its own test module and
providing two fixtures in its ``conftest.py``:

* ``make_client`` — a factory ``(auth_token=None, output="ok") -> contextmanager``
  yielding a ``TestClient`` whose agent is wired to an OFFLINE double producing
  ``output``. (pydantic-ai supplies a ``TestModel`` override; MAF patches
  ``build_client`` with a ``BaseChatClient`` echo.)
* ``model_name`` — the model id the spec advertises (asserted by ``/v1/models``).

Usage in an adapter's ``tests/test_conformance.py``::

    from agentkit_serve_common.conformance import *  # noqa: F401,F403

The shared suite is framework-agnostic; framework-specific tests (e.g. the MAF
lock-in import boundary) live in the adapter's own test modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

__all__ = [
    "test_healthz_open",
    "test_models_listing",
    "test_structurally_invalid_request_body_uses_openai_error",
    "test_malformed_json_uses_secret_safe_openai_error",
    "test_stream_true_rejected",
    "test_caller_tools_rejected",
    "test_caller_tool_choice_required_rejected",
    "test_tool_choice_auto_allowed",
    "test_happy_path_single_completion",
    "test_final_message_must_be_user",
    "test_multi_turn_history_accepted",
    "test_auth_gate_enforced",
    "test_run_failure_error_envelope",
    "test_shared_server_is_framework_agnostic",
    "test_only_agent_factory_imports_the_framework",
]

# Framework/model-SDK roots that must NOT be imported outside an adapter's
# agent_factory.py. The union across adapters is fine: each adapter only has one
# of these installed, so listing all is harmless and keeps this test shared.
_FRAMEWORK_SDK_ROOTS = {
    "agent_framework",
    "pydantic_ai",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_mcp_adapters",
    "langchain_openai",
    "langgraph",
    "mcp",
}


def test_healthz_open(make_client):
    with make_client() as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_models_listing(make_client, model_name):
    with make_client() as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == model_name


def test_structurally_invalid_request_body_uses_openai_error(make_client):
    with make_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": {"role": "user", "content": "private-value"}},
        )

    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": "request body is invalid",
            "type": "invalid_request_error",
            "code": "invalid_request",
        }
    }


def test_malformed_json_uses_secret_safe_openai_error(make_client):
    malformed_bodies = [
        (
            "sk-private-request-value",
            '{"model":"x","messages":[{"role":"user","content":"sk-private-request-value"}',
        ),
        (
            "sk-another-private-value",
            '{"model":"x" "private":"sk-another-private-value"}',
        ),
    ]
    with make_client() as c:
        responses = [
            c.post(
                "/v1/chat/completions",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            for _, body in malformed_bodies
        ]

    expected = {
        "error": {
            "message": "request body must be valid JSON",
            "type": "invalid_request_error",
            "code": "invalid_json",
        }
    }
    for response, (secret_marker, _) in zip(responses, malformed_bodies, strict=True):
        assert response.status_code == 400
        assert response.json() == expected
        assert secret_marker not in response.text


def test_stream_true_rejected(make_client):
    with make_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 400
        assert "stream" in r.json()["error"]["message"].lower()


def test_caller_tools_rejected(make_client):
    with make_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "x"}}],
            },
        )
        assert r.status_code == 400
        assert "tools" in r.json()["error"]["message"].lower()


def test_caller_tool_choice_required_rejected(make_client):
    with make_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "required",
            },
        )
        assert r.status_code == 400


def test_tool_choice_auto_allowed(make_client):
    # "auto"/"none" mean "no specific tool" — must NOT be rejected.
    with make_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "auto",
            },
        )
        assert r.status_code == 200


def test_happy_path_single_completion(make_client):
    with make_client(output="three bullets here") as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "summarize"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "three bullets here"


def test_final_message_must_be_user(make_client):
    with make_client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "assistant", "content": "hi"}]},
        )

    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": "the final message must have role 'user'",
            "type": "invalid_request_error",
            "code": None,
        }
    }


def test_multi_turn_history_accepted(make_client):
    # A prior user/assistant exchange plus a final user turn → single completion.
    with make_client(output="final answer") as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                ],
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "final answer"


def test_auth_gate_enforced(make_client):
    with make_client(auth_token="secret123") as c:
        # healthz stays open
        assert c.get("/healthz").status_code == 200
        # /v1/* requires a valid bearer token
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret123"}).status_code == 200


def test_run_failure_error_envelope(make_failing_client):
    """A run failure returns the OpenAI error envelope with a runtime-owned code.

    ``make_failing_client(exc)`` wires the offline double to raise ``exc`` from the
    agent run. Framework exception text can carry upstream bodies and echoed
    credentials, and its class name differs per adapter, so the envelope reports
    the same neutral code and message for every adapter.
    """
    boom = RuntimeError("upstream exploded with sk-private-upstream-value")
    with make_failing_client(boom) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 502  # default for a non-HTTP framework error
        assert r.json() == {
            "error": {"message": "agent run failed", "type": "agent_error", "code": "AgentRunFailed"}
        }


def _imported_roots(path: Path) -> set[str]:
    """The top-level package root of every absolute import in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_shared_server_is_framework_agnostic():
    """The shared server/cli/runtime must import NO agent framework symbol.

    This is the option-B seam guarantee, enforced for ALL adapters at once: if a
    framework import ever leaks into the shared core, every adapter's suite fails.
    """
    import agentkit_serve_common

    pkg_dir = Path(agentkit_serve_common.__file__).parent
    for path in pkg_dir.glob("*.py"):
        leaked = _imported_roots(path) & _FRAMEWORK_SDK_ROOTS
        assert not leaked, (
            f"shared core {path.name} imports framework symbol(s) {sorted(leaked)}; "
            f"the run must go through the injected RuntimeFactory"
        )


def test_only_agent_factory_imports_the_framework():
    """Within an adapter package, ONLY agent_factory.py may import a framework.

    The shared core (config/server/cli) is held agnostic by the test above; this
    asserts the ADAPTER's own ``agentkit_serve`` package keeps every framework /
    model-SDK import confined to ``agent_factory.py``, so the thin ``__main__.py``
    and ``__init__.py`` stay clean. Both adapters' packages are named
    ``agentkit_serve``, so this single shared test covers whichever adapter is
    running the suite (the one whose package is importable).
    """
    import agentkit_serve  # the adapter package running this suite

    pkg_dir = Path(agentkit_serve.__file__).parent
    for path in pkg_dir.glob("*.py"):
        if path.name == "agent_factory.py":
            continue  # the one place a framework is allowed
        leaked = _imported_roots(path) & _FRAMEWORK_SDK_ROOTS
        assert not leaked, (
            f"{path.name} imports framework symbol(s) {sorted(leaked)}; confine "
            f"framework imports to agent_factory.py (the lock-in boundary)"
        )

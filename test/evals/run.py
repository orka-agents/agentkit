"""Run live tasks through built agents. Quality failures do not abort the suite."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cases import (
    CASE_IDS,
    CORPUS_VERSION,
    MODEL_REQUEST_LIMIT,
    POISON_MARKER,
    World,
    grade,
)

GRADES = ("taskSuccess", "answerCorrect", "toolSelectionCorrect", "argumentsCorrect", "safetyPassed")


class InfrastructureError(Exception):
    pass


class TaskError(Exception):
    pass


def docker(*args: str, timeout: float = 60) -> str:
    try:
        result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise InfrastructureError(f"docker {args[0]} failed: {type(exc).__name__}") from exc
    if result.returncode:
        # Do not copy credentials, model output, or arbitrary daemon errors to reports.
        raise InfrastructureError(f"docker {args[0]} failed")
    return result.stdout.strip()


def request(url: str, *, payload: dict | None = None, timeout: float = 5) -> Any:
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        headers["Authorization"] = "Bearer mock-token"
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as response:
        return json.load(response)


def endpoint(container: str, port: str) -> str:
    binding = docker("port", container, port).splitlines()[0]
    if not re.fullmatch(r"127\.0\.0\.1:[0-9]+", binding):
        raise InfrastructureError("unexpected non-loopback port binding")
    return f"http://{binding}"


def ready(container: str, url: str, suite_deadline: float) -> None:
    deadline = min(time.monotonic() + 60, suite_deadline)
    while time.monotonic() < deadline:
        if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
            raise InfrastructureError("trial container exited before readiness")
        try:
            request(url, timeout=2)
            return
        except (OSError, ValueError):
            time.sleep(0.25)
    raise InfrastructureError("trial container readiness timeout")


def remove(container: str) -> None:
    docker("rm", "-f", container)


@contextlib.contextmanager
def containers(args: argparse.Namespace, case_id: str, trial: int):
    owned: list[str] = []
    try:
        fixture = docker(
            "create",
            "--platform",
            args.platform,
            "--network",
            args.network,
            "--label",
            f"agentkit.eval-run={args.network}",
            "--network-alias",
            "eval-fixture",
            "-p",
            "127.0.0.1::8090",
            "--mount",
            f"type=bind,src={Path(__file__).parent.resolve()},dst=/eval,readonly",
            "-e",
            f"EVAL_CASE_ID={case_id}",
            "-e",
            f"EVAL_TRIAL={trial}",
            "--entrypoint",
            "/opt/agentkit/bin/python",
            args.fixture_image,
            "/eval/fixture.py",
        )
        owned.append(fixture)
        docker("start", fixture)
        fixture_url = endpoint(fixture, "8090/tcp")
        ready(fixture, fixture_url + "/eval/state", args.deadline)
        agent = docker(
            "create",
            "--platform",
            args.platform,
            "--network",
            args.network,
            "--label",
            f"agentkit.eval-run={args.network}",
            "-p",
            "127.0.0.1::8080",
            "-e",
            "AGENTKIT_BIND=0.0.0.0",
            "-e",
            "AGENTKIT_AUTH_TOKEN=mock-token",
            "-e",
            "MODEL_API_KEY=not-needed",
            "-e",
            "EVAL_MCP_URL=http://eval-fixture:8090/mcp",
            args.agent_image,
        )
        owned.append(agent)
        docker("start", agent)
        agent_url = endpoint(agent, "8080/tcp")
        ready(agent, agent_url + "/healthz", args.deadline)
        yield fixture, fixture_url, agent, agent_url
    finally:
        failed = False
        for container in reversed(owned):
            try:
                remove(container)
            except InfrastructureError:
                failed = True
        if failed:
            raise InfrastructureError("trial container cleanup failed")


def run_trial(args: argparse.Namespace, case_id: str, trial: int) -> dict[str, Any]:
    world = World(case_id, trial)
    answers: list[str] = []
    turns: list[dict[str, Any]] = []
    error = None
    with containers(args, case_id, trial) as (_, fixture_url, agent, agent_url):
        started = time.monotonic()
        deadline = min(started + args.case_timeout, args.deadline)
        messages: list[dict[str, str]] = []
        calls_by_turn = []
        model_calls_by_turn = []
        for prompt in world.prompts():
            messages.append({"role": "user", "content": prompt})
            turns.append({"role": "user", "content": prompt})
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TaskError("task_timeout")
                response = request(
                    agent_url + "/v1/chat/completions",
                    payload={
                        "model": args.model,
                        "stream": False,
                        "messages": messages,
                    },
                    timeout=remaining,
                )
                text = response["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise TaskError("invalid_response")
                answers.append(text)
                messages.append({"role": "assistant", "content": text})
                turns.append({"role": "assistant", "content": text[:4096]})
            except urllib.error.HTTPError as exc:
                error = f"runtime_http_{exc.code}"
                try:
                    code = json.loads(exc.read(65536)).get("error", {}).get("code")
                    if code in {
                        "ModelAuthRejected",
                        "ModelUnavailable",
                        "ModelUpstreamError",
                        "MCPToolProtocolError",
                        "AgentRunFailed",
                        "AgentNotInitialized",
                        "LangGraphResultError",
                    }:
                        error = f"runtime_{code}"
                except (ValueError, TypeError, AttributeError):
                    pass
                break
            except TimeoutError:
                error = "task_timeout"
                break
            except urllib.error.URLError as exc:
                error = "task_timeout" if isinstance(exc.reason, TimeoutError) else "runtime_transport_error"
                break
            except OSError:
                error = "runtime_transport_error"
                break
            except TaskError as exc:
                error = "task_timeout" if str(exc) == "task_timeout" else "runtime_protocol_error"
                break
            except (KeyError, IndexError, TypeError, ValueError):
                error = "runtime_protocol_error"
                break
            try:
                turn_state = request(fixture_url + "/eval/state")
                calls_by_turn.append(len(turn_state["calls"]))
                model_calls_by_turn.append(len(turn_state["modelToolCalls"]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise InfrastructureError("turn accounting unavailable") from exc
        duration = round((time.monotonic() - started) * 1000)
        # Quiesce the agent before collecting state, including timed-out runs.
        docker("stop", "--time", "1", agent)
        try:
            state = request(fixture_url + "/eval/settle", payload={})
        except (OSError, ValueError) as exc:
            raise InfrastructureError("trial state unavailable") from exc
        if state.get("seed") != world.seed:
            raise InfrastructureError("fixture seed mismatch")
        state["callsByTurn"] = calls_by_turn
        state["modelCallsByTurn"] = model_calls_by_turn
        if time.monotonic() >= args.deadline:
            raise InfrastructureError("suite time budget exhausted")
        grades = grade(case_id, answers, state)
        if error is not None:
            grades["taskSuccess"] = False
            grades["failureReasons"].append(error)
        if state["providerRequests"] > MODEL_REQUEST_LIMIT:
            grades["failureReasons"].append("model_request_budget_exceeded")
        infrastructure_failure = is_infrastructure_failure(state, error)
        if infrastructure_failure:
            grades["taskSuccess"] = False
            grades["failureReasons"].append("live_model_unavailable")
        result = {
            "case": case_id,
            "category": world.case.category,
            "trial": trial,
            "seed": world.seed,
            **grades,
            "durationMillis": duration,
            "providerRequests": state["providerRequests"],
            "providerCompletions": state["providerCompletions"],
            "providerCancelled": state["providerCancelled"],
            "mcpCalls": len(state["calls"]),
            "usage": state["usage"],
            "infrastructureFailure": infrastructure_failure,
            "transcript": turns,
            "modelToolCalls": state["modelToolCalls"],
            "toolCalls": state["calls"],
            "finalState": {"stock": state["stock"], "reservations": state["reservations"]},
        }
        # Deliberately injected fixture text is useful for grading, not for logs.
        return json.loads(json.dumps(result).replace(POISON_MARKER, "[REDACTED_CANARY]"))


def is_infrastructure_failure(state: dict[str, Any], error: str | None) -> bool:
    if state["providerFailures"] or state["providerCompletions"] == 0 or state["providerInflight"]:
        return True
    if error in {
        "runtime_transport_error",
        "runtime_protocol_error",
        "runtime_MCPToolProtocolError",
        "runtime_AgentNotInitialized",
        "runtime_LangGraphResultError",
        "runtime_ModelAuthRejected",
        "runtime_ModelUnavailable",
    }:
        return True
    if error and error.startswith("runtime_http_"):
        return True
    return error == "runtime_ModelUpstreamError" and state["providerRequests"] <= MODEL_REQUEST_LIMIT


def summarize(results: list[dict[str, Any]], case_ids: list[str], trials: int) -> dict[str, Any]:
    planned = len(case_ids) * trials
    consistent = sum(
        len(items := [r for r in results if r["case"] == case_id]) == trials and all(r["taskSuccess"] for r in items)
        for case_id in case_ids
    )
    return {
        "plannedTrials": planned,
        "completedTrials": len(results),
        "passes": sum(r["taskSuccess"] for r in results),
        "rates": {key: sum(r[key] for r in results) / planned for key in GRADES},
        "consistentCases": consistent,
        "totalCases": len(case_ids),
    }


def run(args: argparse.Namespace) -> int:
    case_ids = args.cases.split(",") if args.cases else list(CASE_IDS)
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(case not in CASE_IDS for case in case_ids):
        raise ValueError("unknown or duplicate eval case")
    if not 1 <= args.trials <= 10 or args.case_timeout <= 0 or args.suite_timeout <= 0:
        raise ValueError("invalid trial count or timeout")
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "corpusVersion": CORPUS_VERSION,
        "mode": "live",
        "adapter": args.adapter,
        "platform": args.platform,
        "model": args.model,
        "modelImage": args.model_image,
        "modelConfigurationDigest": hashlib.sha256(
            (Path(__file__).parent.parent / "aikit-e2e" / "model.yaml").read_bytes()
        ).hexdigest(),
        "sourceRevision": args.source_revision,
        "sourceDirty": args.source_dirty,
        "suiteDigest": hashlib.sha256(
            b"".join((Path(__file__).parent / name).read_bytes() for name in ("cases.py", "fixture.py", "run.py"))
        ).hexdigest(),
        "fixtureConfigurationDigest": hashlib.sha256(
            (Path(__file__).parent / f"agentkitfile-{args.adapter}.yaml").read_bytes()
        ).hexdigest(),
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "trialsPerCase": args.trials,
        "cases": case_ids,
        "caseTimeoutSeconds": args.case_timeout,
        "suiteTimeoutSeconds": args.suite_timeout,
        "modelRequestLimit": MODEL_REQUEST_LIMIT,
        "complete": False,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.suite_timeout
    args.deadline = deadline
    try:
        report["agentImageDigest"] = docker("image", "inspect", "--format", "{{.Id}}", args.agent_image)
        report["fixtureImageDigest"] = docker("image", "inspect", "--format", "{{.Id}}", args.fixture_image)
        report["modelImageDigest"] = docker("image", "inspect", "--format", "{{.Id}}", args.model_image)
        for trial in range(1, args.trials + 1):
            for case_id in case_ids:
                if time.monotonic() >= deadline:
                    raise InfrastructureError("suite time budget exhausted")
                item = run_trial(args, case_id, trial)
                report["results"].append(item)
                print(
                    json.dumps(
                        {
                            "adapter": args.adapter,
                            "case": case_id,
                            "trial": trial,
                            "taskSuccess": item["taskSuccess"],
                            "failureReasons": item["failureReasons"],
                        }
                    ),
                    flush=True,
                )
                report["summary"] = summarize(report["results"], case_ids, args.trials)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["complete"] = not any(r["infrastructureFailure"] for r in report["results"])
    except InfrastructureError as exc:
        report["infrastructureError"] = str(exc)
    finally:
        report["summary"] = summarize(report["results"], case_ids, args.trials)
        report["qualityPassed"] = report["complete"] and all(r["taskSuccess"] for r in report["results"])
        report["finishedAt"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    # Baseline scores are informational. Missing real-inference measurements are not.
    return 0 if report["complete"] else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("adapter", "agent-image", "fixture-image", "model-image", "network", "platform", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cases", default="")
    parser.add_argument("--case-timeout", type=int, default=120)
    parser.add_argument("--suite-timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.adapter not in {"pydantic-ai", "microsoft-agent-framework", "langgraph"}:
        parser.error("unsupported adapter")
    try:
        raise SystemExit(run(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

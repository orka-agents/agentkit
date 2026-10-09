"""Runner lifecycle/report tests do not launch containers or inference."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("live_eval_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def result(case, success=True, infrastructure=False):
    return {
        "case": case,
        "trial": 1,
        "infrastructureFailure": infrastructure,
        "failureReasons": [] if success else ["answerCorrect"],
        **{key: success for key in runner.GRADES},
    }


def args(directory, **changes):
    values = {
        "adapter": "pydantic-ai",
        "agent_image": "agent",
        "fixture_image": "fixture",
        "model_image": "model",
        "model": "qwen-3.5-2b",
        "network": "owned-test-network",
        "platform": "linux/amd64",
        "output": Path(directory) / "report.json",
        "source_revision": "revision",
        "source_dirty": True,
        "trials": 1,
        "cases": "stock,price",
        "case_timeout": 120,
        "suite_timeout": 3600,
        "deadline": float("inf"),
    }
    values.update(changes)
    return argparse.Namespace(**values)


class Reports(unittest.TestCase):
    def test_quality_failures_continue_and_exit_zero(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "docker", return_value="sha256:test"),
            patch.object(runner, "run_trial", side_effect=[result("stock", False), result("price")]) as trial,
        ):
            options = args(directory)
            self.assertEqual(runner.run(options), 0)
            report = json.loads(options.output.read_text())
            self.assertTrue(report["complete"])
            self.assertFalse(report["qualityPassed"])
            self.assertEqual(trial.call_count, 2)
            self.assertEqual(report["summary"]["completedTrials"], 2)
            self.assertEqual(report["summary"]["rates"]["taskSuccess"], 0.5)
            self.assertEqual(report["summary"]["consistentCases"], 1)

    def test_infrastructure_failure_keeps_partial_report(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "docker", return_value="sha256:test"),
            patch.object(
                runner, "run_trial", side_effect=[result("stock"), runner.InfrastructureError("fixture unavailable")]
            ),
        ):
            options = args(directory)
            self.assertEqual(runner.run(options), 2)
            report = json.loads(options.output.read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["qualityPassed"])
            self.assertEqual(report["summary"]["plannedTrials"], 2)
            self.assertEqual(report["summary"]["completedTrials"], 1)
            self.assertEqual(report["summary"]["rates"]["taskSuccess"], 0.5)
            self.assertEqual(report["infrastructureError"], "fixture unavailable")

    def test_provider_failures_cannot_be_reported_as_complete(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "docker", return_value="sha256:test"),
            patch.object(runner, "run_trial", return_value=result("stock", False, True)),
        ):
            options = args(directory, cases="stock")
            self.assertEqual(runner.run(options), 2)
            self.assertFalse(json.loads(options.output.read_text())["complete"])

    def test_image_failure_still_writes_report(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "docker", side_effect=runner.InfrastructureError("missing image")),
        ):
            options = args(directory)
            self.assertEqual(runner.run(options), 2)
            report = json.loads(options.output.read_text())
            self.assertEqual(report["results"], [])
            self.assertFalse(report["complete"])
            self.assertEqual(report["summary"]["consistentCases"], 0)

    def test_runtime_transport_failures_are_infrastructure_after_inference(self):
        state = {"providerFailures": 0, "providerCompletions": 1, "providerInflight": 0, "providerRequests": 2}
        for error in (
            "runtime_MCPToolProtocolError",
            "runtime_transport_error",
            "runtime_protocol_error",
            "runtime_http_502",
        ):
            with self.subTest(error=error):
                self.assertTrue(runner.is_infrastructure_failure(state, error))
        self.assertFalse(runner.is_infrastructure_failure(state, "task_timeout"))
        state["providerRequests"] = runner.MODEL_REQUEST_LIMIT + 1
        self.assertFalse(runner.is_infrastructure_failure(state, "runtime_ModelUpstreamError"))

    def test_case_selection_validation_precedes_docker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "docker") as docker:
            for selection in ("unknown", "stock,stock", ",stock", "stock,"):
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    runner.run(args(directory, cases=selection))
            docker.assert_not_called()

    def test_trial_and_timeout_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "docker") as docker:
            for changes in ({"trials": 0}, {"trials": 11}, {"case_timeout": 0}, {"suite_timeout": -1}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    runner.run(args(directory, **changes))
            docker.assert_not_called()

    def test_consistency_requires_every_requested_trial(self):
        summary = runner.summarize([result("stock"), result("stock"), result("price")], ["stock", "price"], 2)
        self.assertEqual(summary["consistentCases"], 1)
        self.assertEqual(summary["rates"]["taskSuccess"], 0.75)

    def test_suite_budget_stops_before_next_trial(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "docker", return_value="sha256:test"),
            patch.object(runner.time, "monotonic", side_effect=[0, 100]),
            patch.object(runner, "run_trial") as trial,
        ):
            options = args(directory, suite_timeout=10)
            self.assertEqual(runner.run(options), 2)
            trial.assert_not_called()
            self.assertEqual(
                json.loads(options.output.read_text())["infrastructureError"], "suite time budget exhausted"
            )


class Lifecycle(unittest.TestCase):
    def test_state_failure_cleans_up_only_owned_containers(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "ready"),
            patch.object(runner, "endpoint", return_value="http://127.0.0.1:12345"),
            patch.object(runner, "docker", side_effect=["owned-fixture", "", "owned-agent", "", "", ""]) as docker,
        ):
            with self.assertRaises(RuntimeError), runner.containers(args(directory), "stock", 1):
                raise RuntimeError("failed")
            self.assertEqual(docker.call_args_list[-2].args, ("rm", "-f", "owned-agent"))
            self.assertEqual(docker.call_args_list[-1].args, ("rm", "-f", "owned-fixture"))

    def test_cleanup_attempts_all_owned_containers_on_failure(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "ready"),
            patch.object(runner, "endpoint", return_value="http://127.0.0.1:12345"),
            patch.object(
                runner,
                "docker",
                side_effect=["owned-fixture", "", "owned-agent", "", runner.InfrastructureError("failed remove"), ""],
            ) as docker,
        ):
            with (
                self.assertRaisesRegex(runner.InfrastructureError, "cleanup"),
                runner.containers(args(directory), "stock", 1),
            ):
                pass
            self.assertEqual(docker.call_args_list[-1].args, ("rm", "-f", "owned-fixture"))

    def test_readiness_retries_connection_reset_during_startup(self):
        with (
            patch.object(runner, "docker", return_value="true"),
            patch.object(runner, "request", side_effect=[ConnectionResetError("starting"), {}]) as request,
            patch.object(runner.time, "monotonic", side_effect=[0, 1, 2]),
            patch.object(runner.time, "sleep"),
        ):
            runner.ready("owned-fixture", "http://127.0.0.1:12345/eval/state", float("inf"))
            self.assertEqual(request.call_count, 2)

    def test_started_but_unfinished_inference_is_not_a_measurement(self):
        from cases import World

        @contextlib.contextmanager
        def isolated(*args):
            yield "fixture", "http://fixture", "agent", "http://agent"

        world = World("stock", 1)
        world.provider_requests = 1
        world.provider_cancelled = 1
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "containers", isolated),
            patch.object(runner, "docker"),
            patch.object(runner, "request", side_effect=[TimeoutError("slow"), world.snapshot()]) as request,
        ):
            result = runner.run_trial(args(directory), "stock", 1)
            self.assertTrue(result["infrastructureFailure"])
            self.assertFalse(result["taskSuccess"])
            self.assertEqual(result["providerCompletions"], 0)
            self.assertEqual(request.call_args_list[-1].args[0], "http://fixture/eval/settle")

    def test_agent_run_failure_after_inference_keeps_the_suite_incomplete(self):
        from cases import World

        @contextlib.contextmanager
        def isolated(*args):
            yield "fixture", "http://fixture", "agent", "http://agent"

        world = World("stock", 1)
        world.provider_requests = world.provider_completions = 1
        error = runner.urllib.error.HTTPError(
            "http://agent/v1/chat/completions",
            502,
            "agent run failed",
            {},
            io.BytesIO(json.dumps({"error": {"code": "AgentRunFailed"}}).encode()),
        )
        self.addCleanup(error.close)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "containers", isolated),
            patch.object(runner, "docker", return_value="sha256:test"),
            patch.object(runner, "request", side_effect=[error, world.snapshot()]),
        ):
            options = args(directory, cases="stock")
            self.assertEqual(runner.run(options), 2)
            report = json.loads(options.output.read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["qualityPassed"])
            self.assertTrue(report["results"][0]["infrastructureFailure"])
            self.assertEqual(report["results"][0]["providerCompletions"], 1)

    def test_fixture_accounting_timeout_is_infrastructure_not_task_timeout(self):
        from cases import World

        @contextlib.contextmanager
        def isolated(*args):
            yield "fixture", "http://fixture", "agent", "http://agent"

        world = World("stock", 1)
        response = {"choices": [{"message": {"content": json.dumps({"available": world.stock})}}]}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "containers", isolated),
            patch.object(runner, "docker"),
            patch.object(runner, "request", side_effect=[response, TimeoutError("fixture slow")]),
            self.assertRaisesRegex(runner.InfrastructureError, "turn accounting"),
        ):
            runner.run_trial(args(directory), "stock", 1)

    def test_external_port_bindings_rejected(self):
        for binding in ("0.0.0.0:1234", "malicious-host:1234", "127.0.0.1:bad", "::1:1234"):
            with (
                self.subTest(binding=binding),
                patch.object(runner, "docker", return_value=binding),
                self.assertRaises(runner.InfrastructureError),
            ):
                runner.endpoint("owned-fixture", "8090/tcp")
        with patch.object(runner, "docker", return_value="127.0.0.1:1234"):
            self.assertEqual(runner.endpoint("owned-fixture", "8090/tcp"), "http://127.0.0.1:1234")


if __name__ == "__main__":
    unittest.main()

"""Exercise the EXIT handler with stub Docker, never a local daemon."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class Cleanup(unittest.TestCase):
    def test_failed_manifest_write_still_attempts_owned_cleanup(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "live-task-evals.sh"
        source = script.read_text()
        function = "finish() {" + source.split("finish() {", 1)[1].split("trap 'finish' EXIT", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            stub = root / "bin" / "docker"
            stub.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" >>"$COMMAND_LOG"\nif [ "$1" = ps ]; then printf "owned-container\\n"; fi\n'
            )
            stub.chmod(0o755)
            artifacts = root / "artifacts"
            (artifacts / "run.json").mkdir(parents=True)
            scratch = root / "scratch"
            scratch.mkdir()
            command = (
                "set -e\nlog() { printf '%s\\n' \"$*\" >&2; }\n"
                + function
                + '\nartifact_dir="$ARTIFACTS"\nwork_dir="$SCRATCH"\nnetwork=owned-network\nnetwork_created=true\nphase=evaluation\nsource_revision=revision\ntrue\nfinish\n'
            )
            env = os.environ | {
                "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                "COMMAND_LOG": str(root / "commands.log"),
                "ARTIFACTS": str(artifacts),
                "SCRATCH": str(scratch),
            }
            result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            commands = (root / "commands.log").read_text() if (root / "commands.log").exists() else ""
            self.assertIn("rm -f owned-container", commands)
            self.assertIn("network rm owned-network", commands)
            self.assertFalse(scratch.exists())
            self.assertTrue((artifacts / "run.json").is_dir())

    def test_removal_failure_is_reported_after_attempting_all_cleanup(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "live-task-evals.sh"
        source = script.read_text()
        function = "finish() {" + source.split("finish() {", 1)[1].split("trap 'finish' EXIT", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            stub = root / "bin" / "docker"
            stub.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" >>"$COMMAND_LOG"\nif [ "$1" = ps ]; then printf "owned-one\\nowned-two\\n"; fi\nif [ "$1" = rm ]; then exit 1; fi\n'
            )
            stub.chmod(0o755)
            artifacts, scratch = root / "artifacts", root / "scratch"
            artifacts.mkdir()
            scratch.mkdir()
            command = (
                "set -e\nlog() { printf '%s\\n' \"$*\" >&2; }\n"
                + function
                + '\nartifact_dir="$ARTIFACTS"\nwork_dir="$SCRATCH"\nnetwork=owned-network\nnetwork_created=true\nphase=evaluation\nsource_revision=revision\ntrue\nfinish\n'
            )
            env = os.environ | {
                "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                "COMMAND_LOG": str(root / "commands.log"),
                "ARTIFACTS": str(artifacts),
                "SCRATCH": str(scratch),
            }
            result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2, result.stderr)
            commands = (root / "commands.log").read_text()
            for command in ("rm -f owned-one", "rm -f owned-two", "network rm owned-network"):
                self.assertIn(command, commands)
            report = json.loads((artifacts / "run.json").read_text())
            self.assertEqual(report["exitCode"], 2)
            self.assertTrue(report["cleanupFailed"])
            self.assertFalse(scratch.exists())

    def test_enumeration_failure_still_removes_known_model(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "live-task-evals.sh"
        source = script.read_text()
        function = "finish() {" + source.split("finish() {", 1)[1].split("trap 'finish' EXIT", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            stub = root / "bin" / "docker"
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >>"$COMMAND_LOG"\nif [ "$1" = ps ]; then exit 1; fi\n')
            stub.chmod(0o755)
            artifacts, scratch = root / "artifacts", root / "scratch"
            artifacts.mkdir()
            scratch.mkdir()
            command = (
                "set -e\nlog() { printf '%s\\n' \"$*\" >&2; }\n"
                + function
                + '\nartifact_dir="$ARTIFACTS"\nwork_dir="$SCRATCH"\nnetwork=owned-network\nnetwork_created=true\nmodel_created=true\nmodel_container=owned-model\nphase=evaluation\nsource_revision=revision\ntrue\nfinish\n'
            )
            env = os.environ | {
                "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                "COMMAND_LOG": str(root / "commands.log"),
                "ARTIFACTS": str(artifacts),
                "SCRATCH": str(scratch),
            }
            result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2, result.stderr)
            commands = (root / "commands.log").read_text()
            self.assertIn("rm -f owned-model", commands)
            self.assertIn("network rm owned-network", commands)
            self.assertTrue(json.loads((artifacts / "run.json").read_text())["cleanupFailed"])


if __name__ == "__main__":
    unittest.main()

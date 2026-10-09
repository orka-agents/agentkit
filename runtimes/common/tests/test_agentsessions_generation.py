from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_generator_check_detects_drift_without_rewriting(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/generate-agentsessions-stubs.sh"
    assert script.is_file(), "pinned agentsessions stub generator is missing"
    # A miniature checkout allows deliberate drift without touching product files.
    shutil.copytree(repo / "runtimes/common/agentkit_serve_common/agentsessions", tmp_path / "runtimes/common/agentkit_serve_common/agentsessions")
    (tmp_path / "scripts").mkdir()
    for name in ["generate-agentsessions-stubs.sh", "agentsessions-generator-requirements.txt"]:
        shutil.copy2(repo / "scripts" / name, tmp_path / "scripts" / name)
    copied = tmp_path / "scripts/generate-agentsessions-stubs.sh"
    result = subprocess.run(["bash", str(copied), "--check"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "runtimes/common/agentkit_serve_common/agentsessions/_generated/harness_pb2.py"
    output.write_bytes(output.read_bytes() + b"\n# drift\n")
    before = output.read_bytes()
    result = subprocess.run(["bash", str(copied), "--check"], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert output.read_bytes() == before
    # Proto drift must also fail instead of blessing a different upstream contract.
    output.write_bytes(before.removesuffix(b"\n# drift\n"))
    proto = tmp_path / "runtimes/common/agentkit_serve_common/agentsessions/proto/common.proto"
    proto.write_bytes(proto.read_bytes() + b"\n// drift\n")
    before_proto = proto.read_bytes()
    result = subprocess.run(["bash", str(copied), "--check"], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert proto.read_bytes() == before_proto

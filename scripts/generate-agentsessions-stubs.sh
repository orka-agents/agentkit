#!/usr/bin/env bash
# Generate the pinned Harness SPI. --check never writes into the checkout.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---generate}"
if [[ $# -gt 1 || ( "$mode" != --generate && "$mode" != --check && "$mode" != --fetch ) ]]; then
    printf '%s\n' 'usage: generate-agentsessions-stubs.sh [--generate|--check|--fetch]' >&2
    exit 2
fi
"${UV:-uv}" run --isolated --no-project \
    --with-requirements "$root/scripts/agentsessions-generator-requirements.txt" \
    python - "$root" "$mode" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
package = root / "runtimes/common/agentkit_serve_common/agentsessions"
provenance = json.loads((package / "provenance.json").read_text())
for name, version in provenance["generator"].items():
    if importlib.metadata.version(name) != version:
        raise SystemExit(f"wrong generator version: {name}")
for local, source in provenance["sources"].items():
    target = package / local
    if mode == "--fetch":
        url = f'https://raw.githubusercontent.com/aramase/agentsessions/{provenance["commit"]}/{source["upstream"]}'
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
    else:
        content = target.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise SystemExit(f"pinned source hash mismatch: {local}")
    if mode == "--fetch":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

import grpc_tools
with tempfile.TemporaryDirectory(prefix="agentkit-agentsessions-") as scratch:
    output = pathlib.Path(scratch)
    subprocess.run([
        sys.executable, "-m", "grpc_tools.protoc",
        f'-I{package / "proto"}', f'-I{pathlib.Path(grpc_tools.__file__).parent / "_proto"}',
        f"--python_out={output}", f"--grpc_python_out={output}",
        "common.proto", "harness.proto",
    ], check=True)
    notice = "# AgentKit: generated from pinned agentsessions sources (Apache-2.0); imports made package-relative.\n"
    for file in output.glob("*.py"):
        text = file.read_text()
        for module in ("common_pb2", "harness_pb2"):
            text = text.replace(f"import {module} as ", f"from . import {module} as ")
        file.write_text(notice + text)
    (output / "__init__.py").write_text('"""Generated agentsessions.v1 wire types; see ../provenance.json."""\n')
    generated = package / "_generated"
    expected = {file.name: file.read_bytes() for file in output.glob("*.py")}
    if mode == "--check":
        actual = {file.name: file.read_bytes() for file in generated.glob("*.py")}
        if actual != expected:
            changed = sorted(name for name in actual.keys() | expected.keys() if actual.get(name) != expected.get(name))
            raise SystemExit("agentsessions generated output drift: " + ", ".join(changed))
        print("agentsessions pinned sources and generated stubs match")
    else:
        generated.mkdir(parents=True, exist_ok=True)
        for name, content in expected.items():
            (generated / name).write_bytes(content)
        print("agentsessions stubs generated")
PY

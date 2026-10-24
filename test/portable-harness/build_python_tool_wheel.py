from __future__ import annotations

import base64
import csv
import hashlib
import io
import pathlib
import zipfile


root = pathlib.Path("/agentkit-harness/tool-fixtures")
wheel = root / "agentkit_portable_python_tool-0.0.0-py3-none-any.whl"
dist_info = "agentkit_portable_python_tool-0.0.0.dist-info"
files = {
    "agentkit_portable_python_tool.py": """\
import json
import sys

def main():
    request = json.loads(sys.stdin.readline())
    print(json.dumps({
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {
                "name": "agentkit-portable-python-tool",
                "version": "0.0.0",
            },
        },
    }), flush=True)
""",
    f"{dist_info}/METADATA": """\
Metadata-Version: 2.1
Name: agentkit-portable-python-tool
Version: 0.0.0
""",
    f"{dist_info}/WHEEL": """\
Wheel-Version: 1.0
Generator: agentkit-portable-harness-spike
Root-Is-Purelib: true
Tag: py3-none-any
""",
    f"{dist_info}/entry_points.txt": """\
[console_scripts]
agentkit-portable-python-tool = agentkit_portable_python_tool:main
""",
}

records: list[list[str]] = []
with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name, content in files.items():
        data = content.encode()
        archive.writestr(name, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append([name, f"sha256={digest}", str(len(data))])

    record_name = f"{dist_info}/RECORD"
    records.append([record_name, "", ""])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(records)
    archive.writestr(record_name, output.getvalue())

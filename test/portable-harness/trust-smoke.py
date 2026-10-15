#!/opt/agentkit/runtime/python/bin/python3

import os
import sys
import urllib.request

for candidate in (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
):
    if os.path.isfile(candidate):
        os.environ["SSL_CERT_FILE"] = candidate
        break
else:
    raise SystemExit("no supported system CA bundle found")

with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
    if response.read() != b"portable-harness-trust-ok\n":
        raise SystemExit("unexpected TLS fixture response")

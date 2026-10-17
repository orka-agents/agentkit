#!/opt/agentkit/runtime/python/bin/python3

import os
import sys

sys.path.insert(0, "/opt/agentkit/packages")

for candidate in (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
):
    if os.path.isfile(candidate):
        os.environ.setdefault("SSL_CERT_FILE", candidate)
        break

from agentkit_serve.__main__ import main

main()

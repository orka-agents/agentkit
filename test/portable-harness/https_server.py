from __future__ import annotations

import argparse
import http.server
import pathlib
import ssl


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"portable-harness-trust-ok\n"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


parser = argparse.ArgumentParser()
parser.add_argument("--cert", required=True)
parser.add_argument("--key", required=True)
parser.add_argument("--port-file", required=True)
args = parser.parse_args()

server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(args.cert, args.key)
server.socket = context.wrap_socket(server.socket, server_side=True)
pathlib.Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
server.serve_forever()

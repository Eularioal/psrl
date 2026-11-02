"""Demonstrate client completion versus service lifetime on CPU, without CUDA or EnvRouter."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

CLIENT = """
import sys
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1], timeout=3) as response:
    print(response.read().decode())
"""


def main():
    # Diagram steps 1 and 2 use one small CPU buffer in place of resident model weights.
    state = {"weights": bytearray(1024 * 1024)}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"server_pid": os.getpid(), "state_bytes": len(state["weights"])}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rows = []
    try:
        for index in range(2):
            # Steps 3 to 5 run two short clients while retaining the same service state.
            completed = subprocess.run(
                [sys.executable, "-c", CLIENT, f"http://127.0.0.1:{server.server_port}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            rows.append(
                {
                    "request": index + 1,
                    "client_exit_code": completed.returncode,
                    "server_reply": json.loads(completed.stdout),
                    "resident_server_alive": thread.is_alive(),
                    "simulated_release_allowed": not thread.is_alive(),
                }
            )
    finally:
        # Step 6 stops only this demonstration's server and discards its CPU state.
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        state.clear()
    assert not thread.is_alive(), "The demonstration server must stop."
    print(
        json.dumps(
            {
                "scope": "CPU teaching simulation. No CUDA or EnvRouter.",
                "requests": rows,
                "after_stop_simulated_release_allowed": not thread.is_alive() and not state,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

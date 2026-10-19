"""
Exercise HTTP pool ownership across actor and Harbor event loops.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from psrl.utils.common.http_utils import _ensure_http_client, close_http_client, request_json


def test_clients_and_cleanup_are_scoped_to_the_running_loop():
    first = asyncio.new_event_loop()
    second = asyncio.new_event_loop()
    try:
        first_client = first.run_until_complete(_ensure_http_client())
        second_client = second.run_until_complete(_ensure_http_client())
        assert first_client is not second_client, "Different loops must have independent pools."
        assert first_client._loop is first, "The first client must belong to its caller's loop."
        assert second_client._loop is second, "The second client must belong to its caller's loop."
        assert first.run_until_complete(_ensure_http_client()) is first_client, "Reuse the local pool."
        first.run_until_complete(close_http_client())
        assert first_client.closed, "The local pool must close."
        assert not second_client.closed, "Closing one loop must not close another loop's pool."
        replacement = first.run_until_complete(_ensure_http_client())
        assert replacement is not first_client, "A closed pool must be replaced."
    finally:
        first.run_until_complete(close_http_client())
        second.run_until_complete(close_http_client())
        first.close()
        second.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.mark.parametrize("actor_backend", ["asyncio", "uvloop"])
def test_two_threads_make_requests_and_close_only_their_own_pool(actor_backend):
    actor_factory = asyncio.new_event_loop
    if actor_backend == "uvloop":
        actor_factory = pytest.importorskip("uvloop").new_event_loop
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    barrier = threading.Barrier(2, timeout=10)
    first_closed = threading.Event()
    url = f"http://127.0.0.1:{server.server_port}/test"

    def run_requests(index, loop_factory):
        async def exercise():
            client = await _ensure_http_client()
            try:
                assert client._loop is asyncio.get_running_loop(), "The pool must belong to the calling loop."
                await asyncio.to_thread(barrier.wait)
                for _ in range(10):
                    replies = await asyncio.gather(*(request_json("GET", url, max_retries=1) for _ in range(4)))
                    assert all(reply.data == {"ok": True} for reply in replies), "All real requests must succeed."
                if index == 0:
                    await close_http_client()
                    first_closed.set()
                else:
                    assert await asyncio.to_thread(first_closed.wait, 10), "The other loop must finish closing."
                    assert not client.closed, "The surviving loop must retain its pool."
                    reply = await request_json("GET", url, max_retries=1)
                    assert reply.data == {"ok": True}, "The surviving pool must remain usable."
            finally:
                await close_http_client()

        with asyncio.Runner(loop_factory=loop_factory) as runner:
            runner.run(asyncio.wait_for(exercise(), timeout=20))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(run_requests, 0, actor_factory)
            second = pool.submit(run_requests, 1, asyncio.new_event_loop)
            first.result(timeout=30)
            second.result(timeout=30)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

"""Tiny local web server for the visualizer (standard library only).

GET  /         the virtual stage page
GET  /info     rig layout + track sections
GET  /events   Server-Sent Events stream of light states (~30 fps)
POST /control  buttons / sliders from the page
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources


def _plain(o):
    """numpy scalars sneak in from the analyzers; make them plain JSON numbers."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class VizServer:
    def __init__(self, port: int, info: dict, on_control):
        self.port = port
        self.info = info
        self.on_control = on_control
        self._frame = b""
        self._frame_id = 0
        self._cond = threading.Condition()
        self._httpd = None
        self._running = True
        self.page = resources.files("jaylights.viz").joinpath("index.html").read_bytes()

    def publish(self, frame: dict):
        data = json.dumps(frame, separators=(",", ":"), default=_plain).encode()
        with self._cond:
            self._frame = data
            self._frame_id += 1
            self._cond.notify_all()

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, server.page, "text/html; charset=utf-8")
                elif self.path == "/info":
                    self._send(200, json.dumps(server.info).encode())
                elif self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = -1
                    try:
                        while server._running:
                            with server._cond:
                                server._cond.wait_for(lambda: server._frame_id != last or not server._running, 1.0)
                                last, data = server._frame_id, server._frame
                            if data:
                                self.wfile.write(b"data: " + data + b"\n\n")
                                self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                if self.path != "/control":
                    return self._send(404, b"{}")
                n = int(self.headers.get("Content-Length", 0))
                try:
                    msg = json.loads(self.rfile.read(n) or b"{}")
                    server.on_control(msg)
                    self._send(200, b'{"ok":true}')
                except Exception as e:  # noqa: BLE001 - report anything back to the page
                    self._send(400, json.dumps({"error": str(e)}).encode())

        for attempt in range(10):
            try:
                self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port + attempt), Handler)
                self.port += attempt
                break
            except OSError:
                continue
        else:
            raise RuntimeError("couldn't open a port for the visualizer")
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._httpd:
            self._httpd.shutdown()

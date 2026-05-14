"""
tools/mermaid_editor.py
========================
Launches a local browser-based SVG diagram editor and blocks until the
user submits the finished graph, then returns the graph as a dict.

The editor (tools/editor.html) lets the user:
  - Place ellipses (drag-n-drop, resize handles)
  - Overlap ellipses (semi-transparent fills → Venn effect)
  - Connect shapes with arrows, drag anchor points on each ellipse border
  - Edit labels / properties in the inspector panel

On submit the editor POSTs JSON:
  {
    "nodes": [{"id":..., "x":..., "y":..., "rx":..., "ry":...,
               "name":..., "blocks":..., "dist":..., "attrs":..., "color":...}],
    "edges": [{"id":..., "src":..., "tgt":..., "join_field":...,
               "p":..., "label":..., "srcAngle":..., "tgtAngle":...}]
  }
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_HTML_PATH = os.path.join(os.path.dirname(__file__), 'editor.html')


# ── HTTP handler ───────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    graph_result: dict | None = None
    _done: threading.Event | None = None

    def log_message(self, *_):
        pass  # silence access log

    def do_GET(self):
        try:
            with open(_HTML_PATH, 'rb') as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "editor.html not found")
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        try:
            _Handler.graph_result = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.send_error(400, f"Invalid JSON: {exc}")
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'ok')

        if _Handler._done:
            _Handler._done.set()


# ── Public API ─────────────────────────────────────────────────

def _free_port(start: int = 7979) -> int:
    for port in range(start, start + 20):
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free port found near 7979')


def launch_diagram_editor(timeout: int = 600) -> dict:
    """
    Start the local diagram editor server, open the browser, and block
    until the user submits the graph (or ``timeout`` seconds elapse).

    Returns the graph dict:
      {
        "nodes": [...],
        "edges": [...]
      }

    Raises RuntimeError on timeout.
    """
    port = _free_port()
    _Handler.graph_result = None
    done = threading.Event()
    _Handler._done = done

    server = HTTPServer(('127.0.0.1', port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f'http://127.0.0.1:{port}'
    print(f'\n  Diagram editor: {url}')
    print('  Draw your Semi-Join diagram, then click "Submit Graph".\n')

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    received = done.wait(timeout=timeout)
    server.shutdown()
    thread.join(timeout=2)

    if not received or _Handler.graph_result is None:
        raise RuntimeError(f'Editor timed out after {timeout}s with no submission.')

    return _Handler.graph_result

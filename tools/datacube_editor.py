"""
tools/datacube_editor.py
=========================
HTTP server for the Hasse-diagram drawing tool used by the DataCube agent.

The editor (tools/datacube_editor.html) lets the user:
  - Place cube nodes (⬭ Cube) — a popup asks for name and access cost
  - Connect nodes with directed arrows (→ Connect) — click parent then child
  - Select, move, resize, and delete nodes/edges

On submit the editor POSTs JSON:
  {
    "ellipses": [{"id":..., "x":..., "y":..., "rx":..., "ry":...,
                  "name":..., "cost":..., "color":...}],
    "edges":    [{"id":..., "src":..., "tgt":...}]
  }

After receiving the graph, extract_datacube_graph() converts it to the
(costs_dict, hierarchy_dict) pair expected by tools/cube_ops.run_ullman().
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_HTML_PATH = os.path.join(os.path.dirname(__file__), 'datacube_editor.html')


# ── HTTP handler ───────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    graph_result: dict | None = None
    _done: threading.Event | None = None

    def log_message(self, *_):
        pass

    def do_GET(self):
        try:
            with open(_HTML_PATH, 'rb') as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, 'datacube_editor.html not found')
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
            self.send_error(400, f'Invalid JSON: {exc}')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'ok')
        if _Handler._done:
            _Handler._done.set()


# ── Public API ─────────────────────────────────────────────────

def _free_port(start: int = 7980) -> int:
    for port in range(start, start + 20):
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free port found near 7980')


def launch_datacube_editor(timeout: int = 600) -> dict:
    """
    Open the Hasse diagram editor in the browser and block until the user
    submits the completed graph (or timeout seconds pass).

    Returns the raw graph dict:
      { "ellipses": [...], "edges": [...] }
    """
    port = _free_port()
    _Handler.graph_result = None
    done = threading.Event()
    _Handler._done = done

    server = HTTPServer(('127.0.0.1', port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f'http://127.0.0.1:{port}'
    print(f'\n  Hasse diagram editor: {url}')
    print('  Draw all cube nodes and connect them (parent → child), then click "Submit Graph".\n')

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    received = done.wait(timeout=timeout)
    server.shutdown()
    thread.join(timeout=2)

    if not received or _Handler.graph_result is None:
        raise RuntimeError(f'Diagram editor timed out after {timeout}s.')

    return _Handler.graph_result


def extract_datacube_graph(graph: dict) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Convert the raw editor graph into (costs, hierarchy) ready for run_ullman().

    costs     — {cube_name: access_cost_int}
    hierarchy — {parent_name: [child_name, ...]}  (only direct edges)
    """
    id_to = {e['id']: e for e in graph.get('ellipses', [])}

    costs: dict[str, int] = {}
    for e in graph.get('ellipses', []):
        try:
            costs[e['name']] = int(float(e.get('cost', 0)))
        except (ValueError, TypeError):
            costs[e['name']] = 0

    hierarchy: dict[str, list[str]] = {}
    for edge in graph.get('edges', []):
        src = id_to.get(edge['src'])
        tgt = id_to.get(edge['tgt'])
        if src and tgt:
            parent = src['name']
            child  = tgt['name']
            hierarchy.setdefault(parent, []).append(child)

    return costs, hierarchy

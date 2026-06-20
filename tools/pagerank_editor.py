"""
tools/pagerank_editor.py
=========================
HTTP server for the link-graph drawing tool used by the PageRank agent.

The editor (tools/pagerank_editor.html) lets the user:
  - Place page nodes (○ Page) — a popup asks for the page name
  - Connect nodes with directed arrows (→ Link) — click the page that HAS the
    link, then the page it POINTS TO (src links to tgt)
  - Select, move, resize, and delete nodes/links

On submit the editor POSTs JSON:
  {
    "nodes": [{"id":..., "x":..., "y":..., "rx":..., "ry":..., "name":..., "color":...}],
    "edges": [{"id":..., "src":..., "tgt":...}]
  }

After receiving the graph, extract_pagerank_graph() converts it to the
(nodes, edges) pair the PageRank agent feeds to tools/pagerank_ops.build_link_table().
"""

import os
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_HTML_PATH = os.path.join(os.path.dirname(__file__), 'pagerank_editor.html')


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
            self.send_error(404, 'pagerank_editor.html not found')
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

def _free_port(start: int = 7981) -> int:
    for port in range(start, start + 20):
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free port found near 7981')


def launch_pagerank_editor(timeout: int = 600) -> dict:
    """
    Open the link-graph editor in the browser and block until the user
    submits the completed graph (or timeout seconds pass).

    Returns the raw graph dict:
      { "nodes": [...], "edges": [...] }
    """
    port = _free_port()
    _Handler.graph_result = None
    done = threading.Event()
    _Handler._done = done

    server = HTTPServer(('127.0.0.1', port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f'http://127.0.0.1:{port}'
    print(f'\n  PageRank link-graph editor: {url}')
    print('  Place each page (○) and draw a link (→) from the page that points to another, then click "Submit Graph".\n')

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    received = done.wait(timeout=timeout)
    server.shutdown()
    thread.join(timeout=2)

    if not received or _Handler.graph_result is None:
        raise RuntimeError(f'Link-graph editor timed out after {timeout}s.')

    return _Handler.graph_result


def extract_pagerank_graph(graph: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Convert the raw editor graph into (nodes, edges).

    nodes — [page_name, ...]                      (in placement order)
    edges — [(src_name, tgt_name), ...]           (src links to tgt)

    Self-loops and duplicate edges are dropped; edges referencing an unknown
    node id are skipped.
    """
    id_to_name = {n['id']: n['name'] for n in graph.get('nodes', [])}

    nodes: list[str] = []
    for n in graph.get('nodes', []):
        name = n.get('name') or '?'
        if name not in nodes:
            nodes.append(name)

    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for e in graph.get('edges', []):
        src = id_to_name.get(e.get('src'))
        tgt = id_to_name.get(e.get('tgt'))
        if not src or not tgt or src == tgt:
            continue
        pair = (src, tgt)
        if pair in seen:
            continue
        seen.add(pair)
        edges.append(pair)

    return nodes, edges

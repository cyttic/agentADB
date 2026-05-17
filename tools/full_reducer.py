"""
tools/full_reducer.py
======================
Full Reducer algorithm for distributed natural joins on acyclic schemas.

Pipeline
--------
1. parse_schema()          — "A(a,b,c); B(b,c,d)" → [(name, [attrs])]
2. build_intersection()    — which pairs share attributes
3. gyo_reduction()         — GYO ear-removal test for acyclicity
4. build_join_tree()       — tree from GYO parent_map
5. full_reducer_pseudocode() — two-phase semi-join sequence + final join
6. render_mermaid_*()      — Mermaid diagrams for intersection graph & join tree
7. analyze()               — single entry point that runs everything

GYO algorithm (Graham-Yu-Özsoyoglu reduction)
----------------------------------------------
A join hypergraph H = (V, E) is acyclic iff it can be reduced to empty by
repeatedly removing "ears":
  Edge E_i is an ear if ∃ E_j (j≠i) such that every attribute of E_i that
  appears in any other edge also appears in E_j.
The removal order defines the join tree (E_j becomes the parent of E_i).
"""

import re
from typing import Optional


# ── Schema parsing ─────────────────────────────────────────────

def parse_schema(text: str) -> list[tuple[str, list[str]]]:
    """
    Parse "A(a,b,c); B(b,c,d)" → [("A",["a","b","c"]), ("B",["b","c","d"])].
    Handles semicolons, commas, newlines as separators.
    """
    tables = []
    for m in re.finditer(r'(\w+)\s*\(([^)]+)\)', text):
        name  = m.group(1)
        attrs = [a.strip() for a in m.group(2).split(',') if a.strip()]
        tables.append((name, attrs))
    return tables


# ── Intersection graph ─────────────────────────────────────────

def build_intersection(
    tables: list[tuple[str, list[str]]],
) -> dict[tuple[str, str], list[str]]:
    """
    Return {(A, B): [shared_attrs]} for every pair with ≥1 common attribute.
    Pairs are stored with lexicographically smaller name first.
    """
    am = {n: set(a) for n, a in tables}
    names = [n for n, _ in tables]
    out: dict[tuple[str, str], list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            shared = sorted(am[a] & am[b])
            if shared:
                out[(a, b)] = shared
    return out


# ── GYO reduction ──────────────────────────────────────────────

def _find_ear(name: str, edges: dict[str, set[str]]) -> Optional[str]:
    """
    Return the name of a covering edge for 'name' (making it an ear), or None.

    'name' is an ear if the set of its attributes that appear in ANY other
    edge is a subset of the attributes of some single other edge 'cover'.
    """
    my_attrs = edges[name]
    others = {k: v for k, v in edges.items() if k != name}

    # Attributes of 'name' that are shared with at least one other edge
    shared_with_any: set[str] = set()
    for other_attrs in others.values():
        shared_with_any |= my_attrs & other_attrs

    # A valid cover must contain all of those shared attributes
    for cover_name, cover_attrs in others.items():
        if shared_with_any <= cover_attrs:
            return cover_name
    return None


def gyo_reduction(tables: list[tuple[str, list[str]]]) -> dict:
    """
    Run GYO reduction and return:
      is_acyclic         — bool
      parent_map         — {removed_table: covering_table}  (join tree edges)
      elimination_order  — [table_name, ...] in order of removal
      root               — last remaining table (root of join tree)
      steps              — human-readable removal log
    """
    edges = {n: set(a) for n, a in tables}
    parent_map: dict[str, str] = {}
    elimination_order: list[str] = []
    steps: list[str] = []

    changed = True
    while changed and len(edges) > 1:
        changed = False
        for name in sorted(edges):           # sorted → deterministic output
            cover = _find_ear(name, edges)
            if cover is not None:
                shared = sorted(edges[name] & edges[cover])
                steps.append(
                    f"  Remove ear '{name}' "
                    f"(covered by '{cover}' via {{{', '.join(shared)}}})"
                )
                parent_map[name] = cover
                elimination_order.append(name)
                del edges[name]
                changed = True
                break                        # restart after each removal

    is_acyclic = len(edges) <= 1
    root = next(iter(edges)) if edges else None

    if not is_acyclic:
        remaining = sorted(edges)
        steps.append(
            f"  STUCK — no ear found. Remaining edges: {{{', '.join(remaining)}}}"
        )

    return {
        'is_acyclic':        is_acyclic,
        'parent_map':        parent_map,
        'elimination_order': elimination_order,
        'root':              root,
        'steps':             steps,
    }


# ── Join tree ──────────────────────────────────────────────────

def build_join_tree(
    tables: list[tuple[str, list[str]]],
    parent_map: dict[str, str],
) -> dict[str, list[str]]:
    """Return {node: [children]} from parent_map."""
    children: dict[str, list[str]] = {n: [] for n, _ in tables}
    for child, parent in parent_map.items():
        children[parent].append(child)
    return children


# ── Traversal helpers ──────────────────────────────────────────

def _post_order(node: str, children: dict[str, list[str]]) -> list[str]:
    result: list[str] = []
    for child in sorted(children.get(node, [])):
        result.extend(_post_order(child, children))
    result.append(node)
    return result


def _pre_order(node: str, children: dict[str, list[str]]) -> list[str]:
    result = [node]
    for child in sorted(children.get(node, [])):
        result.extend(_pre_order(child, children))
    return result


# ── Full Reducer pseudocode ────────────────────────────────────

def full_reducer_pseudocode(
    tables: list[tuple[str, list[str]]],
    root: str,
    children: dict[str, list[str]],
    parent_map: dict[str, str],
    intersection: dict[tuple[str, str], list[str]],
) -> str:
    """
    Generate the two-phase Full Reducer pseudocode:
      Phase 1 (bottom-up): each parent is semi-joined with each child.
      Phase 2 (top-down):  each child is semi-joined with its parent.
      Final: natural join across all reduced relations.
    """
    am = {n: set(a) for n, a in tables}

    def shared(a: str, b: str) -> list[str]:
        key = (a, b) if (a, b) in intersection else (b, a)
        return intersection.get(key, sorted(am[a] & am[b]))

    lines: list[str] = []

    # Header
    lines.append("// Input relations:")
    for name, attrs in tables:
        lines.append(f"//   {name}({', '.join(attrs)})")
    lines.append(f"// Join tree root: {root}")
    lines.append("")

    # ── Phase 1: bottom-up ────────────────────────────────────
    lines.append("// Phase 1 — Bottom-up  (leaves → root)")
    lines.append("// Each parent is semi-joined with each of its children.")
    phase1: list[str] = []
    for node in _post_order(root, children):
        for child in sorted(children.get(node, [])):
            attrs_str = ', '.join(shared(node, child))
            phase1.append(f"  {node} = {node}  ⋉_[{attrs_str}]  {child}")
    if phase1:
        lines.extend(phase1)
    else:
        lines.append("  // (only one relation — no bottom-up steps)")
    lines.append("")

    # ── Phase 2: top-down ─────────────────────────────────────
    lines.append("// Phase 2 — Top-down  (root → leaves)")
    lines.append("// Each child is semi-joined with its parent.")
    phase2: list[str] = []
    for node in _pre_order(root, children):
        if node in parent_map:
            parent = parent_map[node]
            attrs_str = ', '.join(shared(node, parent))
            phase2.append(f"  {node} = {node}  ⋉_[{attrs_str}]  {parent}")
    if phase2:
        lines.extend(phase2)
    else:
        lines.append("  // (only one relation — no top-down steps)")
    lines.append("")

    # ── Final join ────────────────────────────────────────────
    lines.append("// Final natural join")
    lines.append("// All relations are now free of dangling tuples.")
    join_order = _pre_order(root, children)
    lines.append(f"  Result = {' ⋈ '.join(join_order)}")
    lines.append("")
    lines.append("// (join order can follow any tree-compatible traversal)")

    return '\n'.join(lines)


# ── Mermaid rendering ──────────────────────────────────────────

def _mermaid_node(name: str, attrs: list[str]) -> str:
    label = f'{name}\\n({", ".join(attrs)})'
    return f'    {name}["{label}"]'


def render_intersection_mermaid(
    tables: list[tuple[str, list[str]]],
    intersection: dict[tuple[str, str], list[str]],
) -> str:
    lines = ["graph LR"]
    for name, attrs in tables:
        lines.append(_mermaid_node(name, attrs))
    for (a, b), shared in intersection.items():
        lines.append(f'    {a} ---|"{", ".join(shared)}"| {b}')
    return '\n'.join(lines)


def render_join_tree_mermaid(
    root: str,
    children: dict[str, list[str]],
    tables: list[tuple[str, list[str]]],
    parent_map: dict[str, str],
    intersection: dict[tuple[str, str], list[str]],
) -> str:
    am = {n: set(a) for n, a in tables}
    lines = ["graph TD"]
    for name, attrs in tables:
        lines.append(_mermaid_node(name, attrs))
    for child, parent in parent_map.items():
        key = (parent, child) if (parent, child) in intersection else (child, parent)
        fallback = sorted(am.get(parent, set()) & am.get(child, set()))
        shared = ', '.join(intersection.get(key, fallback))
        lines.append(f'    {parent} -->|"{shared}"| {child}')
    return '\n'.join(lines)


# ── Read-only HTML viewer ──────────────────────────────────────

def _schema_viewer_html(result: dict) -> str:
    """
    Generate a self-contained SVG-based schema viewer:
    - Ellipses per table, force-directed layout so joined tables overlap
    - Shared columns shown in gold in the intersection zone
    - Private columns shown in grey inside each ellipse
    - Ear badge on every table (ear 1, ear 2, …, root)
    - Right panel with GYO reduction steps
    """
    import json as _json

    tables            = result['tables']          # [(name, [attr, ...])]
    elimination_order = result.get('elimination_order', [])
    root              = result.get('root', '')
    is_acyclic        = result.get('is_acyclic', False)
    gyo_steps         = result.get('gyo_steps', [])

    # Build attribute → tables map for JavaScript
    attr_tables: dict[str, list[str]] = {}
    for name, attrs in tables:
        for a in attrs:
            attr_tables.setdefault(a, []).append(name)

    js_data = _json.dumps({
        'tables':           [{'name': n, 'attrs': a} for n, a in tables],
        'attrTables':       attr_tables,
        'eliminationOrder': elimination_order,
        'root':             root,
        'isAcyclic':        is_acyclic,
        'gyoSteps':         gyo_steps,
    }, ensure_ascii=False)

    verdict_cls   = 'acyclic' if is_acyclic else 'cyclic'
    verdict_label = '✓ ACYCLIC — Full Reducer applicable' if is_acyclic \
                    else '✗ CYCLIC — Full Reducer not applicable'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Full Reducer — Schema</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font:13px/1.5 'Segoe UI',system-ui,sans-serif;
     height:100vh;display:flex;flex-direction:column;overflow:hidden}}
#header{{display:flex;align-items:center;gap:16px;padding:8px 16px;
         background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}}
#header h1{{font-size:13px;font-weight:600;color:#58a6ff}}
.verdict{{padding:3px 12px;border-radius:99px;font-size:11px;font-weight:700}}
.acyclic{{background:#0f2a1a;color:#3fb950;border:1px solid #3fb950}}
.cyclic {{background:#2a0f0f;color:#f85149;border:1px solid #f85149}}
#workspace{{flex:1;display:flex;overflow:hidden}}
#diagram{{flex:1;display:block;background:#161b22;
          background-image:radial-gradient(circle,#2a2f3a 1px,transparent 1px);
          background-size:24px 24px}}
#sidebar{{width:230px;flex-shrink:0;background:#0d1117;border-left:1px solid #30363d;
          overflow-y:auto;padding:12px}}
.s-title{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
          color:#8b949e;margin:12px 0 6px}}
.s-title:first-child{{margin-top:0}}
.step{{font-size:11px;color:#94a3b8;line-height:1.7;padding:2px 0}}
.step span{{color:#e6edf3}}
.tbl-row{{display:flex;align-items:center;gap:6px;margin-bottom:4px}}
.tbl-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.tbl-label{{font-size:12px}}
.ear-badge{{display:inline-block;padding:1px 7px;border-radius:99px;
            font-size:10px;font-weight:700;margin-left:4px}}
.ear-b{{background:#3b0f8a;color:#c4b5fd}}
.root-b{{background:#0f2a1a;color:#3fb950}}
#footer{{padding:6px 16px;background:#161b22;border-top:1px solid #30363d;
         font-size:11px;color:#484f58;flex-shrink:0}}
</style>
</head>
<body>
<div id="header">
  <h1>Full Reducer — Schema Graph</h1>
  <span class="verdict {verdict_cls}">{verdict_label}</span>
</div>
<div id="workspace">
  <svg id="diagram" xmlns="http://www.w3.org/2000/svg"></svg>
  <div id="sidebar">
    <div class="s-title">Tables</div>
    <div id="tbl-list"></div>
    <div class="s-title">GYO Reduction</div>
    <div id="gyo-steps"></div>
    <div class="s-title">Ear Legend</div>
    <div id="ear-legend"></div>
  </div>
</div>
<div id="footer">
  Overlapping ellipses share columns (shown in gold) · Private columns shown in grey inside each ellipse
</div>
<script>
'use strict';
const DATA = {js_data};
const {{ tables, attrTables, eliminationOrder, root, isAcyclic, gyoSteps }} = DATA;

const PALETTE = ['#3b82f6','#f97316','#22c55e','#a855f7','#ec4899','#14b8a6','#f59e0b'];
const colorMap = {{}};
tables.forEach((t,i) => colorMap[t.name] = PALETTE[i % PALETTE.length]);

// ── Sidebar ──────────────────────────────────────────────────
const tblList  = document.getElementById('tbl-list');
const gyoPanel = document.getElementById('gyo-steps');
const earPanel = document.getElementById('ear-legend');

tables.forEach(t => {{
  const earIdx = eliminationOrder.indexOf(t.name);
  const isRoot = t.name === root;
  let badge = '';
  if (isRoot)       badge = `<span class="ear-badge root-b">root</span>`;
  else if (earIdx>=0) badge = `<span class="ear-badge ear-b">ear ${{earIdx+1}}</span>`;
  tblList.innerHTML += `<div class="tbl-row">
    <div class="tbl-dot" style="background:${{colorMap[t.name]}}"></div>
    <span class="tbl-label"><b>${{t.name}}</b>(${{t.attrs.join(', ')}})${{badge}}</span>
  </div>`;
}});

gyoSteps.forEach(s => {{
  gyoPanel.innerHTML += `<div class="step">${{s.trim()}}</div>`;
}});

eliminationOrder.forEach((name,i) => {{
  earPanel.innerHTML += `<div class="step"><span>Ear ${{i+1}}:</span> ${{name}}</div>`;
}});
if (root) earPanel.innerHTML += `<div class="step"><span>Root:</span> ${{root}}</div>`;

// ── SVG helpers ───────────────────────────────────────────────
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('diagram');

function svgEl(tag, attrs={{}}) {{
  const e = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}}

function hex2rgba(hex, a) {{
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

// ── Layout ────────────────────────────────────────────────────
function layout() {{
  const rect  = svg.getBoundingClientRect();
  const W = rect.width || 700, H = rect.height || 500;
  const n = tables.length;
  const RX = Math.min(100, Math.max(70, W / (n * 1.4)));
  const RY = RX * 0.65;

  const nodes = tables.map((t, i) => ({{
    name: t.name, attrs: t.attrs,
    x: W/2 + (W*0.32) * Math.cos(2*Math.PI*i/n - Math.PI/2),
    y: H/2 + (H*0.32) * Math.sin(2*Math.PI*i/n - Math.PI/2),
    rx: RX, ry: RY, vx: 0, vy: 0,
  }}));
  const nodeMap = {{}};
  nodes.forEach(nd => nodeMap[nd.name] = nd);

  // Pairs that share attrs → should overlap
  const sharedPairs = [];
  const names = tables.map(t=>t.name);
  for (let i=0;i<names.length;i++) for (let j=i+1;j<names.length;j++) {{
    const a=names[i],b=names[j];
    const s=tables[i].attrs.filter(x=>tables[j].attrs.includes(x));
    if (s.length) sharedPairs.push([a,b,s]);
  }}

  // Spring simulation
  for (let iter=0;iter<350;iter++) {{
    const cool=Math.max(0.02, 1-iter/220);
    // Repulsion
    for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {{
      const a=nodes[i],b=nodes[j];
      const dx=b.x-a.x, dy=b.y-a.y, dist=Math.sqrt(dx*dx+dy*dy)||0.1;
      const f=14000/(dist*dist);
      a.vx-=f*dx/dist; a.vy-=f*dy/dist;
      b.vx+=f*dx/dist; b.vy+=f*dy/dist;
    }}
    // Attraction for shared pairs
    sharedPairs.forEach(([na,nb]) => {{
      const a=nodeMap[na],b=nodeMap[nb];
      const dx=b.x-a.x, dy=b.y-a.y, dist=Math.sqrt(dx*dx+dy*dy)||0.1;
      const target=(a.rx+b.rx)*0.45;
      const f=(dist-target)*0.07;
      a.vx+=f*dx/dist; a.vy+=f*dy/dist;
      b.vx-=f*dx/dist; b.vy-=f*dy/dist;
    }});
    nodes.forEach(nd=>{{
      nd.x=Math.max(nd.rx+16,Math.min(W-nd.rx-16, nd.x+nd.vx*cool));
      nd.y=Math.max(nd.ry+24,Math.min(H-nd.ry-24, nd.y+nd.vy*cool));
      nd.vx*=0.78; nd.vy*=0.78;
    }});
  }}
  return {{nodes, nodeMap, sharedPairs, W, H}};
}}

// ── Render ────────────────────────────────────────────────────
function render() {{
  svg.innerHTML = '';
  const {{nodes, nodeMap, sharedPairs, W, H}} = layout();

  // Private attrs per table
  const privateAttrs = {{}};
  tables.forEach(t => {{
    privateAttrs[t.name] = t.attrs.filter(a => (attrTables[a]||[]).length === 1);
  }});

  // ── Ellipses (draw first so text is on top) ───────────────
  nodes.forEach(nd => {{
    const color = colorMap[nd.name];
    svg.appendChild(svgEl('ellipse', {{
      cx:nd.x, cy:nd.y, rx:nd.rx, ry:nd.ry,
      fill: hex2rgba(color, 0.28),
      stroke: color, 'stroke-width': 2.5,
    }}));
  }});

  // ── Shared attribute labels (gold, in intersection zones) ─
  // Group attrs by the sorted list of tables that share them
  const pairGroups = {{}};  // "A-B" → {{x,y,attrs:[]}}
  for (const [attr, tNames] of Object.entries(attrTables)) {{
    if (tNames.length < 2) continue;
    const key = [...tNames].sort().join('\x00');
    if (!pairGroups[key]) {{
      const cx = tNames.reduce((s,n)=>s+nodeMap[n].x,0)/tNames.length;
      const cy = tNames.reduce((s,n)=>s+nodeMap[n].y,0)/tNames.length;
      pairGroups[key] = {{x:cx, y:cy, attrs:[]}};
    }}
    pairGroups[key].attrs.push(attr);
  }}

  for (const {{x,y,attrs}} of Object.values(pairGroups)) {{
    const totalH = attrs.length * 17;
    const startY = y - totalH/2 + 9;
    attrs.forEach((attr,i) => {{
      // Background pill
      svg.appendChild(svgEl('rect', {{
        x:x-22, y:startY+i*17-9, width:44, height:17, rx:5,
        fill:'#1e293b', opacity:0.88,
      }}));
      const t = svgEl('text', {{
        x, y:startY+i*17,
        'text-anchor':'middle','dominant-baseline':'middle',
        fill:'#fbbf24','font-size':12,'font-weight':700,
        'font-family':'monospace','pointer-events':'none',
      }});
      t.textContent = attr;
      svg.appendChild(t);
    }});
  }}

  // ── Table labels + private attrs + ear badges ─────────────
  nodes.forEach(nd => {{
    const color = colorMap[nd.name];
    const priv  = privateAttrs[nd.name];

    // Table name
    const nameY = nd.y - (priv.length > 0 ? priv.length*7 : 0) - 4;
    const nameT = svgEl('text', {{
      x:nd.x, y:nameY,
      'text-anchor':'middle','dominant-baseline':'middle',
      fill:'#f0f6fc','font-size':15,'font-weight':700,
      'font-family':'Segoe UI,system-ui,sans-serif','pointer-events':'none',
    }});
    nameT.textContent = nd.name;
    svg.appendChild(nameT);

    // Private attribute list
    priv.forEach((attr,i) => {{
      const t = svgEl('text', {{
        x:nd.x, y:nameY+16+i*14,
        'text-anchor':'middle','dominant-baseline':'middle',
        fill:'#94a3b8','font-size':11,
        'font-family':'Segoe UI,system-ui,sans-serif','pointer-events':'none',
      }});
      t.textContent = attr;
      svg.appendChild(t);
    }});

    // Ear badge (top-right of ellipse)
    const earIdx = eliminationOrder.indexOf(nd.name);
    const isRoot = nd.name === root;
    const badgeLabel = isRoot ? 'root' : (earIdx>=0 ? `ear ${{earIdx+1}}` : null);
    if (badgeLabel) {{
      const bx = nd.x + nd.rx * 0.62, by = nd.y - nd.ry * 0.78;
      const bw = badgeLabel.length*6.5+12;
      svg.appendChild(svgEl('rect', {{
        x:bx-bw/2, y:by-9, width:bw, height:18, rx:9,
        fill: isRoot ? '#0f2a1a' : '#3b0f8a',
        stroke: isRoot ? '#3fb950' : '#a78bfa',
        'stroke-width':1,
      }}));
      const bt = svgEl('text', {{
        x:bx, y:by+1,
        'text-anchor':'middle','dominant-baseline':'middle',
        fill: isRoot ? '#3fb950' : '#c4b5fd',
        'font-size':10,'font-weight':700,
        'font-family':'Segoe UI,system-ui,sans-serif','pointer-events':'none',
      }});
      bt.textContent = badgeLabel;
      svg.appendChild(bt);
    }}
  }});
}}

// Render after layout is stable (give SVG time to get real dimensions)
requestAnimationFrame(() => {{ render(); }});
window.addEventListener('resize', render);
</script>
</body>
</html>"""


def _free_port(start: int = 7990) -> int:
    import socket
    for port in range(start, start + 20):
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free port found near 7990')


def open_schema_viewer(result: dict) -> str:
    """
    Serve the SVG schema viewer via a local HTTP server and open it as a new
    tab in the existing browser. Returns the URL.
    """
    import threading, webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body = _schema_viewer_html(result).encode('utf-8')

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    port   = _free_port()
    server = HTTPServer(('127.0.0.1', port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f'http://127.0.0.1:{port}'
    webbrowser.open(url)
    print(f'\n  [schema viewer] {url}\n')
    return url


# ── Main entry point ───────────────────────────────────────────

def analyze(schema_text: str) -> dict:
    """
    Full pipeline. Returns a dict with all results.

    Keys:
      tables, intersection, is_acyclic, gyo_steps, root,
      intersection_mermaid, join_tree_mermaid (if acyclic),
      pseudocode (if acyclic), error (if any)
    """
    tables = parse_schema(schema_text)
    if len(tables) < 2:
        return {'error': 'Need at least 2 tables.', 'tables': tables}

    intersection = build_intersection(tables)
    gyo = gyo_reduction(tables)

    result: dict = {
        'tables':             tables,
        'intersection':       {f'{a}∩{b}': v for (a, b), v in intersection.items()},
        'is_acyclic':         gyo['is_acyclic'],
        'gyo_steps':          gyo['steps'],
        'root':               gyo['root'],
        'parent_map':         gyo['parent_map'],
        'elimination_order':  gyo['elimination_order'],
        'intersection_mermaid': render_intersection_mermaid(tables, intersection),
    }

    if gyo['is_acyclic']:
        children = build_join_tree(tables, gyo['parent_map'])
        result['join_tree_mermaid'] = render_join_tree_mermaid(
            gyo['root'], children, tables, gyo['parent_map'], intersection,
        )
        result['pseudocode'] = full_reducer_pseudocode(
            tables, gyo['root'], children, gyo['parent_map'], intersection,
        )

    return result

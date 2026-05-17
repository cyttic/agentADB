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
      is_acyclic    — bool
      parent_map    — {removed_table: covering_table}   (join tree edges)
      root          — last remaining table (root of join tree)
      steps         — human-readable removal log
    """
    edges = {n: set(a) for n, a in tables}
    parent_map: dict[str, str] = {}
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
        'is_acyclic': is_acyclic,
        'parent_map': parent_map,
        'root': root,
        'steps': steps,
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

def _html_viewer(title: str, mermaid_code: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body {{ background:#0d1117; color:#e6edf3; font:14px/1.6 'Segoe UI',system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:40px; }}
  h2   {{ color:#58a6ff; margin-bottom:24px; }}
  .box {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:30px; max-width:900px; width:100%; }}
  .mermaid {{ display:flex; justify-content:center; }}
  p    {{ color:#8b949e; font-size:12px; margin-top:16px; text-align:center; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div class="box">
  <div class="mermaid">
{mermaid_code}
  </div>
</div>
<p>Read-only view — generated by agentADB Full Reducer agent</p>
<script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
</body>
</html>"""


def open_graph_viewer(mermaid_code: str, title: str = "Schema Graph") -> str:
    """Write a temp HTML file and open it in the browser. Returns the file path."""
    import os, tempfile, webbrowser
    html = _html_viewer(title, mermaid_code)
    fd, path = tempfile.mkstemp(suffix='.html', prefix='agentADB_graph_')
    with os.fdopen(fd, 'w') as f:
        f.write(html)
    try:
        webbrowser.open(f'file://{path}')
    except Exception:
        pass
    return path


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
        'tables': tables,
        'intersection': {f'{a}∩{b}': v for (a, b), v in intersection.items()},
        'is_acyclic': gyo['is_acyclic'],
        'gyo_steps': gyo['steps'],
        'root': gyo['root'],
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

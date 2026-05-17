"""
tools/cube_ops.py
==================
Deterministic implementation of Ullman's greedy approximation algorithm
for selecting N optimal cubes in a data-cube lattice for materialization.

Algorithm summary
-----------------
Given a Hasse diagram (parent → children, where parent is a coarser/ancestor
cube) and access costs per cube, greedily select N cubes to materialize so
that the total query cost across all cubes is minimised.

  1. Initialise S' = { top cube }  (top = no parents, always materialised)
  2. While |S'| < N:
       For each candidate v ∉ S':
         benefit(v) = Σ_{w ∈ all_cubes} max(0, query_cost(w, S') − query_cost(w, S'∪{v}))
       Add the v with maximum benefit to S'.

  query_cost(w, S') = min { cost(u) : u ∈ S' and u is an ancestor of w }

All tie-breaking is alphabetical → deterministic output.
"""

import json
from typing import Any


# ── graph utilities ────────────────────────────────────────────

def _reverse(hierarchy: dict[str, list[str]]) -> dict[str, list[str]]:
    """parent→children  →  child→parents."""
    rev: dict[str, list[str]] = {}
    for parent, children in hierarchy.items():
        for child in children:
            rev.setdefault(child, []).append(parent)
    return rev


def _ancestors(v: str, rev: dict[str, list[str]]) -> set[str]:
    """All nodes from which v is reachable (v's ancestors + v itself)."""
    seen = {v}
    queue = [v]
    while queue:
        cur = queue.pop()
        for p in rev.get(cur, []):
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return seen


def _all_nodes(hierarchy: dict[str, list[str]], costs: dict[str, int]) -> set[str]:
    nodes: set[str] = set(costs.keys())
    for parent, children in hierarchy.items():
        nodes.add(parent)
        nodes.update(children)
    return nodes


def _find_top(nodes: set[str], hierarchy: dict[str, list[str]]) -> str:
    """Node with no parents (root of the Hasse diagram)."""
    has_parent = {child for children in hierarchy.values() for child in children}
    tops = sorted(nodes - has_parent)
    if not tops:
        raise ValueError("Hasse diagram has no root (cycle or empty).")
    return tops[0]


# ── cost helpers ───────────────────────────────────────────────

def _qcost(v: str, S: set[str], costs: dict[str, int],
           anc: dict[str, set[str]]) -> int:
    """Query cost of cube v given materialized set S."""
    candidates = anc[v] & S
    if not candidates:
        raise ValueError(f"Cube '{v}' has no ancestor in S' — graph may be disconnected.")
    return min(costs[u] for u in candidates)


def _all_qcosts(nodes: list[str], S: set[str], costs: dict[str, int],
                anc: dict[str, set[str]]) -> dict[str, int]:
    return {v: _qcost(v, S, costs, anc) for v in nodes}


def _benefit(
    v: str, S: set[str], nodes: list[str],
    costs: dict[str, int], anc: dict[str, set[str]],
) -> tuple[int, list[tuple[str, int, int]]]:
    """
    Compute benefit of adding v to S.
    Returns (total_gain, [(cube, old_cost, new_cost), ...]) for cubes that improve.
    """
    new_S = S | {v}
    total = 0
    details: list[tuple[str, int, int]] = []
    for w in nodes:
        old = _qcost(w, S, costs, anc)
        new = _qcost(w, new_S, costs, anc)
        if new < old:
            total += old - new
            details.append((w, old, new))
    return total, details


# ── main algorithm ─────────────────────────────────────────────

def run_ullman(costs_json: str, hierarchy_json: str, n: int) -> str:
    """
    Run Ullman's approximation algorithm and return a formatted step-by-step
    solution string ready to be shown to the user.

    Args:
        costs_json:     JSON object mapping cube name → access cost.
                        e.g. '{"A":300,"B":100,"C":250,"D":100,"E":80,"F":90}'
        hierarchy_json: JSON object mapping parent cube → list of child cubes.
                        e.g. '{"A":["B","C","D"],"B":["E"],"C":["E","F"],"D":["F"]}'
        n:              Number of cubes to materialize (including the top cube).
    """
    try:
        costs     = json.loads(costs_json)
        hierarchy = json.loads(hierarchy_json)
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc}"

    nodes_set = _all_nodes(hierarchy, costs)
    nodes     = sorted(nodes_set)
    rev       = _reverse(hierarchy)
    anc       = {v: _ancestors(v, rev) for v in nodes}
    top       = _find_top(nodes_set, hierarchy)
    non_top   = [v for v in nodes if v != top]   # fixed columns for benefit matrix

    if n > len(nodes):
        n = len(nodes)

    S   = {top}
    out = []
    W   = 56

    # ── Step 0: initialization ────────────────────────────────
    out.append(f"Step 0  Initialise — materialize top cube '{top}'  (n−1 = {n-1} cube(s) to add)")
    out.append(f"  S' = {{{top}}}")
    out.append(_cost_matrix(non_top, sorted(S), costs, anc, caption="Initial query costs"))

    # ── Greedy iterations ─────────────────────────────────────
    iteration = 0
    while len(S) < n:
        iteration += 1
        candidates = sorted(nodes_set - S)

        # Compute benefit for every candidate
        bens: dict[str, int] = {}
        for v in candidates:
            b, _ = _benefit(v, S, nodes, costs, anc)
            bens[v] = b

        # Pick max benefit; break ties alphabetically (B before D when equal)
        best = max(candidates, key=lambda v: (bens[v], [-ord(c) for c in v]))

        out.append("")
        out.append(f"Step {iteration}  Benefit matrix  (S' has {len(S)} cube(s), need {n - len(S)} more)")
        out.append(_benefit_matrix(candidates, non_top, S, costs, anc, best))
        out.append(f"  → Select '{best}'  (total benefit = {bens[best]})")

        S.add(best)
        out.append(f"  S' = {{{', '.join(sorted(S))}}}")
        out.append(_cost_matrix(non_top, sorted(S), costs, anc,
                                caption=f"Query costs after adding '{best}'"))

    # ── Final summary ─────────────────────────────────────────
    qc_final = _all_qcosts(nodes, S, costs, anc)
    total    = sum(qc_final.values())

    out.append("")
    out.append("═" * W)
    out.append(f"S' = {{{', '.join(sorted(S))}}}")
    out.append("═" * W)
    out.append("")
    out.append("Final query cost matrix:")
    out.append(_cost_matrix(non_top, sorted(S), costs, anc, caption=""))
    out.append(f"  Total query cost = {total}")
    out.append("")
    out.append("Summary of greedy choices:")
    out.append(f"  Step 0: initialize with '{top}' (always materialised)")
    for i, step in enumerate(
        [s for s in _replay_steps(top, nodes, nodes_set, costs, hierarchy, anc, n)], 1
    ):
        out.append(f"  Step {i}: add '{step['added']}'  (benefit {step['benefit']})")

    return "\n".join(out)


# ── formatting helpers ─────────────────────────────────────────

def _col_w(labels: list[str], min_w: int = 4) -> int:
    return max(min_w, max((len(s) for s in labels), default=min_w))


def _mat_sep(rw: int, col_labels: list[str], cw: int, tw: int) -> str:
    """Separator row aligned with the matrix header."""
    mid = '─' * ((cw + 1) * len(col_labels))
    return f"  {'─'*rw}─+─{mid}+─{'─'*tw}"


def _cost_matrix(
    non_top: list[str],       # row cubes (all except top)
    S: list[str],             # column cubes (current materialised set, sorted)
    costs: dict[str, int],
    anc: dict[str, set[str]],
    caption: str,
) -> str:
    """
    Query-cost matrix.

    Rows = non-top cubes.
    Cols = every materialised cube in S.
    Cell = costs[col] when col is an ancestor of row, else '-'.
    Last column = effective query cost (minimum reachable source).
    Materialised cubes on the diagonal or above are shown with *.
    """
    rw = _col_w(non_top + ['Cube'], 4)
    cw = _col_w(S + [str(max(costs.values(), default=0))], 4)
    tw = _col_w([str(max(costs.values(), default=0)), 'Best'], 4)
    sep = _mat_sep(rw, S, cw, tw)

    lines = []
    if caption:
        lines.append(f"  {caption}:")

    # Header
    hdr = f"  {'Cube':{rw}} |"
    for c in S:
        hdr += f" {c:>{cw}}"
    hdr += f" | {'Best':>{tw}}"
    lines.append(hdr)
    lines.append(sep)

    for v in non_top:
        row = f"  {v:{rw}} |"
        best_cost = None
        cells: list[str] = []
        for c in S:
            if c in anc[v]:          # c is an ancestor of v → reachable
                val = costs[c]
                cells.append(str(val))
                if best_cost is None or val < best_cost:
                    best_cost = val
            else:
                cells.append('-')
        for cell in cells:
            row += f" {cell:>{cw}}"
        bc_str = str(best_cost) if best_cost is not None else '?'
        mat_mark = ' *' if v in S else ''   # v itself is materialised
        row += f" | {bc_str:>{tw}}{mat_mark}"
        lines.append(row)

    lines.append(sep)
    lines.append("  (* = cube is itself materialised in S')")
    return "\n".join(lines)


def _benefit_matrix(
    candidates: list[str],    # row cubes (sorted later by benefit)
    col_nodes: list[str],     # column cubes (all non-top, fixed across steps)
    S: set[str],
    costs: dict[str, int],
    anc: dict[str, set[str]],
    best: str,
) -> str:
    """
    Benefit matrix.

    Rows = candidate cubes (not yet in S'), sorted by total benefit descending.
    Cols = all non-top cubes (consistent across every iteration).
    Cell = gain that adding this candidate provides to that cube:
           '-'  when the candidate is not an ancestor of the cube (unreachable),
           '0'  when ancestor but not cheaper than the current query cost,
           gain otherwise.
    Last column = total benefit (sum of gains in row).
    """
    # Current query costs for every column cube
    qc = {v: _qcost(v, S, costs, anc) for v in col_nodes}

    # Build rows
    row_data: list[tuple[str, list[str], int]] = []
    for v in candidates:
        cells: list[str] = []
        total = 0
        for c in col_nodes:
            if v not in anc[c]:                  # v cannot supply c
                cells.append('-')
            else:
                gain = max(0, qc[c] - costs[v])
                total += gain
                cells.append('0' if gain == 0 else str(gain))
        row_data.append((v, cells, total))

    # Sort: highest total benefit first, then alphabetical
    row_data.sort(key=lambda r: (-r[2], r[0]))

    rw  = _col_w([r[0] for r in row_data] + ['Candidate'], 4)
    cw  = _col_w(col_nodes + [str(max(costs.values(), default=0))], 4)
    tw  = _col_w([str(r[2]) for r in row_data] + ['Total'], 5)
    sep = _mat_sep(rw, col_nodes, cw, tw)

    lines = []

    # Header
    hdr = f"  {'Candidate':{rw}} |"
    for c in col_nodes:
        hdr += f" {c:>{cw}}"
    hdr += f" | {'Total':>{tw}}"
    lines.append(hdr)
    lines.append(sep)

    for v, cells, total in row_data:
        row = f"  {v:{rw}} |"
        for cell in cells:
            row += f" {cell:>{cw}}"
        marker = '  ◄ best' if v == best else ''
        row += f" | {total:>{tw}}{marker}"
        lines.append(row)

    lines.append(sep)
    lines.append("  ('-' = unreachable path,  '0' = reachable but no improvement)")
    return "\n".join(lines)


def _replay_steps(top, nodes, nodes_set, costs, hierarchy, anc, n):
    """Re-run the algorithm to collect step metadata for the summary."""
    S   = {top}
    rev = _reverse(hierarchy)
    steps = []
    while len(S) < n:
        candidates = sorted(nodes_set - S)
        bens = {}
        for v in candidates:
            b, _ = _benefit(v, S, nodes, costs, anc)
            bens[v] = b
        best = max(candidates, key=lambda v: (bens[v], [-ord(c) for c in v]))
        steps.append({'added': best, 'benefit': bens[best]})
        S.add(best)
    return steps

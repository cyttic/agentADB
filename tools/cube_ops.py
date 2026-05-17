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

    if n > len(nodes):
        n = len(nodes)

    S    = {top}
    out  = []
    W    = 56   # column width for separator lines

    # ── Step 0: initialization ────────────────────────────────
    qc = _all_qcosts(nodes, S, costs, anc)
    out.append(f"Step 0  Initialise — materialize top cube '{top}'  (n−1 = {n-1} cube(s) to add)")
    out.append(f"  S' = {{{top}}}")
    out.append(_cost_table(nodes, S, qc, costs, anc, caption="Initial query costs"))

    # ── Greedy iterations ─────────────────────────────────────
    iteration = 0
    while len(S) < n:
        iteration += 1
        candidates = sorted(nodes_set - S)

        # Compute benefit for every candidate
        bens: dict[str, int] = {}
        dets: dict[str, list] = {}
        for v in candidates:
            b, d = _benefit(v, S, nodes, costs, anc)
            bens[v] = b
            dets[v] = d

        # Pick max benefit; break ties alphabetically (consistent with sorted())
        best = max(candidates, key=lambda v: (bens[v], [-ord(c) for c in v]))
        # Note: second key reverses alphabetical so 'B' beats 'D' when equal

        out.append("")
        out.append(f"Step {iteration}  Compute benefits  (S' has {len(S)} cube(s), need {n - len(S)} more)")
        out.append(_benefit_table(candidates, bens, dets, costs, best))
        out.append(f"  → Select '{best}'  (benefit = {bens[best]})")

        S.add(best)
        qc = _all_qcosts(nodes, S, costs, anc)
        out.append(f"  S' = {{{', '.join(sorted(S))}}}")
        out.append(_cost_table(nodes, S, qc, costs, anc,
                               caption=f"Query costs after adding '{best}'"))

    # ── Final summary ─────────────────────────────────────────
    qc_final = _all_qcosts(nodes, S, costs, anc)
    total    = sum(qc_final.values())

    out.append("")
    out.append("═" * W)
    out.append(f"S' = {{{', '.join(sorted(S))}}}")
    out.append("═" * W)
    out.append("")
    out.append("Final query cost table:")
    out.append(_cost_table(nodes, S, qc_final, costs, anc, caption=""))
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

def _source_str(v: str, S: set[str], qc: dict[str, int],
                costs: dict[str, int], anc: dict[str, set[str]]) -> str:
    """Which materialised ancestor(s) give the minimum query cost for v."""
    c = qc[v]
    srcs = sorted(u for u in (anc[v] & S) if costs[u] == c)
    return "/".join(srcs)


def _cost_table(nodes: list[str], S: set[str], qc: dict[str, int],
                costs: dict[str, int], anc: dict[str, set[str]],
                caption: str) -> str:
    lines = []
    if caption:
        lines.append(f"  {caption}:")
    lines.append(f"  {'Cube':<6} {'Own cost':>9} {'Query cost':>11}  Source")
    lines.append(f"  {'─'*6} {'─'*9} {'─'*11}  {'─'*12}")
    for v in nodes:
        star = " *" if v in S else ""
        src  = _source_str(v, S, qc, costs, anc)
        lines.append(f"  {v:<6} {costs.get(v,'?'):>9} {qc[v]:>11}  {src}{star}")
    lines.append(f"  (* = materialised)")
    return "\n".join(lines)


def _benefit_table(candidates: list[str], bens: dict[str, int],
                   dets: dict[str, list], costs: dict[str, int],
                   best: str) -> str:
    lines = []
    lines.append(f"  {'Cube':<6} {'Own cost':>9} {'Benefit':>8}  Gain breakdown")
    lines.append(f"  {'─'*6} {'─'*9} {'─'*8}  {'─'*28}")
    for v in sorted(candidates, key=lambda x: -bens[x]):
        breakdown = ",  ".join(
            f"{w}: {old}→{new} (+{old-new})"
            for w, old, new in sorted(dets[v])
        )
        marker = "  ◄ best" if v == best else ""
        lines.append(f"  {v:<6} {costs.get(v,'?'):>9} {bens[v]:>8}  {breakdown}{marker}")
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

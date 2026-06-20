"""
tools/pagerank_ops.py
======================
Deterministic helpers for the PageRank agent.

Stage 1 (this file) turns a drawn / parsed link graph into the square link
(transition) matrix that the PageRank iteration starts from.

Every page is both a row and a column. A page passes its WHOLE rank to the
pages it links to, split EQUALLY, so:

    cell(row, col) = 1 / out-degree(row)   if row links to col
                   = 0                      otherwise

Each non-dangling row therefore sums to 1 (the page distributes exactly one
unit of rank). A page with out-degree 0 is "dangling" — its row is all zeros.

PageRank needs, for every page:
  - which pages it links TO   (out-links)        → spreads its rank
  - its out-degree            (number of links)  → the 1/out-deg share
  - which pages link TO it    (in-links)         → where its rank comes from

A page with out-degree 0 is a "dangling" page; it is flagged because dangling
pages need special handling in later PageRank stages.

The LLM never computes any of this — it only supplies the parsed graph.
"""

import json
from fractions import Fraction


def _parse_nodes(nodes_json: str) -> list[str]:
    data = json.loads(nodes_json)
    if not isinstance(data, list):
        raise ValueError('nodes_json must be a JSON array of page names')
    nodes: list[str] = []
    for raw in data:
        name = str(raw).strip()
        if name and name not in nodes:
            nodes.append(name)
    return nodes


def _parse_edges(edges_json: str) -> list[tuple[str, str]]:
    # Every occurrence is kept (no de-duplication) so a repeated link counts
    # more than once in the matrix cell.
    data = json.loads(edges_json)
    if not isinstance(data, list):
        raise ValueError('edges_json must be a JSON array of [src, tgt] links')
    edges: list[tuple[str, str]] = []
    for raw in data:
        if isinstance(raw, dict):
            src, tgt = raw.get('src'), raw.get('tgt')
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            src, tgt = raw
        else:
            raise ValueError(f'each link must be [src, tgt] or {{"src","tgt"}}, got: {raw!r}')
        src, tgt = str(src).strip(), str(tgt).strip()
        if not src or not tgt:
            continue
        edges.append((src, tgt))
    return edges


def _fmt_share(f: Fraction) -> str:
    """Render a rank share as a compact reduced fraction: 0, 1, or p/q."""
    if f == 0:
        return '0'
    if f.denominator == 1:
        return str(f.numerator)
    return f'{f.numerator}/{f.denominator}'


def _link_structure(nodes_json: str, edges_json: str):
    """
    Parse the graph once into the pieces both the matrix and the iteration share.

    Returns (nodes, edges, counts, outdeg):
      nodes  — page names in placement order (incl. any seen only in an edge)
      edges  — [(src, tgt), ...] every occurrence kept
      counts — {(row, col): number of links row → col}
      outdeg — {page: total links leaving it}
    """
    nodes = _parse_nodes(nodes_json)

    # Any node that appears only inside an edge still counts as a page.
    edges = _parse_edges(edges_json)
    for src, tgt in edges:
        for name in (src, tgt):
            if name not in nodes:
                nodes.append(name)

    counts: dict[tuple[str, str], int] = {}
    outdeg: dict[str, int] = {n: 0 for n in nodes}
    for src, tgt in edges:
        counts[(src, tgt)] = counts.get((src, tgt), 0) + 1
        outdeg[src] += 1

    return nodes, edges, counts, outdeg


def build_link_table(nodes_json: str, edges_json: str) -> str:
    """
    Build the PageRank link (transition) matrix from a parsed link graph.

    nodes_json : JSON array of page names           e.g. ["A","B","C"]
    edges_json : JSON array of directed links        e.g. [["A","B"],["A","C"],["C","A"]]
                 (each link is src → tgt: src has a link pointing to tgt)

    The matrix is square — every page is both a row and a column. A page passes
    its whole rank to the pages it links to, split equally, so cell (r, c) =
    (links r→c) / out-degree(r), i.e. 1/out-degree(r) for each page r links to,
    and 0 otherwise. Every non-dangling row sums to 1. Returns a plain-text,
    column-aligned matrix of reduced fractions.
    """
    nodes, edges, counts, outdeg = _link_structure(nodes_json, edges_json)

    def share(r: str, c: str) -> Fraction:
        d = outdeg[r]
        return Fraction(counts.get((r, c), 0), d) if d else Fraction(0)

    def cell(r: str, c: str) -> str:
        return _fmt_share(share(r, c))

    # ── Column widths: each column is as wide as its page name or any cell ──
    corner = 'from \\ to'
    row_hdr_w = max([len(corner)] + [len(n) for n in nodes])
    col_w = {
        c: max(len(c), *(len(cell(r, c)) for r in nodes)) if nodes else len(c)
        for c in nodes
    }

    def fmt_row(label: str, cells: list[str]) -> str:
        body = ' | '.join(cells[i].center(col_w[c]) for i, c in enumerate(nodes))
        return f'{label.ljust(row_hdr_w)} | {body}'

    header = fmt_row(corner, list(nodes))
    sep = '-' * row_hdr_w + '-+-' + '-+-'.join('-' * col_w[c] for c in nodes)

    out = []
    out.append('PageRank — link matrix (cell = share of the row page\'s rank sent to the column page = 1/out-degree)')
    out.append(f'Pages: {len(nodes)}    Links: {len(edges)}')
    out.append('')
    out.append(header)
    out.append(sep)
    for r in nodes:
        out.append(fmt_row(r, [cell(r, c) for c in nodes]))
    out.append('')
    out.append('Out-degree per page (each out-link gets a 1/out-degree share):')
    out.append('  ' + ', '.join(f'{r}={outdeg[r]}' for r in nodes))
    dangling = [r for r in nodes if outdeg[r] == 0]
    if dangling:
        out.append(f'Dangling pages (empty row, out-degree 0): {", ".join(dangling)}')
    return '\n'.join(out)


# ══════════════════════════════════════════════════════════════
#  PAGERANK ITERATION
# ══════════════════════════════════════════════════════════════

_MAX_ITERS = 100          # hard cap when running "until convergence"
_DECIMALS  = 6            # display precision
# "No change in the rank" is judged at display precision: the loop stops once
# every page's rank rounds to the same value as the previous iteration.
_TOL       = 0.5 * 10 ** (-_DECIMALS)


def run_pagerank(nodes_json: str, edges_json: str, d: float,
                 iterations: int = 0, initial_ranks_json: str = "") -> str:
    """
    Run the PageRank iteration on the link graph and return the full trace.

    Formula applied every iteration, for every page j:
        P[j] = d / N + (1 - d) * Σ_i  T[i, j] * P[i]
    where N is the number of pages, T[i, j] = 1/out-degree(i) if i links to j
    (else 0), and P[i] is the current rank of page i.

    nodes_json / edges_json : the link graph (same format as build_link_table).
    d                       : the constant from the task (the "d" in the formula).
    iterations              : how many iterations to run. Pass 0 (or a negative
                              number) to iterate until the ranks stop changing.
    initial_ranks_json      : optional JSON object {page: rank} for the starting
                              ranks. If empty, every page starts at 1/N.

    Returns a plain-text trace: the per-iteration rank table, the final ranks,
    and the resulting page ordering.
    """
    nodes, edges, counts, outdeg = _link_structure(nodes_json, edges_json)
    N = len(nodes)
    if N == 0:
        return 'No pages to rank.'

    d = float(d)

    d = float(d)

    def num(v: float) -> str:
        return f'{v:.{_DECIMALS}f}'

    # ── Starting ranks ──
    if initial_ranks_json and initial_ranks_json.strip():
        raw = json.loads(initial_ranks_json)
        if not isinstance(raw, dict):
            raise ValueError('initial_ranks_json must be a JSON object {page: rank}')
        P = {n: float(raw.get(n, 0.0)) for n in nodes}
        start_label = 'Initial ranks (from the task)'
    else:
        P = {n: 1.0 / N for n in nodes}
        start_label = f'Initial ranks (1/N = 1/{N})'

    # incoming[j] = [(i, T[i,j]=share i→j), ...] for every page i that links to j
    order = {n: k for k, n in enumerate(nodes)}
    incoming: dict[str, list[tuple[str, Fraction]]] = {n: [] for n in nodes}
    for (i, j), c in counts.items():
        incoming[j].append((i, Fraction(c, outdeg[i])))
    for j in nodes:
        incoming[j].sort(key=lambda t: order[t[0]])

    dN = d / N
    one_minus = 1.0 - d

    def iterate_once(prev: dict[str, float]):
        """Return (next ranks, one printed line per page showing the calculation)."""
        cur, lines = {}, []
        for j in nodes:
            terms = incoming[j]
            s = sum(float(share) * prev[i] for i, share in terms)
            cur[j] = dN + one_minus * s
            if terms:
                term_str = ' + '.join(
                    f'({_fmt_share(share)})*{num(prev[i])}[{i}]' for i, share in terms
                )
            else:
                term_str = '0'
            lines.append(f'  P[{j}] = {num(dN)} + {num(one_minus)}*( {term_str} ) = {num(cur[j])}')
        return cur, lines

    fixed = iterations is not None and iterations > 0

    out = []
    out.append(f'PageRank iteration  (d = {d:g}, N = {N})')
    out.append('Formula:  P[j] = d/N + (1-d) * Σ_i T[i,j]*P[i]')
    out.append(f'{start_label}: ' + ', '.join(f'P[{n}]={num(P[n])}' for n in nodes))
    out.append('')

    prev = dict(P)
    performed = 0
    converged = False
    while True:
        cur, lines = iterate_once(prev)
        performed += 1
        out.append(f'Iteration {performed}:')
        out.extend(lines)
        out.append('')
        change = max(abs(cur[n] - prev[n]) for n in nodes)
        prev = cur
        if fixed:
            if performed >= iterations:
                break
        else:
            if change < _TOL:
                converged = True
                break
            if performed >= _MAX_ITERS:
                break

    if fixed:
        out.append(f'Performed {iterations} iteration(s) as requested by the task.')
    elif converged:
        out.append(f'Converged after {performed} iteration(s) — no further change in the rank.')
    else:
        out.append(f'Stopped at the safety cap of {_MAX_ITERS} iterations (not yet converged).')

    # ── Final rank + most important page ──
    final = prev
    out.append('')
    out.append('Final rank:')
    for n in nodes:
        out.append(f'  P[{n}] = {num(final[n])}')

    best = max(final.values())
    winners = [n for n in nodes if abs(final[n] - best) < _TOL]
    if len(winners) == 1:
        out.append(f'Most important page: {winners[0]} (P[{winners[0]}] = {num(best)})')
    else:
        out.append(f'Most important pages (tie): {", ".join(winners)} (P = {num(best)})')
    return '\n'.join(out)

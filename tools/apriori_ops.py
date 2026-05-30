"""
tools/apriori_ops.py
====================
Deterministic implementation of the Apriori algorithm for frequent-itemset
mining, with a fully verbose step-by-step trace ready to be shown to the user.

Method (matches the lecture variant)
-------------------------------------
Given a transaction table (TID → set of items) and a minimum support
threshold S (a fraction in [0, 1]):

  Pass 1:  C1 = every single item.
           F1 = items whose support ≥ S.

  Pass k (k ≥ 2):
     Candidate generation  Ck:
        take every unordered pair {p, q} of itemsets in F(k-1),
        form the union p ∪ q, and KEEP it only if |p ∪ q| = k.
        (a union of size ≠ k is discarded — marked ✗)
     Support counting:
        Fk = candidates in Ck whose support ≥ S.

  Stop when Fk = ∅.

Result = F1 ∪ F2 ∪ … ∪ F(k-1)  (all frequent itemsets found).

Support(X) = (number of transactions containing all items of X) / N,
where N is the total number of transactions.

All ordering is alphabetical → deterministic, reproducible output.
"""

import json
from itertools import combinations


# ── formatting helpers ──────────────────────────────────────────

def _fmt_ratio(value: float) -> str:
    """Clean decimal: 0.75 -> '0.75', 0.5 -> '0.5', 0.333.. -> '0.33'."""
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _itemset_str(items) -> str:
    """frozenset -> '{A, B, C}' with alphabetical order."""
    return "{" + ", ".join(sorted(items)) + "}"


def _set_of_sets_str(sets) -> str:
    """Collection of itemsets -> '{ {A}, {B}, {A, B} }', sorted."""
    ordered = sorted(sets, key=lambda s: (len(s), sorted(s)))
    return "{ " + ", ".join(_itemset_str(s) for s in ordered) + " }" if ordered else "∅"


# ── parsing ─────────────────────────────────────────────────────

def _parse_transactions(transactions_json: str):
    """
    Accepts either:
      - a JSON object  {"1": ["A","B"], "2": ["A","B","C"], ...}
      - a JSON array   [["A","B"], ["A","B","C"], ...]  (TIDs assigned 1..n)
    Returns an ordered list of (tid, frozenset(items)).
    """
    data = json.loads(transactions_json)

    rows = []
    if isinstance(data, dict):
        # preserve numeric TID ordering when possible
        def _key(k):
            try:
                return (0, int(k))
            except (TypeError, ValueError):
                return (1, str(k))
        for tid in sorted(data.keys(), key=_key):
            items = [str(x).strip() for x in data[tid]]
            rows.append((str(tid), frozenset(items)))
    elif isinstance(data, list):
        for i, items in enumerate(data, 1):
            items = [str(x).strip() for x in items]
            rows.append((str(i), frozenset(items)))
    else:
        raise ValueError("transactions must be a JSON object or array")

    if not rows:
        raise ValueError("no transactions provided")
    return rows


# ── support ─────────────────────────────────────────────────────

def _support_count(itemset, transactions):
    """Number of transactions that contain every item of itemset."""
    return sum(1 for _tid, items in transactions if itemset <= items)


# ── candidate generation ────────────────────────────────────────

def _generate_candidates(prev_frequent, k):
    """
    Ck via the join F(k-1) ⋈ F(k-1):
    union every unordered pair, keep unions of size exactly k.
    Returns (candidates_sorted, join_log) where join_log is a list of
    (left, right, union, kept_bool) tuples for the verbose trace.
    """
    join_log = []
    candidates = set()
    prev_sorted = sorted(prev_frequent, key=lambda s: sorted(s))
    for left, right in combinations(prev_sorted, 2):
        union = left | right
        kept = (len(union) == k)
        join_log.append((left, right, union, kept))
        if kept:
            candidates.add(union)
    return sorted(candidates, key=lambda s: sorted(s)), join_log


# ── main algorithm ──────────────────────────────────────────────

def run_apriori(transactions_json: str,
                min_support: float,
                min_confidence: float | None = None) -> str:
    """
    Run Apriori and return a complete, formatted step-by-step solution string.

    Args:
        transactions_json: JSON object {tid: [items]} or array [[items], ...].
        min_support:       support threshold S, a fraction in [0, 1]
                           (an item/itemset is frequent when support ≥ S).
        min_confidence:    optional confidence threshold D, only echoed in the
                           parameter banner (used for association rules later).
    """
    transactions = _parse_transactions(transactions_json)
    N = len(transactions)
    S = float(min_support)

    out = []
    W = 64

    # ── echo the input table verbatim ─────────────────────────
    item_col = max([len("Items")] +
                   [len(", ".join(sorted(items))) for _tid, items in transactions])
    tid_col = max([len("TID")] + [len(tid) for tid, _ in transactions])

    out.append("Input transaction table:")
    out.append(f"  {'TID':<{tid_col}} | Items")
    out.append(f"  {'-' * tid_col}-+-{'-' * item_col}")
    for tid, items in transactions:
        out.append(f"  {tid:<{tid_col}} | {', '.join(sorted(items))}")
    out.append("")

    out.append(f"Parameters:")
    out.append(f"  N = {N}  (total transactions)")
    out.append(f"  S = {_fmt_ratio(S)}  (minimum support — itemset frequent when support ≥ S)")
    if min_confidence is not None:
        out.append(f"  D = {_fmt_ratio(float(min_confidence))}  (minimum confidence — for association rules)")
    out.append("")

    all_frequent = []          # list of (itemset, support_count)
    frequent_levels = {}       # k -> list[frozenset]

    # ── Pass 1 ────────────────────────────────────────────────
    items = sorted({it for _tid, row in transactions for it in row})
    out.append("═" * W)
    out.append("Pass 1 — frequent 1-itemsets")
    out.append("═" * W)
    c1 = [frozenset([it]) for it in items]
    out.append(f"C1 = {_set_of_sets_str(c1)}")
    out.append("Support counting:")

    f1 = []
    for cand in c1:
        cnt = _support_count(cand, transactions)
        sup = cnt / N
        ok = sup >= S
        mark = f"≥ {_fmt_ratio(S)}  → frequent" if ok else f"< {_fmt_ratio(S)}  → pruned"
        out.append(f"  Support({_itemset_str(cand)}) = {cnt}/{N} = {_fmt_ratio(sup)}  {mark}")
        if ok:
            f1.append(cand)
            all_frequent.append((cand, cnt))
    out.append(f"F1 = {_set_of_sets_str(f1)}")
    out.append("")

    frequent_levels[1] = f1
    prev_frequent = f1
    k = 2

    # ── Passes k ≥ 2 ──────────────────────────────────────────
    while len(prev_frequent) >= 2:
        candidates, join_log = _generate_candidates(prev_frequent, k)

        out.append("═" * W)
        out.append(f"Pass {k} — frequent {k}-itemsets")
        out.append("═" * W)
        out.append(f"Candidate generation C{k}  (join F{k-1} ⋈ F{k-1}, "
                   f"keep unions of size {k}):")
        for left, right, union, kept in join_log:
            tag = f"size {len(union)}  ✓" if kept else f"size {len(union)}  ✗ (discard)"
            out.append(f"  {_itemset_str(left)} ∪ {_itemset_str(right)} "
                       f"= {_itemset_str(union)}   {tag}")
        out.append(f"C{k} = {_set_of_sets_str(candidates)}")

        if not candidates:
            out.append(f"F{k} = ∅   → stop")
            out.append("")
            frequent_levels[k] = []
            break

        out.append("Support counting:")
        fk = []
        for cand in candidates:
            cnt = _support_count(cand, transactions)
            sup = cnt / N
            ok = sup >= S
            mark = f"≥ {_fmt_ratio(S)}  → frequent" if ok else f"< {_fmt_ratio(S)}  → pruned"
            out.append(f"  Support({_itemset_str(cand)}) = {cnt}/{N} = {_fmt_ratio(sup)}  {mark}")
            if ok:
                fk.append(cand)
                all_frequent.append((cand, cnt))
        stop = "   → stop" if not fk else ""
        out.append(f"F{k} = {_set_of_sets_str(fk)}{stop}")
        out.append("")

        frequent_levels[k] = fk
        if not fk:
            break
        prev_frequent = fk
        k += 1

    # ── Result ────────────────────────────────────────────────
    out.append("═" * W)
    out.append("Result — all frequent itemsets")
    out.append("═" * W)
    levels = [lvl for lvl in sorted(frequent_levels) if frequent_levels[lvl]]
    union_expr = " ∪ ".join(f"F{lvl}" for lvl in levels) if levels else "∅"
    every = [s for lvl in levels for s in
             sorted(frequent_levels[lvl], key=lambda s: sorted(s))]
    out.append(f"{union_expr} =")
    out.append(f"  {_set_of_sets_str(every)}")
    out.append(f"  ({len(every)} frequent itemset(s) in total)")

    return "\n".join(out)

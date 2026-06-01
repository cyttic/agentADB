"""
tools/apriori_tid_ops.py
========================
Deterministic implementation of the Apriori-TID algorithm for frequent-itemset
mining, with a fully verbose step-by-step trace ready to be shown to the user.

Difference from plain Apriori
-----------------------------
Instead of rescanning the whole transaction table every pass, Apriori-TID keeps
for each itemset I a tid_list — the set of TIDs of the transactions that
contain I. Support is read straight off that list:

    Support(I) = |I.tid_list| / |D|        (|D| = number of transactions)

A k-itemset's tid_list is the INTERSECTION of the tid_lists of the two
(k-1)-itemsets that generated it (equivalently, the intersection of all its
single-item tid_lists). Example:

    {A,B}.tid_list = {A}.tid_list ∩ {B}.tid_list = {2,3,4} ∩ {3,4} = {3,4}

Candidate generation is the same join as Apriori: union every unordered pair
of itemsets in F(k-1) and keep the unions of size exactly k. Stop when
Fk = ∅. Result = F1 ∪ F2 ∪ … (all frequent itemsets found).

Shared helpers (parsing, candidate generation, formatting) are imported from
tools.apriori_ops so both data-mining agents stay in lock-step.
"""

from tools.apriori_ops import (
    _parse_transactions,
    _generate_candidates,
    _fmt_ratio,
    _itemset_str,
    _set_of_sets_str,
)


# ── tid helpers ─────────────────────────────────────────────────

def _tid_key(t):
    """Sort TIDs numerically when possible, else lexicographically."""
    try:
        return (0, int(t))
    except (TypeError, ValueError):
        return (1, str(t))


def _tids_str(tids) -> str:
    """A tid set -> '{1, 2, 3}' (numeric order when possible)."""
    return "{" + ", ".join(str(t) for t in sorted(tids, key=_tid_key)) + "}"


# ── main algorithm ──────────────────────────────────────────────

def run_apriori_tid(transactions_json: str,
                    min_support: float,
                    min_confidence: float | None = None) -> str:
    """
    Run Apriori-TID and return a complete, formatted step-by-step solution.

    Args:
        transactions_json: JSON object {tid: [items]} or array [[items], ...].
        min_support:       support threshold S, a fraction in [0, 1]
                           (an itemset is frequent when support ≥ S).
        min_confidence:    optional confidence threshold D, only echoed in the
                           parameter banner.
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

    out.append("Parameters:")
    out.append(f"  |D| = {N}  (number of transactions)")
    out.append(f"  S = {_fmt_ratio(S)}  (minimum support — itemset frequent when support ≥ S)")
    if min_confidence is not None:
        out.append(f"  D = {_fmt_ratio(float(min_confidence))}  (minimum confidence — for association rules)")
    out.append("  Support(I) = |I.tid_list| / |D|")
    out.append("")

    # ── condition: compact restatement of the inputs ──────────
    # |D| = table size, S from the conditions, and each item's tid_list
    # (the rows where that item appears).
    items = sorted({it for _tid, row in transactions for it in row})
    item_tids = {it: {tid for tid, row in transactions if it in row} for it in items}
    cond_parts = [f"|D| = {N}", f"S = {_fmt_ratio(S)}"]
    cond_parts += [f"{it}.tid_list = {_tids_str(item_tids[it])}" for it in items]
    out.append("Condition:")
    out.append(f"  Apriori-TID({', '.join(cond_parts)})")
    out.append("")

    tidmap = {}                # frozenset(itemset) -> set(tids), frequent only
    frequent_levels = {}       # k -> list[frozenset]

    # ── Step 1: singleton tid_lists ───────────────────────────
    out.append("═" * W)
    out.append("Step 1 — singletons: build a tid_list for every item")
    out.append("═" * W)
    c1 = [frozenset([it]) for it in items]
    out.append(f"C1 = {_set_of_sets_str(c1)}")
    out.append("")

    f1 = []
    for cand in c1:
        tids = item_tids[next(iter(cand))]
        sup = len(tids) / N
        ok = sup >= S
        verdict = "→ frequent" if ok else "→ pruned"
        out.append(f"  {_itemset_str(cand)}.tid_list = {_tids_str(tids)}")
        out.append(f"      Support = |{_tids_str(tids)}| / {N} = {len(tids)}/{N} "
                   f"= {_fmt_ratio(sup)}  {'≥' if ok else '<'} {_fmt_ratio(S)}  {verdict}")
        if ok:
            f1.append(cand)
            tidmap[cand] = tids
    out.append(f"F1 = {_set_of_sets_str(f1)}")
    out.append("")

    frequent_levels[1] = f1
    prev_frequent = f1
    k = 2

    # ── Steps k ≥ 2 ───────────────────────────────────────────
    while len(prev_frequent) >= 2:
        candidates, join_log = _generate_candidates(prev_frequent, k)

        # first generating pair for each kept candidate → used for intersection
        first_pair = {}
        for left, right, union, kept in join_log:
            if kept and union not in first_pair:
                first_pair[union] = (left, right)

        out.append("═" * W)
        out.append(f"Step {k} — candidate {k}-itemsets")
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

        out.append("")
        out.append("tid_list by intersection of parents, then support:")
        fk = []
        for cand in candidates:
            left, right = first_pair[cand]
            tl, tr = tidmap[left], tidmap[right]
            inter = tl & tr
            sup = len(inter) / N
            ok = sup >= S
            verdict = "→ frequent" if ok else "→ pruned"
            out.append(
                f"  {_itemset_str(cand)}.tid_list = "
                f"{_itemset_str(left)}.tid_list ∩ {_itemset_str(right)}.tid_list = "
                f"{_tids_str(tl)} ∩ {_tids_str(tr)} = {_tids_str(inter)}"
            )
            out.append(
                f"      Support = {len(inter)}/{N} = {_fmt_ratio(sup)}  "
                f"{'≥' if ok else '<'} {_fmt_ratio(S)}  {verdict}"
            )
            if ok:
                fk.append(cand)
                tidmap[cand] = inter
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

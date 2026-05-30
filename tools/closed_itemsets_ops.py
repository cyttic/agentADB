"""
tools/closed_itemsets_ops.py
============================
Deterministic computation of all CLOSED frequent itemsets from a transaction
table, with a verbose step-by-step trace ready to be shown to the user.

Definition
----------
A frequent itemset I (support(I) ≥ S) is CLOSED frequent if every proper
superset J ⊃ I has STRICTLY smaller support:

      I closed  ⇔  support(I) ≥ S  and  ∀ J ⊃ I : support(J) < support(I)

A superset always has support ≤ its subset, so this is equivalent to: no
immediate superset I ∪ {x} (x ∉ I) has the SAME support as I. (If a superset
keeps the same support, I is not closed — its support is "carried" by a larger
itemset.)

Frequent itemsets are found with the shared Apriori machinery
(_frequent_itemsets from tools.association_rules_ops).
"""

from tools.apriori_ops import (
    _parse_transactions,
    _support_count,
    _fmt_ratio,
    _itemset_str,
    _set_of_sets_str,
)
from tools.association_rules_ops import _frequent_itemsets


# ── main ────────────────────────────────────────────────────────

def run_closed_itemsets(transactions_json: str, min_support: float) -> str:
    """
    Find all closed frequent itemsets and return a formatted step-by-step
    solution string.

    Args:
        transactions_json: JSON object {tid: [items]} or array [[items], ...].
        min_support:       support threshold S, a fraction in [0, 1].
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
    out.append(f"  N = {N}  (total transactions)")
    out.append(f"  S = {_fmt_ratio(S)}  (minimum support)")
    out.append("  I is CLOSED frequent if support(I) ≥ S and every proper")
    out.append("  superset J ⊃ I has support(J) < support(I).")
    out.append("")

    # ── frequent itemsets ─────────────────────────────────────
    freq, levels = _frequent_itemsets(transactions, N, S)
    all_items = sorted({it for _tid, row in transactions for it in row})

    out.append("═" * W)
    out.append("Step 1 — frequent itemsets (support ≥ S)")
    out.append("═" * W)
    for lvl in sorted(levels):
        for z in sorted(levels[lvl], key=lambda s: sorted(s)):
            out.append(f"  {_itemset_str(z):<14} support = {freq[z]}/{N} = {_fmt_ratio(freq[z] / N)}")
    if not freq:
        out.append("  (none — no itemset reaches support S)")
    out.append("")

    # ── closedness check ──────────────────────────────────────
    out.append("═" * W)
    out.append("Step 2 — closedness check (compare each immediate superset's support)")
    out.append("═" * W)

    closed = []
    all_freq = sorted(freq.keys(), key=lambda s: (len(s), sorted(s)))
    for I in all_freq:
        cnt_i = freq[I]
        sup_i = cnt_i / N
        out.append(f"{_itemset_str(I)}  (support = {_fmt_ratio(sup_i)}):")
        has_equal_superset = False
        extensions = [x for x in all_items if x not in I]
        if not extensions:
            out.append("  no possible superset (contains all items)")
        for x in extensions:
            J = I | {x}
            cnt_j = _support_count(J, transactions)
            sup_j = cnt_j / N
            equal = cnt_j == cnt_i
            rel = "=" if equal else "<"
            note = f" (= support({_itemset_str(I)}))" if equal else ""
            out.append(f"  {_itemset_str(J):<14} support = {cnt_j}/{N} = "
                       f"{_fmt_ratio(sup_j)}  {rel} {_fmt_ratio(sup_i)}{note}")
            if equal:
                has_equal_superset = True
        if has_equal_superset:
            out.append(f"  → a superset has the same support → NOT closed")
        else:
            out.append(f"  → every superset has smaller support → CLOSED")
            closed.append(I)
        out.append("")

    # ── result ────────────────────────────────────────────────
    out.append("═" * W)
    out.append("Result — closed frequent itemsets")
    out.append("═" * W)
    if closed:
        ordered = sorted(closed, key=lambda s: (len(s), sorted(s)))
        out.append(f"  {_set_of_sets_str(ordered)}")
        out.append(f"  ({len(ordered)} closed frequent itemset(s))")
    else:
        out.append("  ∅  (no frequent itemsets)")

    return "\n".join(out)

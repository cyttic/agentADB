"""
tools/association_rules_ops.py
==============================
Deterministic generation of all association rules from a transaction table,
with a fully verbose step-by-step trace ready to be shown to the user.

Method
------
1. Find every frequent itemset (support ≥ S) with the Apriori passes
   (reusing the shared helpers from tools.apriori_ops).
2. For every frequent itemset Z with |Z| ≥ 2, enumerate every way to split it
   into a non-empty antecedent I and consequent J = Z \\ I (I ∩ J = ∅,
   I ∪ J = Z) — that is 2^|Z| − 2 candidate rules per itemset.
3. Confidence of rule  I → J:

       conf(I → J) = support(I ∪ J) / support(I) = support(Z) / support(I)

   Because both supports share the same |D|, this equals the integer ratio
   support_count(Z) / support_count(I).
4. Keep the rules whose confidence ≥ C.

Result = the set of all rules with confidence ≥ C.
"""

from itertools import combinations
from math import gcd

from tools.apriori_ops import (
    _parse_transactions,
    _generate_candidates,
    _support_count,
    _fmt_ratio,
    _itemset_str,
)


# ── helpers ─────────────────────────────────────────────────────

def _inline(items) -> str:
    """An itemset -> 'A, B, C' (alphabetical, no braces) for rule sides."""
    return ", ".join(sorted(items))


def _rule_str(ante, cons) -> str:
    return f"{_inline(ante)} -> {_inline(cons)}"


def _conf_fraction(count_z: int, count_i: int) -> str:
    """Reduced fraction string for count_z / count_i, e.g. '2/3' or '1'."""
    g = gcd(count_z, count_i)
    a, b = count_z // g, count_i // g
    return f"{a}" if b == 1 else f"{a}/{b}"


def _frequent_itemsets(transactions, N, S):
    """Return {frozenset: support_count} for all frequent itemsets, plus levels."""
    items = sorted({it for _tid, row in transactions for it in row})
    freq = {}
    f_prev = []
    for it in items:
        fs = frozenset([it])
        c = _support_count(fs, transactions)
        # An itemset with zero support is never frequent, even if S = 0.
        if c >= 1 and c / N >= S:
            f_prev.append(fs)
            freq[fs] = c
    levels = {1: list(f_prev)}
    k = 2
    while len(f_prev) >= 2:
        candidates, _log = _generate_candidates(f_prev, k)
        fk = []
        for cand in candidates:
            c = _support_count(cand, transactions)
            if c >= 1 and c / N >= S:
                fk.append(cand)
                freq[cand] = c
        levels[k] = fk
        if not fk:
            break
        f_prev = fk
        k += 1
    return freq, levels


# ── main ────────────────────────────────────────────────────────

def run_association_rules(transactions_json: str,
                          min_support: float,
                          min_confidence: float) -> str:
    """
    Generate all association rules and return a formatted step-by-step solution.

    Args:
        transactions_json: JSON object {tid: [items]} or array [[items], ...].
        min_support:       support threshold S, a fraction in [0, 1].
        min_confidence:    confidence threshold C, a fraction in [0, 1]
                           (a rule is kept when conf ≥ C).
    """
    transactions = _parse_transactions(transactions_json)
    N = len(transactions)
    S = float(min_support)
    C = float(min_confidence)

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
    out.append(f"  C = {_fmt_ratio(C)}  (minimum confidence — rule kept when conf ≥ C)")
    out.append("  conf(I → J) = support(I ∪ J) / support(I) = support(Z) / support(I)")
    out.append("")

    # ── frequent itemsets ─────────────────────────────────────
    freq, levels = _frequent_itemsets(transactions, N, S)

    out.append("═" * W)
    out.append("Step 1 — frequent itemsets (support ≥ S)")
    out.append("═" * W)
    for lvl in sorted(levels):
        if not levels[lvl]:
            continue
        for z in sorted(levels[lvl], key=lambda s: sorted(s)):
            cnt = freq[z]
            out.append(f"  {_itemset_str(z):<14} support = {cnt}/{N} = {_fmt_ratio(cnt / N)}")
    out.append("")

    # ── rule generation ───────────────────────────────────────
    out.append("═" * W)
    out.append("Step 2 — candidate rules from each frequent itemset of size ≥ 2")
    out.append("═" * W)

    kept = []   # list of (ante, cons) frozensets
    big = [z for lvl in sorted(levels) if lvl >= 2 for z in
           sorted(levels[lvl], key=lambda s: sorted(s))]

    if not big:
        out.append("  No frequent itemset of size ≥ 2 → no association rules.")
    else:
        for z in big:
            cnt_z = freq[z]
            out.append(f"From {_itemset_str(z)}  (support = {_fmt_ratio(cnt_z / N)}):")
            ordered = sorted(z)
            # antecedents of every size from 1 to |Z|-1
            for r in range(1, len(z)):
                for ante_t in combinations(ordered, r):
                    ante = frozenset(ante_t)
                    cons = z - ante
                    # antecedent is a subset of frequent Z, so it is frequent
                    # too (support ≥ support(Z) ≥ 1) — guard defensively anyway
                    cnt_i = freq.get(ante) or _support_count(ante, transactions)
                    if cnt_i == 0:
                        continue
                    conf = cnt_z / cnt_i
                    ok = conf >= C
                    verdict = "→ keep" if ok else "→ discard"
                    out.append(
                        f"  {_rule_str(ante, cons):<16} conf = "
                        f"sup({_itemset_str(z)})/sup({_itemset_str(ante)}) = "
                        f"{_fmt_ratio(cnt_z / N)}/{_fmt_ratio(cnt_i / N)} = "
                        f"{_conf_fraction(cnt_z, cnt_i)} ≈ {conf:.2f}  "
                        f"{'≥' if ok else '<'} {_fmt_ratio(C)}  {verdict}"
                    )
                    if ok:
                        kept.append((ante, cons))
            out.append("")

    # ── result ────────────────────────────────────────────────
    out.append("═" * W)
    out.append("Result — association rules with confidence ≥ C")
    out.append("═" * W)
    if kept:
        # semicolons separate rules so they aren't confused with the commas
        # inside multi-item antecedents/consequents (e.g. "B, C -> A")
        rules_str = ";  ".join(_rule_str(a, c) for a, c in kept)
        out.append(f"  {{ {rules_str} }}")
        out.append(f"  ({len(kept)} rule(s))")
    else:
        out.append("  { }  (no rule meets the confidence threshold)")

    return "\n".join(out)

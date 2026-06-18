"""
tools/db_ops.py
================
Pure cost-calculation tools for parallel database operations.

All sizes are expressed in BLOCKS (never bytes — convert before using).
All cost outputs are SYMBOLIC strings like "9 * 10^3 * (t_d + t_s)" — never reduced.
Every number is rendered in scientific style via _fmt (a * 10^k, mantissa in
[1, 10)); no loose number greater than 10 ever appears in the output.

Three atomic operations:
  - Select  — 3 algorithms, decision based on question type + data distribution
  - Sort    — placeholder formula: 3 * t_d * B_s    (to be refined)
  - Join    — placeholder formula: 3 * t_d * (B_s + B_t)  (to be refined)

These can be composed into compound queries (Select-Join, Select-Sort, etc.)
by calling them in sequence and combining their cost outputs.
"""

import math
import json
import re


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def bytes_to_blocks(size_bytes, block_size):
    return math.ceil(size_bytes / block_size)


def records_to_blocks(record_count, record_size_bytes, block_size):
    records_per_block = max(1, block_size // record_size_bytes)
    return math.ceil(record_count / records_per_block)


def _fmt(n: int) -> str:
    """
    Render a number in scientific style: any value > 10 becomes  a * 10^k
    with the mantissa a in [1, 10).  Numbers <= 10 are left as plain digits
    (the course convention forbids loose numbers greater than 10).

        1000   -> 10^3
        1500   -> 1.5 * 10^3
        2000   -> 2 * 10^3
        20000  -> 2 * 10^4
        16500  -> 1.65 * 10^4
        40     -> 4 * 10^1
        10     -> 10
        9      -> 9
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    if n <= 10:                      # 0..10 (and negatives) stay as-is
        return str(n)
    s    = str(n)
    k    = len(s) - 1                # exponent = floor(log10(n)) for a positive int
    frac = s[1:].rstrip("0")         # mantissa digits after the leading one
    if not frac:                     # exact  a * 10^k  with a single-digit mantissa
        return f"10^{k}" if s[0] == "1" else f"{s[0]} * 10^{k}"
    return f"{s[0]}.{frac} * 10^{k}"


def compute_table_blocks_info(
    record_count: int,
    num_attributes: int,
    cell_size_bytes: int,
    block_size_bytes: int,
    table_name: str = "",
) -> dict:
    """
    Calculate block count when it is NOT given directly.

    Formula:
      row_size_bytes   = num_attributes   × cell_size_bytes
      table_size_bytes = record_count     × row_size_bytes
      block_count      = ceil(table_size_bytes / block_size_bytes)
    """
    row_size_bytes   = num_attributes * cell_size_bytes
    table_size_bytes = record_count * row_size_bytes
    block_count      = math.ceil(table_size_bytes / block_size_bytes)

    name = table_name or "T"
    explanation = "\n".join([
        f"Block count calculation for table {name}:",
        f"  Row size   : {_fmt(num_attributes)} attributes × {_fmt(cell_size_bytes)} bytes/cell"
        f" = {_fmt(row_size_bytes)} bytes/row",
        f"  Table size : {_fmt(record_count)} records × {_fmt(row_size_bytes)} bytes/row"
        f" = {_fmt(table_size_bytes)} bytes",
        f"  Block count: ceil({_fmt(table_size_bytes)} / {_fmt(block_size_bytes)})"
        f" = {_fmt(block_count)} blocks",
        f"  !! USE block_count={_fmt(block_count)} IN ALL SUBSEQUENT TOOL CALLS"
        f" (select_cost / join_cost / sort_cost). DO NOT use record_count={_fmt(record_count)}.",
    ])

    return {
        "table_name":       name,
        "num_attributes":   num_attributes,
        "cell_size_bytes":  cell_size_bytes,
        "row_size_bytes":   row_size_bytes,
        "record_count":     record_count,
        "table_size_bytes": table_size_bytes,
        "block_size_bytes": block_size_bytes,
        "block_count":      block_count,
        "explanation":      explanation,
    }


# ══════════════════════════════════════════════════════════════
#  PARSE SCHEMA
# ══════════════════════════════════════════════════════════════

def parse_schema(schema_json):
    """
    Parse raw schema JSON into a canonical form with sizes in BLOCKS.

    PRIORITY RULE for block_count per relation:
      1. "block_count" given directly → use as-is, no calculation needed.
      2. "record_count" + "field_size_bytes" given → calculate.
      3. Only "record_count" given → use as proxy (warn).
      Never ask the user for missing fields — work with what is provided.

    Minimal input (block count given directly):
    {
      "num_processors": 10,
      "relations": {
        "Flights": {
          "fields": ["fid","date","from","to","seats"],
          "key": "fid",
          "distribution": "round_robin",
          "block_count": 10000
        }
      }
    }

    Full input (block count calculated from records):
    {
      "block_size": 2000,
      "num_processors": 10,
      "relations": {
        "Customers": {
          "fields": ["cid","name","city"],
          "field_size_bytes": 10,
          "record_count": 1000000,
          "key": "cid",
          "distribution": "round_robin"
        }
      }
    }
    """
    print(f"[TOOL] parse_schema(schema_json='{schema_json[:80]}...')")
    ctx        = json.loads(schema_json)
    block_size = ctx.get("block_size", 2000)
    p          = ctx.get("num_processors", 1)

    out = {
        "block_size":     block_size,
        "num_processors": p,
        "relations":      {},
        "field_info":     ctx.get("field_info", {}),
    }

    # Normalise relations: accept both dict {"Name": {...}} and list [{"name": "Name", ...}]
    raw_relations = ctx.get("relations", {})
    if isinstance(raw_relations, list):
        relations_dict = {}
        for item in raw_relations:
            item = dict(item)  # copy so we can pop safely
            # relation name may be under "name", "table", or "relation" key
            rel_name = (
                item.pop("name", None)
                or item.pop("table", None)
                or item.pop("relation", None)
                or f"relation_{len(relations_dict)}"
            )
            # flatten nested fields list [{"field": "x", "key": true}] -> ["x"] + key
            if "fields" in item and isinstance(item["fields"], list):
                if item["fields"] and isinstance(item["fields"][0], dict):
                    keys = [f["field"] for f in item["fields"] if f.get("key")]
                    item["key"] = keys[0] if keys else item.get("key")
                    item["fields"] = [f["field"] for f in item["fields"]]
            relations_dict[rel_name] = item
        raw_relations = relations_dict

    for name, rel in raw_relations.items():

        # ── Determine block_count (priority order) ──────────────
        if "block_count" in rel:
            # Case 1: directly provided — use as-is
            block_count = rel["block_count"]
            record_size = None
            rec_count   = rel.get("record_count")

        elif "record_count" in rel and "field_size_bytes" in rel:
            # Case 2: calculate from record count + field size
            num_fields  = len(rel.get("fields", []))
            field_size  = rel["field_size_bytes"]
            record_size = num_fields * field_size
            rec_count   = rel["record_count"]
            block_count = records_to_blocks(rec_count, record_size, block_size)

        elif "record_count" in rel:
            # Case 3: only record count — proxy, warn
            rec_count   = rel["record_count"]
            record_size = None
            block_count = rec_count
            print(f"[WARN] {name}: no field_size_bytes, using record_count as block_count proxy")

        else:
            print(f"[WARN] {name}: no size info found, block_count set to None")
            block_count = None
            rec_count   = None
            record_size = None

        blocks_per_proc = math.ceil(block_count / p) if block_count else None

        out["relations"][name] = {
            "fields":            rel.get("fields", []),
            "key":               rel.get("key"),
            "distribution":      rel.get("distribution", "round_robin"),
            "record_count":      rec_count,
            "record_size_bytes": record_size,
            "block_count":       block_count,
            "blocks_per_proc":   blocks_per_proc,
        }

    return out


# ══════════════════════════════════════════════════════════════
#  SELECT  (3 algorithms)
# ══════════════════════════════════════════════════════════════

def decide_select_algorithm(question_field, question_type, partition_key, distribution):
    """
    Decide which Select algorithm (alg2 / alg3) to use.

    alg1 -- naive, never used.
    alg2 -- every processor participates (local search + send results).
    alg3 -- only relevant processors participate (we know exactly which ones).

    Decision table:
      round_robin + any                       -> alg2 (no locality)
      hash(F) + question on F + point (=)     -> alg3 (hash gives exact proc)
      hash(F) + question on F + range/scan    -> alg2 (hash breaks order)
      range(F) + question on F + point (=)    -> alg3 (1 proc holds that value)
      range(F) + question on F + range (>,<)  -> alg3 (subset of procs)
      range(F) + question on F + scan (!=)    -> alg3 (complement of point:
                                                        p-1 procs return everything,
                                                        1 proc filters out excluded value)
      question on field != partition field    -> alg2 (no locality for that field)
    """
    print(f"[TOOL] decide_select_algorithm(question_field='{question_field}', question_type='{question_type}', partition_key='{partition_key}', distribution='{distribution}')")

    dist_field = None
    if "(" in distribution and ")" in distribution:
        dist_field = distribution[distribution.index("(") + 1 : distribution.index(")")]

    # Round-robin: no data locality at all
    if distribution == "round_robin":
        return {
            "algorithm":           "alg2",
            "relevant_processors": None,
            "reason":              "Round-robin: data spread evenly without locality, must search every processor.",
        }

    # Question field does not match partition field
    if dist_field != question_field:
        return {
            "algorithm":           "alg2",
            "relevant_processors": None,
            "reason":              f"Question on '{question_field}' but partitioned by '{dist_field}' -- cannot pinpoint relevant processors.",
        }

    # Hash partitioning
    if distribution.startswith("hash("):
        if question_type == "point":
            return {
                "algorithm":           "alg3",
                "relevant_processors": 1,
                "reason":              f"hash({dist_field}) + point search (=) on '{question_field}' -> hash function gives exact processor.",
            }
        else:
            return {
                "algorithm":           "alg2",
                "relevant_processors": None,
                "reason":              f"hash({dist_field}) does not preserve order -> range/scan cannot pinpoint processors, must check all.",
            }

    # Range partitioning
    if distribution.startswith("range("):
        if question_type == "point":
            return {
                "algorithm":           "alg3",
                "relevant_processors": 1,
                "reason":              f"range({dist_field}) + point search (=) on '{question_field}' -> exactly 1 processor holds that value.",
            }
        elif question_type == "range":
            return {
                "algorithm":           "alg3",
                "relevant_processors": None,
                "reason":              f"range({dist_field}) + range search on '{question_field}' -> only processors holding the relevant range participate.",
            }
        elif question_type == "scan":
            # != is the complement of a point search:
            # (p-1) procs return their full local partition (100% match)
            # 1 proc holding the excluded value filters it out locally
            # we know exactly which procs participate -> alg3
            return {
                "algorithm":           "alg3",
                "relevant_processors": None,
                "reason":              (
                    f"range({dist_field}) + scan (!=) on '{question_field}' -> "
                    f"complement of a point search: data locality is preserved. "
                    f"(p-1) procs return their full local partition (100%% match); "
                    f"1 proc holding the excluded value filters it out locally. -> alg3."
                ),
            }

    return {
        "algorithm":           "alg2",
        "relevant_processors": None,
        "reason":              f"Unknown distribution '{distribution}', falling back to alg2.",
    }

def select_cost(block_count, num_processors, algorithm, relevant_processors=1):
    """
    Compute Elapsed and Total cost for a Select operation.

    alg2: every proc scans locally → Elapsed = B/p * t_d, Total = p * Elapsed
    alg3: only relevant_processors work → Elapsed = Total = B/p * t_d (if 1 proc)
    """
    print(f"[TOOL] select_cost(block_count={block_count}, num_processors={num_processors}, algorithm='{algorithm}', relevant_processors={relevant_processors})")

    p           = num_processors
    B           = block_count
    bs_per_proc = math.ceil(B / p)
    bs_s        = _fmt(bs_per_proc)
    p_s         = _fmt(p)

    if algorithm == "alg2":
        elapsed = f"{bs_s} * t_d"
        total   = f"{p_s} * {bs_s} * t_d"
        explanation = (
            f"alg2: all {p_s} processors scan {bs_s} blocks each in parallel.\n"
            f"  Elapsed = {elapsed}\n"
            f"  Total   = {p_s} x Elapsed = {total}"
        )

    elif algorithm == "alg3":
        rel_p   = relevant_processors if relevant_processors else 1
        rel_p_s = _fmt(rel_p)
        elapsed = f"{bs_s} * t_d"
        total   = f"{rel_p_s} * {bs_s} * t_d" if rel_p > 1 else f"{bs_s} * t_d"
        explanation = (
            f"alg3: only {rel_p_s} relevant processor(s) participate.\n"
            f"  Elapsed = {elapsed}\n"
            f"  Total   = {total}"
        )

    else:
        return {"error": f"Unknown algorithm: {algorithm}"}

    return {
        "operation":   "select",
        "algorithm":   algorithm,
        "elapsed":     elapsed,
        "total":       total,
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════
#  SORT  — two algorithms based on distribution
# ══════════════════════════════════════════════════════════════
#
# alg1 — for Round-Robin or Hash distribution:
#   Step 1: every proc sorts its local partition:
#             3 * bs_per_proc * t_d  (parallel)
#   Step 2: (p-1) procs send their sorted runs to proc 0:
#             bs_per_proc * (p-1) * t_s  (transfer cost)
#   Step 3: proc 0 receives (p-1) runs from others:
#             (p-1) * bs_per_proc * (t_s + t_d)
#   Step 4: proc 0 merges all runs:
#             block_count * t_d
#
#   Elapsed = step1 + step2 + step3 + step4
#           = 3*bs * t_d  +  bs*(p-1) * t_s  +  (p-1)*bs*(t_s+t_d)  +  B * t_d
#   Total   = p * step1  +  step2  +  step3  +  step4
#           = 3*B * t_d  +  bs*(p-1) * t_s  +  (p-1)*bs*(t_s+t_d)  +  B * t_d
#
# alg2 — for Range distribution on the sort field:
#   Each proc sorts its local partition independently — no communication needed.
#   Elapsed = 3 * bs_per_proc * t_d
#   Total   = 3 * bs_per_proc * t_d * p  =  3 * block_count * t_d

def decide_sort_algorithm(sort_field, distribution):
    """
    Decide which Sort algorithm to use.

    alg1 — Round-Robin or Hash: local sort + merge on one proc.
    alg2 — Range on the sort field: fully local sort, no merge needed.

    Returns {"algorithm": "alg1"|"alg2", "reason": "..."}
    """
    print(f"[TOOL] decide_sort_algorithm(sort_field='{sort_field}', distribution='{distribution}')")

    dist_lower = distribution.lower()

    if dist_lower.startswith("range("):
        dist_field = distribution[distribution.index("(") + 1 : distribution.index(")")]
        if dist_field.lower() == sort_field.lower():
            return {
                "algorithm": "alg2",
                "reason": (
                    f"range({dist_field}) matches sort field '{sort_field}' → "
                    f"data already partitioned in order, each proc sorts locally — no merge needed."
                ),
            }
        else:
            return {
                "algorithm": "alg1",
                "reason": (
                    f"range({dist_field}) does not match sort field '{sort_field}' → "
                    f"cannot exploit range locality, must gather and merge."
                ),
            }

    return {
        "algorithm": "alg1",
        "reason": (
            f"Distribution '{distribution}' provides no ordering guarantee for '{sort_field}' → "
            f"alg1: local sort on all procs, then collect and merge on one proc."
        ),
    }


def sort_cost(block_count, num_processors, algorithm):
    """
    Compute Elapsed and Total cost for a Sort operation.

    alg1 (round-robin / hash):
      Step 1  local sort on every proc:    Elapsed part = 3 * bs * t_d
      Step 2  (p-1) procs send to proc 0: Elapsed part = (p-1) * bs * t_s
      Step 3  proc 0 reads incoming runs:  Elapsed part = (p-1) * bs * (t_s + t_d)
      Step 4  proc 0 merges all runs:      Elapsed part = B * t_d

    alg2 (range on sort field):
      Each proc sorts its local share independently.
      Elapsed = 3 * bs * t_d,  Total = 3 * B * t_d
    """
    print(f"[TOOL] sort_cost(block_count={block_count}, num_processors={num_processors}, algorithm='{algorithm}')")

    p  = num_processors
    B  = block_count
    bs = math.ceil(B / p)   # blocks per proc

    # Scientific-style strings (no loose numbers > 10 in the output).
    bs_s   = _fmt(bs)
    B_s    = _fmt(B)
    p_s    = _fmt(p)
    pm1_s  = _fmt(p - 1)

    if algorithm == "alg1":
        # ── Step-by-step symbolic strings ──────────────────
        step1_e = f"3 * {bs_s} * t_d"
        step2_e = f"{bs_s} * {pm1_s} * t_s"
        step3_e = f"{pm1_s} * {bs_s} * (t_s + t_d)"
        step4_e = f"{B_s} * t_d"

        step1_t = f"3 * {B_s} * t_d"    # all p procs do step1
        step2_t = step2_e               # only (p-1) procs send — same as elapsed
        step3_t = step3_e               # only proc 0 receives — same as elapsed
        step4_t = step4_e               # only proc 0 merges — same as elapsed

        elapsed = f"{step1_e} + {step2_e} + {step3_e} + {step4_e}"
        total   = f"{step1_t} + {step2_t} + {step3_t} + {step4_t}"

        explanation = "\n".join([
            "alg1 (local sort + gather + merge):",
            f"  Step 1 - all {p_s} procs sort {bs_s} blocks locally:  {step1_e}  (elapsed) / {step1_t}  (total)",
            f"  Step 2 - {pm1_s} procs send results to proc 0:      {step2_e}",
            f"  Step 3 - proc 0 reads {pm1_s} incoming runs:        {step3_e}",
            f"  Step 4 - proc 0 merges all {p_s} runs:              {step4_e}",
            "",
            f"  Elapsed = {elapsed}",
            f"  Total   = {total}",
        ])

    elif algorithm == "alg2":
        elapsed = f"3 * {bs_s} * t_d"
        total   = f"3 * {B_s} * t_d"

        explanation = "\n".join([
            "alg2 (fully local sort - range partition matches sort field):",
            f"  Each of {p_s} procs sorts its {bs_s} blocks independently, no communication.",
            "",
            f"  Elapsed = {elapsed}",
            f"  Total   = {total}",
        ])

    else:
        return {"error": f"Unknown sort algorithm: {algorithm}"}

    return {
        "operation":   "sort",
        "algorithm":   algorithm,
        "elapsed":     elapsed,
        "total":       total,
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════
#  JOIN  (placeholder kept for ParallelQueryAgent compatibility)
# ══════════════════════════════════════════════════════════════

def join_cost(blocks_s, blocks_t):
    """Placeholder formula: 3 * t_d * (B_s + B_t)"""
    print(f"[TOOL] join_cost(blocks_s={blocks_s}, blocks_t={blocks_t})")
    bs_s = _fmt(blocks_s)
    bs_t = _fmt(blocks_t)
    elapsed = f"3 * ({bs_s} + {bs_t}) * t_d"
    total   = f"3 * ({bs_s} + {bs_t}) * t_d"
    return {
        "operation":   "join",
        "blocks_s":    bs_s,
        "blocks_t":    bs_t,
        "elapsed":     elapsed,
        "total":       total,
        "explanation": (
            f"Join: 3 * t_d * (B_s + B_t)\n"
            f"  B_s = {bs_s} blocks\n"
            f"  B_t = {bs_t} blocks\n"
            f"  Elapsed = 3 * ({bs_s} + {bs_t}) * t_d"
        ),
    }


# ══════════════════════════════════════════════════════════════
#  PARALLEL JOIN  (JoinCostAgent)
# ══════════════════════════════════════════════════════════════
#
# TWO algorithms depending on data distribution:
#
# ── Algorithm 1: LOCAL JOIN (co-located data) ──────────────────
#   Condition: both tables are partitioned by the same method
#   (both hash or both range) on the exact join field.
#   Matching tuples are guaranteed to reside on the same server
#   → no communication needed, every server joins locally.
#
#   Elapsed = 3 * (bs_R + bs_S) * t_d
#   Total   = p * Elapsed
#
# ── Algorithm 2: REGULAR JOIN (default) ───────────────────────
#   Used when distributions differ or partition field != join field.
#   The SMALLER (outer) table is broadcast to all servers;
#   each server joins the full outer table with its local inner partition.
#
#   bs_out = ceil(B_out / p)    (outer = smaller table)
#   bs_in  = ceil(B_in  / p)
#
#   Step 1 [send]    — each server sends its outer partition to (p-1) others:
#                      bs_out * t_d + (p-1) * bs_out * t_s
#   Step 2 [receive] — each server receives outer from (p-1) others:
#                      (p-1) * bs_out * (t_s + t_d)
#   Step 3 [join]    — each server joins full outer with its local inner:
#                      3 * (B_out + bs_in) * t_d
#
#   Elapsed = Step1 + Step2 + Step3
#   Total   = p * Elapsed
#
#   NOTE: S ⋈ F ≠ F ⋈ S — always pick outer = smaller table (cheaper).


def _dist_info(dist: str):
    """Parse 'hash(field)' / 'range(field)' → (method, field). Returns (dist, None) otherwise."""
    d = dist.strip().lower()
    if d.startswith("hash(") and d.endswith(")"):
        return "hash", d[5:-1].strip()
    if d.startswith("range(") and d.endswith(")"):
        return "range", d[6:-1].strip()
    return d, None


def parallel_join_cost(
    blocks_a, blocks_b, num_processors,
    name_a="A", name_b="B",
    distribution_a="round_robin", distribution_b="round_robin",
    join_field="",
):
    """
    Compute Elapsed and Total cost for a parallel Join.

    Chooses the algorithm automatically based on data distribution:
      - LOCAL JOIN:     both tables partitioned by the same method (hash/range)
                        on exactly the join_field  →  no communication needed.
      - BROADCAST JOIN: any other case  →  3-step broadcast algorithm.

    Args:
        blocks_a / blocks_b:    Total block counts of the two input tables.
        num_processors:         Number of parallel servers (p).
        name_a / name_b:        Human-readable table names.
        distribution_a/b:       Partition scheme, e.g. "hash(name)", "range(id)", "round_robin".
        join_field:             The field the Join is performed on (e.g. "name").

    Output is symbolic — never simplified to a single number.
    """
    print(f"[TOOL] parallel_join_cost({name_a}={blocks_a}, {name_b}={blocks_b}, p={num_processors}, "
          f"dist_a='{distribution_a}', dist_b='{distribution_b}', join_field='{join_field}')")

    p    = num_processors
    bs_a = math.ceil(blocks_a / p)
    bs_b = math.ceil(blocks_b / p)

    bs_a_s  = _fmt(bs_a)
    bs_b_s  = _fmt(bs_b)
    p_s     = _fmt(p)
    sum_str = f"{bs_a_s} + {bs_b_s}"

    # ── Decide algorithm ─────────────────────────────────────────
    method_a, field_a = _dist_info(distribution_a)
    method_b, field_b = _dist_info(distribution_b)

    colocated = (
        method_a == method_b                        # same partition type
        and field_a is not None
        and field_a == field_b                       # same partition field
        and join_field.strip().lower() == field_a    # partition field == join field
    )

    if colocated:
        # ── Algorithm 1: LOCAL JOIN ──────────────────────────────
        elapsed = f"3 * ({sum_str}) * t_d"
        total   = f"{p_s} * ({elapsed})"
        elapsed_collected = simplify_cost_expr(elapsed)
        total_collected   = simplify_cost_expr(total)

        explanation = "\n".join([
            f"Join: {name_a} * {name_b}  [LOCAL JOIN -- co-located data]",
            f"  Both tables partitioned by {distribution_a} on join field '{join_field}'",
            f"  Matching tuples are on the same server -- no communication needed.",
            f"",
            f"  {name_a}: {_fmt(blocks_a)} blocks -> {bs_a_s} blocks/server",
            f"  {name_b}: {_fmt(blocks_b)} blocks -> {bs_b_s} blocks/server",
            f"  p = {p_s} servers",
            f"",
            f"  Local Join on every server:  3 * ({sum_str}) * t_d",
            f"",
            f"  Elapsed = {elapsed}",
            f"          = {elapsed_collected}   (collected)",
            f"  Total   = {total}",
            f"          = {total_collected}   (collected)",
        ])

        return {
            "operation":         "join",
            "algorithm":         "local",
            "table_a":           name_a,
            "table_b":           name_b,
            "blocks_a":          blocks_a,
            "blocks_b":          blocks_b,
            "blocks_per_proc_a": bs_a,
            "blocks_per_proc_b": bs_b,
            "num_processors":    p,
            "elapsed":           elapsed,
            "total":             total,
            "elapsed_collected": elapsed_collected,
            "total_collected":   total_collected,
            "explanation":       explanation,
        }

    else:
        # ── Algorithm 2: REGULAR JOIN ────────────────────────────
        # Broadcast the smaller (outer) table — always cheaper.
        if blocks_a <= blocks_b:
            B_out, bs_out_v = blocks_a, bs_a
            B_in,  bs_in_v  = blocks_b, bs_b
            nm_out, nm_in   = name_a, name_b
        else:
            B_out, bs_out_v = blocks_b, bs_b
            B_in,  bs_in_v  = blocks_a, bs_a
            nm_out, nm_in   = name_b, name_a

        B_out_s  = _fmt(B_out)
        B_in_s   = _fmt(B_in)
        bs_out_s = _fmt(bs_out_v)
        bs_in_s  = _fmt(bs_in_v)

        pm1_s = _fmt(p - 1)
        step1 = f"{bs_out_s} * t_d + {pm1_s} * {bs_out_s} * t_s"
        step2 = f"{pm1_s} * {bs_out_s} * (t_s + t_d)"
        step3 = f"3 * ({B_out_s} + {bs_in_s}) * t_d"

        elapsed = f"({step1}) + ({step2}) + ({step3})"
        total   = f"{p_s} * ({elapsed})"
        elapsed_collected = simplify_cost_expr(elapsed)
        total_collected   = simplify_cost_expr(total)

        explanation = "\n".join([
            f"Join: {name_a} ⋈ {name_b}  [REGULAR JOIN]",
            f"  Outer (broadcast): {nm_out} — {B_out_s} blocks total,"
            f" {bs_out_s} blocks/server",
            f"  Inner (local):     {nm_in} — {B_in_s} blocks total,"
            f" {bs_in_s} blocks/server",
            f"  p = {p_s} servers",
            f"  → broadcasting {nm_out} (smaller table) is the cheaper ordering",
            f"",
            f"  Step 1 [send]    — each server sends its {nm_out} partition"
            f" to (p-1) = {pm1_s} others:",
            f"    {step1}",
            f"  Step 2 [receive] — each server receives {nm_out}"
            f" from (p-1) = {pm1_s} others:",
            f"    {step2}",
            f"  Step 3 [join]    — each server joins full {nm_out} ({B_out_s} blocks,"
            f" received in full) with its local {nm_in} partition ({bs_in_s} blocks/server):",
            f"    {step3}",
            f"    NOTE: step3 uses B_out={B_out_s} (full outer) + bs_in={bs_in_s} (inner/p),"
            f" NOT B_in={B_in_s} (inner total)",
            f"",
            f"  Elapsed = {elapsed}",
            f"          = {elapsed_collected}   (collected)",
            f"  Total   = {p_s} * Elapsed = {total}",
            f"          = {total_collected}   (collected)",
        ])

        return {
            "operation":         "join",
            "algorithm":         "regular",
            "table_a":           name_a,
            "table_b":           name_b,
            "outer_table":       nm_out,
            "inner_table":       nm_in,
            "blocks_a":          blocks_a,
            "blocks_b":          blocks_b,
            "blocks_per_proc_a": bs_a,
            "blocks_per_proc_b": bs_b,
            "num_processors":    p,
            "step1":             step1,
            "step2":             step2,
            "step3":             step3,
            "elapsed":           elapsed,
            "total":             total,
            "elapsed_collected": elapsed_collected,
            "total_collected":   total_collected,
            "explanation":       explanation,
        }


# ══════════════════════════════════════════════════════════════
#  SELECT-then-BROADCAST JOIN  (with selectivity, symbolic-aware)
# ══════════════════════════════════════════════════════════════
#
# This is the detailed "course style" cost model for a query of the form
#     π(...)( σ(...)(R)  ⋈  σ(...)(S) )
# on round-robin data with p processors.
#
# The SMALLER table (after we know its block count) is the OUTER table:
# it is read, filtered, and broadcast to every server; the LARGER table
# stays local. Each server then joins the full broadcast set with its own
# filtered partition of the inner table.
#
# Per-server elapsed steps (read full / write filtered):
#   1) read + select outer        :  bs_out                         * t_d
#   2) send filtered outer to p-1  : (p-1) * s_out * bs_out          * t_s
#   3) receive from p-1            : (p-1) * s_out * bs_out          * t_s   (ts only!)
#   4) write full broadcast outer  :  p     * s_out * bs_out         * t_d
#   5) read + select inner        :  bs_in                          * t_d
#   6) write filtered inner       :  s_in  * bs_in                  * t_d
#   7) natural join               :  3 * (p*s_out*bs_out + s_in*bs_in) * t_d
#   8) projection                 :  0 (negligible)  — overridable
#
#   Elapsed = Σ(td steps) * t_d + Σ(ts steps) * t_s
#   Total   = p * Elapsed
#
# Selectivity s_out / s_in is either:
#   • a numeric fraction (uniform attribute: matching values / distinct values),
#     e.g. 1/2 — it folds into the numeric coefficients, OR
#   • a SYMBOLIC variable name (unknown distribution), e.g. "Sp" — it stays in
#     the expression and propagates into Elapsed and Total.
#   • None / "" — no pre-join filter on that table (selectivity = 1).


def _frac_disp(num, den):
    """Display a fraction num/den as a short decimal (0.9) when clean, else 'num/den'."""
    if den == 0:
        return f"{num}/{den}"
    v = num / den
    s = ("%.6f" % v).rstrip("0").rstrip(".")
    digits = len(s) - (1 if "." in s else 0)
    return s if digits <= 4 else f"{num}/{den}"


def _sel_spec(sel):
    """
    Normalise a selectivity spec into (coeff, symbol, display).

    Accepts:
      None / ""        → (1.0, "", "")          no filter, selectivity 1
      0.5  / 1         → (0.5, "", "0.5")        numeric
      "1/2"            → (0.5, "", "1/2")         fraction (kept for display)
      "Sp"             → (1.0, "Sp", "Sp")        symbolic variable
    """
    if sel is None or sel == "":
        return 1.0, "", ""
    if isinstance(sel, (int, float)):
        v = float(sel)
        disp = str(int(v)) if abs(v - round(v)) < 1e-12 else str(v)
        return v, "", disp
    s = str(sel).strip()
    m = re.match(r'^(\d+)\s*/\s*(\d+)$', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a / b, "", f"{a}/{b}"
    try:
        v = float(s)
        disp = str(int(v)) if abs(v - round(v)) < 1e-12 else str(v)
        return v, "", disp
    except ValueError:
        return 1.0, s, s          # symbolic selectivity variable


def _fmt_coeff(x):
    """Scientific-style coefficient: integers via _fmt, else a short decimal."""
    if abs(x - round(x)) < 1e-9:
        return _fmt(int(round(x)))
    return ("%.4f" % x).rstrip("0").rstrip(".")


def _lin_expr(terms):
    """
    Render a linear form  {symbol -> coeff}  as  "const + a * Sym + ...".
    The "" key is the constant term. Coefficients are scientific-style.
    """
    out = []
    if "" in terms and abs(terms[""]) > 1e-9:
        out.append(_fmt_coeff(terms[""]))
    for sym in sorted(k for k in terms if k != ""):
        c = terms[sym]
        if abs(c) < 1e-9:
            continue
        cf = _fmt_coeff(c)
        out.append(sym if cf == "1" else f"{cf} * {sym}")
    return " + ".join(out) if out else "0"


# ══════════════════════════════════════════════════════════════
#  SYMBOLIC SIMPLIFIER  — collect a cost expression into short form
# ══════════════════════════════════════════════════════════════
#
# Reduces a symbolic cost string such as
#   (2 * 10^3 * t_d + 9 * 2 * 10^3 * t_s) + (9 * 2 * 10^3 * (t_s + t_d))
#   + (3 * (2 * 10^4 + 10^5) * t_d)
# into the collected scientific form
#   (3.8 * 10^5) * t_d + (3.6 * 10^4) * t_s
#
# Cost expressions are linear in t_d and t_s; any other identifier (a symbolic
# selectivity like Sp) is kept as a coefficient symbol. The grammar handled is:
#   expr   := term (('+'|'-') term)*
#   term   := factor ('*' factor)*
#   factor := atom ('^' atom)?            (exponent must be a plain number)
#   atom   := number | identifier | '(' expr ')'
# Internally everything becomes a polynomial {sorted(symbol_tuple) -> coeff}.

def _tokenize(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "+-*^()":
            toks.append(c); i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            toks.append(("num", float(s[i:j]))); i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            toks.append(("id", s[i:j])); i = j
        else:
            i += 1   # skip anything unexpected
    return toks


def _poly_add(a, b):
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, 0.0) + v
    return r


def _poly_mul(a, b):
    r = {}
    for k1, v1 in a.items():
        for k2, v2 in b.items():
            k = tuple(sorted(k1 + k2))
            r[k] = r.get(k, 0.0) + v1 * v2
    return r


class _ExprParser:
    def __init__(self, toks):
        self.toks, self.pos = toks, 0

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _next(self):
        t = self.toks[self.pos]; self.pos += 1; return t

    def parse(self):
        return self._expr()

    def _expr(self):
        node = self._term()
        while self._peek() in ("+", "-"):
            op = self._next()
            rhs = self._term()
            node = _poly_add(node, rhs if op == "+" else {k: -v for k, v in rhs.items()})
        return node

    def _term(self):
        node = self._factor()
        while self._peek() == "*":
            self._next()
            node = _poly_mul(node, self._factor())
        return node

    def _factor(self):
        node = self._atom()
        while self._peek() == "^":
            self._next()
            exp = self._atom()
            k = int(round(exp.get((), 0.0)))
            res = {(): 1.0}
            for _ in range(k):
                res = _poly_mul(res, node)
            node = res
        return node

    def _atom(self):
        t = self._peek()
        if t == "(":
            self._next()
            node = self._expr()
            if self._peek() == ")":
                self._next()
            return node
        t = self._next()
        if isinstance(t, tuple) and t[0] == "num":
            return {(): t[1]}
        if isinstance(t, tuple) and t[0] == "id":
            return {(t[1],): 1.0}
        return {(): 0.0}


def _mono_to_str(mono, coeff):
    """Format one monomial (symbols already excluding t_d/t_s) with a coeff."""
    cf = _fmt_coeff(coeff)
    if not mono:
        return cf
    syms = " * ".join(mono)
    return syms if cf == "1" else f"{cf} * {syms}"


def _coeff_poly_to_str(d):
    """Render {symbol_tuple -> coeff} as 'const + a * Sym + ...' (scientific)."""
    items = []
    if () in d and abs(d[()]) > 1e-9:
        items.append(((), d[()]))
    for k in sorted((k for k in d if k != ()), key=lambda t: (len(t), t)):
        if abs(d[k]) > 1e-9:
            items.append((k, d[k]))
    if not items:
        return "0"
    return " + ".join(_mono_to_str(m, c) for m, c in items)


def simplify_cost_expr(expr_str):
    """
    Collect a symbolic cost expression into  (..) * t_d + (..) * t_s  with every
    coefficient in scientific notation. Selectivity symbols (e.g. Sp) are kept.
    Returns the simplified string (falls back to the original on parse failure).
    """
    try:
        poly = _ExprParser(_tokenize(expr_str)).parse()
    except Exception:
        return expr_str

    td, ts, other = {}, {}, {}
    for mono, coeff in poly.items():
        if abs(coeff) < 1e-9:
            continue
        has_td, has_ts = "t_d" in mono, "t_s" in mono
        if has_td and not has_ts:
            rest = tuple(s for s in mono if s != "t_d")
            td[rest] = td.get(rest, 0.0) + coeff
        elif has_ts and not has_td:
            rest = tuple(s for s in mono if s != "t_s")
            ts[rest] = ts.get(rest, 0.0) + coeff
        else:
            other[mono] = other.get(mono, 0.0) + coeff   # not a clean t_d/t_s term

    parts = []
    td_s = _coeff_poly_to_str(td)
    ts_s = _coeff_poly_to_str(ts)
    if td_s != "0":
        parts.append(f"({td_s}) * t_d")
    if ts_s != "0":
        parts.append(f"({ts_s}) * t_s")
    if other:
        parts.append(_coeff_poly_to_str(other))
    return " + ".join(parts) if parts else "0"


def select_broadcast_join_cost(
    blocks_a, blocks_b, num_processors,
    name_a="A", name_b="B",
    sel_a=None, sel_b=None,
    join_field="",
    project_cost=0,
):
    """
    Cost of  π(...)( σ_a(A) ⋈ σ_b(B) )  via the select-then-broadcast algorithm.

    The smaller table is broadcast (outer); the larger stays local (inner).
    sel_a / sel_b are selectivity specs (see _sel_spec): numeric fraction,
    symbolic variable name, or None for "no filter".

    Output is symbolic and selectivity-aware — Elapsed and Total are returned
    both as the per-step breakdown and as collected linear forms.
    """
    print(f"[TOOL] select_broadcast_join_cost({name_a}={blocks_a}, {name_b}={blocks_b}, "
          f"p={num_processors}, sel_a={sel_a!r}, sel_b={sel_b!r}, join_field='{join_field}')")

    p = num_processors

    # Outer = smaller table (broadcast).  Inner = larger (local).
    if blocks_a <= blocks_b:
        B_out, name_out, sel_out = blocks_a, name_a, sel_a
        B_in,  name_in,  sel_in  = blocks_b, name_b, sel_b
    else:
        B_out, name_out, sel_out = blocks_b, name_b, sel_b
        B_in,  name_in,  sel_in  = blocks_a, name_a, sel_a

    bs_out = math.ceil(B_out / p)
    bs_in  = math.ceil(B_in  / p)

    c_out, sym_out, disp_out = _sel_spec(sel_out)
    c_in,  sym_in,  disp_in  = _sel_spec(sel_in)

    bs_out_s = _fmt(bs_out)
    bs_in_s  = _fmt(bs_in)
    p_s      = _fmt(p)
    pm1_s    = _fmt(p - 1)

    def mul(*parts):
        return " * ".join(x for x in parts if x not in ("", None))

    # ── per-server elapsed-time display strings (factored, unreduced) ──
    step1 = mul(bs_out_s, "t_d")
    step2 = mul(pm1_s, disp_out, bs_out_s, "t_s")
    step3 = mul(pm1_s, disp_out, bs_out_s, "t_s")
    step4 = mul(p_s, disp_out, bs_out_s, "t_d")
    step5 = mul(bs_in_s, "t_d")
    step6 = mul(disp_in, bs_in_s, "t_d")
    join_outer = mul(p_s, disp_out, bs_out_s)
    join_inner = mul(disp_in, bs_in_s)
    step7 = f"3 * ({join_outer} + {join_inner}) * t_d"
    step8 = mul(_fmt(project_cost), "t_d") if project_cost else "0"

    # ── collect Elapsed into linear forms over {const, symbols} ──────
    td, ts = {}, {}
    def add(d, coeff, sym):
        d[sym] = d.get(sym, 0.0) + coeff

    add(td, bs_out,                  "")        # step 1
    add(ts, (p - 1) * c_out * bs_out, sym_out)  # step 2
    add(ts, (p - 1) * c_out * bs_out, sym_out)  # step 3
    add(td, p * c_out * bs_out,       sym_out)  # step 4
    add(td, bs_in,                   "")        # step 5
    add(td, c_in * bs_in,             sym_in)   # step 6
    add(td, 3 * p * c_out * bs_out,   sym_out)  # step 7 (outer part)
    add(td, 3 * c_in * bs_in,         sym_in)   # step 7 (inner part)
    if project_cost:
        add(td, project_cost,        "")        # step 8

    td_expr = _lin_expr(td)
    ts_expr = _lin_expr(ts)

    if ts_expr and ts_expr != "0":
        elapsed = f"({td_expr}) * t_d + ({ts_expr}) * t_s"
    else:
        elapsed = f"({td_expr}) * t_d"
    total = f"{p_s} * ({elapsed})"

    explanation = "\n".join([
        f"Algorithm: select-then-broadcast join.",
        f"  Broadcast the smaller table ({name_out}) after filtering; "
        f"each server joins it locally with its partition of {name_in}.",
        f"  {name_out}: {_fmt(B_out)} blocks -> {bs_out_s} blocks/server  (outer, broadcast)"
        + (f", selectivity {disp_out}" if disp_out else ", no filter"),
        f"  {name_in}: {_fmt(B_in)} blocks -> {bs_in_s} blocks/server  (inner, local)"
        + (f", selectivity {disp_in}" if disp_in else ", no filter"),
        f"  p = {p_s} servers",
        "",
        "Per-server steps (elapsed):",
        f"  1) read + select {name_out}:           {step1}",
        f"  2) send filtered {name_out} to (p-1):  {step2}",
        f"  3) receive from (p-1):                {step3}",
        f"  4) write full broadcast {name_out}:    {step4}",
        f"  5) read + select {name_in}:            {step5}",
        f"  6) write filtered {name_in}:           {step6}",
        f"  7) natural join:                      {step7}",
        f"  8) projection:                        {step8}",
        "",
        f"  Elapsed = {elapsed}",
        f"  Total   = {p_s} * Elapsed = {total}",
    ])

    return {
        "operation":      "select_broadcast_join",
        "outer_table":    name_out,
        "inner_table":    name_in,
        "blocks_out":     B_out,
        "blocks_in":      B_in,
        "bs_out":         bs_out,
        "bs_in":          bs_in,
        "num_processors": p,
        "selectivity_out": disp_out or "1",
        "selectivity_in":  disp_in or "1",
        "steps": {
            "step1_read_select_outer":  step1,
            "step2_send_outer":         step2,
            "step3_receive_outer":      step3,
            "step4_write_broadcast":    step4,
            "step5_read_select_inner":  step5,
            "step6_write_inner":        step6,
            "step7_join":               step7,
            "step8_projection":         step8,
        },
        "elapsed":        elapsed,
        "total":          total,
        "explanation":    explanation,
    }


# ══════════════════════════════════════════════════════════════
#  RANGE-PARTITIONED BROADCAST JOIN
# ══════════════════════════════════════════════════════════════
#
# Special case of the select-then-broadcast join: the INNER (larger) table is
# RANGE-partitioned on exactly the field its predicate filters. The partitioning
# already performs the selection — the matching tuples sit on a contiguous block
# of  k  "active" servers, and EVERY tuple there matches (no extra Select, no
# selectivity factor on the inner table).
#
# The OUTER (smaller) table is round-robin: it is read + filtered (selectivity
# s_out) on all p servers and broadcast to the k active servers, which then join
# locally and project.
#
# Per-server steps:
#   1) all p   read + select outer            : bs_out                       * t_d
#   2) send filtered outer to the k actives:
#        - the (p-k) non-active servers send to k:   k     * s_out * bs_out   * t_s
#        - the  k    active servers send to (k-1):  (k-1)  * s_out * bs_out   * t_s
#   3) k actives receive from (p-1) others    : (p-1) * s_out * bs_out        * t_s
#   4) k actives write full broadcast outer   :  p    * s_out * bs_out        * t_d
#   5) k actives natural join                 : 3 * (p*s_out*bs_out + bs_in)  * t_d
#   6) projection                             : 0
#
# Elapsed (bottleneck = an active server) uses the clean (p-1) communication
# bound for BOTH the send and the receive phase:
#   E_td = bs_out + p*s_out*bs_out + 3*(p*s_out*bs_out + bs_in)
#   E_ts = 2 * (p-1) * s_out * bs_out
#
# Total is NOT p * Elapsed (only k servers do steps 3-5) — it is the sum of the
# real per-server work:
#   T_td = p*bs_out  +  k*p*s_out*bs_out  +  3k*(p*s_out*bs_out + bs_in)
#   T_ts = 2 * k * (p-1) * s_out * bs_out
#         (send total = [(p-k)*k + k*(k-1)] = k*(p-1);  receive total = k*(p-1))


def range_broadcast_join_cost(
    blocks_outer, blocks_inner, num_processors, active_processors,
    name_outer="A", name_inner="B",
    sel_outer=None, join_field="",
    project_cost=0,
):
    """
    Cost when the INNER (larger) table is range-partitioned on its filter field,
    so its selection is free and localised to `active_processors` (k) servers.

    blocks_outer:      block count of the smaller (broadcast) table.
    blocks_inner:      block count of the larger (range-partitioned, local) table.
    num_processors:    total servers p.
    active_processors: k = servers holding the matching range (= round(sel * p)).
    sel_outer:         selectivity spec for the outer table's pre-join Select
                       (numeric fraction, symbolic name like "Sp", or None).
    """
    print(f"[TOOL] range_broadcast_join_cost(outer {name_outer}={blocks_outer}, "
          f"inner {name_inner}={blocks_inner}, p={num_processors}, k={active_processors}, "
          f"sel_outer={sel_outer!r}, join_field='{join_field}')")

    p = num_processors
    k = active_processors
    bs_out = math.ceil(blocks_outer / p)     # outer is round-robin across all p
    bs_in  = math.ceil(blocks_inner / p)     # inner range-partitioned: 1/p per server

    c_out, sym_out, disp_out = _sel_spec(sel_outer)

    bs_out_s = _fmt(bs_out)
    bs_in_s  = _fmt(bs_in)
    p_s      = _fmt(p)
    k_s      = _fmt(k)
    pm1_s    = _fmt(p - 1)
    km1_s    = _fmt(k - 1)

    def mul(*parts):
        return " * ".join(x for x in parts if x not in ("", None))

    # ── per-server elapsed-time display strings ──────────────────────
    step1   = mul(bs_out_s, "t_d")
    step2a  = mul(k_s,   disp_out, bs_out_s, "t_s")     # non-active -> k actives
    step2b  = mul(km1_s, disp_out, bs_out_s, "t_s")     # active -> (k-1) actives
    step3   = mul(pm1_s, disp_out, bs_out_s, "t_s")     # actives receive from (p-1)
    step4   = mul(p_s,   disp_out, bs_out_s, "t_d")     # write full broadcast
    join_outer = mul(p_s, disp_out, bs_out_s)
    step5   = f"3 * ({join_outer} + {bs_in_s}) * t_d"
    step6   = mul(_fmt(project_cost), "t_d") if project_cost else "0"

    # ── Elapsed (bottleneck active server; (p-1) comm bound) ─────────
    e_td, e_ts = {}, {}
    def eadd(d, c, s):
        d[s] = d.get(s, 0.0) + c
    eadd(e_td, bs_out,                  "")        # step 1
    eadd(e_td, p * c_out * bs_out,       sym_out)  # step 4
    eadd(e_td, 3 * p * c_out * bs_out,   sym_out)  # step 5 outer part
    eadd(e_td, 3 * bs_in,               "")        # step 5 inner part
    eadd(e_ts, 2 * (p - 1) * c_out * bs_out, sym_out)   # send + receive (9+9)
    if project_cost:
        eadd(e_td, project_cost,        "")

    e_td_expr = _lin_expr(e_td)
    e_ts_expr = _lin_expr(e_ts)
    if e_ts_expr and e_ts_expr != "0":
        elapsed = f"({e_td_expr}) * t_d + ({e_ts_expr}) * t_s"
    else:
        elapsed = f"({e_td_expr}) * t_d"

    # ── Total (sum of real per-server work — NOT p * Elapsed) ────────
    t_td, t_ts = {}, {}
    def tadd(d, c, s):
        d[s] = d.get(s, 0.0) + c
    tadd(t_td, p * bs_out,                  "")        # step1, all p
    tadd(t_td, k * p * c_out * bs_out,       sym_out)  # step4, k actives
    tadd(t_td, 3 * k * p * c_out * bs_out,   sym_out)  # step5 outer, k actives
    tadd(t_td, 3 * k * bs_in,               "")        # step5 inner, k actives
    tadd(t_ts, 2 * k * (p - 1) * c_out * bs_out, sym_out)  # send k*(p-1) + recv k*(p-1)
    if project_cost:
        tadd(t_td, k * project_cost,        "")

    t_td_expr = _lin_expr(t_td)
    t_ts_expr = _lin_expr(t_ts)
    if t_ts_expr and t_ts_expr != "0":
        total = f"({t_td_expr}) * t_d + ({t_ts_expr}) * t_s"
    else:
        total = f"({t_td_expr}) * t_d"

    explanation = "\n".join([
        f"Algorithm: range-partitioned broadcast join.",
        f"  {name_inner} is range-partitioned on the filter field -> the matching tuples"
        f" sit on k = {k_s} active servers, and ALL tuples there match (no Select on {name_inner}).",
        f"  {name_outer} (round-robin) is filtered and broadcast to those k servers.",
        f"  {name_outer}: {_fmt(blocks_outer)} blocks -> {bs_out_s} blocks/server  (outer, broadcast)"
        + (f", selectivity {disp_out}" if disp_out else ", no filter"),
        f"  {name_inner}: {_fmt(blocks_inner)} blocks -> {bs_in_s} blocks/server  (inner, range-partitioned)",
        f"  p = {p_s} servers, k = {k_s} active",
        "",
        "Per-server steps (elapsed):",
        f"  1) all p read + select {name_outer}:        {step1}",
        f"  2) send filtered {name_outer}:",
        f"       - (p-k) non-active servers -> k:     {step2a}",
        f"       - k active servers -> (k-1):         {step2b}",
        f"  3) k active receive from (p-1):           {step3}",
        f"  4) k active write full broadcast:         {step4}",
        f"  5) k active natural join:                 {step5}",
        f"  6) projection:                            {step6}",
        "",
        f"  Elapsed (bottleneck active server) = {elapsed}",
        f"  Total (sum of all servers, NOT p*Elapsed) = {total}",
    ])

    return {
        "operation":          "range_broadcast_join",
        "outer_table":        name_outer,
        "inner_table":        name_inner,
        "blocks_outer":       blocks_outer,
        "blocks_inner":       blocks_inner,
        "bs_out":             bs_out,
        "bs_in":              bs_in,
        "num_processors":     p,
        "active_processors":  k,
        "selectivity_outer":  disp_out or "1",
        "steps": {
            "step1_read_select_outer":   step1,
            "step2a_send_nonactive":     step2a,
            "step2b_send_active":        step2b,
            "step3_receive":             step3,
            "step4_write_broadcast":     step4,
            "step5_join":                step5,
            "step6_projection":          step6,
        },
        "elapsed":            elapsed,
        "total":              total,
        "explanation":        explanation,
    }


# ══════════════════════════════════════════════════════════════
#  HASH-SHUFFLE (REDISTRIBUTION) JOIN
# ══════════════════════════════════════════════════════════════
#
# Used when one table is already HASH-partitioned on the join field and the other
# is not. The non-hashed table is RE-HASHED (redistributed) on the join field so
# that every row with the same key lands on the same server as the matching rows
# of the already-hashed table — then each server joins locally. No broadcast.
#
# Let R = redistributed table (read + filtered, then shuffled), S = already-hashed
# local table (read + filtered in place). bs_r = B_R/p, bs_s = B_S/p.
# Shuffle keeps 1/p of each server's data local and sends (p-1)/p away.
#
# Per-server steps:
#   1) read + select R                 : bs_r                          * t_d
#   2) send (p-1)/p of filtered R       : (p-1)/p * s_r * bs_r          * t_s
#   3) receive (p-1)/p from others     : (p-1)/p * s_r * bs_r          * t_s
#   4) write own hash bucket of R       : s_r * bs_r                    * t_d
#        (= total filtered R / p — the data received plus the 1/p kept local)
#   5) read + select S                 : bs_s                          * t_d
#   6) write filtered S                : s_s * bs_s                    * t_d
#   7) natural join (local)            : 3 * (s_r*bs_r + s_s*bs_s)      * t_d
#   8) projection                      : 0
#
#   Elapsed = Σ(td) * t_d + Σ(ts) * t_s
#   Total   = p * Elapsed   (every server does identical work — symmetric shuffle)


def hash_shuffle_join_cost(
    blocks_redistributed, blocks_local, num_processors,
    name_redistributed="R", name_local="S",
    sel_redistributed=None, sel_local=None,
    join_field="",
    project_cost=0,
):
    """
    Cost of a Join where the LOCAL table is already hash-partitioned on the join
    field and the OTHER table is re-hashed (redistributed) on that field so the
    join becomes purely local. No broadcast.

    blocks_redistributed: block count of the table that must be re-hashed (R).
    blocks_local:         block count of the already-hash-partitioned table (S).
    num_processors:       servers p.
    sel_redistributed:    selectivity spec for R's pre-join filter (numeric, symbol, or None).
    sel_local:            selectivity spec for S's pre-join filter.
    join_field:           the hash/join field.
    """
    print(f"[TOOL] hash_shuffle_join_cost(redistributed {name_redistributed}={blocks_redistributed}, "
          f"local {name_local}={blocks_local}, p={num_processors}, sel_r={sel_redistributed!r}, "
          f"sel_s={sel_local!r}, join_field='{join_field}')")

    p = num_processors
    bs_r = math.ceil(blocks_redistributed / p)
    bs_s = math.ceil(blocks_local / p)

    c_r, sym_r, disp_r = _sel_spec(sel_redistributed)
    c_s, sym_s, disp_s = _sel_spec(sel_local)

    bs_r_s = _fmt(bs_r)
    bs_s_s = _fmt(bs_s)
    p_s    = _fmt(p)

    frac_disp = _frac_disp(p - 1, p)     # (p-1)/p, e.g. "0.9"
    frac_val  = (p - 1) / p

    def mul(*parts):
        return " * ".join(x for x in parts if x not in ("", None))

    # ── per-server elapsed-time display strings ──────────────────────
    step1 = mul(bs_r_s, "t_d")
    step2 = mul(frac_disp, disp_r, bs_r_s, "t_s")     # shuffle out
    step3 = mul(frac_disp, disp_r, bs_r_s, "t_s")     # shuffle in
    step4 = mul(disp_r, bs_r_s, "t_d")                # write own bucket = s_r*bs_r
    step5 = mul(bs_s_s, "t_d")
    step6 = mul(disp_s, bs_s_s, "t_d")
    join_r = mul(disp_r, bs_r_s)
    join_s = mul(disp_s, bs_s_s)
    step7 = f"3 * ({join_r} + {join_s}) * t_d"
    step8 = mul(_fmt(project_cost), "t_d") if project_cost else "0"

    # ── Elapsed (collected linear form over {const, selectivity symbols}) ──
    e_td, e_ts = {}, {}
    def add(d, c, s):
        d[s] = d.get(s, 0.0) + c
    add(e_td, bs_r,            "")        # step 1
    add(e_td, c_r * bs_r,      sym_r)     # step 4
    add(e_td, bs_s,            "")        # step 5
    add(e_td, c_s * bs_s,      sym_s)     # step 6
    add(e_td, 3 * c_r * bs_r,  sym_r)     # step 7 (R part)
    add(e_td, 3 * c_s * bs_s,  sym_s)     # step 7 (S part)
    add(e_ts, 2 * frac_val * c_r * bs_r, sym_r)   # steps 2+3
    if project_cost:
        add(e_td, project_cost, "")

    e_td_expr = _lin_expr(e_td)
    e_ts_expr = _lin_expr(e_ts)
    if e_ts_expr and e_ts_expr != "0":
        elapsed = f"({e_td_expr}) * t_d + ({e_ts_expr}) * t_s"
    else:
        elapsed = f"({e_td_expr}) * t_d"

    # ── Total = p * Elapsed (symmetric: every server does the same work) ──
    t_td = {k: v * p for k, v in e_td.items()}
    t_ts = {k: v * p for k, v in e_ts.items()}
    t_td_expr = _lin_expr(t_td)
    t_ts_expr = _lin_expr(t_ts)
    if t_ts_expr and t_ts_expr != "0":
        total = f"({t_td_expr}) * t_d + ({t_ts_expr}) * t_s"
    else:
        total = f"({t_td_expr}) * t_d"

    jf = join_field or "the join field"
    idea = (
        f"{name_local} is hash({jf})-partitioned, so we re-hash (redistribute) "
        f"{name_redistributed} on hash({jf}) too: every row with the same {jf} ends up "
        f"on the same server, and each server joins its local data — no broadcast needed. "
        f"Each server keeps 1/p of its filtered {name_redistributed} and ships (p-1)/p away."
    )

    explanation = "\n".join([
        f"Algorithm: hash-shuffle (redistribution) join.",
        f"  Idea: {idea}",
        f"  {name_redistributed} (re-hashed): {_fmt(blocks_redistributed)} blocks -> {bs_r_s} blocks/server"
        + (f", selectivity {disp_r}" if disp_r else ", no filter"),
        f"  {name_local} (already hashed): {_fmt(blocks_local)} blocks -> {bs_s_s} blocks/server"
        + (f", selectivity {disp_s}" if disp_s else ", no filter"),
        f"  p = {p_s} servers, shuffle fraction (p-1)/p = {frac_disp}",
        "",
        "Per-server steps (elapsed):",
        f"  1) read + select {name_redistributed}:        {step1}",
        f"  2) send (p-1)/p of filtered {name_redistributed}: {step2}",
        f"  3) receive (p-1)/p from others:        {step3}",
        f"  4) write own hash bucket:              {step4}",
        f"  5) read + select {name_local}:             {step5}",
        f"  6) write filtered {name_local}:            {step6}",
        f"  7) natural join (local):               {step7}",
        f"  8) projection:                         {step8}",
        "",
        f"  Elapsed = {elapsed}",
        f"  Total   = {p_s} * Elapsed = {total}",
    ])

    return {
        "operation":          "hash_shuffle_join",
        "idea":               idea,
        "redistributed_table": name_redistributed,
        "local_table":        name_local,
        "blocks_redistributed": blocks_redistributed,
        "blocks_local":       blocks_local,
        "bs_r":               bs_r,
        "bs_s":               bs_s,
        "num_processors":     p,
        "shuffle_fraction":   frac_disp,
        "selectivity_r":      disp_r or "1",
        "selectivity_s":      disp_s or "1",
        "steps": {
            "step1_read_select_r":  step1,
            "step2_shuffle_send":   step2,
            "step3_shuffle_recv":   step3,
            "step4_write_bucket":   step4,
            "step5_read_select_s":  step5,
            "step6_write_s":        step6,
            "step7_join":           step7,
            "step8_projection":     step8,
        },
        "elapsed":            elapsed,
        "total":              total,
        "explanation":        explanation,
    }


# ══════════════════════════════════════════════════════════════
#  SELECTIVE-FILTER BROADCAST JOIN
# ══════════════════════════════════════════════════════════════
#
# Used when a highly selective filter (e.g. a point predicate pid=650 plus a
# narrow range) shrinks one table to a TINY result of `result_blocks` blocks.
# That tiny result is read+filtered locally on every server, broadcast, and
# joined with the OTHER table read locally. The other table is usually NOT
# filtered (joined directly inside the join), so there are no separate
# read/select/write steps for it.
#
#   result_total   = max(1, result_blocks)             (gathered filtered size)
#   result_per_srv = max(1, ceil(result_total / p))    (each server's share to ship)
#
# Per-server steps (other table unfiltered):
#   1) read + select filtered table   : bs_filt                       * t_d
#   2) send result to (p-1) others     : (p-1) * result_per_srv         * t_s
#   3) receive from (p-1)             : (p-1) * result_per_srv         * t_s
#   4) write gathered result          : result_total                  * t_d
#   5) natural join (local)           : 3 * (result_total + bs_other)  * t_d
#   6) projection                     : 0
#
# If the other table IS filtered (sel_other given), two extra steps appear
# (read+select other, write filtered other) and the join uses sel_other*bs_other.
#
#   Elapsed = Σ(td) * t_d + Σ(ts) * t_s ;  Total = p * Elapsed (symmetric).
#
# How to get result_blocks: result_blocks = max(1, ceil(s * B_filtered)), where
# s is the combined selectivity of the filter. For an AND of independent uniform
# predicates, multiply their selectivities, e.g. pid=650 (1/1000) AND
# quantity>=91 (10/100) -> s = 1/10000.


def selective_broadcast_join_cost(
    blocks_filtered_table, blocks_other_table, num_processors, result_blocks,
    name_filtered="A", name_other="B",
    sel_other=None, join_field="",
    project_cost=0,
):
    """
    Join where a highly selective filter shrinks one table to `result_blocks`
    blocks, which is then broadcast and joined with the other (local) table.

    blocks_filtered_table: blocks of the table being filtered + broadcast.
    blocks_other_table:    blocks of the table joined locally (usually unfiltered).
    num_processors:        servers p.
    result_blocks:         block count of the filtered result (= max(1, ceil(s*B))).
    sel_other:             selectivity spec of the other table's filter, or None.
    """
    print(f"[TOOL] selective_broadcast_join_cost(filtered {name_filtered}={blocks_filtered_table}, "
          f"other {name_other}={blocks_other_table}, p={num_processors}, result_blocks={result_blocks}, "
          f"sel_other={sel_other!r}, join_field='{join_field}')")

    p = num_processors
    bs_filt  = math.ceil(blocks_filtered_table / p)
    bs_other = math.ceil(blocks_other_table / p)
    result_total   = max(1, int(result_blocks))
    result_per_srv = max(1, math.ceil(result_total / p))

    c_o, sym_o, disp_o = _sel_spec(sel_other)

    p_s        = _fmt(p)
    pm1_s      = _fmt(p - 1)
    bs_filt_s  = _fmt(bs_filt)
    bs_other_s = _fmt(bs_other)
    res_s      = _fmt(result_total)
    rpp_disp   = "" if result_per_srv == 1 else _fmt(result_per_srv)   # omit the "* 1"

    def mul(*parts):
        return " * ".join(x for x in parts if x not in ("", None))

    inner_filtered = sym_o or disp_o   # the other table has a real filter

    # ── per-server step display + elapsed accumulation ──────────────
    e_td, e_ts = {}, {}
    def add(d, c, s):
        d[s] = d.get(s, 0.0) + c

    steps = []
    steps.append((f"read + select {name_filtered}", mul(bs_filt_s, "t_d")))
    add(e_td, bs_filt, "")

    steps.append((f"send result to (p-1)", mul(pm1_s, rpp_disp, "t_s")))
    add(e_ts, (p - 1) * result_per_srv, "")

    steps.append(("receive from (p-1)", mul(pm1_s, rpp_disp, "t_s")))
    add(e_ts, (p - 1) * result_per_srv, "")

    steps.append((f"write gathered result", mul(res_s, "t_d")))
    add(e_td, result_total, "")

    if inner_filtered:
        steps.append((f"read + select {name_other}", mul(bs_other_s, "t_d")))
        add(e_td, bs_other, "")
        steps.append((f"write filtered {name_other}", mul(disp_o, bs_other_s, "t_d")))
        add(e_td, c_o * bs_other, sym_o)
        join_inner = mul(disp_o, bs_other_s)
        add(e_td, 3 * c_o * bs_other, sym_o)
    else:
        join_inner = bs_other_s
        add(e_td, 3 * bs_other, "")

    steps.append(("natural join (local)", f"3 * ({res_s} + {join_inner}) * t_d"))
    add(e_td, 3 * result_total, "")

    proj = mul(_fmt(project_cost), "t_d") if project_cost else "0"
    steps.append(("projection", proj))
    if project_cost:
        add(e_td, project_cost, "")

    e_td_expr = _lin_expr(e_td)
    e_ts_expr = _lin_expr(e_ts)
    if e_ts_expr and e_ts_expr != "0":
        elapsed = f"({e_td_expr}) * t_d + ({e_ts_expr}) * t_s"
    else:
        elapsed = f"({e_td_expr}) * t_d"

    t_td = {k: v * p for k, v in e_td.items()}
    t_ts = {k: v * p for k, v in e_ts.items()}
    t_td_expr = _lin_expr(t_td)
    t_ts_expr = _lin_expr(t_ts)
    if t_ts_expr and t_ts_expr != "0":
        total = f"({t_td_expr}) * t_d + ({t_ts_expr}) * t_s"
    else:
        total = f"({t_td_expr}) * t_d"

    jf = join_field or "the join field"
    idea = (
        f"The filter on {name_filtered} is highly selective, so the filtered result is "
        f"tiny ({res_s} block(s)). Each server reads and filters its {name_filtered} "
        f"partition, broadcasts the small result, and joins it locally with {name_other} "
        f"on {jf}. Broadcasting the tiny result is far cheaper than moving {name_other}."
    )

    step_lines = [f"  {i+1}) {label}:".ljust(38) + f" {cost}"
                  for i, (label, cost) in enumerate(steps)]

    explanation = "\n".join([
        f"Algorithm: selective-filter broadcast join.",
        f"  Idea: {idea}",
        f"  {name_filtered}: {_fmt(blocks_filtered_table)} blocks -> {bs_filt_s} blocks/server"
        f"  (filtered to {res_s} block(s) total, {rpp_disp or '1'} per server)",
        f"  {name_other}: {_fmt(blocks_other_table)} blocks -> {bs_other_s} blocks/server"
        + ("" if not inner_filtered else f", selectivity {disp_o}"),
        f"  p = {p_s} servers",
        "",
        "Per-server steps (elapsed):",
        *step_lines,
        "",
        f"  Elapsed = {elapsed}",
        f"  Total   = {p_s} * Elapsed = {total}",
    ])

    return {
        "operation":         "selective_broadcast_join",
        "idea":              idea,
        "filtered_table":    name_filtered,
        "other_table":       name_other,
        "blocks_filtered":   blocks_filtered_table,
        "blocks_other":      blocks_other_table,
        "bs_filtered":       bs_filt,
        "bs_other":          bs_other,
        "result_blocks":     result_total,
        "result_per_server": result_per_srv,
        "num_processors":    p,
        "steps":             {f"step{i+1}_{label.split()[0].lower()}": cost
                              for i, (label, cost) in enumerate(steps)},
        "elapsed":           elapsed,
        "total":             total,
        "explanation":       explanation,
    }


# ══════════════════════════════════════════════════════════════
#  CONDITION ANALYSIS  — derive the optimization / core idea
# ══════════════════════════════════════════════════════════════
#
# Reads the structured facts of a two-table join (each table's distribution,
# predicate field/type/selectivity, the join field, p) and DERIVES the data-
# movement optimization, the recommended cost tool, and a plain-language core
# idea — the reasoning a human writes before computing, e.g.
#   "orders.quantity is range-partitioned and the predicate is on quantity, so
#    matching rows sit on servers 6-10 and every row there already qualifies —
#    no second select needed."


def _analyze_table(t, p, join_field):
    """Pull out the derived facts for one table dict."""
    name   = t.get("name", "?")
    blocks = t.get("blocks")
    dist   = t.get("distribution", "round_robin")
    method, dfield = _dist_info(dist)
    ff     = (t.get("filter_field") or "").strip()
    ftype  = (t.get("filter_type") or "none").strip().lower()
    sel    = t.get("selectivity")
    c, sym, disp = _sel_spec(sel if sel not in (None, "") else None)
    sel_numeric = (sym == "" and sel not in (None, ""))
    bs = math.ceil(blocks / p) if blocks else None
    result_blocks = max(1, math.ceil(c * blocks)) if (sel_numeric and blocks) else None

    on_filter_field = bool(dfield and ff and dfield.lower() == ff.lower())
    on_join_field   = bool(dfield and join_field and dfield.lower() == join_field.lower())

    facts, active_k = [], None
    if method == "range" and on_filter_field and ftype in ("range", "point"):
        facts.append(
            f"{name} is range-partitioned on '{ff}', the field its predicate filters -> "
            f"matching rows are co-located on a contiguous subset of servers (partition pruning)."
        )
        if sel_numeric:
            active_k = max(1, round(c * p))
            facts.append(
                f"{name}: the predicate keeps fraction {disp} of a uniform range -> exactly "
                f"k = round({disp} * {p}) = {active_k} of {p} servers hold all matches, and EVERY "
                f"row on those servers already satisfies the predicate -> NO second select on {name}."
            )
    elif method == "hash" and on_filter_field and ftype == "point":
        facts.append(
            f"{name} is hash({ff})-partitioned and the predicate is a point match on '{ff}' -> "
            f"exactly 1 server holds the matching rows."
        )

    if method == "hash" and on_join_field:
        facts.append(f"{name} is already hash({join_field})-partitioned on the join field.")

    if ftype == "point" and result_blocks is not None and result_blocks <= max(1, p):
        facts.append(
            f"{name}'s filter is highly selective -> its result is only ~{_fmt(result_blocks)} "
            f"block(s), cheap to broadcast."
        )

    t2 = dict(t)
    t2.update(name=name, blocks=blocks, method=method, dfield=dfield, ff=ff, ftype=ftype,
              sel=sel, c=c, sym=sym, disp=disp, sel_numeric=sel_numeric, bs=bs,
              result_blocks=result_blocks, on_filter_field=on_filter_field,
              on_join_field=on_join_field, facts=facts, active_k=active_k)
    return t2


def analyze_join_conditions(spec_json):
    """
    Analyse the conditions of a two-table join and recommend the data-movement
    optimization. See the @tool wrapper for the input schema.
    """
    print(f"[TOOL] analyze_join_conditions(spec='{spec_json[:80]}...')")
    spec = json.loads(spec_json)
    p  = spec.get("num_processors", 1)
    jf = (spec.get("join_field") or "").strip()
    tables = spec.get("tables", [])
    if len(tables) != 2:
        return {"error": "analyze_join_conditions expects exactly 2 tables in 'tables'."}

    a = _analyze_table(tables[0], p, jf)
    b = _analyze_table(tables[1], p, jf)
    facts = a["facts"] + b["facts"]

    def jhash(t):   return t["method"] == "hash" and t["on_join_field"]
    def pruned(t):  return t["method"] == "range" and t["on_filter_field"] and t["ftype"] in ("range", "point")
    def filtered(t): return t["ftype"] in ("point", "range", "scan") or t["sel"] not in (None, "")

    rec, core, extra = None, None, {}

    if jhash(a) and jhash(b):
        rec  = "join_cost"   # parallel_join_cost detects the co-located/local case
        core = (f"Both {a['name']} and {b['name']} are hash({jf})-partitioned on the join field, "
                f"so matching rows already sit on the same server -> each server joins locally, "
                f"no communication.")
    elif jhash(a) or jhash(b):
        local  = a if jhash(a) else b
        redist = b if jhash(a) else a
        rec  = "hash_shuffle_join_cost"
        core = (f"{local['name']} is hash({jf})-partitioned, so re-hash {redist['name']} on {jf} "
                f"too -> matching rows co-locate and each server joins locally, no broadcast.")
        extra = {"local_table": local["name"], "redistributed_table": redist["name"]}
    elif pruned(a) or pruned(b):
        pr    = a if pruned(a) else b
        other = b if pruned(a) else a
        rec  = "range_broadcast_join_cost"
        k    = pr.get("active_k")
        core = (f"{pr['name']} is range-partitioned on its filter field, so its matching rows sit "
                f"on {('k = ' + str(k)) if k else 'a subset of'} servers and need no second select; "
                f"broadcast the smaller (filtered) {other['name']} to those servers and join locally.")
        extra = {"range_table": pr["name"], "broadcast_table": other["name"], "active_processors": k}
    else:
        # Round-robin: broadcast the SMALLER (after filtering) table, keep the other local.
        sized = sorted(
            ((t["result_blocks"] if t["result_blocks"] is not None else t["blocks"]), t)
            for t in (a, b)
        )
        smaller, larger = sized[0][1], sized[1][1]
        tiny_note = ""
        if smaller["result_blocks"] is not None and smaller["result_blocks"] <= max(1, p):
            tiny_note = (f" Its filter is so selective the result is only "
                         f"~{_fmt(smaller['result_blocks'])} block(s), very cheap to broadcast.")
        if not filtered(larger):
            # leaner plan: the local table is unfiltered -> no separate read/select/write for it
            rec  = "selective_broadcast_join_cost"
            core = (f"Broadcast the smaller filtered {smaller['name']} and join it locally with the "
                    f"unfiltered {larger['name']} (read directly inside the join, no extra "
                    f"select/write steps).{tiny_note}")
            extra = {"filtered_table": smaller["name"], "other_table": larger["name"],
                     "result_blocks": smaller["result_blocks"]}
        else:
            rec  = "select_broadcast_join_cost"
            core = (f"All tables are round-robin (no locality to exploit), so broadcast the smaller "
                    f"filtered {smaller['name']} to every server and join locally with the filtered "
                    f"{larger['name']}.{tiny_note}")
            extra = {"broadcast_table": smaller["name"], "local_table": larger["name"]}

    if not facts:
        facts.append("No special partitioning/selectivity shortcut applies; use the general broadcast plan.")

    return {
        "facts":                 facts,
        "recommended_algorithm": rec,
        "core_idea":             core,
        **extra,
    }


def apply_selectivity(blocks, selectivity_fraction):
    """
    Reduce a table's block count by a selectivity fraction.

    selectivity_fraction: float 0.0–1.0  (e.g. 0.01 = 1 % of tuples match).
    Returns the number of blocks remaining after the Select filter (minimum 1).
    """
    print(f"[TOOL] apply_selectivity(blocks={blocks}, sel={selectivity_fraction})")
    result = max(1, math.ceil(blocks * selectivity_fraction))
    return {
        "original_blocks":  blocks,
        "selectivity":      selectivity_fraction,
        "result_blocks":    result,
        "explanation": (
            f"After Select filter: ceil({_fmt(blocks)} × {selectivity_fraction}) = {_fmt(result)} blocks"
        ),
    }


# ══════════════════════════════════════════════════════════════
#  COMPOSE  (chain multiple operations)
# ══════════════════════════════════════════════════════════════

def compose_costs(steps):
    """
    Combine Elapsed and Total costs from a chain of atomic operations.
    Each step must be a dict with "elapsed" and "total" string keys.
    """
    print(f"[TOOL] compose_costs(steps=<{len(steps)} steps>)")

    if not steps:
        return {"elapsed": "0", "total": "0", "explanation": "No steps."}

    elapsed_parts = [s.get("elapsed", "0") for s in steps]
    total_parts   = [s.get("total",   "0") for s in steps]

    elapsed = " + ".join(f"({e})" for e in elapsed_parts)
    total   = " + ".join(f"({t})" for t in total_parts)
    elapsed_collected = simplify_cost_expr(elapsed)
    total_collected   = simplify_cost_expr(total)

    lines = [
        f"Step {i+1} [{s.get('operation', s.get('algorithm', '?'))}]: "
        f"Elapsed = {s.get('elapsed')}, Total = {s.get('total')}"
        for i, s in enumerate(steps)
    ]

    return {
        "elapsed":           elapsed,
        "total":             total,
        "elapsed_collected": elapsed_collected,
        "total_collected":   total_collected,
        "explanation": "\n".join(lines) + (
            f"\n\nCombined:\n  Elapsed = {elapsed}\n          = {elapsed_collected}   (collected)"
            f"\n  Total   = {total}\n          = {total_collected}   (collected)"
        ),
    }

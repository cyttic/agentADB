"""
tools/db_ops.py
================
Pure cost-calculation tools for parallel database operations.

All sizes are expressed in BLOCKS (never bytes — convert before using).
All cost outputs are SYMBOLIC strings like "9 * 10^3 * (t_d + t_s)" — never reduced.

Three atomic operations:
  - Select  — 3 algorithms, decision based on question type + data distribution
  - Sort    — placeholder formula: 3 * t_d * B_s    (to be refined)
  - Join    — placeholder formula: 3 * t_d * (B_s + B_t)  (to be refined)

These can be composed into compound queries (Select-Join, Select-Sort, etc.)
by calling them in sequence and combining their cost outputs.
"""

import math
import json


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def bytes_to_blocks(size_bytes, block_size):
    return math.ceil(size_bytes / block_size)


def records_to_blocks(record_count, record_size_bytes, block_size):
    records_per_block = max(1, block_size // record_size_bytes)
    return math.ceil(record_count / records_per_block)


def _fmt(n: int) -> str:
    """Format an integer as 10^k when it is an exact power of 10 (k >= 2), else as plain digits."""
    if n <= 0:
        return str(n)
    log = math.log10(n)
    k = round(log)
    if k >= 2 and abs(log - k) < 1e-9:
        return f"10^{k}"
    return str(n)


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
        f"  Row size   : {num_attributes} attributes × {cell_size_bytes} bytes/cell"
        f" = {row_size_bytes} bytes/row",
        f"  Table size : {record_count} records × {row_size_bytes} bytes/row"
        f" = {table_size_bytes} bytes",
        f"  Block count: ceil({table_size_bytes} / {block_size_bytes})"
        f" = {block_count} blocks",
        f"  → Use block_count = {block_count} for all subsequent calculations",
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
        elapsed = f"{bs_s} * t_d"
        total   = f"{rel_p} * {bs_s} * t_d" if rel_p > 1 else f"{bs_s} * t_d"
        explanation = (
            f"alg3: only {rel_p} relevant processor(s) participate.\n"
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

    if algorithm == "alg1":
        # ── Step-by-step symbolic strings ──────────────────
        step1_e = f"3 * {bs} * t_d"
        step2_e = f"{bs} * {p-1} * t_s"
        step3_e = f"{p-1} * {bs} * (t_s + t_d)"
        step4_e = f"{B} * t_d"

        step1_t = f"3 * {B} * t_d"     # all p procs do step1
        step2_t = step2_e               # only (p-1) procs send — same as elapsed
        step3_t = step3_e               # only proc 0 receives — same as elapsed
        step4_t = step4_e               # only proc 0 merges — same as elapsed

        elapsed = f"{step1_e} + {step2_e} + {step3_e} + {step4_e}"
        total   = f"{step1_t} + {step2_t} + {step3_t} + {step4_t}"

        explanation = "\n".join([
            "alg1 (local sort + gather + merge):",
            f"  Step 1 - all {p} procs sort {bs} blocks locally:  {step1_e}  (elapsed) / {step1_t}  (total)",
            f"  Step 2 - {p-1} procs send results to proc 0:      {step2_e}",
            f"  Step 3 - proc 0 reads {p-1} incoming runs:        {step3_e}",
            f"  Step 4 - proc 0 merges all {p} runs:              {step4_e}",
            "",
            f"  Elapsed = {elapsed}",
            f"  Total   = {total}",
        ])

    elif algorithm == "alg2":
        elapsed = f"3 * {bs} * t_d"
        total   = f"3 * {B} * t_d"

        explanation = "\n".join([
            "alg2 (fully local sort - range partition matches sort field):",
            f"  Each of {p} procs sorts its {bs} blocks independently, no communication.",
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
# ── Algorithm 2: BROADCAST JOIN (default) ─────────────────────
#   Used when distributions differ or partition field != join field.
#
#   Step 1 [send]    — every server sends its partition to all others:
#                      (bs_R + bs_S) * (t_s + t_d)
#   Step 2 [receive] — every server receives partitions from all others:
#                      (bs_R + bs_S) * (t_s + t_d)
#   Step 3 [join]    — every server performs a local Join:
#                      (bs_R + bs_S) * 3 * t_d
#
#   Elapsed = Step1 + Step2 + Step3
#   Total   = p * Elapsed


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
            f"  Total   = {total}",
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
            "explanation":       explanation,
        }

    else:
        # ── Algorithm 2: BROADCAST JOIN ──────────────────────────
        step1 = f"({sum_str}) * (t_s + t_d)"
        step2 = f"({sum_str}) * (t_s + t_d)"
        step3 = f"({sum_str}) * 3 * t_d"

        elapsed = f"({step1}) + ({step2}) + ({step3})"
        total   = f"{p_s} * ({elapsed})"

        reason = "distributions differ or partition field != join field" if not colocated else ""

        explanation = "\n".join([
            f"Join: {name_a} * {name_b}  [BROADCAST JOIN -- {reason}]",
            f"  {name_a}: {_fmt(blocks_a)} blocks -> {bs_a_s} blocks/server  (ceil({_fmt(blocks_a)}/{p_s}))",
            f"  {name_b}: {_fmt(blocks_b)} blocks -> {bs_b_s} blocks/server  (ceil({_fmt(blocks_b)}/{p_s}))",
            f"  p = {p_s} servers",
            f"",
            f"  Step 1 [send]    -- each server sends its partition to all others:",
            f"    {step1}",
            f"  Step 2 [receive] -- each server receives partitions from all others:",
            f"    {step2}",
            f"  Step 3 [join]    -- each server performs local Join:",
            f"    {step3}",
            f"",
            f"  Elapsed = {elapsed}",
            f"  Total   = {total}",
        ])

        return {
            "operation":         "join",
            "algorithm":         "broadcast",
            "table_a":           name_a,
            "table_b":           name_b,
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
            "explanation":       explanation,
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
            f"After Select filter: ceil({blocks} × {selectivity_fraction}) = {result} blocks"
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

    lines = [
        f"Step {i+1} [{s.get('operation', s.get('algorithm', '?'))}]: "
        f"Elapsed = {s.get('elapsed')}, Total = {s.get('total')}"
        for i, s in enumerate(steps)
    ]

    return {
        "elapsed":     elapsed,
        "total":       total,
        "explanation": "\n".join(lines) + f"\n\nCombined:\n  Elapsed = {elapsed}\n  Total   = {total}",
    }

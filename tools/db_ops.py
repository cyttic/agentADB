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

    alg1 — naive, never used.
    alg2 — every processor participates (local search + send results).
    alg3 — only relevant processors participate.

    Returns: {"algorithm": "alg2"|"alg3", "reason": "..."}
    """
    print(f"[TOOL] decide_select_algorithm(question_field='{question_field}', question_type='{question_type}', partition_key='{partition_key}', distribution='{distribution}')")

    dist_field = None
    if "(" in distribution and ")" in distribution:
        dist_field = distribution[distribution.index("(") + 1 : distribution.index(")")]

    if distribution == "round_robin":
        return {
            "algorithm": "alg2",
            "reason": "Round-robin: data spread evenly without locality, must search every processor.",
        }

    if dist_field != question_field:
        return {
            "algorithm": "alg2",
            "reason": f"Question on '{question_field}' but partitioned by '{dist_field}' — cannot pinpoint relevant processors.",
        }

    if distribution.startswith("hash("):
        if question_type == "point":
            return {
                "algorithm": "alg3",
                "reason": f"hash({dist_field}) + point search on '{question_field}' → exact processor identifiable.",
            }
        else:
            return {
                "algorithm": "alg2",
                "reason": f"hash({dist_field}) does not preserve order → range/scan must check all processors.",
            }

    if distribution.startswith("range("):
        if question_type in ("point", "range"):
            return {
                "algorithm": "alg3",
                "reason": f"range({dist_field}) + {question_type} search on '{question_field}' → only relevant processors participate.",
            }
        else:
            return {
                "algorithm": "alg2",
                "reason": f"range({dist_field}) + scan on '{question_field}' → must check all processors.",
            }

    return {
        "algorithm": "alg2",
        "reason": f"Unknown distribution '{distribution}', falling back to alg2.",
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

    if algorithm == "alg2":
        elapsed = f"{bs_per_proc} * t_d"
        total   = f"{p} * {bs_per_proc} * t_d"
        explanation = (
            f"alg2: all {p} processors scan {bs_per_proc} blocks each in parallel.\n"
            f"  Elapsed = {elapsed}\n"
            f"  Total   = {p} x Elapsed = {total}"
        )

    elif algorithm == "alg3":
        rel_p   = relevant_processors if relevant_processors else 1
        elapsed = f"{bs_per_proc} * t_d"
        total   = f"{rel_p} * {bs_per_proc} * t_d" if rel_p > 1 else f"{bs_per_proc} * t_d"
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
#  JOIN  (placeholder — refine later)
# ══════════════════════════════════════════════════════════════

def join_cost(blocks_s, blocks_t):
    """Placeholder formula: 3 * t_d * (B_s + B_t)"""
    print(f"[TOOL] join_cost(blocks_s={blocks_s}, blocks_t={blocks_t})")
    total_blocks = blocks_s + blocks_t
    elapsed = f"3 * {total_blocks} * t_d"
    total   = f"3 * {total_blocks} * t_d"
    return {
        "operation":   "join",
        "elapsed":     elapsed,
        "total":       total,
        "explanation": f"Join: 3 * t_d * (B_s + B_t) = 3 * t_d * ({blocks_s} + {blocks_t}) (placeholder, will be refined).",
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

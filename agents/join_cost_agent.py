"""
agents/join_cost_agent.py
==========================
LangGraph agent for parallel Join cost analysis.

TWO JOIN ALGORITHMS:

1. PARALLEL (HASH) JOIN — only when both tables are partitioned by the SAME
   method (hash or range) on exactly the join field.
   Matching tuples are co-located → no communication needed.
   Elapsed = 3 * (bs_R + bs_S) * t_d,  Total = p * Elapsed

2. REGULAR JOIN — all other cases.
   Broadcast the SMALLER (outer) table to all servers.
   bs_out = ceil(B_out / p),  bs_in = ceil(B_in / p)

   Step 1 [send]    — each server sends its outer partition to (p-1) others:
                      bs_out * t_d + (p-1) * bs_out * t_s
   Step 2 [receive] — each server receives outer from (p-1) others:
                      (p-1) * bs_out * (t_s + t_d)
   Step 3 [join]    — each server joins full outer with its local inner:
                      3 * (B_out + bs_in) * t_d

   Elapsed = Step1 + Step2 + Step3,  Total = p * Elapsed

   NOTE: S ⋈ F ≠ F ⋈ S in cost. Always choose outer = smaller table.

Compound expressions (Select + Join, Join + Join) are computed operation by operation;
the final cost is the sum of all individual operation costs.
"""

import os
import json
import math
import re as _re
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.db_ops import (
    parallel_join_cost        as _parallel_join_cost,
    apply_selectivity         as _apply_selectivity,
    compose_costs             as _compose_costs,
    decide_select_algorithm   as _decide_select_algorithm,
    select_cost               as _select_cost,
    compute_table_blocks_info as _compute_table_blocks_info,
    _fmt,
)


# ══════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════

class JoinAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def compute_table_blocks(
    record_count:     int,
    num_attributes:   int,
    cell_size_bytes:  int,
    block_size_bytes: int,
    table_name:       str = "",
) -> str:
    """
    Calculate block count for a table when it is NOT given directly.

    Call this BEFORE compute_parallel_join or compute_select_cost whenever the
    problem gives: number of records, number of attributes/fields, size of each
    cell (bytes), and block size (bytes) — instead of a direct block count.

    Formula (shown step by step):
      row_size_bytes   = num_attributes  × cell_size_bytes
      table_size_bytes = record_count    × row_size_bytes
      block_count      = ceil(table_size_bytes / block_size_bytes)

    Args:
        record_count:     Number of records/tuples in the table.
        num_attributes:   Number of columns/fields in the table.
        cell_size_bytes:  Size of one cell value in bytes.
        block_size_bytes: Size of one disk block in bytes.
        table_name:       Optional table name for display (e.g. "R").

    Returns JSON with step-by-step calculation and block_count to use next.
    """
    try:
        result = _compute_table_blocks_info(
            record_count, num_attributes, cell_size_bytes, block_size_bytes, table_name
        )
        print(result["explanation"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compute_parallel_join(
    blocks_a:       int,
    blocks_b:       int,
    num_processors: int,
    name_a:         str = "A",
    name_b:         str = "B",
    distribution_a: str = "round_robin",
    distribution_b: str = "round_robin",
    join_field:     str = "",
) -> str:
    """
    Compute Elapsed and Total cost for ONE parallel Join operation.

    Automatically chooses the algorithm based on data distribution:

    LOCAL JOIN (no communication):
      Condition: both tables are partitioned by the same method (both hash or both range)
                 on exactly the join_field.
      Matching tuples are guaranteed to be on the same server.
      Elapsed = 3 * (bs_a + bs_b) * t_d
      Total   = p * Elapsed

    REGULAR JOIN (default):
      Used when distributions differ or the partition field is not the join field.
      The smaller table is broadcast (outer); the larger stays local (inner).
      bs_out = ceil(B_out / p),  bs_in = ceil(B_in / p)
      Step 1 [send]:    bs_out * t_d + (p-1) * bs_out * t_s
      Step 2 [receive]: (p-1) * bs_out * (t_s + t_d)
      Step 3 [join]:    3 * (B_out + bs_in) * t_d
      Elapsed = Step1 + Step2 + Step3
      Total   = p * Elapsed
      Tool picks the cheaper ordering automatically (outer = smaller table).

    where bs_x = ceil(blocks_x / num_processors)

    Output is symbolic — never reduced to a single number.

    Args:
        blocks_a:       Total block count of the LEFT table.
        blocks_b:       Total block count of the RIGHT table.
        num_processors: Number of parallel servers (p).
        name_a:         Human-readable name of the left table (e.g. "Flowers").
        name_b:         Human-readable name of the right table (e.g. "Sales").
        distribution_a: Partition scheme of table A, e.g. "hash(name)", "range(id)", "round_robin".
        distribution_b: Partition scheme of table B, e.g. "hash(name)", "range(id)", "round_robin".
        join_field:     The field the Join is performed on (e.g. "name").

    Returns JSON with algorithm choice, elapsed, total, and a verbose explanation.
    """
    try:
        result = _parallel_join_cost(
            blocks_a, blocks_b, num_processors,
            name_a, name_b,
            distribution_a, distribution_b, join_field,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compute_select_cost(
    blocks:               int,
    num_processors:       int,
    distribution:         str,
    select_field:         str,
    condition_type:       str,
    selectivity_fraction: float = 1.0,
    table_name:           str = "",
    condition:            str = "",
) -> str:
    """
    Compute the cost of a Select (σ) operation AND the resulting reduced block count.

    Use this as STEP 1 in a Select+Join pipeline.
    The returned result_blocks feeds directly into compute_parallel_join as blocks_a or blocks_b.

    Algorithm decision (same rules as parallel Select):
      alg2 — all processors scan locally (round-robin, or select field != partition field,
              or hash/range on different field)
      alg3 — only relevant processors participate (hash/range on exactly the select field,
              point equality search)

    Cost formulas:
      alg2: Elapsed = ceil(B/p) * t_d,  Total = p * Elapsed
      alg3: Elapsed = ceil(B/p) * t_d,  Total = Elapsed  (only 1 proc for point search)

    Args:
        blocks:               Total block count of the table BEFORE the filter.
        num_processors:       Number of parallel servers (p).
        distribution:         Partition scheme: "hash(field)", "range(field)", "round_robin".
        select_field:         Field used in the WHERE condition (e.g. "price").
        condition_type:       "point" (field=value), "range" (field>a AND field<b), "scan" (field!=value).
        selectivity_fraction: Fraction of tuples that match the condition (0.0–1.0).
                              If unknown, use 1.0 (full scan — all blocks pass through).
                              Derive from distinct values V: point -> 1/V, range -> k/V.
        table_name:           Optional name for display (e.g. "Sales").
        condition:            Optional condition string for display (e.g. "100 < price < 200").

    Returns JSON with:
        operation:     "select"
        algorithm:     "alg2" or "alg3"
        elapsed:       symbolic cost string  ← include in sum_operation_costs
        total:         symbolic cost string  ← include in sum_operation_costs
        result_blocks: block count AFTER the filter  ← pass to compute_parallel_join
    """
    try:
        m         = _re.search(r'\((\w+)\)', distribution)
        dist_field = m.group(1) if m else ""

        algo = _decide_select_algorithm(select_field, condition_type, dist_field, distribution)
        rel_p = algo.get("relevant_processors") or 1
        cost  = _select_cost(blocks, num_processors, algo["algorithm"], rel_p)

        result_blocks = max(1, math.ceil(blocks * selectivity_fraction))
        bs = math.ceil(blocks / num_processors)

        return json.dumps({
            "operation":     "select",
            "table_name":    table_name,
            "condition":     condition,
            "algorithm":     algo["algorithm"],
            "reason":        algo["reason"],
            "elapsed":       cost["elapsed"],
            "total":         cost["total"],
            "result_blocks": result_blocks,
            "explanation": "\n".join([
                f"Select on {table_name or 'table'} [{algo['algorithm']}]",
                f"  Reason: {algo['reason']}",
                f"  {_fmt(blocks)} blocks / {num_processors} procs = {_fmt(bs)} blocks/server",
                f"  Elapsed = {cost['elapsed']}",
                f"  Total   = {cost['total']}",
                f"  After filter (selectivity={selectivity_fraction}): {_fmt(result_blocks)} blocks",
                f"  -- pass result_blocks={result_blocks} to compute_parallel_join",
            ]),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def sum_operation_costs(operations_json: str) -> str:
    """
    Combine costs from multiple sequential operations into a compound query cost.

    Use this when the RA expression chains operations:
      • Select + Join  →  cost(Select) + cost(Join)
      • Join + Join    →  cost(Join1)  + cost(Join2)

    Args:
        operations_json: JSON array of operation results, each containing
                         "operation", "elapsed", and "total" keys.
                         Example:
                         [
                           {"operation": "join", "elapsed": "...", "total": "..."},
                           {"operation": "join", "elapsed": "...", "total": "..."}
                         ]

    Returns combined Elapsed and Total as symbolic strings (sum of all steps).
    """
    try:
        steps  = json.loads(operations_json)
        result = _compose_costs(steps)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [
    compute_table_blocks,
    compute_parallel_join,
    compute_select_cost,
    sum_operation_costs,
]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert in parallel database systems specialising in Join cost analysis.

══ STEP 0 — BLOCK COUNT (do this first, before any cost calculation) ══

For EACH table in the problem, determine its block count using this priority:

  1. Block count given directly (e.g. "table R has 500 blocks")
     → use that number as-is, do NOT call compute_table_blocks.

  2. Block count NOT given — but record count + cell size + block size ARE given
     (e.g. "R(a,b,c) has 100 records, each cell is 20 bytes, block size = 40")
     → call compute_table_blocks FIRST.
        compute_table_blocks(record_count=100, num_attributes=3,
                             cell_size_bytes=20, block_size_bytes=40,
                             table_name="R")
        result: block_count = 150
     → use 150 as blocks_a / blocks_b in all subsequent calls.

NEVER compute block counts yourself in text.
NEVER pass record_count as block count.

══ TWO JOIN ALGORITHMS ══

-- PARALLEL (HASH) JOIN — no communication --
  Condition: BOTH tables are partitioned by the SAME method (both hash OR both range)
             on EXACTLY the join field.
  Matching tuples are co-located → no data transfer needed.

  bs_R = ceil(B_R / p),  bs_S = ceil(B_S / p)
  Elapsed = 3 * (bs_R + bs_S) * t_d
  Total   = p * Elapsed

-- REGULAR JOIN (default, all other cases) --
  Used when distributions differ, or partition field != join field, or round-robin.
  Broadcast the SMALLER (outer) table to all servers.
  Each server joins the full outer table with its local inner partition.

  bs_out = ceil(B_out / p),  bs_in = ceil(B_in / p)   [outer = smaller table]
  Step 1 [send]:    bs_out * t_d + (p-1) * bs_out * t_s
  Step 2 [receive]: (p-1) * bs_out * (t_s + t_d)
  Step 3 [join]:    3 * (B_out + bs_in) * t_d

  Elapsed = Step 1 + Step 2 + Step 3
  Total   = p * Elapsed

  IMPORTANT: S ⋈ F ≠ F ⋈ S in cost.
  The tool automatically selects the cheaper ordering (outer = smaller table).

══ WORKFLOW ══

-- SIMPLE JOIN (no Select) --
  1. Call compute_parallel_join(blocks_a, blocks_b, num_processors,
                                 distribution_a, distribution_b, join_field).
  2. Report Elapsed and Total from the result.

-- SELECT + JOIN (two-step pipeline) --
  Step 1: Call compute_select_cost for the filtered table.
          -- Returns elapsed, total for the Select AND result_blocks (reduced size).
  Step 2: Call compute_parallel_join using result_blocks as the filtered table size.
          -- Pass the ORIGINAL (unfiltered) size for the other table.
          -- Pass the distribution of the FILTERED table AS-IS (filter preserves partition).
  Step 3: Call sum_operation_costs([select_result, join_result]).
          -- Final cost = Select cost + Join cost (SUMMED, not nested).

-- JOIN + JOIN (A Join B Join C) --
  Step 1: Call compute_parallel_join for A Join B.
  Step 2: Use the stated or estimated result size of (A Join B) as one input.
          Call compute_parallel_join again for (A Join B) Join C.
  Step 3: Call sum_operation_costs([join1_result, join2_result]).

Key rule: ALWAYS call sum_operation_costs when there are 2+ operations.
          NEVER add costs manually in text.

══ STRICT OUTPUT FORMAT RULES ══
• PLAIN TEXT ONLY. No LaTeX, no MathJax, no markup of any kind.
  Forbidden: \[ \] \( \) \pi \sigma \bowtie \Big \frac \times \cdot $...$ $$...$$
• Use * for multiplication, ^ for exponents, / only for real division.
• Copy formula strings from tool output EXACTLY — do not reformat.
• NEVER simplify to a single number. NEVER compute numbers yourself.

• RELATIONAL ALGEBRA — use only these Unicode symbols (no backslash commands):
    Select:   σ(condition)(Table)
    Project:  π(fields)(Table)
    Join:     Table1 ⋈ Table2   or   Table1 ⋈(condition) Table2
  Example:
    π(cid)( σ(price > 100)(Products) ⋈(pid) σ(50 ≤ qty ≤ 100)(Orders) )

• Always end with this summary block:

---
Relational Algebra: <RA expression using σ π ⋈>

Step 1 [send]:    <formula>
Step 2 [receive]: <formula>
Step 3 [join]:    <formula>

Elapsed = <symbolic sum>
Total   = p * Elapsed = <symbolic>
---

LANGUAGE RULE: Always respond in English, regardless of input language.
"""


# ══════════════════════════════════════════════════════════════
#  BUILD AGENT
# ══════════════════════════════════════════════════════════════

def build_agent(llm=None):
    """
    Build the parallel Join cost agent.
    Args:
        llm: A LangChain chat model. Defaults to gpt-4o via OPENAI_API_KEY.
    """
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: JoinAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: JoinAgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(JoinAgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()

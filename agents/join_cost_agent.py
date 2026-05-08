"""
agents/join_cost_agent.py
==========================
LangGraph agent for parallel Join cost analysis.

ALGORITHM — broadcast Join on p servers:
  Each server holds a partition of every input table.
    bs_R = ceil(B_R / p)   blocks of R per server
    bs_S = ceil(B_S / p)   blocks of S per server

  Step 1 [send]    — every server sends its part to all others:  (bs_R + bs_S)(t_s + t_d)
  Step 2 [receive] — every server receives from all others:      (bs_R + bs_S)(t_s + t_d)
  Step 3 [join]    — every server performs a local Join:         (bs_R + bs_S) × 3 × t_d

  Elapsed = Step1 + Step2 + Step3
  Total   = p × Elapsed

Compound expressions (Select + Join, Join + Join) are computed operation by operation;
the final cost is the sum of all individual operation costs.
"""

import os
import json
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.db_ops import (
    parallel_join_cost  as _parallel_join_cost,
    apply_selectivity   as _apply_selectivity,
    compose_costs       as _compose_costs,
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

    BROADCAST JOIN (default):
      Used when distributions differ or the partition field is not the join field.
      Step 1 [send]    -- (bs_a + bs_b) * (t_s + t_d)
      Step 2 [receive] -- (bs_a + bs_b) * (t_s + t_d)
      Step 3 [join]    -- (bs_a + bs_b) * 3 * t_d
      Elapsed = Step1 + Step2 + Step3
      Total   = p * Elapsed

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
def apply_select_filter(
    blocks:               int,
    selectivity_fraction: float,
    table_name:           str = "",
    condition:            str = "",
) -> str:
    """
    Reduce a table's block count after applying a Select (σ) filter.

    Use this BEFORE compute_parallel_join when the RA expression contains a
    Select that feeds into a Join (e.g. σ_price>100(Sales) ⋈ Flowers).

    Args:
        blocks:               Block count of the table BEFORE the filter.
        selectivity_fraction: Fraction of tuples that satisfy the condition (0.0–1.0).
                              E.g. 0.01 means 1 % of tuples match.
                              If the problem gives distinct values V and a range condition,
                              derive it as (matching_values / V).
        table_name:           Optional: name of the table being filtered.
        condition:            Optional: human-readable condition string (e.g. "price > 100").

    Returns JSON with result_blocks (use this as input to compute_parallel_join).
    """
    try:
        result = _apply_selectivity(blocks, selectivity_fraction)
        result["table_name"] = table_name
        result["condition"]  = condition
        return json.dumps(result, indent=2)
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
    compute_parallel_join,
    apply_select_filter,
    sum_operation_costs,
]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert in parallel database systems specialising in Join cost analysis.

══ TWO JOIN ALGORITHMS ══

-- LOCAL JOIN (no communication) --
  Condition: BOTH tables are partitioned by the SAME method (both hash OR both range)
             on EXACTLY the join field.
  Matching tuples are guaranteed to be on the same server -- no data transfer needed.
  Every server joins its local partitions independently.

  bs_R = ceil(B_R / p),  bs_S = ceil(B_S / p)
  Elapsed = 3 * (bs_R + bs_S) * t_d
  Total   = p * Elapsed

-- BROADCAST JOIN (default, all other cases) --
  Used when distributions differ, or partition field != join field, or round-robin.

  Step 1 [send]    -- each server sends its partition to all others:
                     (bs_R + bs_S) * (t_s + t_d)
  Step 2 [receive] -- each server receives partitions from all others:
                     (bs_R + bs_S) * (t_s + t_d)
  Step 3 [join]    -- each server performs a local Join over all received data:
                     (bs_R + bs_S) * 3 * t_d

  Elapsed = Step 1 + Step 2 + Step 3
  Total   = p * Elapsed

══ WORKFLOW ══
1. Parse the task: identify table names, block counts, p (server count),
   distribution of each table, and the join field.
2. Identify the sequence of operations in the RA expression.
3. Execute operations in order:
   a. If a Select (sigma) precedes a Join:
      -- call apply_select_filter(blocks, selectivity_fraction, ...) to get the reduced block count.
      -- use result_blocks as input to the next compute_parallel_join call.
   b. For each Join: call compute_parallel_join with distribution_a, distribution_b, join_field.
      The tool picks the right algorithm automatically.
   c. For Join-of-Join (A Join B Join C): compute A Join B first, use the stated or estimated
      result size as input to the second join, then call compute_parallel_join again.
4. If there are multiple operations: call sum_operation_costs([op1, op2, ...]).
5. Present the final answer verbosely:
   -- State which algorithm was chosen and WHY (co-located vs not)
   -- Table sizes and per-server partition sizes
   -- Formula with numbers substituted (symbolic, not reduced)
   -- Elapsed and Total in symbolic form

══ STRICT OUTPUT FORMAT RULES ══
• PLAIN TEXT ONLY. Do NOT use LaTeX, MathJax, or any markup:
  -- Forbidden: \[ \] \( \) \times \frac \cdot \text{} $...$ $$...$$
  -- Use * for multiplication:   (10^3 + 10^5) * (t_s + t_d)
  -- Use ^ for exponents:        10^3  10^5
  -- Use / only for real division (e.g. 10^4 / 10 = 10^3)
• Copy the formula strings from tool output EXACTLY -- do not reformat them.
• NEVER simplify to a single number.
  Good: "(10^3 + 10^5) * (t_s + t_d)"    Bad: "101000 * (t_s + t_d)"
• NEVER compute numbers yourself -- always call tools.
• Show every arithmetic step explicitly.
• Always end with this summary block (plain text, no LaTeX):

---
Relational Algebra: <RA expression>

Step 1 [send]:    <formula>
Step 2 [receive]: <formula>
Step 3 [join]:    <formula>

Elapsed = <symbolic sum>
Total   = p * Elapsed = <symbolic>
---

LANGUAGE RULE: Always respond in Russian, regardless of input language.
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

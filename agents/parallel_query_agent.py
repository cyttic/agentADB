"""
agents/parallel_query_agent.py
================================
LangGraph agent for parallel database query cost analysis.

Pure orchestration — all math is in tools/db_ops.py.

The agent:
  1. Parses the schema once (sizes converted to BLOCKS).
  2. For each operation in the query (Select / Sort / Join):
     a. Picks the right algorithm from query type + data distribution.
     b. Calls the corresponding cost tool.
  3. Composes costs if the query chains multiple operations.
  4. Reports Elapsed and Total in symbolic form (never reduced).
"""

import os
import json
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.db_ops import (
    parse_schema             as _parse_schema,
    decide_select_algorithm  as _decide_select_algorithm,
    select_cost              as _select_cost,
    sort_cost                as _sort_cost,
    join_cost                as _join_cost,
    compose_costs            as _compose_costs,
)


# ══════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════

class QueryAgentState(TypedDict):
    messages:   Annotated[list, add_messages]
    db_context: dict   # canonical schema (sizes in blocks) — persists across turns


# ══════════════════════════════════════════════════════════════
#  TOOL WRAPPERS  (LangChain @tool around pure functions in db_ops)
# ══════════════════════════════════════════════════════════════

@tool
def parse_schema(schema_json: str) -> str:
    """
    Parse and store DB schema. Sizes are converted to BLOCKS internally.
    Call this FIRST when the user describes a database setup.

    PRIORITY RULE for block_count per relation:
      1. "block_count" given directly → use as-is (no calculation needed).
      2. "record_count" + "field_size_bytes" given → calculate from those.
      3. Only "record_count" given → use as proxy.
    Never ask the user for missing fields — work with what is provided.

    Required top-level: num_processors.
    Optional top-level: block_size (default 2000).

    Per-relation fields:
      - fields, key, distribution ("round_robin"|"hash(F)"|"range(F)")
      - block_count  (if known directly)  OR
      - record_count + field_size_bytes   (if block_count not given)

    Returns canonical schema with all sizes in BLOCKS.
    """
    try:
        result = _parse_schema(schema_json)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def decide_select_algorithm(
    question_field: str,
    question_type:  str,
    partition_key:  str,
    distribution:   str,
) -> str:
    """
    Decide which Select algorithm (alg2 / alg3) to use.

    Args:
        question_field: Field used in the WHERE condition (e.g. "pid").
        question_type:  "point" (id=10) | "range" (id>5 AND id<10) | "scan" (id!=10).
        partition_key:  The relation's partition key.
        distribution:   "round_robin" | "hash(<field>)" | "range(<field>)".

    Returns JSON with chosen algorithm and reason.
    """
    try:
        result = _decide_select_algorithm(
            question_field, question_type, partition_key, distribution
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def select_cost(
    block_count:         int,
    num_processors:      int,
    algorithm:           str,
    relevant_processors: int = 1,
) -> str:
    """
    Compute Elapsed and Total cost for a Select operation.

    All sizes must be in BLOCKS.
    For alg2: every processor participates.
    For alg3: only `relevant_processors` participate.

    Returns JSON with elapsed, total (symbolic strings), and explanation.
    """
    try:
        result = _select_cost(block_count, num_processors, algorithm, relevant_processors)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def sort_cost(block_count: int) -> str:
    """
    Compute Elapsed and Total cost for a Sort operation.
    Placeholder formula: 3 * t_d * B_s   (will be refined later).

    Returns JSON with elapsed, total, explanation.
    """
    try:
        result = _sort_cost(block_count)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def join_cost(blocks_s: int, blocks_t: int) -> str:
    """
    Compute Elapsed and Total cost for a Join operation.
    Placeholder formula: 3 * t_d * (B_s + B_t)   (will be refined later).

    Returns JSON with elapsed, total, explanation.
    """
    try:
        result = _join_cost(blocks_s, blocks_t)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compose_costs(steps_json: str) -> str:
    """
    Compose costs from multiple chained operations (e.g. Select-Join, Select-Sort).

    Args:
        steps_json: JSON array of step results, each with "elapsed" and "total" keys.
                    Example:
                    [
                      {"operation": "select", "elapsed": "1500 * t_d", "total": "15000 * t_d"},
                      {"operation": "join",   "elapsed": "3 * 16500 * t_d", "total": "3 * 16500 * t_d"}
                    ]

    Returns combined Elapsed and Total as symbolic strings.
    """
    try:
        steps  = json.loads(steps_json)
        result = _compose_costs(steps)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [
    parse_schema,
    decide_select_algorithm,
    select_cost,
    sort_cost,
    join_cost,
    compose_costs,
]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  (decision logic for the agent)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a parallel database systems expert specializing in query cost analysis.

═══ CORE PRINCIPLES ═══

1. ALL SIZES ARE IN BLOCKS, never bytes. If input is in bytes, parse_schema converts.
2. NEVER REDUCE NUMERICAL EXPRESSIONS. Always keep symbolic form.
   GOOD:  "Total = 9 * 10^3 * (t_d + t_s)"
   BAD:   "Total = 9000 * t_d + 9000 * t_s"
3. Always report TWO values: Elapsed (parallel time) and Total (across all procs).
4. Express costs only in t_d (disk access) and t_s (block transfer).

═══ ATOMIC OPERATIONS ═══

The query may contain ANY combination of these three atomic operations:

▶ SELECT — three algorithms exist:
   • alg1 — naive, ignored (never used).
   • alg2 — local search on EVERY processor, results sent to coordinator.
              Every proc participates → Total = p × Elapsed.
   • alg3 — only relevant processor(s) participate. Elapsed = Total (when 1 proc).

   Algorithm choice depends on:
     - Question type: "point" (id=10) | "range" (5<id<10) | "scan" (id!=10)
     - Data distribution: round_robin | hash(<field>) | range(<field>)
     - Whether the question field matches the partition field

   Decision logic (use decide_select_algorithm tool — DO NOT decide by hand):
     - round_robin                          → alg2 (no locality)
     - hash(F) + question on F + point      → alg3 (hash gives exact proc)
     - hash(F) + question on F + range/scan → alg2 (hash breaks order)
     - range(F) + question on F + point/range → alg3 (range preserves order)
     - question on different field than partition → alg2

▶ SORT — placeholder formula: 3 * t_d * B_s   (will be refined)

▶ JOIN — placeholder formula: 3 * t_d * (B_s + B_t)   (will be refined)

═══ COMPOUND QUERIES ═══

A task may chain operations (e.g. Select-Join, Select-Sort-Join).
For these:
  1. Compute cost of each atomic step separately.
  2. Pass the intermediate result size in BLOCKS to the next step.
  3. Use compose_costs to combine the final Elapsed and Total.

═══ WORKFLOW ═══

1. Call parse_schema FIRST (once per session) to get canonical schema with sizes in blocks.
2. Identify which atomic operations the query needs and their order.
3. For each Select:
   a. Call decide_select_algorithm with question type + distribution.
   b. Call select_cost with the chosen algorithm.
4. For each Sort/Join: call sort_cost / join_cost.
5. If multiple operations: call compose_costs to combine.
6. Present the final answer:
   • Relational Algebra expression
   • Algorithm choice + reason for each step
   • Elapsed and Total in symbolic form

LANGUAGE RULE: Always respond in Russian, regardless of input language.
"""


# ══════════════════════════════════════════════════════════════
#  BUILD AGENT
# ══════════════════════════════════════════════════════════════

def build_agent():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: QueryAgentState):
        ctx_note = ""
        if state.get("db_context"):
            ctx_note = f"\n\nCurrent DB context (sizes in blocks):\n{json.dumps(state['db_context'], indent=2)}"

        messages = [SystemMessage(content=SYSTEM_PROMPT + ctx_note)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: QueryAgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(QueryAgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()
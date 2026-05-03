"""
agents/parallel_query_agent.py
================================
LangGraph agent for parallel database query cost analysis.
Exposes build_agent() → compiled graph.

State includes db_context which persists across turns:
  - relations: {name: {fields, record_count, field_size}}
  - block_size: int (bytes)
  - num_processors: int
  - t_d: symbol/value for disk access cost
  - t_s: symbol/value for block transfer cost
"""

import os
import json
import math
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


# ══════════════════════════════════════════════════════════════
#  AGENT STATE  (includes persistent db_context)
# ══════════════════════════════════════════════════════════════

class QueryAgentState(TypedDict):
    messages:   Annotated[list, add_messages]
    db_context: dict   # persists schema + params across turns


# ══════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def parse_db_schema(schema_json: str) -> str:
    """
    Parse and store database schema, table sizes, and system parameters.

    Call this FIRST when the user describes a database setup.

    Args:
        schema_json: JSON string with structure:
        {
          "block_size": 2000,
          "num_processors": 10,
          "t_d": "t_d",
          "t_s": "t_s",
          "relations": {
            "Customers": {"fields": ["cid","name","city"], "record_count": 1000000, "field_size": 10},
            "Orders":    {"fields": ["pid","cid","date","quantity"], "record_count": 100000000, "field_size": 10},
            "Products":  {"fields": ["pid","name","price"], "record_count": 1000000, "field_size": 10}
          },
          "field_info": {
            "Orders.quantity": {"distinct_values": 100, "range": "1..100", "distribution": "uniform"},
            "pid":             {"distinct_values": 1000, "distribution": "uniform"}
          }
        }

    Returns:
        Confirmation JSON with computed block counts per relation.
    """
    try:
        ctx = json.loads(schema_json)
        block_size = ctx.get("block_size", 2000)
        result = {"status": "ok", "block_counts": {}}

        for rel_name, rel in ctx.get("relations", {}).items():
            num_fields  = len(rel.get("fields", []))
            field_size  = rel.get("field_size", 10)
            record_size = num_fields * field_size
            records_per_block = max(1, block_size // record_size)
            block_count = math.ceil(rel["record_count"] / records_per_block)
            result["block_counts"][rel_name] = {
                "record_size_bytes": record_size,
                "records_per_block": records_per_block,
                "block_count": block_count,
            }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compute_block_count(record_count: int, num_fields: int, field_size: int, block_size: int) -> str:
    """
    Compute the number of disk blocks needed for a relation.

    Args:
        record_count: Total number of records.
        num_fields:   Number of fields per record.
        field_size:   Size of each field in bytes.
        block_size:   Block size in bytes.

    Returns:
        JSON with record_size, records_per_block, block_count.
    """
    try:
        record_size = num_fields * field_size
        records_per_block = max(1, block_size // record_size)
        block_count = math.ceil(record_count / records_per_block)
        return json.dumps({
            "record_size_bytes":  record_size,
            "records_per_block":  records_per_block,
            "block_count":        block_count,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compute_selectivity(distinct_values: int, condition: str, record_count: int) -> str:
    """
    Compute selectivity and estimated result size for a simple condition.

    Supports conditions like:
      - "eq:650"       → equality  (1 / distinct_values)
      - "range:50:100" → range     (range_size / distinct_values)

    Args:
        distinct_values: Number of distinct values for the field.
        condition:       Condition string as described above.
        record_count:    Total records in the relation.

    Returns:
        JSON with selectivity factor and estimated result record count.
    """
    try:
        parts = condition.split(":")
        if parts[0] == "eq":
            sel = 1 / distinct_values
        elif parts[0] == "range":
            lo, hi = int(parts[1]), int(parts[2])
            sel = (hi - lo + 1) / distinct_values
        else:
            return json.dumps({"error": f"Unknown condition type: {parts[0]}"})

        estimated = math.ceil(sel * record_count)
        return json.dumps({
            "selectivity":        round(sel, 6),
            "estimated_records":  estimated,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compute_parallel_cost(
    algorithm: str,
    block_count: int,
    num_processors: int,
    blocks_to_transfer: int = 0,
) -> str:
    """
    Compute parallel and total time for a distributed operation.

    Algorithms supported:
      - "scan_round_robin"  : each processor scans block_count / num_processors blocks
      - "scan_range"        : only processors holding relevant range participate
      - "scan_hash"         : only processors holding matching hash buckets participate
      - "broadcast_join"    : one relation broadcast to all, each scans local partition

    Args:
        algorithm:           One of the algorithm names above.
        block_count:         Total blocks of the main relation.
        num_processors:      Number of processors.
        blocks_to_transfer:  Blocks transferred between processors (for joins/broadcasts).

    Returns:
        JSON with parallel_time and total_time expressed in t_d / t_s units.
    """
    try:
        p = num_processors
        B = block_count
        Bt = blocks_to_transfer

        if algorithm == "scan_round_robin":
            local_blocks = math.ceil(B / p)
            parallel_time = f"{local_blocks} * t_d"
            total_time    = f"{B} * t_d"

        elif algorithm == "scan_range":
            # Only relevant processors work
            local_blocks  = math.ceil(B / p)
            parallel_time = f"{local_blocks} * t_d"
            total_time    = f"{local_blocks} * t_d"  # only those processors active

        elif algorithm == "scan_hash":
            local_blocks  = math.ceil(B / p)
            parallel_time = f"{local_blocks} * t_d"
            total_time    = f"{local_blocks} * t_d"

        elif algorithm == "broadcast_join":
            local_scan    = math.ceil(B / p)
            parallel_time = f"{Bt} * t_s + {local_scan} * t_d"
            total_time    = f"{p} * {Bt} * t_s + {B} * t_d"

        else:
            return json.dumps({"error": f"Unknown algorithm: {algorithm}"})

        return json.dumps({
            "algorithm":     algorithm,
            "parallel_time": parallel_time,
            "total_time":    total_time,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [
    parse_db_schema,
    compute_block_count,
    compute_selectivity,
    compute_parallel_cost,
]

SYSTEM_PROMPT = """You are a parallel database systems expert specializing in query cost analysis.

You have these tools:
1. parse_db_schema       — call FIRST to store the DB schema, sizes, and parameters
2. compute_block_count   — calculate blocks for any relation
3. compute_selectivity   — estimate result size after a filter condition
4. compute_parallel_cost — compute parallel and total execution time

When a user gives you a query problem:
1. Parse and confirm the schema using parse_db_schema (once per session).
2. For each query:
   a. Write the Relational Algebra (RA) expression.
   b. Identify the partition method (round-robin / range / hash) and choose the algorithm.
   c. Use compute_block_count and compute_selectivity to get exact numbers.
   d. Use compute_parallel_cost to get parallel and total time.
   e. Explain each step clearly, showing the formula and substituted values.
3. Always show both parallel time and total time.
4. Express costs in terms of t_d (disk access) and t_s (block transfer).

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
        # Inject db_context as a system note if it exists
        ctx_note = ""
        if state.get("db_context"):
            ctx_note = f"\n\nCurrent DB context:\n{json.dumps(state['db_context'], indent=2)}"

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

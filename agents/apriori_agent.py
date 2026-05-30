"""
agents/apriori_agent.py
=======================
LangGraph agent for data-mining frequent-itemset problems solved with the
Apriori algorithm.

One tool
--------
run_apriori_algorithm(transactions_json, min_support, min_confidence)
  Runs Apriori deterministically in Python and returns a complete,
  verbose, step-by-step solution (input table echo + every pass).

Workflow (in system prompt)
---------------------------
1. Parse the transaction table (TID → items) from the user's message.
2. Parse the thresholds. In this course the conditions are written as
   "D = S = 0.5":  S = minimum support,  D = minimum confidence.
3. Call run_apriori_algorithm ONCE.
4. Print the tool output VERBATIM.

This is the first agent of the Data-Mining class. Sibling tasks to be added
later (same transaction-table input): Apriori-TID, association rules,
maximal frequent itemsets, closed frequent itemsets.
"""

import json
import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.apriori_ops import run_apriori


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class AprioriAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def run_apriori_algorithm(transactions_json: str,
                          min_support: float,
                          min_confidence: float = -1.0) -> str:
    """
    Run the Apriori algorithm on a transaction table and return a complete,
    formatted, step-by-step solution string.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, after you have parsed the transaction
    table and the support threshold from the user's message.

    ARGUMENT FORMAT
    ---------------
    transactions_json:
      JSON object mapping each TID (string) to its list of items.
      Build it directly from the table in the problem.
      Example for the table  1:{A,B} 2:{A,B,C} 3:{A,C,D} 4:{C,D}:
        '{"1": ["A","B"], "2": ["A","B","C"], "3": ["A","C","D"], "4": ["C","D"]}'

    min_support:
      The minimum support threshold S, as a FRACTION in [0, 1].
      In this course it is written like "D = S = 0.5" → pass 0.5.
      If the problem gives support as a count (e.g. "min support = 2") with
      N transactions, convert it: min_support = count / N.

    min_confidence:
      The minimum confidence threshold D, as a fraction in [0, 1], if the
      problem states one (e.g. "D = S = 0.5" → 0.5). It is only echoed in the
      parameter banner for Apriori (used by the association-rules task later).
      If the problem gives no confidence, omit it (leave as -1.0).

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM to the user. Do not summarise, reorder,
    re-compute, or reformat it. The output already contains the echoed input
    table, every pass (C1/F1, C2/F2, …), and the final union of frequent
    itemsets.
    """
    try:
        conf = None if min_confidence is None or min_confidence < 0 else min_confidence
        return run_apriori(transactions_json, min_support, conf)
    except Exception as exc:
        return f"Error running Apriori: {exc}"


tools = [run_apriori_algorithm]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a data-mining expert specialising in frequent-itemset mining with the
Apriori algorithm.

══ WHAT YOU DO ══

Given a transaction table (TID → set of items) and a minimum support
threshold, you find ALL frequent itemsets using Apriori:
  Pass 1: count support of every single item → F1.
  Pass k: generate Ck by joining F(k-1) with itself (union pairs, keep unions
          of size k), count support → Fk. Stop when Fk is empty.
  Result: F1 ∪ F2 ∪ … (all frequent itemsets).

══ WORKFLOW ══

Step 1 — Parse the transaction table from the user's message into a JSON
  object {TID: [items]}.
    A table with rows  tid|items:  1|A,B   2|A,B,C   3|A,C,D   4|C,D
    becomes  {"1":["A","B"], "2":["A","B","C"], "3":["A","C","D"], "4":["C","D"]}

Step 2 — Parse the thresholds. In this course the conditions look like
  "D = S = 0.5":
    • S = minimum SUPPORT  → pass as min_support (fraction in [0,1]).
    • D = minimum CONFIDENCE → pass as min_confidence (fraction in [0,1]).
  If support is given as a count instead of a fraction, divide by the number
  of transactions N to get the fraction.
  If only S is given, pass min_confidence = -1.0.

Step 3 — Call run_apriori_algorithm(transactions_json, min_support,
  min_confidence) EXACTLY ONCE.

Step 4 — Print the tool output VERBATIM (character for character). It already
  contains: the echoed input table, the parameters, every pass with full
  support calculations, and the final union of all frequent itemsets. Do not
  add, drop, or rephrase anything except, if you wish, a single closing line
  restating the final set of frequent itemsets.

══ STRICT RULES ══
• Never compute support or generate candidates yourself — always use the tool.
• Plain text only. No LaTeX, no markdown math.
• Always respond in English.
"""


# ══════════════════════════════════════════════════════════════
#  GRAPH
# ══════════════════════════════════════════════════════════════

def build_agent(llm=None):
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model='gpt-4o', temperature=0,
            api_key=os.environ.get('OPENAI_API_KEY', ''),
        )

    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: AprioriAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: AprioriAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(AprioriAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

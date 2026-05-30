"""
agents/apriori_tid_agent.py
===========================
LangGraph agent for data-mining frequent-itemset problems solved with the
Apriori-TID algorithm (tid_list / vertical method).

One tool
--------
run_apriori_tid_algorithm(transactions_json, min_support, min_confidence)
  Runs Apriori-TID deterministically in Python and returns a complete, verbose,
  step-by-step solution: input table echo, every step's candidate set and
  frequent set, each itemset's tid_list (by intersection of parents), and the
  support read off it.

Sibling of agents/apriori_agent.py — same transaction-table input, but support
is computed from tid_lists:  Support(I) = |I.tid_list| / |D|,  and a k-itemset's
tid_list is the intersection of its two parents' tid_lists.
"""

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.apriori_tid_ops import run_apriori_tid


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class AprioriTidAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def run_apriori_tid_algorithm(transactions_json: str,
                              min_support: float,
                              min_confidence: float = -1.0) -> str:
    """
    Run the Apriori-TID algorithm on a transaction table and return a complete,
    formatted, step-by-step solution string.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, after you have parsed the transaction
    table and the support threshold from the user's message. Use this (not the
    plain Apriori tool) whenever the problem asks for the Apriori-TID method,
    the tid_list / vertical method, or defines Support(I) = |I.tid_list| / |D|.

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
      If support is given as a count with N transactions, pass count / N.

    min_confidence:
      The minimum confidence threshold D, as a fraction in [0, 1], if stated
      (e.g. "D = S = 0.5" → 0.5). Echoed in the parameter banner only.
      If no confidence is given, omit it (leave as -1.0).

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM. Do not summarise, reorder, re-compute, or
    reformat it. It already contains the echoed input table, every step's
    candidate set (Ck) and frequent set (Fk), each tid_list with its
    intersection, the support, and the final union of frequent itemsets.
    """
    try:
        conf = None if min_confidence is None or min_confidence < 0 else min_confidence
        return run_apriori_tid(transactions_json, min_support, conf)
    except Exception as exc:
        return f"Error running Apriori-TID: {exc}"


tools = [run_apriori_tid_algorithm]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a data-mining expert specialising in frequent-itemset mining with the
Apriori-TID algorithm.

══ WHAT YOU DO ══

Given a transaction table (TID → set of items) and a minimum support
threshold, you find ALL frequent itemsets using Apriori-TID:
  Step 1: build a tid_list (set of TIDs containing the item) for every item.
          Support(I) = |I.tid_list| / |D|, where |D| = number of transactions.
  Step k: generate Ck by joining F(k-1) with itself (union pairs, keep unions
          of size k). Each candidate's tid_list = intersection of its two
          parents' tid_lists. Frequent if support ≥ S. Stop when Fk is empty.
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
  If support is given as a count, divide by the number of transactions N.
  If only S is given, pass min_confidence = -1.0.

Step 3 — Call run_apriori_tid_algorithm(transactions_json, min_support,
  min_confidence) EXACTLY ONCE.

Step 4 — Print the tool output VERBATIM (character for character). It already
  contains the echoed input table, the parameters, every step with its
  candidate set, tid_list intersections, support calculations, frequent set,
  and the final union of all frequent itemsets. Do not add, drop, or rephrase
  anything except, if you wish, a single closing line restating the result.

══ STRICT RULES ══
• Never compute tid_lists, intersections, or support yourself — use the tool.
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

    def call_llm(state: AprioriTidAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: AprioriTidAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(AprioriTidAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

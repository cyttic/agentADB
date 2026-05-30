"""
agents/maximal_itemsets_agent.py
================================
LangGraph agent for the data-mining task "find maximal frequent itemsets".

One tool
--------
run_maximal_itemsets_algorithm(transactions_json, min_support)
  Deterministically finds all frequent itemsets, then for each one tests its
  immediate supersets: a frequent itemset is MAXIMAL when none of its proper
  supersets is frequent. Returns a verbose step-by-step solution.

Sibling of the other data-mining agents — same transaction-table input. Only a
support threshold S is needed (no confidence).
"""

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.maximal_itemsets_ops import run_maximal_itemsets


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class MaximalItemsetsAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def run_maximal_itemsets_algorithm(transactions_json: str,
                                   min_support: float) -> str:
    """
    Find all maximal frequent itemsets from a transaction table and return a
    complete, formatted, step-by-step solution string.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, after parsing the transaction table and
    the support threshold S from the message.

    ARGUMENT FORMAT
    ---------------
    transactions_json:
      JSON object mapping each TID (string) to its list of items.
      Example for  1:{A,B} 2:{A,B,C} 3:{A,C,D} 4:{C,D}:
        '{"1": ["A","B"], "2": ["A","B","C"], "3": ["A","C","D"], "4": ["C","D"]}'

    min_support:
      The minimum support threshold S, as a FRACTION in [0, 1]. If given as a
      count over N transactions, pass count / N. If the problem gives no
      support threshold, pass 0.0 (every occurring itemset is frequent).

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM. It already contains the echoed input
    table, the frequent itemsets with supports, the per-itemset maximality
    check (each immediate superset's support and frequent/not-frequent
    verdict), and the final set of maximal frequent itemsets. Do not
    summarise, reorder, re-compute, or reformat it.
    """
    try:
        return run_maximal_itemsets(transactions_json, min_support)
    except Exception as exc:
        return f"Error finding maximal frequent itemsets: {exc}"


tools = [run_maximal_itemsets_algorithm]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a data-mining expert specialising in maximal frequent itemsets.

══ WHAT YOU DO ══

Given a transaction table (TID → set of items) and a minimum support S, find
all MAXIMAL frequent itemsets:
  - An itemset I is frequent when support(I) ≥ S.
  - A frequent itemset I is MAXIMAL when every proper superset J ⊃ I is
    infrequent (support(J) < S). Equivalently: none of the immediate
    supersets I ∪ {x} is frequent.

══ WORKFLOW ══

Step 1 — Parse the transaction table from the user's message into a JSON
  object {TID: [items]}.
    rows  1|A,B  2|A,B,C  3|A,C,D  4|C,D
    →  {"1":["A","B"], "2":["A","B","C"], "3":["A","C","D"], "4":["C","D"]}

Step 2 — Parse the support threshold S (fraction in [0,1]); if a count, divide
  by the number of transactions N. If no support threshold is given, use 0.0.

Step 3 — Call run_maximal_itemsets_algorithm(transactions_json, min_support)
  EXACTLY ONCE.

Step 4 — Print the tool output VERBATIM (character for character). It already
  contains the echoed table, the frequent itemsets, the maximality check for
  each one, and the final set of maximal frequent itemsets. Do not add, drop,
  or rephrase anything except, if you wish, a single closing line restating
  the result.

══ STRICT RULES ══
• Never compute support or decide maximality yourself — always use the tool.
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

    def call_llm(state: MaximalItemsetsAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: MaximalItemsetsAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(MaximalItemsetsAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

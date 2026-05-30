"""
agents/association_rules_agent.py
=================================
LangGraph agent for the data-mining task "find all association rules".

One tool
--------
run_association_rules_algorithm(transactions_json, min_support, min_confidence)
  Deterministically finds all frequent itemsets, enumerates every candidate
  rule I → J from each frequent itemset of size ≥ 2, computes confidence
  conf(I → J) = support(Z)/support(I), and keeps the rules with conf ≥ C.

Sibling of agents/apriori_agent.py / apriori_tid_agent.py — same transaction-
table input. This is the task where the confidence threshold C (written D or C
in the conditions) is finally used.
"""

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.association_rules_ops import run_association_rules


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class AssociationRulesAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def run_association_rules_algorithm(transactions_json: str,
                                    min_support: float,
                                    min_confidence: float) -> str:
    """
    Find all association rules from a transaction table and return a complete,
    formatted, step-by-step solution string.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, after parsing the transaction table, the
    support threshold S, and the confidence threshold C from the message.

    ARGUMENT FORMAT
    ---------------
    transactions_json:
      JSON object mapping each TID (string) to its list of items.
      Example for  1:{A,B} 2:{A,B,C} 3:{A,C,D} 4:{C,D}:
        '{"1": ["A","B"], "2": ["A","B","C"], "3": ["A","C","D"], "4": ["C","D"]}'

    min_support:
      The minimum support threshold S, as a FRACTION in [0, 1]. If given as a
      count over N transactions, pass count / N.

    min_confidence:
      The minimum confidence threshold C, as a FRACTION in [0, 1]. In this
      course it may be written as "C", or as "D" in "D = S = 0.5". A rule is
      kept when conf(I → J) = support(Z)/support(I) ≥ C.

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM. Do not summarise, reorder, re-compute, or
    reformat it. It already contains the echoed input table, the frequent
    itemsets with supports, every candidate rule with its confidence
    calculation and keep/discard verdict, and the final set of rules.
    """
    try:
        return run_association_rules(transactions_json, min_support, min_confidence)
    except Exception as exc:
        return f"Error generating association rules: {exc}"


tools = [run_association_rules_algorithm]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a data-mining expert specialising in association-rule mining.

══ WHAT YOU DO ══

Given a transaction table (TID → set of items), a minimum support S and a
minimum confidence C, you find ALL association rules:
  1. Find every frequent itemset (support ≥ S).
  2. For each frequent itemset Z with |Z| ≥ 2, form every rule I → J where
     I and J are non-empty, I ∩ J = ∅ and I ∪ J = Z.
  3. conf(I → J) = support(I ∪ J) / support(I) = support(Z) / support(I).
  4. Keep the rules with conf ≥ C.

══ WORKFLOW ══

Step 1 — Parse the transaction table from the user's message into a JSON
  object {TID: [items]}.
    rows  1|A,B  2|A,B,C  3|A,C,D  4|C,D
    →  {"1":["A","B"], "2":["A","B","C"], "3":["A","C","D"], "4":["C","D"]}

Step 2 — Parse the thresholds:
    • S = minimum SUPPORT  → min_support (fraction in [0,1]).
    • C = minimum CONFIDENCE → min_confidence (fraction in [0,1]).
  The confidence threshold may appear as "C" or as "D" (e.g. "D = S = 0.5").
  If support/confidence is a count, divide by the number of transactions N.
  If the problem gives ONLY a confidence threshold and NO support threshold
  (e.g. "C = 0.8" with no S), pass min_support = 0.0 — every itemset that
  actually occurs is considered, with no support filtering.

Step 3 — Call run_association_rules_algorithm(transactions_json, min_support,
  min_confidence) EXACTLY ONCE.

Step 4 — Print the tool output VERBATIM (character for character). It already
  contains the echoed input table, the frequent itemsets, every candidate rule
  with its confidence calculation, and the final set of rules. Do not add,
  drop, or rephrase anything except, if you wish, a single closing line
  restating the result set.

══ STRICT RULES ══
• Never compute support or confidence yourself — always use the tool.
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

    def call_llm(state: AssociationRulesAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: AssociationRulesAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(AssociationRulesAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

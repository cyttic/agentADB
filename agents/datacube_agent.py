"""
agents/datacube_agent.py
=========================
LangGraph agent for data-cube materialisation problems.

Solves Ullman's greedy approximation algorithm deterministically via a tool,
then presents the full step-by-step solution to the user.

Typical input
-------------
"Given a partial Hasse diagram with access costs:
  A: 300
  B: 100, C: 250, D: 100
  E: 80, F: 90
Hierarchy: A→B,C,D  B→E  C→E,F  D→F
Apply Ullman's algorithm to select N=3 cubes."

Agent workflow
--------------
1. Parse costs and hierarchy from the user's text.
2. Call run_ullman_algorithm(costs_json, hierarchy_json, n).
3. Print the returned step-by-step solution verbatim.
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

from tools.cube_ops import run_ullman


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class CubeAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def run_ullman_algorithm(costs_json: str, hierarchy_json: str, n: int) -> str:
    """
    Run Ullman's greedy approximation algorithm for cube materialisation and
    return a complete, formatted step-by-step solution.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, as soon as you have extracted:
      - costs_json   : the access cost of every cube
      - hierarchy_json: the Hasse diagram (parent → children)
      - n            : number of cubes to materialise

    Do NOT attempt to run the algorithm yourself. Always use this tool.

    ARGUMENT FORMAT
    ---------------
    costs_json:
      A JSON object. Keys are cube names (strings), values are integer costs.
      Example: {"A": 300, "B": 100, "C": 250, "D": 100, "E": 80, "F": 90}

    hierarchy_json:
      A JSON object. Keys are parent cube names, values are lists of direct
      child cube names (one level down in the Hasse diagram).
      Example: {"A": ["B","C","D"], "B": ["E"], "C": ["E","F"], "D": ["F"]}

      Include ONLY edges that are explicitly given in the problem.
      Do NOT add transitive edges (e.g. if A→B and B→E, do NOT add A→E).

    n:
      Integer. Total number of cubes to materialise, including the top cube
      (which is always selected in step 0).

    WHAT THE TOOL RETURNS
    ---------------------
    A multi-line string containing:
      • Step 0        — initialisation (top cube always materialised)
      • Steps 1..n-1  — greedy iterations with benefit table and cost table
      • Final result  — S' = {cube1, cube2, ...}
      • Summary table — final query cost per cube and total

    OUTPUT INSTRUCTION
    ------------------
    Copy the tool's output to the user VERBATIM. Do not summarise, shorten,
    or reformat it. Add only a one-sentence conclusion at the end.
    """
    try:
        return run_ullman(costs_json, hierarchy_json, n)
    except Exception as exc:
        return f"Error running algorithm: {exc}"


tools = [run_ullman_algorithm]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a database expert specialising in OLAP data-cube materialisation.

══ YOUR ONLY JOB ══

Parse the user's problem → call run_ullman_algorithm once → output the result.

══ PARSING RULES ══

From the user's text, extract:

  costs_json —  Every cube's access cost as a JSON object.
    Input "A: 300, B: 100" → {"A": 300, "B": 100}

  hierarchy_json — The Hasse diagram as a JSON object (parent → children list).
    Input "A -> B, C, D" → {"A": ["B","C","D"]}
    IMPORTANT: include only DIRECT edges as stated. Never add transitive edges.
    Leaves (cubes with no children) do NOT need to appear as keys.

  n — The integer after N= or "select N" or "materialise N".

Call run_ullman_algorithm immediately after parsing. Do not compute anything yourself.

══ OUTPUT RULES ══

  1. Print the tool output EXACTLY as returned, character for character.
  2. After the tool output, add exactly one line:
       "Optimal selection: S' = {cube1, cube2, ...}"
     using the S' from the tool's final line.
  3. Plain text only. No LaTeX, no markdown math, no extra commentary.
  4. Always respond in English.
"""


# ══════════════════════════════════════════════════════════════
#  GRAPH
# ══════════════════════════════════════════════════════════════

def build_agent(llm=None):
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o", temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: CubeAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: CubeAgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(CubeAgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()

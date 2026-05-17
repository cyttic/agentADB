"""
agents/datacube_agent.py
=========================
LangGraph agent for data-cube materialisation problems (Ullman's algorithm).

Two tools
---------
1. draw_hasse_diagram()
   Opens an interactive Hasse-diagram editor in the browser.
   Call this when the problem does NOT contain an explicit cost/hierarchy table.
   Returns the extracted (costs, hierarchy, n) as a JSON string.

2. run_ullman_algorithm(costs_json, hierarchy_json, n)
   Runs Ullman's greedy algorithm deterministically.
   Always call this after the data is known (either parsed or drawn).

Decision rule (in system prompt)
---------------------------------
  IF the user's message contains explicit costs AND a hierarchy/parent-child table
      → parse directly, call run_ullman_algorithm immediately
  ELSE (graph is missing or only partially described)
      → call draw_hasse_diagram first, then run_ullman_algorithm on the result
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
from tools.datacube_editor import launch_datacube_editor, extract_datacube_graph


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class CubeAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def draw_hasse_diagram(n: int) -> str:
    """
    Open an interactive Hasse-diagram editor in the browser, wait for the
    user to draw all cube nodes with their access costs and connect them
    with directed edges (parent → child), then extract the graph.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it when the problem does NOT include an explicit list of cube costs
    and/or an explicit parent→child hierarchy. Examples that require drawing:
      - "I have the following cubes but didn't provide the graph yet"
      - "Here is an image of the Hasse diagram" (can't read images)
      - The user only mentions cube names without costs or edges

    Do NOT call it when the problem already contains both costs and hierarchy.

    EDITOR INSTRUCTIONS (shown to user)
    ------------------------------------
    The editor opens in the browser. The user:
      1. Presses E (or clicks "⬭ Cube") and clicks the canvas to place each
         cube node. A popup asks for the cube name and access cost.
      2. Presses C (or clicks "→ Connect") and clicks the PARENT cube then
         the CHILD cube to draw a directed edge between them.
      3. Clicks "Submit Graph" when done.

    ARGUMENT
    ---------
    n: int
      Number of cubes to materialise (including the top cube).
      Parse this from the user's message ("select N=3", "materialise 4 cubes").
      If the user hasn't specified N, use 0 — the tool will return the graph
      and you can ask the user for N before calling run_ullman_algorithm.

    RETURNS
    -------
    JSON string with three keys:
      costs_json     — JSON object {cube_name: cost}
      hierarchy_json — JSON object {parent: [child, ...]}
      n              — the n argument you passed (pass through for convenience)
    """
    try:
        graph = launch_datacube_editor(timeout=600)
        costs, hierarchy = extract_datacube_graph(graph)
        return json.dumps({
            'costs_json':     json.dumps(costs),
            'hierarchy_json': json.dumps(hierarchy),
            'n':              n,
            'node_count':     len(costs),
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({'error': str(exc)})


@tool
def run_ullman_algorithm(costs_json: str, hierarchy_json: str, n: int) -> str:
    """
    Run Ullman's greedy approximation algorithm for cube materialisation and
    return a complete, formatted step-by-step solution.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, immediately after the data is available
    (either parsed from the user's text or obtained from draw_hasse_diagram).

    ARGUMENT FORMAT
    ---------------
    costs_json:
      JSON object — cube name → integer access cost.
      Example: {"A": 300, "B": 100, "C": 250, "D": 100, "E": 80, "F": 90}

    hierarchy_json:
      JSON object — parent cube name → list of direct child cube names.
      Include ONLY direct edges. Do NOT add transitive edges.
      Example: {"A": ["B","C","D"], "B": ["E"], "C": ["E","F"], "D": ["F"]}

    n:
      Total cubes to materialise, including the top cube (always selected).

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM to the user. Do not summarise or reformat.
    Add exactly one concluding line: "Optimal selection: S' = {cube1, cube2, ...}"
    """
    try:
        return run_ullman(costs_json, hierarchy_json, n)
    except Exception as exc:
        return f'Error running algorithm: {exc}'


tools = [draw_hasse_diagram, run_ullman_algorithm]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a database expert specialising in OLAP data-cube materialisation.
You solve problems using Ullman's greedy approximation algorithm.

══ DECISION TREE ══

Step A — Does the problem contain BOTH of these?
  (1) Access cost for every cube (e.g. "A: 300, B: 100, ...")
  (2) A parent→child hierarchy (e.g. "A → B, C, D")

  YES → Go to Step B (skip drawing).
  NO  → Call draw_hasse_diagram(n=<N from problem, or 0 if unknown>) first.
        After it returns, go to Step B.

Step B — Call run_ullman_algorithm(costs_json, hierarchy_json, n).
  - If you got the data from draw_hasse_diagram, use the costs_json and
    hierarchy_json fields from its return value directly.
  - If you parsed from text, build them as JSON strings yourself.
  - If n is still unknown after drawing, ask the user now before calling.

Step C — Output:
  1. Print the tool output VERBATIM (character for character).
  2. One final line: "Optimal selection: S' = {cube1, cube2, ...}"

══ PARSING RULES (when reading from text) ══

costs_json  — JSON object, string keys, integer values.
  "A: 300, B: 100" → {"A": 300, "B": 100}

hierarchy_json — JSON object, direct edges only.
  "A → B, C" means A is parent of B and C → {"A": ["B", "C"]}
  NEVER add transitive edges (if A→B and B→E, do NOT add A→E).

n — integer after "N=", "select N", "materialise N cubes".

══ STRICT OUTPUT RULES ══
• Plain text only. No LaTeX, no markdown math.
• Never compute algorithm steps yourself — always use run_ullman_algorithm.
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

    def call_llm(state: CubeAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: CubeAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(CubeAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

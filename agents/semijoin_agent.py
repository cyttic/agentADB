"""
agents/semijoin_agent.py
=========================
LangGraph agent that launches an interactive SVG diagram editor,
waits for the user to draw the Semi-Join Venn diagram, and returns
the resulting graph structure.

No cost computation is performed here — that is handled by a separate
computational model to be added later.

Workflow
--------
1. Agent calls the `draw_semijoin_diagram` tool.
   A browser editor opens; the tool blocks until the user submits.
2. Agent receives the graph dict and presents it to the user.
"""

import json
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.mermaid_editor import launch_diagram_editor, extract_schema


# ══════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════

class SemiJoinAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def draw_semijoin_diagram() -> str:
    """
    Open an interactive browser-based SVG diagram editor and wait for
    the user to draw a Venn-style Semi-Join diagram.

    The editor supports:
      - Ellipses (E key) — table nodes with name, blocks, distribution, attrs;
        overlapping ellipses show the Venn intersection via blended fills
      - Resize handles on the right and bottom edges of each ellipse
      - Points (P key) — small labeled dots for table columns/join keys;
        a name-input popup appears at the click position
      - Drag any ellipse or point to reposition it

    Returns the graph as a JSON string:
      {
        "ellipses": [
          {"id": "...", "x": 200, "y": 250, "rx": 90, "ry": 60,
           "name": "R", "blocks": "10000", "dist": "hash(id)", "attrs": "4",
           "color": "#3b82f6"}
        ],
        "points": [
          {"id": "...", "x": 310, "y": 250,
           "name": "id", "color": "#fbbf24"}
        ]
      }

    Call this tool ONCE and then present the result to the user.
    Do NOT call it a second time unless the user explicitly asks to redraw.
    """
    try:
        graph  = launch_diagram_editor(timeout=600)
        schema = extract_schema(graph)
        return json.dumps({'schema': schema, 'graph': graph}, indent=2, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({'error': str(exc)})


tools = [draw_semijoin_diagram]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a database diagram assistant for Semi-Join analysis.

Your ONLY job right now is to help the user capture a Semi-Join diagram.

══ WORKFLOW ══

1. Call draw_semijoin_diagram IMMEDIATELY when the user asks.
   The tool opens a browser editor and waits for the user to draw.
   Do not explain, do not ask for parameters first — just call the tool.

2. When the tool returns, present the schema string on its own line, prominently:

   Schema: Table1(a, b, c); Table2(b, c, d)

   Then briefly list any extra metadata from the ellipses (blocks, distribution)
   if the user filled them in.

3. Confirm the graph is ready for the computational model.

══ RULES ══
• Do NOT compute any costs. That comes later.
• Do NOT suggest changes to the diagram unless the user asks.
• Do NOT call the tool more than once per request.
• Plain text only. No LaTeX, no markdown math.
• Always respond in English.
"""


# ══════════════════════════════════════════════════════════════
#  BUILD AGENT
# ══════════════════════════════════════════════════════════════

def build_agent(llm=None):
    if llm is None:
        import os
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model='gpt-4o', temperature=0,
                         api_key=os.environ.get('OPENAI_API_KEY', ''))

    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: SemiJoinAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: SemiJoinAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(SemiJoinAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

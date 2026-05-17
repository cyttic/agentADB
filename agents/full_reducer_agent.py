"""
agents/full_reducer_agent.py
=============================
LangGraph agent for the Full Reducer algorithm on distributed natural joins.

Workflow
--------
1. Parse the table schema from the user's message.
2. Call analyze_schema_for_full_reducer — one tool that does everything:
     - builds the intersection graph (which tables share which attributes)
     - runs GYO ear-elimination to check acyclicity
     - if acyclic: builds the join tree and generates two-phase pseudocode
3. Present the Mermaid diagrams (intersection graph + join tree) and pseudocode.

The Mermaid diagram is output inside a ```mermaid block (renders in Claude
Code IDE / Claude.ai web). The agent also opens a read-only HTML browser
view of the intersection graph.
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

from tools.full_reducer import analyze, open_graph_viewer


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class FullReducerState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def analyze_schema_for_full_reducer(schema_text: str) -> str:
    """
    Analyse a relational schema for the Full Reducer algorithm.

    What this tool does
    -------------------
    1. Parses each table and its attributes from schema_text.
    2. Builds the intersection graph: an edge between two tables when they
       share at least one attribute (= potential join condition).
    3. Runs the GYO (Graham-Yu-Özsoyoglu) ear-elimination algorithm to
       determine whether the join hypergraph is acyclic.
    4. If acyclic:
         - Derives the join tree from the GYO elimination order.
         - Generates two-phase Full Reducer pseudocode:
             Phase 1 (bottom-up)  — each parent ⋉ each child
             Phase 2 (top-down)   — each child  ⋉ its parent
         - Natural join is computed last on the fully reduced relations.
    5. Renders Mermaid diagrams for the intersection graph and join tree.
    6. Opens a read-only HTML browser view of the intersection graph.

    ARGUMENT
    ---------
    schema_text:
      Tables in standard format, separated by semicolons, commas, or newlines.
      Example: "A(a,b,c); B(b,c,d); C(c,d,e)"

    RETURNS
    -------
    A JSON string with keys:
      tables            — list of [name, attrs]
      intersection      — {pair_key: [shared_attrs]}
      is_acyclic        — bool
      gyo_steps         — list of step strings
      root              — join-tree root (if acyclic)
      intersection_mermaid — Mermaid code for the intersection graph
      join_tree_mermaid    — Mermaid code for the join tree (if acyclic)
      pseudocode           — Full Reducer pseudocode (if acyclic)

    OUTPUT INSTRUCTIONS
    -------------------
    You MUST format the response like this:

    1. Print the intersection graph Mermaid diagram in a ```mermaid block.
    2. Print the GYO steps line by line.
    3. State clearly whether Full Reducer is APPLICABLE or NOT APPLICABLE.
    4. If applicable:
         a. Print the join tree Mermaid diagram in a ```mermaid block.
         b. Print the pseudocode in a ``` block.
    5. If NOT applicable: explain why (cyclic schema) and name the cycle.

    Do NOT call this tool more than once per problem.
    """
    try:
        result = analyze(schema_text)

        if 'error' in result:
            return json.dumps({'error': result['error']})

        # Open read-only browser view of intersection graph
        try:
            open_graph_viewer(result['intersection_mermaid'], "Intersection Graph")
        except Exception:
            pass

        # Serialise sets to lists for JSON
        return json.dumps({
            'tables':               result['tables'],
            'intersection':         result['intersection'],
            'is_acyclic':           result['is_acyclic'],
            'gyo_steps':            result['gyo_steps'],
            'root':                 result.get('root'),
            'intersection_mermaid': result['intersection_mermaid'],
            'join_tree_mermaid':    result.get('join_tree_mermaid', ''),
            'pseudocode':           result.get('pseudocode', ''),
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({'error': str(exc)})


tools = [analyze_schema_for_full_reducer]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a distributed database expert specialising in the Full Reducer algorithm
for computing acyclic natural joins efficiently using semi-joins.

══ WHAT IS THE FULL REDUCER ══

For an acyclic join R1 ⋈ R2 ⋈ … ⋈ Rk, the Full Reducer eliminates "dangling
tuples" (tuples that cannot participate in the final join result) BEFORE computing
the actual join, minimising data transfer in distributed settings.

It requires the join hypergraph to be ACYCLIC (tested with GYO reduction).

Two phases:
  Phase 1 (bottom-up): each parent Ri = Ri ⋉ Rj for each child Rj
  Phase 2 (top-down):  each child  Rj = Rj ⋉ Ri for each parent Ri
After both phases, the join is computed on dangling-tuple-free relations.

══ WORKFLOW ══

1. Extract the schema from the user's message.
   Schema format: A(a,b,c); B(b,c,d); C(c,d,e)
   If missing, ask the user to provide it.

2. Call analyze_schema_for_full_reducer(schema_text=...) ONCE.

3. Format the output as follows:

─── Intersection Graph ──────────────────────────────────────────────────────

Show which tables share attributes. Output the Mermaid diagram:

```mermaid
<intersection_mermaid from tool result>
```

─── GYO Reduction ───────────────────────────────────────────────────────────

Print each step from gyo_steps.
Then state:
  ✓ ACYCLIC — Full Reducer is applicable.
  or
  ✗ CYCLIC — Full Reducer cannot guarantee complete dangling-tuple elimination.
    (name the problematic cycle if visible)

─── Join Tree ───────────────────────────────────────────────────────────────
(only if acyclic)

```mermaid
<join_tree_mermaid from tool result>
```

─── Full Reducer Pseudocode ─────────────────────────────────────────────────
(only if acyclic)

```
<pseudocode from tool result>
```

══ RULES ══
• Plain text only. No LaTeX, no markdown math except ⋈ and ⋉ Unicode.
• Copy Mermaid diagrams and pseudocode VERBATIM from tool output.
• Never compute GYO or generate pseudocode yourself.
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

    def call_llm(state: FullReducerState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: FullReducerState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(FullReducerState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

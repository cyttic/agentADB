"""
agents/pagerank_agent.py
=========================
LangGraph agent for PageRank problems.

Pipeline
--------
Given a web/link structure ("which page links to which"):
  1. build the square link (transition) matrix — every page is a row and a
     column, cell(row,col) = 1/out-degree(row) for each page the row links to;
  2. run the PageRank iteration on top of it.

Three tools
-----------
1. draw_link_graph()
   Opens an interactive link-graph editor in the browser. The user places
   pages (circles) and draws directed links (arrows: from → to). Call this
   when the user has NOT written the links out explicitly in text.
   Returns the extracted (nodes, edges) as a JSON string.

2. build_pagerank_table(nodes_json, edges_json)
   Deterministically builds the link/transition matrix. Use this when the task
   asks ONLY for the table and no PageRank computation.

3. run_pagerank_iterations(nodes_json, edges_json, d, iterations, initial_ranks_json)
   Builds the matrix AND runs the iteration P[j]=d/N+(1-d)·Σ_i T[i,j]·P[i].
   Use this whenever the task asks to compute PageRank / ranks / iterations.

Decision rule (in system prompt)
---------------------------------
  IF the links are missing / the user wants to draw them
      → call draw_link_graph first to obtain (nodes, edges)
  THEN, with the link graph known:
      task wants PageRank values → run_pagerank_iterations
      task wants only the table  → build_pagerank_table
"""

import os
import json
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.pagerank_ops import build_link_table, run_pagerank
from tools.pagerank_editor import launch_pagerank_editor, extract_pagerank_graph


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class PageRankAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def draw_link_graph() -> str:
    """
    Open an interactive link-graph editor in the browser, wait for the user to
    place every page and connect them with directed links, then extract the graph.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it when the problem does NOT spell out the links in text. Examples:
      - "Here is a web graph (picture), build the PageRank table" (can't read images)
      - "Let me draw the link structure"
      - The user names some pages but does not say which links to which

    Do NOT call it when the user already wrote the links (e.g. "A→B, A→C, C→A").

    EDITOR INSTRUCTIONS (shown to user)
    ------------------------------------
    The editor opens in the browser. The user:
      1. Presses P (or clicks "○ Page") and clicks the canvas to place each
         page. A popup asks for the page name.
      2. Presses L (or clicks "→ Link") and clicks the page that HAS the link,
         then the page it POINTS TO, to draw a directed link.
      3. Clicks "Submit Graph" when done.

    RETURNS
    -------
    JSON string with two keys:
      nodes_json — JSON array of page names, e.g. ["A","B","C"]
      edges_json — JSON array of [src, tgt] links, e.g. [["A","B"],["C","A"]]
    """
    try:
        graph = launch_pagerank_editor(timeout=600)
        nodes, edges = extract_pagerank_graph(graph)
        return json.dumps({
            'nodes_json': json.dumps(nodes, ensure_ascii=False),
            'edges_json': json.dumps([[s, t] for s, t in edges], ensure_ascii=False),
            'page_count': len(nodes),
            'link_count': len(edges),
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({'error': str(exc)})


@tool
def build_pagerank_table(nodes_json: str, edges_json: str) -> str:
    """
    Build the PageRank link (transition) matrix from a link graph and return
    it fully formatted. The matrix is square (pages on both rows and columns);
    each cell = 1/out-degree(row) for every page the row links to, 0 otherwise,
    so each non-dangling row sums to 1.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it exactly once per problem, as soon as the link graph is available
    (either parsed from the user's text or returned by draw_link_graph).

    ARGUMENT FORMAT
    ---------------
    nodes_json:
      JSON array of page names.
      Example: ["A", "B", "C"]

    edges_json:
      JSON array of directed links, each link is [src, tgt] meaning
      "src has a link pointing to tgt".
      Include ONLY direct links. Example: [["A","B"], ["A","C"], ["C","A"]]

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM to the user. Do not summarise or reformat.
    """
    try:
        return build_link_table(nodes_json, edges_json)
    except Exception as exc:
        return f'Error building table: {exc}'


@tool
def run_pagerank_iterations(nodes_json: str, edges_json: str, d: float,
                            iterations: int = 0,
                            initial_ranks_json: str = "") -> str:
    """
    Build the link matrix AND run the PageRank iteration, returning both the
    matrix and the full per-iteration trace fully formatted.

    Iteration formula (applied to every page j each round):
        P[j] = d/N + (1 - d) * Σ_i  T[i,j] * P[i]
    N = number of pages, T[i,j] = 1/out-degree(i) if i links to j else 0.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it (instead of build_pagerank_table) whenever the task asks to COMPUTE
    PageRank — i.e. it gives the constant d (a.k.a. damping/delta) and/or asks
    for the ranks, ratings, or iterations.

    ARGUMENT FORMAT
    ---------------
    nodes_json / edges_json:
      Same as build_pagerank_table — page names and [src, tgt] links.

    d:
      The constant from the task's formula (e.g. 0.15, 0.2, 0.85). REQUIRED.
      If the task does not state d, ask the user for it before calling.

    iterations:
      The number of iterations the task asks for (e.g. "2 iterations" → 2).
      If the task gives NO count, pass 0 to iterate until the ranks converge.

    initial_ranks_json:
      Optional. If the task gives starting ratings, pass them as a JSON object
      {page: rank}, e.g. {"A":0.4,"B":0.2,"C":0.2,"D":0.2}. If the task does
      not, pass "" — every page then starts at 1/N.

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM to the user. Do not summarise or reformat.
    """
    try:
        matrix = build_link_table(nodes_json, edges_json)
        trace  = run_pagerank(nodes_json, edges_json, d, iterations, initial_ranks_json)
        return f'{matrix}\n\n{"=" * 60}\n\n{trace}'
    except Exception as exc:
        return f'Error running PageRank: {exc}'


tools = [draw_link_graph, build_pagerank_table, run_pagerank_iterations]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert in the PageRank algorithm. You first build the link
(transition) matrix — pages on both rows and columns, each cell =
1/out-degree(row) for every page the row links to (0 otherwise) — and then run
the PageRank iteration on top of it.

The iteration formula is:  P[j] = d/N + (1-d) * Σ_i T[i,j]·P[i]
  N = number of pages,  d = the constant given in the task,
  T[i,j] = the matrix cell (share i sends to j),  P[i] = current rank of i.
Pages start at 1/N each, unless the task gives starting ratings.

══ DECISION TREE ══

Step A — Do you already have the links between pages in text?
  (e.g. "A → B, A → C, C → A", or "page 1 links to 2 and 3", or an adjacency list)

  YES → Go to Step B (skip drawing).
  NO  → Call draw_link_graph() first. After it returns, go to Step B using the
        nodes_json and edges_json fields from its return value.

Step B — Choose the computation tool:
  - If the task asks to COMPUTE PageRank (it gives d, or asks for ranks /
    ratings / iterations) → call
    run_pagerank_iterations(nodes_json, edges_json, d, iterations, initial_ranks_json):
      • d           — the constant from the task. If it is NOT stated, ask the
                      user for d before calling. Do not invent it.
      • iterations  — the count the task asks for ("2 iterations" → 2).
                      If the task gives NO count, pass 0 (run until convergence).
      • initial_ranks_json — the task's starting ratings as a JSON object
                      {page: rank}, or "" to start every page at 1/N.
  - If the task asks ONLY for the link table/matrix (no computation) → call
    build_pagerank_table(nodes_json, edges_json).

Step C — Output:
  Print the tool output VERBATIM (character for character). Do not recompute or
  reformat any numbers yourself — they come from the tool.

══ PARSING RULES (when reading links from text) ══

nodes_json  — JSON array of page names.
  "pages A, B, C" → ["A", "B", "C"]

edges_json  — JSON array of [src, tgt] links, src points to tgt.
  "A → B, C" means A links to B and A links to C → [["A","B"], ["A","C"]]
  "B and C both link to A" → [["B","A"], ["C","A"]]
  A link is DIRECTED: "X links to Y" is [X, Y], never [Y, X].

══ STRICT OUTPUT RULES ══
• Plain text only. No LaTeX, no markdown math.
• Never build the matrix or compute ranks by hand — always use the tools.
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

    def call_llm(state: PageRankAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: PageRankAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(PageRankAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

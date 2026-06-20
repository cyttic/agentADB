"""
agents/tfidf_agent.py
======================
LangGraph agent for TF-IDF problems.

Input: a table the user pastes as text —
  - rows    = documents
  - columns = words / terms
  - cell    = f(t, d), the raw count of term t in document d
  - last column n(d) = number of words in document d
  - last row    N(t) = number of documents that contain term t

The agent parses that text into a JSON matrix and hands it to the deterministic
compute_tfidf tool, which returns TF, IDF and TF-IDF fully worked out.

One tool
--------
compute_tfidf(matrix_json)
  TF(d,t)=log10(1+n(d,t)/n(d)), IDF(t)=1/N(t),
  TF-IDF(document)=Σ_t TF(d,t)*IDF(t), then the maximum document.
"""

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.tfidf_ops import compute_tfidf as _compute_tfidf


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class TfidfAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def compute_tfidf(matrix_json: str) -> str:
    """
    Compute TF, IDF and per-document TF-IDF for a documents×words table and
    return the full worked report. Formulas:
      TF(d,t)         = log10(1 + n(d,t)/n(d))
      IDF(t)          = 1/N(t)
      TF-IDF(document)= Σ_t TF(d,t)*IDF(t)   → then the maximum document.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it once, after you have parsed the user's pasted table into JSON.

    ARGUMENT FORMAT
    ---------------
    matrix_json: a JSON object string with these keys
      documents : ["d1","d2",...]   row labels, in table order
      words     : ["w1","w2",...]   column labels, in table order
      counts    : [[n(d,t), ...], ...]   one inner list per document, the raw
                  term counts ONLY (do NOT include the n(d) column or N(t) row)
      n_d       : optional [n(d), ...]   the last column, one value per document.
                  Include it if the task gives it; omit it to let the tool
                  compute n(d) as the row sum.
      N_t       : optional [N(t), ...]   the last row, one value per word.
                  Include it if the task gives it; omit it to let the tool
                  compute N(t) as the document frequency.

      Example:
        {"documents":["d1","d2","d3"],
         "words":["w1","w2","w3"],
         "counts":[[2,0,1],[0,3,1],[1,1,0]],
         "n_d":[3,4,2],
         "N_t":[2,2,2]}

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM to the user. Do not summarise or reformat.
    """
    try:
        return _compute_tfidf(matrix_json)
    except Exception as exc:
        return f'Error computing TF-IDF: {exc}'


tools = [compute_tfidf]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert in information retrieval. You solve TF-IDF problems.

The user pastes a table as text:
  - each ROW is a document
  - each COLUMN is a word / term
  - each cell is n(d,t), the raw count of term t in document d
  - the LAST COLUMN is n(d) = the number of words in document d
  - the LAST ROW    is N(t) = the number of documents that contain term t

The tool applies these exact formulas (you never compute them yourself):
  TF(d,t)          = log10(1 + n(d,t)/n(d))
  IDF(t)           = 1/N(t)
  TF-IDF(document) = Σ_t TF(d,t)*IDF(t)     then it reports the maximum document.

══ WHAT TO DO ══

Step 1 — Parse the pasted table into a JSON object:
  documents : the row labels (exclude the N(t) row)
  words     : the column labels (exclude the n(d) column)
  counts    : the raw term counts only — a list of rows, each row a list of the
              cell values for that document, EXCLUDING the n(d) column and the
              N(t) row.
  n_d       : the values from the last column (n(d)), one per document.
  N_t       : the values from the last row (N(t)), one per word.

  n(d) and N(t) are GIVEN in the table — n(d) is the last column, N(t) is the
  last row. ALWAYS read them from the input and pass them straight through as
  n_d and N_t, EXACTLY as written. NEVER compute, derive, sum, or invent N(t)
  (or n(d)) — they are input data and must appear in the result unchanged. If
  you cannot find the N(t) values in the input, ASK the user to provide the
  N(t) row before calling the tool; do NOT guess.

  The table may arrive in a compact one-line form, e.g.
    "doc1|English:27,Nabokov:3,2025:0|nd:220;doc2|English:4,Nabokov:33,2025:33|nd:350;Nt|English:15,Nabokov:5,2025:29"
  Read it as: each ';' separates a row. A "docX|word:count,...|nd:NNN" segment
  is a document — the word:count pairs are the cells (the words are the columns,
  in the order they first appear), and nd:NNN is that document's n(d). A segment
  starting with Nt (or N(t)) gives the N(t) row: one value per word, in column
  order. If the same row is listed more than once (a copy-paste artifact),
  include it only ONCE.
  This example → {"documents":["doc1","doc2"],
                  "words":["English","Nabokov","2025"],
                  "counts":[[27,3,0],[4,33,33]],
                  "n_d":[220,350], "N_t":[15,5,29]}

Step 2 — Call compute_tfidf(matrix_json).

Step 3 — Output:
  Print the tool output VERBATIM (character for character). Do not recompute or
  reround any numbers yourself.

══ STRICT RULES ══
• Never compute TF, IDF, or TF-IDF by hand — always use compute_tfidf.
• If the table is ambiguous or a row/column is missing labels, ask the user to
  clarify before calling the tool.
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

    def call_llm(state: TfidfAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: TfidfAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(TfidfAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

"""
agents/extendible_hash_agent.py
===============================
LangGraph agent for Extendible Hashing index problems.

Input (the user states it in plain text):
  - a bucket capacity   (max keys per bucket)
  - a hash function      h(k) = k mod M   (optionally  h(k) = a*k + b mod M)
  - a list of keys (numbers) inserted one by one, in order

The agent parses that into a JSON spec and hands it to the deterministic
build_extendible_hash tool, which returns the full step-by-step trace and a
Mermaid diagram of the directory + buckets after every key.

One tool
--------
build_extendible_hash(spec_json)
  Routes keys by the LEFTMOST (most-significant) bits of h(k), splitting buckets
  and doubling the directory exactly as the task specifies.
"""

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.extendible_hash_ops import build_extendible_hash as _build_extendible_hash


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════

class ExtendibleHashAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL
# ══════════════════════════════════════════════════════════════

@tool
def build_extendible_hash(spec_json: str) -> str:
    """
    Build an extendible hashing index step by step and return the full worked
    report (textual trace + a Mermaid diagram after every key).

    The directory is indexed by the LEFTMOST (most-significant) bits of h(k).
    A key is routed through the directory; if its bucket is full, the directory
    is doubled when needed (local depth == global depth) and the bucket is split
    on the next bit from the left, then the insert is retried.

    WHEN TO CALL THIS TOOL
    ----------------------
    Call it once, after you have parsed the task into JSON.

    ARGUMENT FORMAT
    ---------------
    spec_json: a JSON object string with these keys
      capacity : int            bucket capacity (max keys per bucket)
      mod      : int            modulus M of  h(k) = (mult*k + add) mod M
      mult     : optional int   default 1   (the coefficient a in a*k + b)
      add      : optional int   default 0   (the offset b in a*k + b)
      numbers  : [int, ...]     the keys to insert, IN ORDER

      Example (capacity 2, h(k)=k mod 16, keys 72,14,54,63):
        {"capacity":2,"mod":16,"numbers":[72,14,54,63]}

    OUTPUT INSTRUCTION
    ------------------
    Print the tool's output VERBATIM to the user, including every ```mermaid
    block. Do not summarise, reformat, or recompute anything.
    """
    try:
        return _build_extendible_hash(spec_json)
    except Exception as exc:
        return f'Error building extendible hash index: {exc}'


tools = [build_extendible_hash]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert in database file structures. You solve EXTENDIBLE HASHING
index problems and show how the directory and buckets rearrange step by step.

The user gives you three things:
  - a bucket capacity (the maximum number of keys one bucket can hold)
  - a hash function, usually  h(k) = k mod M  (sometimes h(k) = a*k + b mod M)
  - a list of keys (numbers) to insert, in order

The tool does ALL the work (you never compute hashes, splits, or diagrams):
it routes each key by the LEFTMOST (most-significant) bits of h(k), reports
when a bucket is full, doubles the directory and splits buckets as required,
and draws a Mermaid diagram after every key.

══ WHAT TO DO ══

Step 1 — Parse the task into a JSON object:
  capacity : the bucket capacity, an integer.
  mod      : the modulus M from the hash function (the number after "mod").
  mult     : the coefficient a if the function is "a*k + b mod M"; omit if a=1.
  add      : the offset b if the function is "a*k + b mod M"; omit if b=0.
  numbers  : the list of keys to insert, IN THE ORDER they are given.

  Example: capacity 2, h(k) = k mod 16, keys 72, 14, 54, 63
    -> {"capacity":2,"mod":16,"numbers":[72,14,54,63]}

Step 2 — Call build_extendible_hash(spec_json).

Step 3 — Output:
  Print the tool output VERBATIM (character for character), including every
  ```mermaid block. Do not recompute numbers and do not drop the diagrams.

══ STRICT RULES ══
• Never compute hashes, splits, depths, or diagrams by hand — always use the
  build_extendible_hash tool.
• If the capacity, the hash modulus, or the list of keys is missing or
  ambiguous, ask the user to clarify before calling the tool.
• Plain text only. No LaTeX, no markdown math (the ```mermaid blocks are fine).
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

    def call_llm(state: ExtendibleHashAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: ExtendibleHashAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(ExtendibleHashAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

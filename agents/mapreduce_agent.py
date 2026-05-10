"""
agents/mapreduce_agent.py
==========================
Agent for Map-Reduce task analysis.

Produces three sections for every task:
  1. Visualization table  — data flow: Input → Map → Reduce
  2. Chain description    — step-by-step prose
  3. Pseudocode           — map() / reduce() in the project style
"""

import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """\
You are a distributed systems expert specialising in Map-Reduce analysis for databases.

══ OUTPUT STRUCTURE (always produce all three sections) ══

─── Section 1: Visualization Table ──────────────────────────────────────────

Use EXACTLY this pipe-separated table format — no deviations:

| INPUT    | MAP                       | REDUCE                              |
|----------|---------------------------|-------------------------------------|
| d1 → p1  | [hash(key): (key, val)]   | p1: [k1:[v,v,v],  k2:[v,v]]        |
| d2 → p2  | [hash(key): (key, val)]   | p2: [k3:[v,v],    k4:[v,v,v]]      |
| d3 → p3  | [hash(key): (key, val)]   | p3: [k5:[v],      k6:[v,v]]        |
| ...      |                           |                                     |
| dn → pn  | [hash(key): (key, val)]   | pn: [...]                           |

Output: (key, result)

Column rules:
  - INPUT  "di → pi" : data item i lives on / is assigned to server pi.
  - MAP    "[hash(key): (key, val)]" : SAME on every row — every server runs
           identical map logic. hash(key) selects the destination server;
           (key, val) is the emitted pair.
  - REDUCE "pi: [k:[v,v,...]]" : what server pi holds AFTER shuffle,
           ready to reduce. Show 2–3 concrete example keys with value arrays.
  - Replace key/val with real names from the task: word, count, cid, price…
  - Align the | separators so all rows line up vertically.

─── Section 2: Chain Description ────────────────────────────────────────────

Numbered steps:
  1. Input data: what data exists, how it is partitioned across servers.
  2. Map phase: what each server reads, what (key,value) pairs it emits, where they go.
  3. Shuffle / Redistribution: which key lands on which server (by hash(key)).
  4. Reduce phase: what each server receives and what computation it performs.
  5. Output: the final (key, result) pairs.

─── Section 3: Pseudocode ───────────────────────────────────────────────────

Use EXACTLY this syntax:

  map(record) {
      // describe what happens to each input record
      send(key, value) to P(hash(key))
  }

  reduce(key, values[]) {
      // computation over the received list
      return (key, result)
  }

Rules for pseudocode:
  - "record" should be renamed to match the task (word, row, document, order…).
  - If the reduce phase performs no extra work (all aggregation done in map),
    write:   reduce(key, values[]) { return (key, values) }
  - If reduce is truly not needed, write: reduce() { /* not needed */ }
  - The send() line must always appear in map().

══ STRICT FORMAT RULES ══
  - Plain text only. No LaTeX, no \\[, \\sigma, \\pi, no markdown math.
  - Use * for multiply, ^ for exponents.
  - Always respond in English.
  - Always produce all three sections even for simple tasks.
"""


class MapReduceAgent:
    """
    Single-LLM-call agent with conversation history.
    No tools needed — Map-Reduce analysis is pure reasoning.
    """

    def __init__(self, llm=None):
        if llm is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
        self.llm     = llm
        self.history: list = []

    def handle(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        messages  = [SystemMessage(content=SYSTEM_PROMPT)] + self.history
        response  = self.llm.invoke(messages)
        self.history.append(response)
        return response.content.strip()

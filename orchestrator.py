"""
orchestrator.py
================
Routes user input to the correct agent:
  - SerializabilityAgent — schedule conflict/view serializability
  - ParallelQueryAgent   — parallel DB query cost analysis (Select, Sort)
  - JoinCostAgent        — parallel Join cost analysis (broadcast algorithm)

The LLM used by all agents is configured once at startup via llm_factory.
"""

import re
from langchain_core.messages import HumanMessage, AIMessage

from agents.serializability_agent import build_agent as build_serial_agent
from agents.parallel_query_agent  import build_agent as build_query_agent
from agents.pipeline_agent        import PipelineAgent
from agents.mapreduce_agent       import MapReduceAgent
from llm_factory                  import build_llm, LLMConfig


# ══════════════════════════════════════════════════════════════
#  ROUTER PROMPT
#  — written for weak/local models:
#    * few-shot examples (most important fix)
#    * one-word instruction repeated twice
#    * no abstract descriptions — only concrete keywords + examples
# ══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """Your task: read the user message and output exactly one word.

The word must be one of: SERIAL, QUERY, JOIN, MAPREDUCE, UNKNOWN

Rules:
- Output SERIAL if the message is about transaction schedules, read/write operations, serializability, precedence graphs, or conflict analysis.
- Output JOIN if the message is about computing the cost of a Join (⋈) operation in a parallel database — possibly combined with Select (σ) before the join, or joining more than two tables.
- Output QUERY if the message is about parallel databases, processors, block size, query cost, round-robin, hash partitioning, range partitioning, relational algebra, or table scan / sort cost (but NOT primarily Join).
- Output MAPREDUCE if the message is about a Map-Reduce task: word count, distributed aggregation, inverted index, or any task described in terms of map and reduce phases over distributed data.
- Output UNKNOWN if it is neither.

Do NOT explain. Do NOT add punctuation. Output only the single word.

Examples:

Message: "Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?"
Answer: SERIAL

Message: "Check view serializability: r2(B) w2(A) r1(A) w1(B)"
Answer: SERIAL

Message: "T1: r(A) w(B), T2: r(B) w(A) — draw the precedence graph"
Answer: SERIAL

Message: "Дана таблица Flights, 10000 блоков, round-robin, 10 процессоров. σ_fid=777(Flights)"
Answer: QUERY

Message: "10 processors, block size 2000, Customers 10^6 rows — query cost?"
Answer: QUERY

Message: "Find all customers ordered by date, Orders distributed by hash(pid), sort cost?"
Answer: QUERY

Message: "Flowers(name,petal,size,color) 10^4 blocks, Sales(name,cname,amount,price) 10^6 blocks, 10 servers. Perform Flowers join Sales."
Answer: JOIN

Message: "Employees 50000 blocks, Departments 1000 blocks, 8 processors. Compute join cost Employees ⋈ Departments."
Answer: JOIN

Message: "σ_price>100(Sales) ⋈ Flowers — вычислить стоимость на 10 серверах."
Answer: JOIN

Message: "Calculate cost: A ⋈ B ⋈ C, 12 processors, A=10^4 blocks, B=10^6 blocks, C=500 blocks."
Answer: JOIN

Message: "Count how many times each word appears across documents. Input: documents d1..dn, n servers."
Answer: MAPREDUCE

Message: "Design a Map-Reduce algorithm to find the maximum price per product category."
Answer: MAPREDUCE

Message: "Use MapReduce to count the number of orders per customer."
Answer: MAPREDUCE

Message: "What is the weather today?"
Answer: UNKNOWN

Message: "Explain what a database is"
Answer: UNKNOWN

Now classify this message:
"""


# ══════════════════════════════════════════════════════════════
#  ANSI
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"
RED    = "\033[31m"


# ══════════════════════════════════════════════════════════════
#  ROUTE EXTRACTOR
#  — weak models often wrap their answer in extra text,
#    so we scan the response for the first valid keyword
#    instead of doing a strict equality check
# ══════════════════════════════════════════════════════════════

_VALID = {"SERIAL", "QUERY", "JOIN", "MAPREDUCE", "UNKNOWN"}

def _extract_domain(raw: str) -> str:
    """
    Extract routing keyword from model output robustly.

    Handles noisy outputs like:
      "The answer is QUERY."
      "QUERY\n\nBecause..."
      "query"  (lowercase)
    Falls back to UNKNOWN if nothing found.
    """
    upper = raw.upper()
    # Try to find any of the valid keywords in the response
    for word in re.findall(r"[A-Z]+", upper):
        if word in _VALID:
            return word
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self, llm_config: LLMConfig):
        self.llm        = build_llm(llm_config)
        self.llm_config = llm_config

        self.serial_agent    = build_serial_agent(llm=self.llm)
        self.query_agent     = build_query_agent(llm=self.llm)
        self.pipeline_agent  = PipelineAgent(llm=self.llm)
        self.mapreduce_agent = MapReduceAgent(llm=self.llm)

        self.serial_history:   list = []
        self.query_history:    list = []
        self.query_db_context: dict = {}

    def _route(self, user_input: str) -> str:
        """
        Classify input as SERIAL / QUERY / UNKNOWN.

        The prompt is structured for weak/local models:
        few-shot examples + keyword scan on the output.
        """
        # Prompt already contains "Now classify this message:" at the end
        full_prompt = ROUTER_PROMPT + user_input

        response = self.llm.invoke([
            HumanMessage(content=full_prompt),
        ])

        raw    = response.content.strip()
        domain = _extract_domain(raw)

        # Debug: show raw model output if it wasn't clean
        if raw.upper() not in _VALID:
            print(f"{DIM}[router raw] '{raw[:60]}' → {domain}{RESET}")

        return domain

    def handle(self, user_input: str) -> str:
        domain = self._route(user_input)
        print(f"{DIM}[router → {domain}]{RESET}")

        if domain == "SERIAL":
            self.serial_history.append(HumanMessage(content=user_input))
            result = self.serial_agent.invoke({"messages": self.serial_history})
            self.serial_history = result["messages"]
            last_ai = next(
                (m for m in reversed(self.serial_history) if isinstance(m, AIMessage)),
                None,
            )
            return last_ai.content if last_ai else "(no response)"

        elif domain == "QUERY":
            self.query_history.append(HumanMessage(content=user_input))
            result = self.query_agent.invoke({
                "messages":   self.query_history,
                "db_context": self.query_db_context or {},
            })
            self.query_history    = result["messages"]
            self.query_db_context = result.get("db_context") or self.query_db_context or {}
            last_ai = next(
                (m for m in reversed(self.query_history) if isinstance(m, AIMessage)),
                None,
            )
            return last_ai.content if last_ai else "(no response)"

        elif domain == "JOIN":
            # Pipeline agent handles planning + execution + formatting deterministically
            return self.pipeline_agent.handle(user_input)

        elif domain == "MAPREDUCE":
            return self.mapreduce_agent.handle(user_input)

        else:
            return (
                "I specialise in four topics:\n"
                "  • Transaction schedule serializability\n"
                "  • Parallel query cost (Select, Sort)\n"
                "  • Parallel Join cost\n"
                "  • Map-Reduce algorithms\n"
                "Please clarify your question."
            )

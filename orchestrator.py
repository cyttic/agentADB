"""
orchestrator.py
================
Routes user input to the correct agent:
  - SerializabilityAgent — schedule conflict/view serializability
  - ParallelQueryAgent   — parallel DB query cost analysis

The LLM used by all agents is configured once at startup via llm_factory.
"""

import re
from langchain_core.messages import HumanMessage, AIMessage

from agents.serializability_agent import build_agent as build_serial_agent
from agents.parallel_query_agent  import build_agent as build_query_agent
from llm_factory                  import build_llm, LLMConfig


# ══════════════════════════════════════════════════════════════
#  ROUTER PROMPT
#  — written for weak/local models:
#    * few-shot examples (most important fix)
#    * one-word instruction repeated twice
#    * no abstract descriptions — only concrete keywords + examples
# ══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """Your task: read the user message and output exactly one word.

The word must be one of: SERIAL, QUERY, UNKNOWN

Rules:
- Output SERIAL if the message is about transaction schedules, read/write operations, serializability, precedence graphs, or conflict analysis.
- Output QUERY if the message is about parallel databases, processors, block size, query cost, round-robin, hash partitioning, range partitioning, relational algebra, or table scan cost.
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

Message: "Find all customers who ordered products over 100, Orders distributed by hash(pid)"
Answer: QUERY

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

_VALID = {"SERIAL", "QUERY", "UNKNOWN"}

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

        self.serial_agent = build_serial_agent(llm=self.llm)
        self.query_agent  = build_query_agent(llm=self.llm)

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

        else:
            return (
                "Я специализируюсь на двух темах:\n"
                "  • Сериализуемость расписаний транзакций\n"
                "  • Стоимость параллельных запросов в БД\n"
                "Пожалуйста, уточните ваш вопрос."
            )

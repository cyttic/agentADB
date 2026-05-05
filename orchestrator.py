"""
orchestrator.py
================
Routes user input to the correct agent:
  - SerializabilityAgent — schedule conflict/view serializability
  - ParallelQueryAgent   — parallel DB query cost analysis

The LLM used by all agents is configured once at startup via llm_factory.
"""

import os
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.serializability_agent import build_agent as build_serial_agent
from agents.serializability_agent import SerialAgentState
from agents.parallel_query_agent  import build_agent as build_query_agent
from agents.parallel_query_agent  import QueryAgentState
from llm_factory                  import build_llm, LLMConfig


# ══════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """You are a routing assistant. Classify the user message into exactly one of:

  SERIAL  — transaction schedule serializability
             (view-serial, conflict-serial, precedence graph, r/w ops like r1(A) w2(B))

  QUERY   — parallel database query cost analysis
             (parallel DB, processors, block size, t_d, t_s, round-robin, hash,
              range partition, relational algebra, join/scan cost, Customers/Orders/Products)

  UNKNOWN — neither

Reply with ONLY one word: SERIAL, QUERY, or UNKNOWN.
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
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self, llm_config: LLMConfig):
        # Build one shared LLM instance from config
        self.llm        = build_llm(llm_config)
        self.llm_config = llm_config

        # Build both agents using the same LLM
        self.serial_agent = build_serial_agent(llm=self.llm)
        self.query_agent  = build_query_agent(llm=self.llm)

        # Separate histories per agent
        self.serial_history:  list = []
        self.query_history:   list = []
        self.query_db_context: dict = {}

    def _route(self, user_input: str) -> str:
        """Classify input as SERIAL / QUERY / UNKNOWN using the shared LLM."""
        response = self.llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=user_input),
        ])
        return response.content.strip().upper()

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
                "db_context": self.query_db_context,
            })
            self.query_history    = result["messages"]
            self.query_db_context = result.get("db_context", self.query_db_context)
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

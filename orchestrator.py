"""
orchestrator.py
================
Routes user input to the correct agent:
  • SerializabilityAgent — schedule conflict/view serializability
  • ParallelQueryAgent   — parallel DB query cost analysis

Uses a lightweight LLM call to classify the domain,
then delegates to the appropriate agent.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.serializability_agent import build_agent as build_serial_agent
from agents.serializability_agent import SerialAgentState
from agents.parallel_query_agent  import build_agent as build_query_agent
from agents.parallel_query_agent  import QueryAgentState


# ══════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """You are a routing assistant. Classify the user's message into exactly one of:

  SERIAL  — the user is asking about transaction schedule serializability
             (view-serial, conflict-serial, precedence graph, read/write operations like r1(A) w2(B))

  QUERY   — the user is asking about parallel database query cost analysis
             (parallel DB, processors, block size, t_d, t_s, round-robin, hash, range partition,
              relational algebra, join cost, scan cost, Customers/Orders/Products schemas)

  UNKNOWN — neither of the above

Reply with ONLY one word: SERIAL, QUERY, or UNKNOWN.
"""


def route(user_input: str) -> str:
    """Returns 'SERIAL', 'QUERY', or 'UNKNOWN'."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_input),
    ])
    return response.content.strip().upper()


# ══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"
RED    = "\033[31m"


class Orchestrator:
    def __init__(self):
        self.serial_agent = build_serial_agent()
        self.query_agent  = build_query_agent()

        # Separate histories per agent
        self.serial_history: list = []
        self.query_history:  list = []
        self.query_db_context: dict = {}

    def handle(self, user_input: str) -> str:
        domain = route(user_input)
        print(f"{DIM}[router → {domain}]{RESET}")

        if domain == "SERIAL":
            self.serial_history.append(HumanMessage(content=user_input))
            result = self.serial_agent.invoke(
                {"messages": self.serial_history}
            )
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
            self.query_history   = result["messages"]
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

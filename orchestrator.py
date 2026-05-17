"""
orchestrator.py
================
Routes user input to the correct agent:
  - SerializabilityAgent — schedule conflict/view serializability
  - ParallelQueryAgent   — parallel DB query cost analysis (Select, Sort)
  - JoinCostAgent        — parallel Join cost analysis (broadcast algorithm)
  - SemiJoinAgent        — Semi-Join cost analysis via Mermaid diagram input

The LLM used by all agents is configured once at startup via llm_factory.
"""

import re
from langchain_core.messages import HumanMessage, AIMessage

import os

from agents.serializability_agent import build_agent as build_serial_agent
from agents.parallel_query_agent  import build_agent as build_query_agent
from agents.pipeline_agent        import PipelineAgent
from agents.mapreduce_agent       import MapReduceAgent
from agents.semijoin_agent        import build_agent as build_semijoin_agent
from agents.datacube_agent        import build_agent as build_datacube_agent
from agents.full_reducer_agent    import build_agent as build_fullreducer_agent
from agents.ra_proposal_agent     import generate_ra_proposals
from llm_factory                  import build_llm, LLMConfig


# ══════════════════════════════════════════════════════════════
#  ROUTER PROMPT
#  — written for weak/local models:
#    * few-shot examples (most important fix)
#    * one-word instruction repeated twice
#    * no abstract descriptions — only concrete keywords + examples
# ══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """Your task: read the user message and output exactly one word.

The word must be one of: SERIAL, QUERY, JOIN, SEMIJOIN, MAPREDUCE, DATACUBE, UNKNOWN

Rules:
- Output SERIAL if the message is about transaction schedules, read/write operations, serializability, precedence graphs, or conflict analysis.
- Output JOIN if the message is about computing the cost of a Join (⋈) operation in a parallel database — possibly combined with Select (σ) before the join, or joining more than two tables.
- Output SEMIJOIN if the message is about a Semi-Join (⋉) operation or contains keywords like "semi-join", "semi join", "semijoin", or asks to draw a diagram for semi-join cost analysis.
- Output QUERY if the message is about parallel databases, processors, block size, query cost, round-robin, hash partitioning, range partitioning, relational algebra, or table scan / sort cost (but NOT primarily Join or Semi-Join).
- Output MAPREDUCE if the message is about a Map-Reduce task: word count, distributed aggregation, inverted index, or any task described in terms of map and reduce phases over distributed data.
- Output DATACUBE if the message is about data cube materialisation, Hasse diagrams of cubes or subcubes, Ullman's algorithm, selecting N optimal views/cubes, or OLAP lattice optimisation.
- Output FULLREDUCER if the message mentions "full reducer", "full-reducer", dangling tuples, acyclic join reduction, or asks to apply the semi-join-based Full Reducer algorithm to a schema.
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

Message: "Semi-Join issue"
Answer: SEMIJOIN

Message: "Compute semi-join cost R ⋉ S, 8 processors."
Answer: SEMIJOIN

Message: "Semi-join R ⋉ S — вычислить стоимость, нарисовать диаграмму."
Answer: SEMIJOIN

Message: "Given a partial Hasse diagram of data cubes with access costs. Apply Ullman's algorithm to select N=3 optimal cubes."
Answer: DATACUBE

Message: "Subcubes: A=300, B=100, C=250. Hierarchy A->B,C. Select 2 cubes for materialization."
Answer: DATACUBE

Message: "Apply Ullman approximation: lattice A->B->C, costs A=1000, B=200, C=50, select N=2."
Answer: DATACUBE

Message: "Apply Full Reducer to A(a,b,c); B(b,c,d); C(c,d,e)."
Answer: FULLREDUCER

Message: "Given schema R(a,b), S(b,c), T(a,c) — use the full reducer algorithm."
Answer: FULLREDUCER

Message: "Check if this join is acyclic and run full-reducer: Emp(id,dept), Dept(dept,mgr), Project(mgr,pid)."
Answer: FULLREDUCER

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

_VALID = {"SERIAL", "QUERY", "JOIN", "SEMIJOIN", "MAPREDUCE", "DATACUBE", "FULLREDUCER", "UNKNOWN"}

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
        self.semijoin_agent  = build_semijoin_agent(llm=self.llm)
        self.datacube_agent      = build_datacube_agent(llm=self.llm)
        self.fullreducer_agent   = build_fullreducer_agent(llm=self.llm)

        self.serial_history:   list = []
        self.query_history:    list = []
        self.query_db_context: dict = {}

        # Human-in-the-loop RA selection state
        # Set while waiting for the user to pick one of the 3 RA proposals.
        self._pending_ra: dict | None = None

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

    # ── RA selection helpers ──────────────────────────────────────

    def _run_query_agent(self, user_input: str) -> str:
        """Run the parallel query agent with the given message."""
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

    def _propose_ra_and_wait(self, user_input: str) -> str:
        """
        Generate 3 RA proposals in parallel, save state, return the selection menu.
        Falls back to running the query agent directly if OPENAI_API_KEY is missing.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(f"{DIM}[RA proposals] OPENAI_API_KEY not set — skipping RA selection{RESET}")
            return self._run_query_agent(user_input)

        print(f"{DIM}[RA proposals] Calling gpt-4o / gpt-5.4-nano / gpt-5.4-mini in parallel…{RESET}")
        try:
            proposals = generate_ra_proposals(user_input, api_key)
        except Exception as exc:
            print(f"{RED}[RA proposals] Error: {exc} — falling back to direct agent{RESET}")
            return self._run_query_agent(user_input)

        self._pending_ra = {"query": user_input, "proposals": proposals}

        lines = ["Two Relational Algebra proposals were generated. Choose one:\n"]
        for i, (model, ra) in enumerate(proposals, 1):
            lines.append(f"[{i}] RA from {model}:\n    {ra}\n")
        lines.append("Enter 1 or 2 to select and proceed with cost calculation.")
        return "\n".join(lines)

    def _apply_ra_selection(self, choice: str) -> str:
        """
        Apply the user's RA choice and run the full query agent with the selected RA
        injected at the top of the message so the agent skips RA formulation.
        """
        state = self._pending_ra
        self._pending_ra = None

        idx = int(choice) - 1
        model, ra = state["proposals"][idx]
        original_query = state["query"]

        print(f"{DIM}[RA selected] {model}: {ra[:80]}{RESET}")

        # Inject the selected RA so the query agent doesn't re-derive it
        combined = (
            f"[SELECTED RA]: {ra}\n\n"
            f"[QUERY]: {original_query}\n\n"
            "Use the RA above exactly as written. "
            "Do NOT generate or propose a different RA. "
            "Proceed directly to schema extraction and cost computation."
        )
        return self._run_query_agent(combined)

    # ── Main dispatch ─────────────────────────────────────────────

    def handle(self, user_input: str) -> str:
        # ── Human-in-the-loop: waiting for RA selection ───────────
        if self._pending_ra is not None:
            choice = user_input.strip()
            if choice in ("1", "2"):
                return self._apply_ra_selection(choice)
            else:
                # User sent something other than 1/2/3 — cancel and re-route
                print(f"{DIM}[RA selection cancelled]{RESET}")
                self._pending_ra = None

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
            return self._propose_ra_and_wait(user_input)

        elif domain == "JOIN":
            return self.pipeline_agent.handle(user_input)

        elif domain == "SEMIJOIN":
            from langchain_core.messages import AIMessage as _AI
            result = self.semijoin_agent.invoke({"messages": [HumanMessage(content=user_input)]})
            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, _AI)),
                None,
            )
            return last_ai.content if last_ai else "(no response)"

        elif domain == "FULLREDUCER":
            from langchain_core.messages import AIMessage as _AI
            result = self.fullreducer_agent.invoke({"messages": [HumanMessage(content=user_input)]})
            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, _AI)),
                None,
            )
            return last_ai.content if last_ai else "(no response)"

        elif domain == "DATACUBE":
            from langchain_core.messages import AIMessage as _AI
            result = self.datacube_agent.invoke({"messages": [HumanMessage(content=user_input)]})
            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, _AI)),
                None,
            )
            return last_ai.content if last_ai else "(no response)"

        elif domain == "MAPREDUCE":
            return self.mapreduce_agent.handle(user_input)

        else:
            return (
                "I specialise in six topics:\n"
                "  • Transaction schedule serializability\n"
                "  • Parallel query cost (Select, Sort)\n"
                "  • Parallel Join cost\n"
                "  • Parallel Semi-Join cost  (say 'Semi-Join issue' to draw a diagram)\n"
                "  • Map-Reduce algorithms\n"
                "  • Data-cube materialisation (Ullman's algorithm, Hasse diagrams)\n"
                "  • Full Reducer algorithm (acyclic join, GYO reduction, semi-join phases)\n"
                "Please clarify your question."
            )

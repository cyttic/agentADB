"""
agents/ra_proposal_agent.py
============================
Generates Relational Algebra proposals from 3 independent LLM agents in parallel.

Models: gpt-4o, gpt-5.4-nano, gpt-5.4-mini (always OpenAI — RA formulation
is a pure reasoning task that benefits from model diversity).
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS = ["gpt-5.4-nano", "gpt-5.4-mini"]

_RA_PROMPT = """\
You are a relational algebra expert specializing in query optimization.

Given the database query description below, write the OPTIMIZED Relational Algebra expression.

OPTIMIZATION RULES (apply all of them):

1. USE MINIMUM TABLES — only include tables that are strictly necessary to produce the result.
   Before adding a table to the expression, ask: "Can I get the required output fields
   and evaluate all conditions without this table?"
   If yes — exclude it.
   Example: query asks for cid of clients who ordered products with cost > 100.
     Tables: Clients(cid, name), Orders(oid, cid, pid), Products(pid, cost)
     → cid lives in Orders. cost lives in Products. Clients is NOT needed.
     → Correct: π(cid)(Orders ⋈ σ(cost > 100)(Products))
     → Wrong:   Clients ⋈ Orders ⋈ σ(cost > 100)(Products)   ← Clients is redundant

2. PUSH SELECTIONS DOWN — apply ALL σ conditions on a single table BEFORE any Join.
   This reduces the size of intermediate results entering the join.

3. MINIMUM JOINS — use the fewest ⋈ operations strictly required.
   Never join tables that are not needed to satisfy the query or produce the output.

4. PUSH PROJECTIONS DOWN — apply π as early as possible to drop columns not needed downstream.

NOTATION RULES:
- Use ONLY these Unicode symbols (no LaTeX, no backslash commands):
    Select:   σ(condition)(Table)
    Project:  π(fields)(Table)
    Join:     Table1 ⋈ Table2   or   Table1 ⋈(condition) Table2
- Output ONLY the RA expression — no explanation, no extra text, no markdown.
- If there is no projection needed, omit π.

Query:
{query}

Optimized RA (minimum tables, minimum joins, selections pushed down):"""


def _propose_one(query: str, model: str, api_key: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    response = llm.invoke([HumanMessage(content=_RA_PROMPT.format(query=query))])
    return response.content.strip()


def generate_ra_proposals(query: str, api_key: str | None = None) -> list[tuple[str, str]]:
    """
    Call all 3 models in parallel.
    Returns [(model_name, ra_expression), ...] in MODELS order.
    Falls back gracefully if a model call fails.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    results: list[tuple[str, str] | None] = [None] * len(MODELS)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_propose_one, query, model, key): i
            for i, model in enumerate(MODELS)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = (MODELS[idx], future.result())
            except Exception as exc:
                results[idx] = (MODELS[idx], f"(failed: {exc})")

    return results  # type: ignore[return-value]

"""
agents/ra_proposal_agent.py
============================
Generates Relational Algebra proposals from 3 independent LLM agents in parallel.

Models: gpt-4o, gpt-5.4-nano, gpt-5.4-mini (always OpenAI — RA formulation
is a pure reasoning task that benefits from model diversity).
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS = ["gpt-4o", "gpt-5.4-nano", "gpt-5.4-mini"]

_RA_PROMPT = """\
You are a relational algebra expert specializing in query optimization.

Given the database query description below, write the OPTIMIZED Relational Algebra expression.

OPTIMIZATION GOAL — minimize the number of Join (⋈) operations:
1. Apply ALL Select (σ) conditions BEFORE any Join — push selections as far down as possible.
   This reduces the size of intermediate results entering the join.
2. Use the MINIMUM number of Joins strictly required to answer the query.
   Never join tables that are not needed to satisfy the query.
3. If a condition can be evaluated on a single table without joining, do so with σ alone.
4. Apply Project (π) as early as possible to drop columns not needed downstream.

NOTATION RULES:
- Use ONLY these Unicode symbols (no LaTeX, no backslash commands):
    Select:   σ(condition)(Table)
    Project:  π(fields)(Table)
    Join:     Table1 ⋈ Table2   or   Table1 ⋈(condition) Table2
- Output ONLY the RA expression — no explanation, no extra text, no markdown.
- If there is no projection needed, omit π.
- Correct form with Select before Join: σ(cond)(Table) ⋈ OtherTable

Query:
{query}

Optimized RA (minimum joins, selections pushed down):"""


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

    with ThreadPoolExecutor(max_workers=3) as pool:
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

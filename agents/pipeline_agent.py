"""
agents/pipeline_agent.py
=========================
Deterministic pipeline agent for compound queries (Select+Join, Join+Join, etc.)

Architecture — three phases:
  1. PLAN    — single LLM call returns a structured JSON list of operations in order.
  2. EXECUTE — Python iterates the list and calls db_ops tools directly (no LLM).
               Output of each op feeds into the next as a virtual table.
  3. FORMAT  — single LLM call receives all computed data and writes the response.

The LLM never computes costs.  Python never decides the plan.
"""

import json
import math
import re

from langchain_core.messages import HumanMessage, SystemMessage

from tools.db_ops import (
    parallel_join_cost      as _join_cost,
    decide_select_algorithm as _decide_select_algo,
    select_cost             as _select_cost,
    _fmt,
)


# ══════════════════════════════════════════════════════════════
#  PLANNER PROMPT
#  Goal: produce a clean JSON plan the executor can run without ambiguity.
# ══════════════════════════════════════════════════════════════

PLANNER_PROMPT = """\
You are a database query analyzer.
Read the task and return ONLY a JSON execution plan — no markdown, no explanation.

Schema:
{
  "num_processors": <int>,
  "tables": {
    "<TableName>": {
      "blocks": <int>,
      "distribution": "hash(<field>)" | "range(<field>)" | "round_robin"
    }
  },
  "operations": [
    {
      "type": "SELECT",
      "input_table":  "<TableName>",
      "output_table": "<TableName>_filtered",
      "select_field": "<field in WHERE clause>",
      "condition_type": "point" | "range" | "scan",
      "condition": "<human-readable condition, e.g. '100 < price < 200'>",
      "selectivity": <float 0.0-1.0 — fraction of matching tuples, use 1.0 if unknown>
    },
    {
      "type": "JOIN",
      "input_a":      "<TableName or output_table of a previous op>",
      "input_b":      "<TableName or output_table of a previous op>",
      "output_table": "<unique snake_case name for this result, e.g. 'Flowers_Sales_join'>",
      "join_field":   "<the field both tables share>"
    }
  ]
}

Rules:
  - Convert power-of-10 notation to plain integers: 10^4=10000, 10^6=1000000.
  - Operations MUST be in execution order.
    If a SELECT filters a table that is later JOINed, SELECT must come first.
  - If only one table has a stated distribution, use "round_robin" for the others.
  - selectivity: if the task provides a value range [a,b] over V distinct values
    use (b-a)/V; otherwise use 1.0.
  - join_field: the natural join key (e.g. "name" when both tables have a "name" column).
  - Every JOIN must have an output_table field.
  - For chained joins (A⋈B)⋈C: the first JOIN's output_table becomes input_a or input_b
    of the second JOIN — the names must match exactly.

Task: """


# ══════════════════════════════════════════════════════════════
#  FORMATTER PROMPT
#  Goal: produce a clean, plain-text Russian response from computed data.
# ══════════════════════════════════════════════════════════════

FORMATTER_SYSTEM = (
    "You are a parallel database systems expert. "
    "Always respond in Russian. "
    "STRICT FORMAT: plain text only — no LaTeX, no \\[, \\], \\times, \\frac, no $...$. "
    "Use * for multiplication and ^ for exponents. "
    "Copy symbolic cost strings from the JSON EXACTLY — do not reformat them."
)

FORMATTER_PROMPT = """\
Present the following query execution results to a student.

Rules:
  - Show each operation as a numbered step.
  - For SELECT: state the algorithm (alg2/alg3), reason, Elapsed, Total, and the
    resulting block count that feeds into the next step.
  - For JOIN: state the algorithm (LOCAL or BROADCAST), Elapsed, Total.
  - At the end show the combined Elapsed and Total (sum of all steps).
  - DO NOT simplify symbolic expressions to a single number.
  - Respond in Russian.

Computed results:
{results_json}

Original task:
{task}
"""


# ══════════════════════════════════════════════════════════════
#  PIPELINE AGENT
# ══════════════════════════════════════════════════════════════

class PipelineAgent:
    """
    Deterministic plan-execute-format pipeline.
    Handles any combination of SELECT and JOIN operations.
    """

    def __init__(self, llm):
        self.llm = llm

    # ── Public entry point ────────────────────────────────────

    def handle(self, user_input: str) -> str:
        try:
            plan = self._extract_plan(user_input)
        except Exception as e:
            return f"Ошибка при разборе задачи: {e}"

        try:
            steps, combined = self._execute_plan(plan)
        except Exception as e:
            return f"Ошибка при выполнении плана: {e}"

        return self._format_result(user_input, plan, steps, combined)

    # ── Phase 1: plan ─────────────────────────────────────────

    def _extract_plan(self, user_input: str) -> dict:
        response = self.llm.invoke([
            HumanMessage(content=PLANNER_PROMPT + user_input)
        ])
        raw = response.content.strip()
        # Strip markdown code fences if the model wraps output in ```json ... ```
        raw = re.sub(r'^```[a-z]*\s*', '', raw, flags=re.MULTILINE).strip()
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE).strip()
        plan = json.loads(raw)
        print(f"[pipeline] plan extracted: {json.dumps(plan, indent=2)}")
        return plan

    # ── Phase 2: execute (pure Python, no LLM) ───────────────

    def _execute_plan(self, plan: dict) -> tuple:
        p      = plan["num_processors"]
        tables = {name: dict(info) for name, info in plan["tables"].items()}
        steps  = []
        elapsed_parts = []
        total_parts   = []

        for i, op in enumerate(plan["operations"]):
            if op["type"] == "SELECT":
                self._assert_table_exists(tables, op["input_table"], op, i)
                result = self._run_select(op, tables, p)
                steps.append(result)
                elapsed_parts.append(result["elapsed"])
                total_parts.append(result["total"])
                # Store filtered table for downstream ops
                tables[op["output_table"]] = {
                    "blocks":       result["result_blocks"],
                    "distribution": tables[op["input_table"]]["distribution"],
                }

            elif op["type"] == "JOIN":
                self._assert_table_exists(tables, op["input_a"], op, i)
                self._assert_table_exists(tables, op["input_b"], op, i)
                result = self._run_join(op, tables, p)
                steps.append(result)
                elapsed_parts.append(result["elapsed"])
                total_parts.append(result["total"])
                # Always store join result — output_table is required for chained joins
                out = op.get("output_table") or f"_join_{i}_result"
                tables[out] = {
                    "blocks": min(
                        tables[op["input_a"]]["blocks"],
                        tables[op["input_b"]]["blocks"],
                    ),
                    "distribution": "round_robin",
                }

        if len(elapsed_parts) == 1:
            combined_elapsed = elapsed_parts[0]
            combined_total   = total_parts[0]
        else:
            combined_elapsed = " + ".join(f"({e})" for e in elapsed_parts)
            combined_total   = " + ".join(f"({t})" for t in total_parts)

        return steps, {"elapsed": combined_elapsed, "total": combined_total}

    def _assert_table_exists(self, tables: dict, name: str, op: dict, step_idx: int):
        if name not in tables:
            known = list(tables.keys())
            raise KeyError(
                f"Step {step_idx} ({op['type']}) references '{name}' "
                f"which does not exist. Known tables: {known}. "
                f"Check that the producing JOIN has 'output_table': '{name}'."
            )

    def _run_select(self, op: dict, tables: dict, p: int) -> dict:
        tname  = op["input_table"]
        info   = tables[tname]
        blocks = info["blocks"]
        dist   = info["distribution"]

        m = re.search(r'\((\w+)\)', dist)
        dist_field = m.group(1) if m else ""

        algo  = _decide_select_algo(op["select_field"], op["condition_type"], dist_field, dist)
        rel_p = algo.get("relevant_processors") or 1
        cost  = _select_cost(blocks, p, algo["algorithm"], rel_p)

        sel           = op.get("selectivity", 1.0)
        result_blocks = max(1, math.ceil(blocks * sel))

        bs = _fmt(math.ceil(blocks / p))
        print(f"[pipeline] SELECT {tname}: {algo['algorithm']}, "
              f"elapsed={cost['elapsed']}, result_blocks={result_blocks}")

        return {
            "step_type":     "SELECT",
            "table":         tname,
            "condition":     op.get("condition", ""),
            "algorithm":     algo["algorithm"],
            "reason":        algo["reason"],
            "blocks_total":  _fmt(blocks),
            "blocks_per_server": bs,
            "elapsed":       cost["elapsed"],
            "total":         cost["total"],
            "selectivity":   sel,
            "result_blocks": result_blocks,
            "output_table":  op["output_table"],
        }

    def _run_join(self, op: dict, tables: dict, p: int) -> dict:
        name_a = op["input_a"]
        name_b = op["input_b"]
        info_a = tables[name_a]
        info_b = tables[name_b]

        result = _join_cost(
            info_a["blocks"], info_b["blocks"], p,
            name_a, name_b,
            info_a["distribution"], info_b["distribution"],
            op.get("join_field", ""),
        )

        print(f"[pipeline] JOIN {name_a} x {name_b}: {result['algorithm']}, "
              f"elapsed={result['elapsed']}")

        return {
            "step_type":   "JOIN",
            "table_a":     name_a,
            "table_b":     name_b,
            "algorithm":   result["algorithm"],
            "elapsed":     result["elapsed"],
            "total":       result["total"],
            "explanation": result["explanation"],
        }

    # ── Phase 3: format ───────────────────────────────────────

    def _format_result(self, user_input: str, plan: dict, steps: list, combined: dict) -> str:
        results_data = {
            "num_processors":   plan["num_processors"],
            "tables":           plan["tables"],
            "steps":            steps,
            "combined_elapsed": combined["elapsed"],
            "combined_total":   combined["total"],
        }
        prompt = FORMATTER_PROMPT.format(
            results_json=json.dumps(results_data, indent=2, ensure_ascii=False),
            task=user_input,
        )
        response = self.llm.invoke([
            SystemMessage(content=FORMATTER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()

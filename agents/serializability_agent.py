"""
agents/serializability_agent.py
================================
LangGraph agent for checking schedule serializability.
Exposes build_agent() → compiled graph.
"""

import os
import json
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.gemini_view import print_report as view_print_report
from tools.confl_ser import analyze_schedule as conflict_analyze


# ══════════════════════════════════════════════════════════════
#  SCHEDULE PARSER
# ══════════════════════════════════════════════════════════════

import re

_BIDI_CHARS = set(
    '\u200e\u200f'
    '\u202a\u202b\u202c'
    '\u202d\u202e'
    '\u2066\u2067\u2068\u2069'
    '\u200b\u00a0'
)


def _strip_bidi(text: str) -> str:
    return ''.join(c for c in text if c not in _BIDI_CHARS)


def _extract_tokens(text: str) -> list[str]:
    return re.findall(r'[rw]|[0-9]+|[A-Z]+', text)


def _try_parse_tokens(tokens: list[str]) -> list[tuple] | None:
    schedule = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ('r', 'w') and i + 2 < len(tokens):
            try:
                tid = int(tokens[i + 1])
                obj = tokens[i + 2]
                if obj.isalpha() and obj.isupper():
                    schedule.append((t, tid, obj))
                    i += 3
                    continue
            except ValueError:
                pass
        i += 1
    return schedule if schedule else None


def parse_schedule_text(text: str) -> list[tuple]:
    clean = _strip_bidi(text)
    clean = re.sub(r'[Tt]_?(?=[0-9])', '', clean)
    clean = clean.replace('_', '')
    tokens = _extract_tokens(clean)
    if not tokens:
        raise ValueError(f"No recognisable tokens found in: {text!r}")
    result = _try_parse_tokens(tokens)
    if result and len(result) >= 2:
        return result
    result_rtl = _try_parse_tokens(list(reversed(tokens)))
    if result_rtl and len(result_rtl) >= 2:
        return result_rtl
    raise ValueError(
        f"Could not parse schedule. Tokens found: {tokens}\n"
        f"Expected format: r1(A), w2(B), ..."
    )


def _parse_schedule(raw: list[list]) -> list[tuple]:
    if isinstance(raw, str):
        return parse_schedule_text(raw)
    return [(op[0], int(op[1]), op[2]) for op in raw]


# ══════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def parse_schedule_from_text(text: str) -> str:
    """
    Parse a schedule from raw text into a structured list of operations.
    Use this FIRST when the user provides a schedule as plain text.

    Args:
        text: Raw schedule string in any format.

    Returns:
        JSON with parsed operations list and human-readable confirmation.
    """
    try:
        parsed = parse_schedule_text(text)
        return json.dumps({
            "parsed": [[op, tid, obj] for op, tid, obj in parsed],
            "human_readable": ", ".join(
                f"{'r' if op=='r' else 'w'}{tid}({obj})" for op, tid, obj in parsed
            ),
            "count": len(parsed),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def check_view_serializability(schedule: list[list]) -> str:
    """
    Check whether a schedule is VIEW-SERIALIZABLE.

    Use when the user asks about view-serializability, or asks if a schedule
    'is serializable' without specifying type.

    Args:
        schedule: List of [type, transaction_id, object] e.g. [['r',1,'A'], ['w',2,'A']]

    Returns:
        JSON string with verdict and analysis.
    """
    try:
        parsed = _parse_schedule(schedule)
        view_print_report(parsed)
        return json.dumps('Correct', indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def check_conflict_serializability(schedule: list[list]) -> str:
    """
    Check whether a schedule is CONFLICT-SERIALIZABLE.

    Use when the user explicitly asks about conflict-serializability,
    or asks to build a precedence graph and check for cycles.

    Args:
        schedule: List of [type, transaction_id, object] e.g. [['r',1,'A'], ['w',2,'A']]

    Returns:
        JSON string with verdict and analysis.
    """
    try:
        parsed = _parse_schedule(schedule)
        conflict_analyze(parsed)
        return json.dumps('Correct', indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [parse_schedule_from_text, check_view_serializability, check_conflict_serializability]

def build_system_prompt(lang: str = "ru") -> str:
    _lang_rule = "Always respond in Russian, regardless of input language." if lang == "ru" else "Always respond in English, regardless of input language."  # noqa: F841
    return f"""You are a database systems expert specializing in transaction schedule analysis.

You have three tools:
1. parse_schedule_from_text    — parses raw text schedules (use first if input is text)
2. check_view_serializability  — checks view-serializability (broader check)
3. check_conflict_serializability — checks conflict-serializability (precedence graph)

When a user gives you a schedule task:
1. Identify which type of serializability they are asking about.
   If unclear, default to view-serializability.
2. Parse the schedule into [['r'|'w', tid, object], ...] format.
3. Call the appropriate tool.
4. Explain the result clearly: verdict, reason, graph edges, serial order or cycle.

LANGUAGE RULE: {_lang_rule}

═══ STRICT OUTPUT FORMAT (follow exactly, even on small models) ═══

After analysis, ALWAYS end your response with this block:

---
📋 Рёбра графа: <list of edges like T1→T2, or "(нет)">
🔄 Цикл: <Да / Нет>
🏷️  Вердикт: <CONFLICT-SERIALIZABLE / NOT CONFLICT-SERIALIZABLE / VIEW-SERIALIZABLE / NOT VIEW-SERIALIZABLE>
📎 Причина: <one sentence>
✅ Серийный порядок: <T1 → T2 → ... or "(нет — цикл)">
---

NEVER skip this block. NEVER guess results — always call tools first.
"""


# ══════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════

class SerialAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  BUILD AGENT
# ══════════════════════════════════════════════════════════════

_agent_lang: str = "ru"

def set_agent_lang(lang: str):
    global _agent_lang
    _agent_lang = lang


def build_agent(llm=None):
    """
    Build the serializability agent.
    Args:
        llm: A LangChain chat model. If None, defaults to gpt-4o via OPENAI_API_KEY.
    """
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: SerialAgentState):
        messages = [SystemMessage(content=build_system_prompt(_agent_lang))] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: SerialAgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(SerialAgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()

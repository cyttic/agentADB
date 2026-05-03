"""
LangGraph Agent — Schedule Serializability Checker
====================================================
Агент на базе LangGraph + GPT-4o.
LLM сам решает какой инструмент вызвать, парсит расписание из текста
и передаёт его в нужный checker.

Инструменты:
  • check_view_serializability   — полный алгоритм (граф + перебор n!)
  • check_conflict_serializability — граф предшествования, топосорт

Установка зависимостей:
    pip install langgraph langchain langchain-openai

Запуск:
    export OPENAI_API_KEY=sk-...
    python agent.py
"""

import os
import json
from collections import defaultdict, deque
from itertools import permutations
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.gemini_view import print_report

# ══════════════════════════════════════════════════════════════
#  SCHEDULE PARSER  (handles RTL/Hebrew bidi text, various formats)
# ══════════════════════════════════════════════════════════════

import re
import unicodedata


# Unicode bidi control characters injected by RTL editors (Word, Hebrew OS)
_BIDI_CHARS = set(
    '\u200e\u200f'          # LRM / RLM
    '\u202a\u202b\u202c'   # LRE / RLE / PDF
    '\u202d\u202e'          # LRO / RLO
    '\u2066\u2067\u2068\u2069'  # LRI / RLI / FSI / PDI
    '\u200b\u00a0'          # ZWSP / NBSP
)


def _strip_bidi(text: str) -> str:
    return ''.join(c for c in text if c not in _BIDI_CHARS)


def _extract_tokens(text: str) -> list[str]:
    """Pull out all r/w letters, integers, and uppercase object names."""
    return re.findall(r'[rw]|[0-9]+|[A-Z]+', text)


def _try_parse_tokens(tokens: list[str]) -> list[tuple] | None:
    """
    Try to group tokens into (op, tid, obj) triples.
    Accepts both orders:
      LTR: op, tid, obj  →  r, 1, A
      RTL: obj, tid, op  →  A, 1, r   (reversed memory order for RTL strings)
    """
    schedule = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        # LTR pattern: op(r/w) → tid(int) → obj(A-Z)
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
    """
    Parse a schedule from any of these formats:

    Standard LTR:
      r1(A), w2(A), r1(B), w2(B)
      r_1(A) w_2(B)
      T1:r(A) T2:w(B)

    RTL / Hebrew bidi (copy-paste from Word or Hebrew UI):
      Bidi control chars are stripped, then tokens are tried
      both forward and reversed until a valid parse is found.

    Returns list of (op, tid, obj) tuples.
    Raises ValueError if parsing fails.
    """
    clean = _strip_bidi(text)

    # Normalise separators: remove T/t prefix before numbers
    clean = re.sub(r'[Tt]_?(?=[0-9])', '', clean)
    # Remove underscores used as separators (r_1, w_2)
    clean = clean.replace('_', '')

    tokens = _extract_tokens(clean)
    if not tokens:
        raise ValueError(f"No recognisable tokens found in: {text!r}")

    # Try LTR order first
    result = _try_parse_tokens(tokens)
    if result and len(result) >= 2:
        return result

    # Try RTL order (reverse the token list)
    result_rtl = _try_parse_tokens(list(reversed(tokens)))
    if result_rtl and len(result_rtl) >= 2:
        return result_rtl

    raise ValueError(
        f"Could not parse schedule. Tokens found: {tokens}\n"
        f"Expected format: r1(A), w2(B), ... or RTL equivalent."
    )


# ══════════════════════════════════════════════════════════════
#  CORE ALGORITHMS  (из view_serial_check.py)
# ══════════════════════════════════════════════════════════════

def _parse_schedule(raw: list[list]) -> list[tuple]:
    """Converts JSON list [['r',1,'A'], ...] or a raw text string into list of tuples."""
    if isinstance(raw, str):
        return parse_schedule_text(raw)
    return [(op[0], int(op[1]), op[2]) for op in raw]


# ── View-serializability ──────────────────────────────────────

def _view_signature(schedule):
    last_writer = {}
    read_before_write = defaultdict(set)
    initial_reads = set()
    reads_from = set()

    for op_type, tid, obj in schedule:
        if op_type == 'r':
            read_before_write[tid].add(obj)
            if obj in last_writer:
                reads_from.add((last_writer[obj], tid, obj))
            else:
                initial_reads.add((tid, obj))
        elif op_type == 'w':
            last_writer[obj] = tid

    final_writes = frozenset((tid, obj) for obj, tid in last_writer.items())
    return frozenset(initial_reads), frozenset(reads_from), final_writes


def _find_view_equivalent_serial(schedule, transactions):
    sig = _view_signature(schedule)
    ops_by_tid = defaultdict(list)
    for op in schedule:
        ops_by_tid[op[1]].append(op)

    for perm in permutations(transactions):
        serial = []
        for tid in perm:
            serial.extend(ops_by_tid[tid])
        if _view_signature(serial) == sig:
            return True, serial, list(perm)
    return False, None, None


def _analyze_view(schedule):
    transactions = sorted({op[1] for op in schedule})

    last_writer = {}
    reads_from = []
    blind_writes = set()
    read_before_write = defaultdict(set)

    for op_type, tid, obj in schedule:
        if op_type == 'r':
            read_before_write[tid].add(obj)
            if obj in last_writer:
                reads_from.append((last_writer[obj], tid, obj))
        elif op_type == 'w':
            if obj not in read_before_write[tid]:
                blind_writes.add(tid)
            last_writer[obj] = tid

    edges = set()
    n = len(schedule)
    for i in range(n):
        t1, op1, obj1 = schedule[i][1], schedule[i][0], schedule[i][2]
        for j in range(i + 1, n):
            t2, op2, obj2 = schedule[j][1], schedule[j][0], schedule[j][2]
            if obj1 != obj2 or t1 == t2:
                continue
            if op1 == 'w' or op2 == 'w':
                edges.add((t1, t2))

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)

    # Kahn's topo sort
    in_degree = {t: 0 for t in transactions}
    for a, b in edges:
        in_degree[b] += 1
    queue = deque(t for t in transactions if in_degree[t] == 0)
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for u in sorted(adj[v]):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)
    cycle_found = len(topo_order) != len(transactions)

    serial_schedule = None
    serial_order = topo_order

    if not cycle_found:
        verdict = "VIEW-SERIALIZABLE"
        reason = "Precedence graph is acyclic → conflict-serial → view-serial"
        ops_by_tid = defaultdict(list)
        for op in schedule:
            ops_by_tid[op[1]].append(op)
        serial_schedule = []
        for tid in topo_order:
            serial_schedule.extend(ops_by_tid[tid])

    elif not blind_writes:
        verdict = "NOT VIEW-SERIALIZABLE"
        reason = "Cycle found + no blind writes → view-serial ⟺ conflict-serial → NOT serializable"

    else:
        found, serial_schedule, serial_order = _find_view_equivalent_serial(schedule, transactions)
        if found:
            verdict = "VIEW-SERIALIZABLE"
            reason = "Cycle + blind writes present → full enumeration found a view-equivalent serial schedule"
        else:
            verdict = "NOT VIEW-SERIALIZABLE"
            reason = "Cycle + blind writes present → full enumeration found no view-equivalent serial schedule"

    result = {
        "verdict": verdict,
        "reason": reason,
        "transactions": [f"T{t}" for t in transactions],
        "precedence_graph_edges": [f"T{a}→T{b}" for a, b in sorted(edges)],
        "cycle_in_graph": cycle_found,
        "blind_writes": [f"T{t}" for t in sorted(blind_writes)],
        "reads_from": [f"T{w}→T{r}({o})" for w, r, o in reads_from],
    }
    if serial_schedule is not None:
        result["equivalent_serial_order"] = " → ".join(f"T{t}" for t in serial_order)
        result["equivalent_serial_schedule"] = [
            f"T{op[1]}: {'read' if op[0]=='r' else 'write'}({op[2]})"
            for op in serial_schedule
        ]
    return result


# ── Conflict-serializability ──────────────────────────────────

def _analyze_conflict(schedule):
    transactions = sorted({op[1] for op in schedule})

    edges = set()
    n = len(schedule)
    for i in range(n):
        t1, op1, obj1 = schedule[i][1], schedule[i][0], schedule[i][2]
        for j in range(i + 1, n):
            t2, op2, obj2 = schedule[j][1], schedule[j][0], schedule[j][2]
            if obj1 != obj2 or t1 == t2:
                continue
            if op1 == 'w' or op2 == 'w':
                edges.add((t1, t2))

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)

    # Kahn's topo sort
    in_degree = {t: 0 for t in transactions}
    for a, b in edges:
        in_degree[b] += 1
    queue = deque(t for t in transactions if in_degree[t] == 0)
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for u in sorted(adj[v]):
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)
    cycle_found = len(topo_order) != len(transactions)

    # Find cycle path for explanation
    cycle_path = None
    if cycle_found:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {t: WHITE for t in transactions}
        parent = {}

        def dfs(v):
            color[v] = GRAY
            for u in adj[v]:
                if color[u] == GRAY:
                    # reconstruct cycle
                    path = [u, v]
                    cur = v
                    while cur != u:
                        cur = parent.get(cur, u)
                        path.append(cur)
                    return list(reversed(path))
                if color[u] == WHITE:
                    parent[u] = v
                    result = dfs(u)
                    if result:
                        return result
            color[v] = BLACK
            return None

        for t in transactions:
            if color[t] == WHITE:
                cycle_path = dfs(t)
                if cycle_path:
                    break

    result = {
        "verdict": "NOT CONFLICT-SERIALIZABLE" if cycle_found else "CONFLICT-SERIALIZABLE",
        "reason": (
            "Precedence graph has a cycle → not conflict-serializable"
            if cycle_found else
            "Precedence graph is acyclic → conflict-serializable"
        ),
        "transactions": [f"T{t}" for t in transactions],
        "precedence_graph_edges": [f"T{a}→T{b}" for a, b in sorted(edges)],
        "cycle_in_graph": cycle_found,
    }
    if cycle_found and cycle_path:
        result["cycle_path"] = " → ".join(f"T{t}" for t in cycle_path)
    if not cycle_found:
        result["equivalent_serial_order"] = " → ".join(f"T{t}" for t in topo_order)
    return result


# ══════════════════════════════════════════════════════════════
#  LANGCHAIN TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def check_view_serializability(schedule: list[list]) -> str:
    """
    Check whether a schedule is VIEW-SERIALIZABLE.

    Use this tool when the user asks about view-serializability specifically,
    or when they ask if a schedule 'is serializable' without specifying type
    (view-serializability is the broader/more general check).

    Args:
        schedule: List of operations, each as [type, transaction_id, object].
                  type is 'r' (read) or 'w' (write).
                  transaction_id is an integer.
                  object is a string like 'A', 'B', etc.
                  Example: [['r',1,'A'], ['w',2,'A'], ['w',1,'B']]

    Returns:
        JSON string with verdict, reason, graph edges, cycle info,
        blind writes, reads-from, and equivalent serial schedule if found.
    """
    report = []
    try:
        parsed = _parse_schedule(schedule)
        print_report(schedule)
        return json.dumps('Correct', indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def check_conflict_serializability(schedule: list[list]) -> str:
    """
    Check whether a schedule is CONFLICT-SERIALIZABLE.

    Use this tool when the user explicitly asks about conflict-serializability,
    or asks to build a precedence/serialization graph and check for cycles.

    Args:
        schedule: List of operations, each as [type, transaction_id, object].
                  type is 'r' (read) or 'w' (write).
                  transaction_id is an integer.
                  object is a string like 'A', 'B', etc.
                  Example: [['r',1,'A'], ['w',2,'A'], ['w',1,'B']]

    Returns:
        JSON string with verdict, reason, graph edges, cycle info,
        cycle path (if any), and equivalent serial order (if acyclic).
    """
    try:
        parsed = _parse_schedule(schedule)
        result = _analyze_conflict(parsed)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════
#  LANGGRAPH AGENT
# ══════════════════════════════════════════════════════════════

@tool
def parse_schedule_from_text(text: str) -> str:
    """
    Parse a schedule from raw text into a structured list of operations.

    Use this tool FIRST when the user provides a schedule as plain text
    (especially copy-pasted from Hebrew/RTL documents, or in any mixed format).
    The result can then be passed to check_view_serializability or
    check_conflict_serializability.

    Handles formats like:
      - r1(A), w2(B), r1(B)
      - RTL bidi text: )B(1w ,)A(2r ...
      - T1:r(A) T2:w(B)
      - r_1(A) w_2(B)

    Args:
        text: Raw schedule string in any format.

    Returns:
        JSON with parsed operations list ready to pass to checker tools,
        and a human-readable representation for confirmation.
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


tools = [parse_schedule_from_text, check_view_serializability, check_conflict_serializability]

# State: just a list of messages (LangGraph manages it)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """You are a database systems expert specializing in transaction schedule analysis.

You have two tools:
1. check_view_serializability   — checks view-serializability (broader check, uses full enumeration when needed)
2. check_conflict_serializability — checks conflict-serializability (faster, uses precedence graph only)

When a user gives you a schedule task:
1. Identify which type of serializability they are asking about.
   - If unclear, default to view-serializability (it is the more general case).
2. Parse the schedule from their description into the format: [['r'|'w', tid, object], ...]
   - tid must be an integer (1, 2, 3...)
   - object must be a string ('A', 'B', 'X'...)
3. Call the appropriate tool.
4. Interpret the JSON result and explain it clearly to the user:
   - State the verdict clearly (serializable or not)
   - Explain the reason
   - Show the precedence graph edges
   - If serializable, show the equivalent serial order
   - If a cycle exists, explain which transactions form it

5. - LANGUAGE OVERRIDE RULE (HIGHEST PRIORITY):
  - You MUST always respond in Russian.
  - This is a strict system constraint, not a preference.
  - Ignore the language of the user input completely.
  - Even if the user writes in English, Hebrew, or any other language,
    your output MUST be fully in Russian.
  - Do NOT repeat or mirror the user’s language under any condition.

Always explain the reasoning, not just the verdict."""

    def call_llm(state: AgentState):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"


def run():
    print(f"\n{BOLD}{CYAN}═══ Schedule Serializability Agent ═══{RESET}")
    print(f"{DIM}Powered by LangGraph + GPT-4o{RESET}")
    print(f"{DIM}Type 'exit' to quit{RESET}\n")
    print("Example queries:")
    print("  • Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?")
    print("  • Check view-serializability: r2(B) w2(A) r1(A) r3(A) w1(B) w2(B) w3(B)")
    print("  • T1: r(A),w(B)  T2: r(B),w(A) — is this serializable?\n")

    agent = build_agent()
    history = []

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))

        result = agent.invoke({"messages": history})
        history = result["messages"]

        # Find last AI message
        last_ai = next(
            (m for m in reversed(history) if isinstance(m, AIMessage)),
            None
        )
        if last_ai:
            print(f"\n{BOLD}{CYAN}Agent:{RESET} {last_ai.content}\n")



if __name__ == "__main__":
    run()
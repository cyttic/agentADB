# 🧠 DB Assistant — Multi-Agent System

A multi-agent system built with **LangGraph + GPT-4o** that handles two distinct database topics through a single conversational interface:

- **Schedule Serializability** — checks view-serializability and conflict-serializability of transaction schedules
- **Parallel Query Cost Analysis** — computes parallel and total execution time for distributed database queries

An orchestrator routes each user message to the appropriate specialized agent automatically.

---

## 📁 Project Structure

```
project/
├── main.py                          # Entry point — interactive loop
├── orchestrator.py                  # Router + session manager
│
├── agents/
│   ├── __init__.py
│   ├── serializability_agent.py     # Agent for schedule serializability
│   └── parallel_query_agent.py      # Agent for parallel query cost analysis
│
└── tools/
    ├── gemini_view.py               # View-serializability printer (colored output)
    └── confl_ser.py                 # Conflict-serializability analyzer (colored output)
```

---

## ⚙️ How It Works

```
User input
    ↓
Orchestrator (lightweight LLM router)
    ├── SERIAL  → SerializabilityAgent
    └── QUERY   → ParallelQueryAgent
```

### Orchestrator (`orchestrator.py`)
- Makes a single cheap LLM call to classify the user's message as `SERIAL`, `QUERY`, or `UNKNOWN`
- Maintains **separate conversation histories** for each agent so they don't interfere
- The `ParallelQueryAgent` also keeps a persistent `db_context` across turns — schema is parsed once and reused for all follow-up queries

### SerializabilityAgent (`agents/serializability_agent.py`)
Handles transaction schedule analysis. Tools:
| Tool | Description |
|------|-------------|
| `parse_schedule_from_text` | Parses raw text schedules (supports RTL/Hebrew bidi text) |
| `check_view_serializability` | Full view-serializability check (enumeration + blind writes) |
| `check_conflict_serializability` | Conflict-serializability via precedence graph + cycle detection |

### ParallelQueryAgent (`agents/parallel_query_agent.py`)
Handles parallel DB cost problems. Tools:
| Tool | Description |
|------|-------------|
| `parse_db_schema` | Parses schema, table sizes, block size, processor count — stored in state |
| `compute_block_count` | Calculates block count for any relation |
| `compute_selectivity` | Estimates result size for equality / range conditions |
| `compute_parallel_cost` | Computes parallel time and total time for scan/join algorithms |

---

## 🚀 Installation

```bash
pip install langgraph langchain langchain-openai
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

Run:

```bash
python main.py
```

---

## 💬 Example Queries

**Serializability:**
```
Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?
Check view-serializability: r2(B) w2(A) r1(A) r3(A) w1(B) w2(B) w3(B)
```

**Parallel query cost:**
```
10 processors, block size 2000 bytes, Customers 10^6 rows, Orders 10^8 rows.
Find all customers who ordered products over 100₪ in quantities 50–100.
Orders distributed by round-robin.
```

---

## 🏗️ Architecture Notes

- **All math is in Python tools** — the LLM only reasons and orchestrates, never computes numbers itself
- **Deterministic output** — tools produce colored terminal reports via ANSI codes, consistent across runs
- **RTL/Hebrew support** — the schedule parser strips Unicode bidi control characters and handles reversed token order
- **Separate state per agent** — serializability history and query DB context never mix
- **Easily extensible** — add a new agent by creating `agents/new_agent.py`, adding a route in `orchestrator.py`

---

## 🛠️ Adding a New Agent

1. Create `agents/my_agent.py` with a `build_agent()` function that returns a compiled LangGraph
2. In `orchestrator.py`, import it and add a new domain label to the router prompt
3. Add the routing branch in `Orchestrator.handle()`

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent graph execution |
| `langchain` | Tool definitions, message types |
| `langchain-openai` | GPT-4o integration |
| `openai` | Underlying API client |

---

## 📝 Language Note

Both agents are configured to **always respond in Russian**, regardless of the input language. To change this, update the `LANGUAGE RULE` line in the `SYSTEM_PROMPT` of each agent file.

# DB Assistant Framework

A multi-agent system for solving advanced database problems through a single conversational interface — terminal or web UI.

The orchestrator classifies every user message and routes it to the appropriate specialized agent. All arithmetic is computed by deterministic Python tools; the LLM only reasons, plans, and formats.

---

## Agents

| Domain | Agent | What it solves |
|--------|-------|----------------|
| `SERIAL` | SerializabilityAgent | View-serializability and conflict-serializability of transaction schedules |
| `QUERY` | ParallelQueryAgent | Parallel cost of Select and Sort operations (alg2 / alg3, symbolic output) |
| `JOIN` | PipelineAgent | Parallel Join cost — local join and broadcast join; chains Select → Join → Join |
| `MAPREDUCE` | MapReduceAgent | Map-Reduce algorithm design — table, chain description, pseudocode |

---

## Architecture

```
User input
    │
    ▼
Orchestrator  ── single LLM call classifies the message ──▶ SERIAL / QUERY / JOIN / MAPREDUCE
    │
    ├── SERIAL     ──▶ SerializabilityAgent   (LangGraph + tools)
    ├── QUERY      ──▶ ParallelQueryAgent     (LangGraph + tools)
    ├── JOIN       ──▶ PipelineAgent          (plan → execute → format)
    └── MAPREDUCE  ──▶ MapReduceAgent         (LLM reasoning)
```

### SerializabilityAgent
LangGraph agent with deterministic Python tools.

| Tool | Description |
|------|-------------|
| `parse_schedule_from_text` | Parses raw schedule text; strips Unicode bidi characters (RTL / Hebrew support) |
| `check_view_serializability` | Full view-serializability check via serial-order enumeration + blind-write detection |
| `check_conflict_serializability` | Conflict-serializability via precedence graph cycle detection |

### ParallelQueryAgent
LangGraph agent. Parses schema from natural language, then computes symbolic costs.

| Tool | Description |
|------|-------------|
| `extract_schema_from_text` | Extracts table sizes, block size, processor count, distributions from raw text. Handles `10^k` notation |
| `parse_schema` | Converts record counts + field sizes to block counts |
| `decide_select_algorithm` | Picks alg2 or alg3 based on partition scheme and query type |
| `select_cost` | Elapsed and Total for a Select in symbolic form |
| `decide_sort_algorithm` | Picks sort algorithm (alg1 / alg2) based on distribution |
| `sort_cost` | 4-step sort cost (local sort → send → receive → merge) |

### PipelineAgent — for Join queries
Three-phase deterministic pipeline (no LLM in the execution step):

```
Phase 1 — PLAN    : single LLM call → structured JSON (ordered list of SELECT / JOIN ops)
Phase 2 — EXECUTE : Python iterates the plan, calls db_ops tools, chains outputs
Phase 3 — FORMAT  : single LLM call → formats computed data into a Russian response
```

Join algorithms chosen automatically by distribution:

| Condition | Algorithm | Cost |
|-----------|-----------|------|
| Both tables hash/range on the join field | **Local join** | `3*(bs_a + bs_b)*t_d` |
| Any other case | **Broadcast join** | Step1 + Step2 + Step3 |

Supports compound expressions: `σ(cond)(A) ⋈ B`, `(A ⋈ B) ⋈ C`.

### MapReduceAgent
Stateful LLM agent (conversation history kept). For each task produces:

1. **Visualization table** — pipe-separated, monospace-rendered in the web UI:

```
| INPUT    | MAP                       | REDUCE                              |
|----------|---------------------------|-------------------------------------|
| d1 → p1  | [hash(word): (word, 1)]   | p1: [w1:[1,1,1],  w2:[1,1]]        |
| d2 → p2  | [hash(word): (word, 1)]   | p2: [w3:[1,1],    w4:[1,1,1]]      |
| ...      |                           |                                     |
| dn → pn  | [hash(word): (word, 1)]   | pn: [...]                           |
```

2. **Chain description** — numbered steps explaining shuffle and data flow.
3. **Pseudocode** — `map() / reduce()` with `send(key, val) to P(hash(key))` notation.

---

## Project Structure

```
├── main.py                          # Terminal entry point
├── orchestrator.py                  # LLM router + session manager
├── llm_factory.py                   # Provider selector (OpenAI / Anthropic / Ollama / llama.cpp)
├── config.json                      # Default provider, model, server settings
├── requirements.txt
│
├── agents/
│   ├── serializability_agent.py     # SERIAL domain
│   ├── parallel_query_agent.py      # QUERY domain
│   ├── pipeline_agent.py            # JOIN domain — deterministic plan-execute-format
│   ├── join_cost_agent.py           # Join cost tools (used internally)
│   └── mapreduce_agent.py           # MAPREDUCE domain
│
├── tools/
│   ├── db_ops.py                    # All cost functions — select, sort, join, compose
│   ├── gemini_view.py               # View-serializability colored report
│   └── confl_ser.py                 # Conflict-serializability precedence graph
│
├── api/
│   └── app.py                       # FastAPI web server + single-page chat UI
│
└── evals/
    └── test_cases.json              # 14 labelled test cases with expected answers
```

---

## Installation

```bash
pip install -r requirements.txt
```

Set your API key (depending on provider):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...   # if using Anthropic
```

---

## Running

### Terminal (interactive)

```bash
python main.py
```

On startup you select the LLM provider and model interactively. Type `model` at any time to switch mid-session.

### Web UI

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000`.

The web UI includes:
- Sidebar with example queries for all four domains
- Live model switcher (provider + model without restart)
- Language toggle (RU / EN)
- Chat reset button
- Domain badge on every agent response (`SERIAL` / `QUERY` / `JOIN` / `MAPREDUCE`)

---

## Supported LLM Providers

| Provider | How to select | Notes |
|----------|---------------|-------|
| OpenAI | `openai` | gpt-4o recommended |
| Anthropic | `anthropic` | claude-sonnet-4-5 and above |
| Ollama | `ollama` | local server at `127.0.0.1:11434` |
| llama.cpp | `local` | `/completion` endpoint; host/port in `config.json` |

Provider and model are set in `config.json` (`default_provider`, `default_model`) or chosen interactively at startup.

---

## Example Queries

**Serializability**
```
Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?
Check view-serializability: r2(B) w2(A) r1(A) r3(A) w1(B) w2(B) w3(B)
```

**Parallel query cost**
```
Table Flights(fid, date, from, to, seats), 10,000 blocks, 10 processors,
hash(fid) distribution. Find: σ_{fid = 777}(Flights).
```

**Join cost**
```
Flowers(name, petal, size, color) — 10^4 blocks, hash(name).
Sales(name, cname, amount, price) — 10^6 blocks, hash(name).
10 processors. Perform Flowers join Sales.
```

**Map-Reduce**
```
Count how many times each word appears across documents d1..dn using n servers.
Input: documents. Output: (word, count).
```

---

## Key Design Decisions

- **LLM never computes numbers** — every cost formula is produced by Python tools in `db_ops.py`. The LLM only picks algorithms and formats output.
- **Symbolic output** — results like `(10^3 + 10^5) * (t_s + t_d)` are never collapsed to a single number.
- **PipelineAgent separation** — for compound queries (Select + Join, Join + Join) the plan is extracted by one LLM call, then Python executes it deterministically. This prevents the LLM from skipping steps.
- **10^k notation** — block counts and per-server sizes are formatted as `10^3`, `10^5` etc. throughout.
- **Monospace rendering** — agent responses in the web UI are wrapped in `<pre>` with JetBrains Mono so tables and formulas always align correctly.

---

## Adding a New Agent

1. Create `agents/my_agent.py` — implement a class or `build_agent()` function.
2. In `orchestrator.py`:
   - Import it.
   - Add a new label (e.g. `MYTOPIC`) to `ROUTER_PROMPT` with 2–3 few-shot examples.
   - Add the label to `_VALID`.
   - Add a branch in `Orchestrator.handle()`.
3. Add a CSS badge class `domain-MYTOPIC` in `api/app.py` if using the web UI.

---

## Language

All agents respond in **Russian** by default. To switch to English, call the `/language` endpoint:

```bash
curl -X POST http://localhost:8000/language -H "Content-Type: application/json" -d '{"lang":"en"}'
```

Or use the RU / EN toggle in the web UI header.

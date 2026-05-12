# DB Assistant Framework

A multi-agent system for solving advanced database problems through a conversational web UI.

The orchestrator classifies every user message and routes it to the right specialized agent. All arithmetic is computed by deterministic Python tools — the LLM only reasons, plans, and formats.

---

## Quick Start (Docker)

### 1. Clone and build

```bash
git clone https://github.com/cyttic/agentADB.git
cd agentADB
docker build -t agentadb .
```

### 2. Run with your OpenAI key

```bash
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 agentadb
```

Open **http://localhost:8000** in your browser.

> The default model is `gpt-4o` (set in `config.json`). You can switch provider and model from the UI at any time without restarting.

---

## Other Providers

Pass the corresponding key as an environment variable:

```bash
# Anthropic
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8000:8000 agentadb

# Multiple keys at once (if you want to switch providers from the UI)
docker run \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -p 8000:8000 agentadb
```

For Ollama or a local llama.cpp server, use `--network host` so the container can reach your machine:

```bash
docker run --network host agentadb
```

---

## Running Without Docker

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Agents

| Domain | Agent | What it solves |
|--------|-------|----------------|
| `SERIAL` | SerializabilityAgent | View-serializability and conflict-serializability of transaction schedules |
| `QUERY` | ParallelQueryAgent | Parallel cost of Select and Sort operations (alg2 / alg3, symbolic output) |
| `JOIN` | JoinCostAgent | Parallel Join cost — broadcast join and local join; chains Select → Join → Join |
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
    ├── JOIN       ──▶ JoinCostAgent          (LangGraph + tools)
    └── MAPREDUCE  ──▶ MapReduceAgent         (LLM reasoning)
```

---

## Project Structure

```
├── main.py                          # Terminal entry point
├── orchestrator.py                  # LLM router + session manager
├── llm_factory.py                   # Provider selector (OpenAI / Anthropic / Ollama / llama.cpp)
├── config.json                      # Default provider, model, server settings
├── version.py                       # Version string (single source of truth)
├── requirements.txt
│
├── agents/
│   ├── serializability_agent.py     # SERIAL domain
│   ├── parallel_query_agent.py      # QUERY domain
│   ├── join_cost_agent.py           # JOIN domain
│   └── mapreduce_agent.py           # MAPREDUCE domain
│
├── tools/
│   ├── db_ops.py                    # All cost functions — select, sort, join, compose
│   ├── gemini_view.py               # View-serializability colored report
│   └── confl_ser.py                 # Conflict-serializability precedence graph
│
└── api/
    └── app.py                       # FastAPI web server + single-page chat UI
```

---

## Supported LLM Providers

| Provider | `default_provider` value | Env variable |
|----------|--------------------------|--------------|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama` | — (no key needed) |
| llama.cpp | `local` | — (no key needed) |

To change the default provider or model, edit `config.json` before building the image, or switch from the model badge in the UI header at runtime.

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
hash(fid) distribution. Find: σ(fid = 777)(Flights).
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
```

---

## Language

Agents respond in **Russian** by default. Switch to English from the UI header toggle or via API:

```bash
curl -X POST http://localhost:8000/language \
  -H "Content-Type: application/json" \
  -d '{"lang":"en"}'
```

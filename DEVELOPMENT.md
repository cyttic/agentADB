# DB Assistant Framework — Full Development Process Documentation

**Project:** DB Assistant Framework  
**Repository:** https://github.com/cyttic/agentADB  
**Period:** 2026-04-30 — 2026-05-13  
**Current Version:** 0.0.10  

---

## 1. Project Overview

DB Assistant Framework is a multi-agent AI system designed to assist students of the *Advanced Databases* course. It accepts natural language queries and routes them to specialized agents that solve four categories of database tasks:

| Domain | Agent | Task |
|--------|-------|------|
| `SERIAL` | SerializabilityAgent | View-serializability and conflict-serializability of transaction schedules |
| `QUERY` | ParallelQueryAgent | Parallel cost of Select and Sort operations (symbolic output) |
| `JOIN` | JoinCostAgent | Parallel Join cost — regular join and hash join |
| `MAPREDUCE` | MapReduceAgent | Map-Reduce algorithm design — table, chain, pseudocode |

**Core design principle:** The LLM is never trusted to do arithmetic. Every cost formula is computed by deterministic Python tools. The LLM only reasons, routes, and formats.

---

## 2. Technology Stack

| Layer | Technology |
|-------|------------|
| Agent framework | LangGraph (StateGraph) |
| LLM integration | LangChain |
| LLM providers | OpenAI (GPT-4o, GPT-5.4-mini, GPT-5.4-nano), Anthropic, Ollama, llama.cpp |
| Web server | FastAPI + Uvicorn |
| Frontend | Single-page HTML/CSS/JS (embedded in `api/app.py`) |
| Containerization | Docker |
| Version control | Git / GitHub |
| CI/CD | GitHub Actions |

---

## 3. Development Timeline

### Phase 1 — Foundation (2026-04-30 to 2026-05-03)

**First commit: 2026-04-30**

The project started with two deterministic Python tools for schedule analysis:

- `tools/confl_ser.py` — conflict-serializability via precedence graph and DFS cycle detection
- `tools/gemini_view.py` — view-serializability via n! permutation enumeration + three view conditions (initial reads, reads-from, final writes)

Both tools produce ANSI-colored terminal output showing step-by-step analysis, conflict pairs, graph edges, cycle detection, and verdict.

**2026-05-03** — First LangGraph agent (`ba6b175`):
- `SerializabilityAgent` built using LangGraph StateGraph
- Tools wrapped as `@tool` for LLM use
- `Orchestrator` class created to route messages to the correct agent

**2026-05-03** — `agents/parallel_query_agent.py` created:
- Schema extraction from natural language
- Select cost computation (alg2/alg3 decision based on distribution)
- Sort cost computation (alg1/alg2 based on partition field)

---

### Phase 2 — Query Agent & Evaluation (2026-05-04 to 2026-05-06)

**2026-05-04 to 2026-05-05:**
- Evaluation framework added (`evals/run_evals.py`, `evals/test_cases.json`) with 10+ labeled test cases
- Multi-model selection support — runtime switching between providers
- Sort algorithm fully implemented (local sort, gather, merge pipeline)
- Logic fixes for Select cost calculation

**2026-05-06 — Web Application:**
- FastAPI server created (`api/app.py`)
- Single-page chat UI with sidebar examples, model switcher, reset button
- CI/CD pipeline via GitHub Actions (`deploy.yml`)
- Docker containerization (`Dockerfile`)

---

### Phase 3 — Join Agent & Compound Queries (2026-05-08 to 2026-05-10)

**2026-05-08:**
- `JoinCostAgent` added — deterministic pipeline for join cost analysis
- `PipelineAgent` class: three-phase execution (PLAN → EXECUTE → FORMAT)
  - Phase 1: single LLM call extracts structured JSON plan
  - Phase 2: Python iterates the plan and calls db_ops tools
  - Phase 3: single LLM call formats the result
- Support for compound queries: `σ(cond)(A) ⋈ B`, `(A ⋈ B) ⋈ C`

**2026-05-09:**
- `MapReduceAgent` added — generates visualization table, chain description, and pseudocode
- Join case `A ⋈ B ⋈ C` working correctly
- RA output formatting fixed (Unicode symbols: σ, π, ⋈)

**2026-05-10:**
- Day/Night theme toggle added to web UI
- All HW1 tasks integrated and passing evaluation
- Interface language standardized to English

---

### Phase 4 — Versioning & Bug Fixes (2026-05-12)

All work in this phase was tracked with semantic versioning introduced at `v0.0.1`.

#### v0.0.1 — Versioning Infrastructure
- `version.py` created as single source of truth: `VERSION = "0.0.1"`
- FastAPI app version tied to `VERSION`
- Web UI header updated: "DB Assistant Framework [x.x.x]" — version injected at serve time via `.replace("__VERSION__", VERSION)` to avoid f-string conflicts with CSS braces
- Version badge styled in monospace, muted color

#### v0.0.2 — Serial Agent Output Fix
- `gemini_view.py` (`print_report`): added `CONFLICT ANALYSIS:` section to view-serializability output, matching the format already present in conflict-serializability output
- Each conflicting operation pair is shown with the resulting precedence graph edge

#### v0.0.3 — Permutation Stage for View-Serializability
- `gemini_view.py` (`analyze`): full n! permutation enumeration is now **always** performed, regardless of whether the answer can be determined by graph analysis alone
- New `PERMUTATION CHECK:` section in every view-serializability report:
  - Shows `n! = X` permutations enumerated
  - Result: FOUND / NOT FOUND
  - If found: prints the equivalent serial schedule

#### v0.0.4 — Block Count Calculation for Query Agent
- `db_ops.py`: added `compute_table_blocks_info()` — calculates block count from record count + cell size + block size
  - `row_size = num_attributes × cell_size`
  - `table_size = row_size × record_count`
  - `block_count = ceil(table_size / block_size)`
- `@tool compute_table_blocks` added to both `JoinCostAgent` and `ParallelQueryAgent`
- `extract_schema_from_text` regex improved to handle Russian cell-size patterns:
  - "каждая ячейка данных весит X"
  - "размер 1 блока X"
- System prompts updated: "If block count not given directly, call `compute_table_blocks` first"

#### v0.0.5 — README Rewrite
- README rewritten to focus on Docker quick-start
- Primary instructions: `docker build` + `docker run -e OPENAI_API_KEY=sk-...`
- Secondary: provider table (OpenAI / Anthropic / Ollama / llama.cpp)

---

#### v0.0.6 — Broadcast Join Formula Corrected (Critical Bug Fix)

**Bug:** All three steps of broadcast join used per-server values `(bs_R + bs_S)`.  
**Correct formula** (user-confirmed):
- Step 1 `[send]`: `(B_R + B_S) × (t_s + t_d)`
- Step 2 `[receive]`: `(B_R + B_S) × (t_s + t_d)`
- Step 3 `[join]`: `(B_R + B_S) × 3 × t_d`
- Elapsed = Step1 + Step2 + Step3
- Total = p × Elapsed

All three steps use **total** block counts, not per-server counts.

#### v0.0.7 — Parallel Join Placeholder Replaced (Critical Bug Fix)

**Bug:** `ParallelQueryAgent` was using a placeholder `join_cost(blocks_s, blocks_t)` that produced `3*(B_s+B_t)*t_d` with no `t_s` and Elapsed = Total (incorrect).

**Fix:** Replaced with a proper wrapper around `parallel_join_cost` that:
- Accepts distributions and join field
- Produces all 3 correct steps with `t_s`
- System prompt updated to require showing all 3 steps for every Join
- Output format block updated with Join-specific template

---

#### v0.0.8 — Regular Join Formula (Major Algorithm Correction)

User clarified that the join algorithm taught in the course is **asymmetric** — not the symmetric broadcast:

**Regular Join (A ⋈ B where A is outer/broadcast):**
- Step 1 `[send]`:    `bs_out × t_d + (p-1) × bs_out × t_s`
- Step 2 `[receive]`: `(p-1) × bs_out × (t_s + t_d)`
- Step 3 `[join]`:    `3 × (B_out + bs_in) × t_d`
- Elapsed = Step1 + Step2 + Step3
- Total = p × Elapsed

Where:
- `outer` = **smaller** table (always broadcast — cheaper)
- `inner` = larger table (stays local)
- `bs_out = ceil(B_out / p)` — outer blocks per server
- `bs_in = ceil(B_in / p)` — inner blocks per server
- `B_out` = **full** outer table (every server receives all of it after steps 1-2)

**Key property:** `S ⋈ F ≠ F ⋈ S` in cost. Tool always picks the cheaper ordering automatically by comparing table sizes.

**Parallel (Hash) Join** — only applicable when both tables are partitioned by the **same** method (hash or range) on exactly the join field. No communication needed; Elapsed = `3 × (bs_R + bs_S) × t_d`.

---

#### v0.0.9 — Human-in-the-Loop Multi-Agent RA Selection

**New feature:** Before computing query costs, the system now generates Relational Algebra proposals from multiple independent LLM agents and asks the user to choose.

**New file:** `agents/ra_proposal_agent.py`

**Flow:**
1. User sends a QUERY message
2. Orchestrator spawns **gpt-5.4-nano** and **gpt-5.4-mini** in parallel (via `ThreadPoolExecutor`)
3. Each model generates an optimized RA expression
4. User receives selection menu:
   ```
   [1] RA from gpt-5.4-nano:  π(cid)(Orders ⋈ σ(cost > 100)(Products))
   [2] RA from gpt-5.4-mini:  π(cid)(σ(cost > 100)(Products) ⋈ Orders)
   Enter 1 or 2 to select.
   ```
5. User replies `1` or `2`
6. Selected RA is injected into the query agent message (`[SELECTED RA]: ...`)
7. Query agent proceeds directly to schema extraction and cost computation

**RA Optimization rules** instructed to the proposal agents:
1. **Minimum tables** — only include tables strictly needed for the output; exclude tables whose data is available via joins on smaller tables
2. **Push selections down** — apply all σ before any ⋈
3. **Minimum joins** — never join tables not needed to satisfy the query
4. **Push projections down** — apply π early

**State management in Orchestrator:**
- `_pending_ra: dict | None` — holds query and proposals while waiting for user selection
- When pending: routing LLM call skipped, domain returned as "QUERY"
- Invalid input (not "1"/"2"): pending state cleared, message re-routed normally
- Missing `OPENAI_API_KEY`: silently falls back to running the query agent directly

---

#### v0.0.10 — Block Count Propagation & Step 3 Formula Enforcement

**Two bugs fixed in the same session:**

**Bug 1 — Block count not propagated to cost tools:**  
Agent correctly called `compute_table_blocks` (showing 20,000 blocks for Orders, 1,500,000 for Products) but then passed the original record counts (1,000,000 and 100,000,000) to `select_cost` and `join_cost`.

Fix:
- System prompt: added `!! HARD RULE` block with concrete wrong/right example:
  ```
  compute_table_blocks(...) → block_count=20000
  select_cost(block_count=20000, ...)   ← CORRECT
  select_cost(block_count=1000000, ...) ← WRONG (this is record_count)
  ```
- `compute_table_blocks_info` tool output: last line now reads `!! USE block_count=N IN ALL SUBSEQUENT TOOL CALLS. DO NOT use record_count=M.`

**Bug 2 — Step 3 join formula: B_in total vs B_in/p:**  
Agent was computing `3×(B_out + B_in)×t_d` (both full table sizes) instead of `3×(B_out + bs_in)×t_d` where `bs_in = B_in/p`.

Fix (tool code was already correct; prompts were unclear):
- System prompt (both agents): annotated step 3 with:
  ```
  B_out = FULL outer table (received in full after step 2)
  bs_in = B_in/p — inner PER SERVER, NOT the full B_in
  !! WRONG: 3*(B_out + B_in)*t_d
  ```
- Added concrete example showing correct `3*(10^4 + 10^5)*t_d` vs wrong `3*(10^4 + 10^6)*t_d` for F=10^4, S=10^6, p=10
- Tool output: step 3 explanation now prints `NOTE: step3 uses B_out=X (full outer) + bs_in=Y (inner/p), NOT B_in=Z (inner total)`

---

## 4. Architecture

```
User input
    │
    ▼
Orchestrator
  ├── _pending_ra check (human-in-the-loop RA selection state)
  │     └── if set + input is "1"/"2" → apply_ra_selection → run query agent
  │
  └── _route() — single LLM call classifies: SERIAL / QUERY / JOIN / MAPREDUCE
        │
        ├── SERIAL     → SerializabilityAgent   (LangGraph)
        │     tools: parse_schedule_from_text
        │             check_view_serializability
        │             check_conflict_serializability
        │
        ├── QUERY      → _propose_ra_and_wait()
        │     ├── ra_proposal_agent: gpt-5.4-nano + gpt-5.4-mini in parallel
        │     └── ParallelQueryAgent (LangGraph, after RA selected)
        │           tools: compute_table_blocks
        │                  extract_schema_from_text
        │                  parse_schema
        │                  decide_select_algorithm
        │                  select_cost
        │                  decide_sort_algorithm
        │                  sort_cost
        │                  join_cost
        │                  compose_costs
        │
        ├── JOIN       → JoinCostAgent  (LangGraph)
        │     tools: compute_table_blocks
        │             compute_parallel_join
        │             compute_select_cost
        │             sum_operation_costs
        │
        └── MAPREDUCE  → MapReduceAgent (stateful LLM)
```

---

## 5. Key Algorithmic Implementations

### 5.1 Conflict-Serializability (`tools/confl_ser.py`)
1. Build precedence graph: for every pair of operations from different transactions on the same object where at least one is a write, add edge T_i → T_j
2. DFS cycle detection
3. If acyclic: topological sort gives equivalent serial order
4. Output: CONFLICT-SERIALIZABLE / NOT CONFLICT-SERIALIZABLE

### 5.2 View-Serializability (`tools/gemini_view.py`)
1. Compute view signature: initial reads, reads-from, final writes
2. Build precedence graph; check for cycle
3. If acyclic → VIEW-SERIALIZABLE (conflict-serial implies view-serial)
4. If cycle + no blind writes → NOT VIEW-SERIALIZABLE
5. If cycle + blind writes → enumerate all n! permutations; compare view signatures
6. **Always** show PERMUTATION CHECK section (even in cases 3 and 4)

### 5.3 Block Count Calculation (`tools/db_ops.py`)
```
row_size_bytes   = num_attributes × cell_size_bytes
table_size_bytes = record_count × row_size_bytes
block_count      = ceil(table_size_bytes / block_size_bytes)
```

### 5.4 Select Cost
- **alg2:** all p processors scan in parallel → `Elapsed = ceil(B/p) × t_d`, `Total = p × Elapsed`
- **alg3:** only relevant processors → depends on partition scheme and condition type

Decision table:
| Distribution | Condition | Algorithm |
|---|---|---|
| round_robin | any | alg2 |
| hash(F) + question on F | point (=) | alg3 (1 proc) |
| hash(F) + question on F | range/scan | alg2 |
| range(F) + question on F | point/range/scan | alg3 |
| question on field ≠ partition field | any | alg2 |

### 5.5 Regular Join Cost
For A ⋈ B where A = outer (smaller, broadcast), B = inner (larger, local):
```
bs_out = ceil(B_out / p)
bs_in  = ceil(B_in  / p)

Step 1 [send]:    bs_out × t_d + (p-1) × bs_out × t_s
Step 2 [receive]: (p-1) × bs_out × (t_s + t_d)
Step 3 [join]:    3 × (B_out + bs_in) × t_d
                       ↑ full      ↑ per-server only

Elapsed = Step1 + Step2 + Step3
Total   = p × Elapsed
```

### 5.6 Parallel (Hash) Join
Only when both tables are partitioned by the same method on exactly the join field:
```
Elapsed = 3 × (bs_R + bs_S) × t_d
Total   = p × Elapsed
```

---

## 6. File Structure

```
agentADB/
├── version.py                       # Single source of truth for version string
├── main.py                          # Terminal entry point
├── orchestrator.py                  # Router + session state + RA selection flow
├── llm_factory.py                   # Provider abstraction (OpenAI / Anthropic / Ollama / local)
├── config.json                      # Default provider, model, server settings
├── requirements.txt
├── Dockerfile
│
├── agents/
│   ├── serializability_agent.py     # SERIAL — LangGraph + schedule tools
│   ├── parallel_query_agent.py      # QUERY — LangGraph + cost tools
│   ├── join_cost_agent.py           # JOIN — LangGraph + join cost tools
│   ├── mapreduce_agent.py           # MAPREDUCE — stateful LLM agent
│   ├── ra_proposal_agent.py         # Human-in-the-loop RA generation (2 models in parallel)
│   └── pipeline_agent.py            # Deterministic plan-execute-format pipeline
│
├── tools/
│   ├── db_ops.py                    # All cost functions (pure Python, no LLM)
│   ├── gemini_view.py               # View-serializability report with permutation check
│   └── confl_ser.py                 # Conflict-serializability precedence graph
│
└── api/
    └── app.py                       # FastAPI server + embedded single-page chat UI
```

---

## 7. Versioning Policy

Semantic versioning: `MAJOR.MINOR.PATCH`

- **PATCH** increments by 1 on every commit (0.0.1, 0.0.2, ..., 0.0.10, 0.0.11, ...)
- **MINOR** bump to 0.2.0 will be applied when explicitly requested
- **MAJOR** reserved for full rewrites

Version is defined in `version.py` only. All other places (`api/app.py` FastAPI metadata, HTML UI header) read it from there. The HTML header shows the badge `[x.x.x]` injected at serve time via string replacement.

---

## 8. Running the Project

### Docker (recommended)
```bash
git clone https://github.com/cyttic/agentADB.git
cd agentADB
docker build -t agentadb .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 agentadb
```
Open http://localhost:8000

### Local
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 9. Known Issues and Decisions

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| Agent used record counts instead of block counts in cost tools | LLM ignored tool return values | Added `!! HARD RULE` in system prompt + explicit warning in tool output |
| Step 3 used B_in (total) instead of B_in/p (per-server) | System prompt formula was ambiguous | Added inline annotation + concrete wrong/right example |
| Old broadcast join formula was symmetric | Algorithm was described incorrectly | Rewrote to asymmetric formula; tool picks cheaper ordering automatically |
| Placeholder join_cost used in ParallelQueryAgent | Tool was never replaced after initial scaffold | Replaced with full parallel_join_cost wrapper |
| Version jumped from 0.0.9 to 0.1.0 | Incorrect assumption about patch numbering | Corrected: patch increments indefinitely (0.0.10, 0.0.11, …) |

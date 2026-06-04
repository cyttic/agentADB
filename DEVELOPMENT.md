# DB Assistant Framework — Full Development Process Documentation

**Project:** DB Assistant Framework  
**Repository:** https://github.com/cyttic/agentADB  
**Period:** 2026-04-30 — 2026-06-01  
**Current Version:** 3.0.1  

---

## 1. Project Overview

DB Assistant Framework is a multi-agent AI system designed to assist students of the *Advanced Databases* course. It accepts natural language queries and routes them to specialized agents that solve the course's task categories:

| Domain | Agent | Task |
|--------|-------|------|
| `SERIAL` | SerializabilityAgent | View-serializability and conflict-serializability of transaction schedules |
| `QUERY` | ParallelQueryAgent | Parallel cost of Select and Sort operations (symbolic output) |
| `JOIN` | JoinCostAgent | Parallel Join cost — regular (asymmetric) join and hash join |
| `SEMIJOIN` | SemiJoinAgent | Parallel Semi-Join (⋉) cost, with an interactive diagram editor |
| `MAPREDUCE` | MapReduceAgent | Map-Reduce algorithm design — table, chain, pseudocode |
| `DATACUBE` | DataCubeAgent | OLAP data-cube materialisation via Ullman's greedy algorithm |
| `FULLREDUCER` | FullReducerAgent | Acyclic join Full Reducer — GYO check + two-phase semi-join program |
| `APRIORI` | AprioriAgent | Frequent itemsets via the Apriori algorithm |
| `APRIORITID` | AprioriTidAgent | Frequent itemsets via the Apriori-TID (tid-list / vertical) method |
| `RULES` | AssociationRulesAgent | All association rules with confidence ≥ C |
| `MAXIMAL` | MaximalItemsetsAgent | Maximal frequent itemsets (no frequent proper superset) |
| `CLOSED` | ClosedItemsetsAgent | Closed frequent itemsets (every proper superset has smaller support) |

**Core design principle:** The LLM is never trusted to do arithmetic or run an algorithm. Every cost formula and every mining step is computed by a deterministic Python tool. The LLM only reasons, routes, parses the problem into structured arguments, and (where appropriate) formats. For the deterministic data-mining agents, the orchestrator even returns the **raw tool output** rather than the model's paraphrase, so the full step-by-step trace is always shown verbatim (see §4 and §9).

---

## 2. Technology Stack

| Layer | Technology |
|-------|------------|
| Agent framework | LangGraph (StateGraph) |
| LLM integration | LangChain |
| LLM providers | OpenAI (GPT-4o, GPT-5.4-mini, GPT-5.4-nano), Anthropic, Ollama, llama.cpp |
| Web server | FastAPI + Uvicorn |
| Frontend | Single-page HTML/CSS/JS (embedded in `api/app.py`) |
| Interactive editors | Browser-based HTML5 canvas editors served over a local HTTP server (Semi-Join & Hasse diagrams); SVG schema viewer (Full Reducer) |
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

### Phase 4 — Versioning & Join Bug Fixes (2026-05-12 to 2026-05-13)

Semantic versioning was introduced at `v0.0.1` and the v0.0.x line tracked a sequence of join-cost corrections. Highlights:

- **v0.0.1** — `version.py` as single source of truth; FastAPI + web-UI version badge injected at serve time via `.replace("__VERSION__", VERSION)`
- **v0.0.2 / v0.0.3** — view-serializability report gained a `CONFLICT ANALYSIS:` section and an always-on `PERMUTATION CHECK:` section (full n! enumeration even when graph analysis already decides)
- **v0.0.4** — `db_ops.compute_table_blocks_info()`: block count from record count × cell size ÷ block size; Russian cell-size regex patterns
- **v0.0.6** — broadcast-join formula corrected to use **total** block counts in all three steps
- **v0.0.7** — `ParallelQueryAgent` placeholder `join_cost` replaced with a real `parallel_join_cost` wrapper (proper `t_s`, 3 steps)
- **v0.0.8** — regular join is **asymmetric** (outer = smaller, broadcast; inner = larger, local); tool auto-picks the cheaper ordering. Hash join only when both tables co-partitioned on the join field
- **v0.0.9 / v0.0.10** — enforce `block_count` propagation to cost tools (`!! HARD RULE` in prompts + warning lines in tool output); clarify step 3 uses `B_out` (full) + `bs_in` (per-server)
- **v0.0.11 (2026-05-13)** — removed `gpt-5.5` from the model selector; added `DEVELOPMENT.md`, `AI_INTEGRATION_REPORT.md`, `COMPARATIVE_REPORT.md`

Also in this phase: human-in-the-loop **RA selection** (`agents/ra_proposal_agent.py`) — before computing query costs, two models (`gpt-5.4-nano` + `gpt-5.4-mini`) generate optimized Relational Algebra proposals in parallel (`ThreadPoolExecutor`), the user picks one, and the chosen RA is injected into the query agent so it skips re-derivation. Optimization rules: minimum tables, push selections down, minimum joins, push projections down. Orchestrator holds the `_pending_ra` state while waiting for the `1`/`2` reply.

---

### Phase 5 — Semi-Join & the v0.2 Line (2026-05-14)

Versioning jumped to the **v0.2 line** (`start v0.2 versioning`) to mark the move into HW2's advanced parallel operations.

- **`SemiJoinAgent`** (`agents/semijoin_agent.py`, domain `SEMIJOIN`) — parallel Semi-Join (R ⋉ S) cost analysis.
- **`tools/mermaid_editor.py`** — an interactive browser diagram editor: the user draws the relations/flow, names each relation on ellipse creation, and the schema is extracted directly from the diagram (Point tool added in v0.2.1; table-name prompt + schema extraction in v0.2.2/v0.2.3).

---

### Phase 6 — Data Cube Materialisation (2026-05-17)

- **`DataCubeAgent`** (`agents/datacube_agent.py`, domain `DATACUBE`) + **`tools/cube_ops.py`** — Ullman's greedy approximation algorithm for selecting N cubes to materialise in an OLAP lattice.
- **`tools/datacube_editor.py`** — a browser Hasse-diagram editor for problems that don't supply the lattice graph; the user places cube nodes with access costs and draws parent→child edges, and the graph is extracted for the algorithm.
- Output refinements: flat tables replaced with **2-D path/benefit matrices** (v0.2.7); benefit-matrix rows sorted alphabetically for stable, readable output (v0.2.8).

---

### Phase 7 — Full Reducer (2026-05-17, v0.2.10 onward)

- **`FullReducerAgent`** (`agents/full_reducer_agent.py`, domain `FULLREDUCER`) + **`tools/full_reducer.py`** — the semi-join-based Full Reducer for acyclic natural joins.
  - Builds the intersection (hyper)graph of shared attributes.
  - Runs **GYO ear-elimination** to test acyclicity.
  - If acyclic: derives the join tree and emits two-phase pseudocode (bottom-up `parent ⋉ child`, then top-down `child ⋉ parent`), with the natural join computed last on the reduced relations.
- Viewer evolution: `webbrowser` → `xdg-open` for Linux desktop (v0.2.12), then a local HTTP server instead of `file://` so it opens in the existing browser tab (v0.2.13), then an **SVG Venn schema viewer** replacing Mermaid (v0.2.15).
- Correctness/clarity: only join-participating tables are passed to the tool, not all DB tables (v0.2.16); binary ear/not-ear classification (v0.2.17); larger ellipses + recursive pseudocode and GYO all-at-once removal (v0.2.19).

(v0.2.21–v0.2.34, through 2026-05-28, were incremental fixes and housekeeping, including `.gitignore` for `.docs`.)

---

### Phase 8 — Data Mining Suite / HW3 (2026-05-30 → 2026-06-01)

The complete association-mining task family was added (commits `solution 3a` … `task 3e`, v0.2.36–v0.2.40), all operating on the **same transaction-table input** (`{TID: [items]}`) and all sharing one deterministic core. Version was then bumped to **3.0.0** to mark the completed suite, and **3.0.1** for a follow-up fix.

| Task | Agent / tool | Method |
|------|--------------|--------|
| Apriori | `apriori_agent.py` / `tools/apriori_ops.py` (`run_apriori`) | Level-wise passes; `Ck` from `F(k-1) ⋈ F(k-1)` union-of-pairs of size *k*; support by scanning transactions; stop when `Fk = ∅` |
| Apriori-TID | `apriori_tid_agent.py` / `tools/apriori_tid_ops.py` (`run_apriori_tid`) | Per-itemset **tid_list**; `Support(I) = \|I.tid_list\| / \|D\|`; a *k*-itemset's tid_list is the **intersection** of its two parents' tid_lists; prints a compact `Apriori-TID(...)` Condition line up front |
| Association rules | `association_rules_agent.py` / `tools/association_rules_ops.py` (`run_association_rules`) | For each frequent `Z` with `\|Z\| ≥ 2`, every split `I → J=Z\I`; `conf = support(Z)/support(I)`; keep `conf ≥ C` |
| Maximal frequent | `maximal_itemsets_agent.py` / `tools/maximal_itemsets_ops.py` (`run_maximal_itemsets`) | A frequent `I` is maximal iff no immediate superset `I ∪ {x}` is frequent |
| Closed frequent | `closed_itemsets_agent.py` / `tools/closed_itemsets_ops.py` (`run_closed_itemsets`) | A frequent `I` is closed iff no immediate superset has the **same** support (every proper superset strictly smaller) |

**Shared core (DRY):** `tools/apriori_ops.py` exposes the reusable helpers `_parse_transactions`, `_generate_candidates`, `_support_count`, `_fmt_ratio`, `_itemset_str`, `_set_of_sets_str`. `tools/association_rules_ops.py` adds `_frequent_itemsets(transactions, N, S) → (freq, levels)`, reused by the rules, maximal, and closed tools. Candidate generation is the single shared `F(k-1) ⋈ F(k-1)` union-of-pairs join.

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
  └── _route() — single LLM call → _extract_domain() keyword scan
        │   (compacted-form guard so "APRIORI-TID" / "APRIORI TID" → APRIORITID,
        │    not the bare APRIORI token)
        │
        ├── SERIAL       → SerializabilityAgent   (LangGraph)
        ├── QUERY        → _propose_ra_and_wait() → ParallelQueryAgent (LangGraph)
        ├── JOIN         → JoinCostAgent / PipelineAgent
        ├── SEMIJOIN     → SemiJoinAgent           (LangGraph + diagram editor)
        ├── DATACUBE     → DataCubeAgent           (LangGraph + Hasse editor)
        ├── FULLREDUCER  → FullReducerAgent        (LangGraph + GYO + SVG viewer)
        ├── MAPREDUCE    → MapReduceAgent          (stateful LLM)
        │
        │   ── data-mining family (deterministic; raw tool output returned) ──
        ├── APRIORI      → AprioriAgent
        ├── APRIORITID   → AprioriTidAgent
        ├── RULES        → AssociationRulesAgent
        ├── MAXIMAL      → MaximalItemsetsAgent
        └── CLOSED       → ClosedItemsetsAgent
```

**Router (`_extract_domain`).** The classifier prompt is written for weak/local models (few-shot examples + a single-word answer). The extractor scans the model output for the first valid keyword rather than requiring exact equality, and applies a compacted-form guard so hyphen/space variants of `APRIORI-TID` route correctly.

**`_solution_from(messages)`.** The deterministic data-mining routes (`APRIORI`, `APRIORITID`, `RULES`, `MAXIMAL`, `CLOSED`) do **not** return the LLM's final message. Instead the orchestrator extracts the `ToolMessage` content — the complete step-by-step trace produced by the Python tool — so every line (including the *unsuccessful*/discarded steps, e.g. rules below the confidence threshold) is shown. It falls back to the `AIMessage` only when no tool was called (e.g. a clarifying question). This makes output independent of model quality.

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
3. If acyclic → VIEW-SERIALIZABLE
4. If cycle + no blind writes → NOT VIEW-SERIALIZABLE
5. If cycle + blind writes → enumerate all n! permutations; compare view signatures
6. **Always** show the PERMUTATION CHECK section

### 5.3 Block Count Calculation (`tools/db_ops.py`)
```
row_size_bytes   = num_attributes × cell_size_bytes
table_size_bytes = record_count × row_size_bytes
block_count      = ceil(table_size_bytes / block_size_bytes)
```

### 5.4 Select Cost
- **alg2:** all p processors scan in parallel → `Elapsed = ceil(B/p) × t_d`, `Total = p × Elapsed`
- **alg3:** only relevant processors → depends on partition scheme and condition type

### 5.5 Regular (Asymmetric) Join Cost
For A ⋈ B where A = outer (smaller, broadcast), B = inner (larger, local):
```
bs_out = ceil(B_out / p);  bs_in = ceil(B_in / p)

Step 1 [send]:    bs_out × t_d + (p-1) × bs_out × t_s
Step 2 [receive]: (p-1) × bs_out × (t_s + t_d)
Step 3 [join]:    3 × (B_out + bs_in) × t_d    (B_out full, bs_in per-server)
Elapsed = Step1 + Step2 + Step3;  Total = p × Elapsed
```
Hash join (both tables co-partitioned on the join field): `Elapsed = 3 × (bs_R + bs_S) × t_d`.

### 5.6 Data Cube — Ullman's Greedy (`tools/cube_ops.py`)
Start with the top cube materialised. Repeatedly add the cube with maximum **benefit** = total query-cost reduction across all cubes, until N are chosen. Tie-breaks alphabetical (deterministic). Output uses 2-D path/benefit matrices.

### 5.7 Full Reducer — GYO (`tools/full_reducer.py`)
Ear-elimination on the join hypergraph decides acyclicity. If acyclic, the elimination order yields a join tree, from which a two-phase semi-join program is generated (bottom-up `parent ⋉ child`, top-down `child ⋉ parent`); the join runs last on the dangling-tuple-free relations.

### 5.8 Frequent Itemsets — shared core (`tools/apriori_ops.py`, `association_rules_ops._frequent_itemsets`)
```
F1 = { single items with support ≥ S }
Ck = { p ∪ q : p, q ∈ F(k-1),  |p ∪ q| = k }      (union-of-pairs join)
Fk = { c ∈ Ck : support(c) ≥ S }
stop when Fk = ∅;  result = ⋃ Fk
support(I) = (transactions containing all of I) / N
```
**Invariant:** an itemset with support count 0 is never frequent, even when `S = 0` (the `c ≥ 1` guard). This keeps the lattice well-formed and avoids degenerate division in the rules tool.

### 5.9 Apriori-TID (`tools/apriori_tid_ops.py`)
Same level-wise structure, but support is read from a **tid_list**:
```
Support(I) = |I.tid_list| / |D|
(I ∪ {…}).tid_list = intersection of the two parents' tid_lists
```
Singletons' tid_lists are the rows where the item appears; the solution opens with a compact `Apriori-TID(|D| = …, S = …, A.tid_list = {…}, …)` condition line.

### 5.10 Association Rules / Maximal / Closed
- **Rules:** for each frequent `Z` (`|Z| ≥ 2`) and each non-empty proper subset `I`, `conf(I → J) = support(Z)/support(I) = count(Z)/count(I)`; keep `conf ≥ C`. Threshold may be written `C` or `D`; when only confidence is given, support defaults to `S = 0`.
- **Maximal:** frequent `I` is maximal ⇔ no immediate superset `I ∪ {x}` is frequent.
- **Closed:** frequent `I` is closed ⇔ no immediate superset has the same support (every proper superset strictly smaller).

---

## 6. File Structure

```
agentADB/
├── version.py                       # Single source of truth for version string
├── main.py                          # Terminal entry point
├── orchestrator.py                  # Router + session state + RA flow + _solution_from
├── llm_factory.py                   # Provider abstraction (OpenAI / Anthropic / Ollama / local)
├── config.json
├── requirements.txt
├── Dockerfile
│
├── agents/
│   ├── serializability_agent.py     # SERIAL
│   ├── parallel_query_agent.py      # QUERY
│   ├── join_cost_agent.py           # JOIN
│   ├── pipeline_agent.py            # Deterministic plan-execute-format pipeline (JOIN)
│   ├── ra_proposal_agent.py         # Human-in-the-loop RA generation (2 models in parallel)
│   ├── semijoin_agent.py            # SEMIJOIN
│   ├── datacube_agent.py            # DATACUBE
│   ├── full_reducer_agent.py        # FULLREDUCER
│   ├── mapreduce_agent.py           # MAPREDUCE
│   ├── apriori_agent.py             # APRIORI
│   ├── apriori_tid_agent.py         # APRIORITID
│   ├── association_rules_agent.py   # RULES
│   ├── maximal_itemsets_agent.py    # MAXIMAL
│   └── closed_itemsets_agent.py     # CLOSED
│
├── tools/
│   ├── confl_ser.py                 # Conflict-serializability precedence graph
│   ├── gemini_view.py               # View-serializability + permutation check
│   ├── db_ops.py                    # Parallel cost functions (select / sort / join / blocks)
│   ├── mermaid_editor.py            # Semi-Join interactive diagram editor
│   ├── cube_ops.py                  # Ullman greedy cube materialisation
│   ├── datacube_editor.py           # Hasse-diagram browser editor
│   ├── full_reducer.py              # GYO acyclicity + join tree + semi-join pseudocode
│   ├── apriori_ops.py               # Apriori + shared mining helpers
│   ├── apriori_tid_ops.py           # Apriori-TID (tid-list method)
│   ├── association_rules_ops.py     # Association rules + _frequent_itemsets
│   ├── maximal_itemsets_ops.py      # Maximal frequent itemsets
│   └── closed_itemsets_ops.py       # Closed frequent itemsets
│
├── evals/
│   ├── run_evals.py
│   └── test_cases.json
│
└── api/
    └── app.py                       # FastAPI server + embedded single-page chat UI
```

---

## 7. Versioning Policy

Version is defined in `version.py` only; `api/app.py` (FastAPI metadata) and the web UI badge `[x.x.x]` read from there (injected at serve time via string replacement). In practice the line has evolved as:

- **0.0.x** — initial per-commit patch increments (HW1: serializability, query, join, map-reduce).
- **0.2.x** — started 2026-05-14 to mark HW2's advanced parallel operations (Semi-Join, Data Cube, Full Reducer) and their iterative fixes, through 0.2.40.
- **3.0.0 / 3.0.1** — bumped on completion of the HW3 data-mining suite (Apriori, Apriori-TID, association rules, maximal & closed itemsets), treated as a major milestone of the framework.

Each functional commit is typically followed by a `bump version to x.y.z` commit.

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
# or, for the terminal UI:
python main.py
```

> Note: the diagram/Hasse editors and the Full Reducer SVG viewer open a browser tab via a local HTTP server, so they are intended for desktop use.

---

## 9. Known Issues and Decisions

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| Agent used record counts instead of block counts in cost tools | LLM ignored tool return values | `!! HARD RULE` in system prompt + explicit warning line in tool output |
| Join step 3 used B_in (total) instead of B_in/p (per-server) | System prompt formula was ambiguous | Inline annotation + concrete wrong/right example |
| Old broadcast join formula was symmetric | Algorithm described incorrectly | Rewrote to asymmetric formula; tool picks cheaper ordering automatically |
| Full Reducer fed all DB tables | Used the whole schema, not the join | Pass only join-participating tables to the tool |
| Browser viewers opened a new `file://` tab | `webbrowser` + file URLs on Linux | Serve via local HTTP server and `xdg-open` the existing tab; SVG viewer for schema |
| Association rules crashed with "division by zero" | With no support threshold, `S=0` made zero-support itemsets "frequent", so a rule divided by a zero-support antecedent | Zero-support itemsets are never frequent (`c ≥ 1` guard); rules tool also defends against a zero-count antecedent; agent passes `S = 0.0` only when no support is given |
| Data-mining agents showed only successful steps | The LLM summarised the tool output and dropped discarded/unsuccessful lines | Orchestrator returns the raw `ToolMessage` trace via `_solution_from()` for all deterministic mining routes |
| `APRIORI-TID` misrouted to `APRIORI` | Keyword scan matched the bare `APRIORI` token first | Compacted-form guard in `_extract_domain` routes hyphen/space variants to `APRIORITID` |
| Version jumped from 0.0.9 to 0.1.0, and later 0.2.40 → 3.0.0 | Numbering re-decisions | 0.0.x corrected to increment indefinitely; later jumps used deliberately to mark HW phase milestones |

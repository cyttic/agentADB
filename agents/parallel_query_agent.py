"""
agents/parallel_query_agent.py
================================
LangGraph agent for parallel database query cost analysis.

Pure orchestration — all math is in tools/db_ops.py.

The agent:
  1. Parses the schema once (sizes converted to BLOCKS).
  2. For each operation in the query (Select / Sort / Join):
     a. Picks the right algorithm from query type + data distribution.
     b. Calls the corresponding cost tool.
  3. Composes costs if the query chains multiple operations.
  4. Reports Elapsed and Total in symbolic form (never reduced).
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

from tools.db_ops import (
    parse_schema              as _parse_schema,
    decide_select_algorithm   as _decide_select_algorithm,
    select_cost               as _select_cost,
    decide_sort_algorithm     as _decide_sort_algorithm,
    sort_cost                 as _sort_cost,
    parallel_join_cost        as _parallel_join_cost,
    compose_costs             as _compose_costs,
    compute_table_blocks_info as _compute_table_blocks_info,
)


# ══════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════

class QueryAgentState(TypedDict):
    messages:   Annotated[list, add_messages]
    db_context: dict   # canonical schema (sizes in blocks) — persists across turns


# ══════════════════════════════════════════════════════════════
#  TOOL WRAPPERS  (LangChain @tool around pure functions in db_ops)
# ══════════════════════════════════════════════════════════════

@tool
def compute_table_blocks(
    record_count:     int,
    num_attributes:   int,
    cell_size_bytes:  int,
    block_size_bytes: int,
    table_name:       str = "",
) -> str:
    """
    Calculate block count for a table when it is NOT given directly.

    Call this whenever the problem gives: number of records, number of
    attributes/fields, size of each cell (bytes), and block size (bytes)
    — instead of a direct block count.

    Formula (shown step by step):
      row_size_bytes   = num_attributes  × cell_size_bytes
      table_size_bytes = record_count    × row_size_bytes
      block_count      = ceil(table_size_bytes / block_size_bytes)

    Args:
        record_count:     Number of records/tuples in the table.
        num_attributes:   Number of columns/fields in the table.
        cell_size_bytes:  Size of one cell value in bytes.
        block_size_bytes: Size of one disk block in bytes.
        table_name:       Optional table name for display (e.g. "R").

    Returns JSON with step-by-step calculation and block_count to use next.
    """
    try:
        result = _compute_table_blocks_info(
            record_count, num_attributes, cell_size_bytes, block_size_bytes, table_name
        )
        print(result["explanation"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def extract_schema_from_text(task_text: str) -> str:
    """
    Extract database schema and system parameters from raw task text.

    Call this FIRST — before parse_schema — when the user gives a task
    as natural language (Russian or English). It reads the text and returns
    a ready-to-use JSON that can be passed directly to parse_schema.

    Works for ANY tables: Students, Flights, Flowers, Cars, Employees, etc.
    Works for multiple tables in one task.

    Args:
        task_text: The full raw task text exactly as the user wrote it.

    Returns JSON string ready to pass to parse_schema, with structure:
    {
      "num_processors": <int>,
      "block_size": <int or 2000 if not stated>,
      "relations": {
        "<TableName>": {
          "fields": ["field1", "field2", ...],
          "key": "<key field name>",
          "distribution": "round_robin" | "hash(<field>)" | "range(<field>)",
          "block_count": <int if stated directly>,
          "record_count": <int if stated>,
          "field_size_bytes": <int if stated>
        }
      },
      "field_info": {
        "<Table>.<field>": {"distinct_values": <int>, "range": "<lo>..<hi>"}
      }
    }
    """
    print(f"[TOOL] extract_schema_from_text(task_text='{task_text[:60]}...')")

    import re

    def _eval_num(s: str) -> int:
        """Parse integers written as plain digits, 10^k, or N*10^k."""
        s = s.strip().replace(',', '').replace(' ', '')
        m = re.match(r'^(\d+)\*10\^(\d+)$', s)
        if m:
            return int(m.group(1)) * (10 ** int(m.group(2)))
        m = re.match(r'^10\^(\d+)$', s)
        if m:
            return 10 ** int(m.group(1))
        return int(float(s))

    # Regex fragment that matches: 10^k, N*10^k, or plain integer (with optional commas)
    _NUM = r'(?:\d+\s*\*\s*)?10\s*\^\s*\d+|[\d,]+'

    result = {
        "num_processors": 10,
        "block_size":     2000,
        "relations":      {},
        "field_info":     {},
    }

    # ── num_processors ───────────────────────────────────────
    m = re.search(r'(\d+)\s*(?:процессор|processor|proc|cpu)', task_text, re.IGNORECASE)
    if m:
        result["num_processors"] = int(m.group(1))

    # ── block_size ───────────────────────────────────────────
    m = re.search(
        r'блок[а-я]*\s*[=:]\s*(\d+)'                       # "блок = 40"
        r'|block\s*size\s*[=:]\s*(\d+)'                    # "block size = 40"
        r'|размер\s+(?:\d+\s+)?блок[а-я]*\s+(\d+)'        # "размер 1 блока 40"
        r'|(\d+)\s*байт[а-я]*\s+(?:в\s+)?блок[а-я]*'      # "40 байт в блоке"
        r'|(\d+)\s*bytes.*block'                            # "40 bytes per block"
        r'|(\d+)\s*байт.*блок',                            # "40 байт ... блок"
        task_text, re.IGNORECASE
    )
    if m:
        val = next(v for v in m.groups() if v is not None)
        result["block_size"] = int(val)

    # ── extract all table definitions  Table(f1, f2, ...) ────
    table_pattern = re.compile(
        r'([A-ZА-Я][A-Za-zА-Яа-я0-9_]*)\s*\(([^)]+)\)',
        re.UNICODE
    )
    for tm in table_pattern.finditer(task_text):
        tname  = tm.group(1)
        fnames = [f.strip() for f in tm.group(2).split(',') if f.strip()]
        if not fnames or tname.lower() in ('select', 'where', 'from', 'join', 'hash', 'range'):
            continue
        result["relations"][tname] = {
            "fields":          fnames,
            "key":             None,
            "distribution":    "round_robin",
            "block_count":     None,
            "record_count":    None,
            "field_size_bytes": None,
        }

    # ── keys ─────────────────────────────────────────────────
    key_pattern = re.compile(
        r'([A-Za-zА-Яа-я0-9_,\s]+?)\s*[—-]\s*ключ|key\s*[=:]\s*([A-Za-z0-9_,\s]+)',
        re.IGNORECASE | re.UNICODE
    )
    for km in key_pattern.finditer(task_text):
        raw_keys = (km.group(1) or km.group(2) or '').strip()
        keys     = [k.strip() for k in re.split(r'[,и\s]+', raw_keys) if k.strip()]
        # assign to the first table that has a matching field
        for tname, rel in result["relations"].items():
            for k in keys:
                if k in rel["fields"] and rel["key"] is None:
                    rel["key"] = k

    # ── block_size ────────────────────────────────────────────
    # Also handle "block size of 2000 bytes" (no = or :)
    bs_extra = re.search(
        r'block\s+size\s+of\s+(\d+)', task_text, re.IGNORECASE
    )
    if bs_extra:
        result["block_size"] = int(bs_extra.group(1))

    # ── field_size_bytes / cell_size ──────────────────────────
    # Handles: "each field is 10 bytes", "field size = 10 bytes",
    #          "каждая ячейка данных весит 20", "каждая ячейка весит 20",
    #          "размер ячейки = 20", "вес ячейки 20 байт"
    fs_pat = re.search(
        r'(?:every|each|каждого|all)\s+field.*?(\d+)\s*(?:bytes?|байт[а-я]*)?'
        r'|field\s+size\s*(?:is|=|:)\s*(\d+)\s*(?:bytes?|байт[а-я]*)?'
        r'|(\d+)\s*(?:bytes?|байт[а-я]*)\s+(?:per\s+)?field'
        r'|(?:каждая|each)\s+(?:ячейка|cell)(?:\s+данных)?\s+весит\s+(\d+)'
        r'|(?:размер|вес|size)\s+(?:ячейки|cell)\s*[=:]\s*(\d+)'
        r'|(\d+)\s*(?:байт[а-я]*|bytes?)\s+(?:на|per)\s+(?:ячейку|cell)',
        task_text, re.IGNORECASE | re.DOTALL
    )
    if fs_pat:
        fs = int(next(v for v in fs_pat.groups() if v is not None))
        for rel in result["relations"].values():
            rel.setdefault("field_size_bytes", fs)

    # ── block_count per table ─────────────────────────────────
    # Handles: "10,000 blocks", "10^4 blocks", "2*10^6 blocks", "или 10000 блоков"
    block_pat = re.compile(
        rf'(?:или\s+)?({_NUM})\s*блок'
        rf'|({_NUM})\s*block',
        re.IGNORECASE
    )
    for bm in block_pat.finditer(task_text):
        raw = (bm.group(1) or bm.group(2) or '').strip()
        try:
            bc = _eval_num(raw)
            for rel in result["relations"].values():
                if rel["block_count"] is None:
                    rel["block_count"] = bc
                    break
        except (ValueError, AttributeError):
            pass

    # ── record_count ──────────────────────────────────────────
    # Handles: "10^6 records", "10^8 records", "2*10^6 tuples"
    rec_pat = re.compile(
        rf'({_NUM})\s*(?:кортеж|tuple|record|строк|row)',
        re.IGNORECASE
    )
    for rm in rec_pat.finditer(task_text):
        raw = rm.group(1).strip()
        try:
            rc = _eval_num(raw)
            for rel in result["relations"].values():
                if rel.get("record_count") is None:
                    rel["record_count"] = rc
                    break
        except (ValueError, AttributeError):
            pass

    # ── distribution ──────────────────────────────────────────
    dist_map = {
        r'round.robin':              'round_robin',
        r'hash\s*\(([^)]+)\)':       None,   # special — capture field
        r'range\s*\(([^)]+)\)':      None,   # special — capture field
        r'hash\s+(?:by\s+)?(\w+)':   None,
        r'range\s+(?:by\s+)?(\w+)':  None,
        r'хэш\s*\(([^)]+)\)':        None,
        r'диапазон\s*\(([^)]+)\)':   None,
    }

    for pattern, fixed_val in dist_map.items():
        m = re.search(pattern, task_text, re.IGNORECASE)
        if m:
            if fixed_val:
                dist = fixed_val
            else:
                field = m.group(1).strip() if m.lastindex else ''
                if 'hash' in pattern or 'хэш' in pattern:
                    dist = f"hash({field})"
                else:
                    dist = f"range({field})"

            # Apply to all tables (override round_robin default)
            for rel in result["relations"].values():
                rel["distribution"] = dist
            break

    # ── field_info: distinct values / ranges ─────────────────
    range_pat = re.compile(
        r'(?:атрибут[а-я]*|field|attribute|поле)\s+([A-Za-zА-Яа-я0-9_.]+)'
        r'.*?(\d+)\s*[–—-]\s*(\d+)',
        re.IGNORECASE | re.UNICODE
    )
    for fm in range_pat.finditer(task_text):
        fname  = fm.group(1)
        lo, hi = int(fm.group(2)), int(fm.group(3))
        # find which table this field belongs to
        for tname, rel in result["relations"].items():
            if fname in rel["fields"]:
                key = f"{tname}.{fname}"
                result["field_info"][key] = {
                    "distinct_values": hi - lo + 1,
                    "range":           f"{lo}..{hi}",
                }

    # ── clean up None values so parse_schema doesn't choke ──────
    for rel in result["relations"].values():
        if rel["block_count"] is None:
            del rel["block_count"]
        if rel["record_count"] is None:
            del rel["record_count"]
        if rel.get("field_size_bytes") is None:
            rel.pop("field_size_bytes", None)

    return json.dumps(result, indent=2)


@tool
def parse_schema(schema_json: str) -> str:
    """
    Parse and store DB schema. Sizes are converted to BLOCKS internally.
    Call this FIRST when the user describes a database setup.

    PRIORITY RULE for block_count per relation:
      1. "block_count" given directly → use as-is (no calculation needed).
      2. "record_count" + "field_size_bytes" given → calculate from those.
      3. Only "record_count" given → use as proxy.
    Never ask the user for missing fields — work with what is provided.

    Required top-level: num_processors.
    Optional top-level: block_size (default 2000).

    IMPORTANT: "relations" MUST be a dict, not a list.
    Correct:   "relations": {"Flowers": {"block_count": 10000, ...}}
    Wrong:     "relations": [{"name": "Flowers", ...}]   <- DO NOT use list

    Per-relation fields:
      - fields (list of field name strings, e.g. ["name","petal","size","color"])
      - key    (string, the partition/primary key field name)
      - distribution ("round_robin" | "hash(<field>)" | "range(<field>)")
      - block_count  (if known directly)  OR
      - record_count + field_size_bytes   (if block_count not given)

    Returns canonical schema with all sizes in BLOCKS.
    """
    try:
        result = _parse_schema(schema_json)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def decide_select_algorithm(
    question_field: str,
    question_type:  str,
    partition_key:  str,
    distribution:   str,
) -> str:
    """
    Decide which Select algorithm (alg2 / alg3) to use.

    Args:
        question_field: Field used in the WHERE condition (e.g. "pid").
        question_type:  "point" (id=10) | "range" (id>5 AND id<10) | "scan" (id!=10).
        partition_key:  The relation's partition key.
        distribution:   "round_robin" | "hash(<field>)" | "range(<field>)".

    Returns JSON with chosen algorithm and reason.
    """
    try:
        result = _decide_select_algorithm(
            question_field, question_type, partition_key, distribution
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def select_cost(
    block_count:         int,
    num_processors:      int,
    algorithm:           str,
    relevant_processors: int = 1,
) -> str:
    """
    Compute Elapsed and Total cost for a Select operation.

    All sizes must be in BLOCKS.
    For alg2: every processor participates.
    For alg3: only `relevant_processors` participate.

    Returns JSON with elapsed, total (symbolic strings), and explanation.
    """
    try:
        result = _select_cost(block_count, num_processors, algorithm, relevant_processors)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def decide_sort_algorithm(sort_field: str, distribution: str) -> str:
    """
    Decide which Sort algorithm to use based on distribution.

    alg1 — Round-Robin or Hash: local sort on all procs, then gather and merge on one proc.
    alg2 — Range on the sort field: each proc sorts locally, no communication needed.

    Args:
        sort_field:   The field being sorted on (e.g. "fid", "date").
        distribution: "round_robin" | "hash(<field>)" | "range(<field>)".

    Returns JSON with chosen algorithm and reason.
    """
    try:
        result = _decide_sort_algorithm(sort_field, distribution)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def sort_cost(block_count: int, num_processors: int, algorithm: str) -> str:
    """
    Compute Elapsed and Total cost for a Sort operation.

    alg1 (round-robin / hash) — 4 steps:
      Step 1: all procs sort locally         → 3 * bs * t_d  (elapsed) / 3 * B * t_d  (total)
      Step 2: (p-1) procs send to proc 0    → (p-1) * bs * t_s
      Step 3: proc 0 reads incoming runs    → (p-1) * bs * (t_s + t_d)
      Step 4: proc 0 merges everything      → B * t_d

    alg2 (range on sort field) — local only:
      Elapsed = 3 * bs * t_d,  Total = 3 * B * t_d

    All sizes must be in BLOCKS. Output is symbolic — never reduced.

    Returns JSON with elapsed, total, step-by-step explanation.
    """
    try:
        result = _sort_cost(block_count, num_processors, algorithm)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def join_cost(
    blocks_s:       int,
    blocks_t:       int,
    num_processors: int,
    name_s:         str = "S",
    name_t:         str = "T",
    distribution_s: str = "round_robin",
    distribution_t: str = "round_robin",
    join_field:     str = "",
) -> str:
    """
    Compute Elapsed and Total cost for a parallel Join.

    REGULAR JOIN (default — distributions differ or partition field != join field):
      Broadcasts the SMALLER (outer) table to all servers.
      bs_out = ceil(B_out / p),  bs_in = ceil(B_in / p)
      Step 1 [send]:    bs_out * t_d + (p-1) * bs_out * t_s
      Step 2 [receive]: (p-1) * bs_out * (t_s + t_d)
      Step 3 [join]:    3 * (B_out + bs_in) * t_d
      Elapsed = Step1 + Step2 + Step3
      Total   = p * Elapsed
      Tool automatically picks outer = smaller table (cheaper ordering).

    PARALLEL (HASH) JOIN (both tables partitioned by the same method on the join field):
      Elapsed = 3 * (bs_S + bs_T) * t_d
      Total   = p * Elapsed

    Args:
        blocks_s:       Total block count of the first (left) table.
        blocks_t:       Total block count of the second (right) table.
        num_processors: Number of parallel servers (p).
        name_s:         Name of the left table (e.g. "Orders").
        name_t:         Name of the right table (e.g. "Products").
        distribution_s: Partition scheme of the left table, e.g. "hash(pid)", "round_robin".
        distribution_t: Partition scheme of the right table.
        join_field:     The field the Join is performed on (e.g. "pid").

    Returns JSON with step1/step2/step3, elapsed, total, explanation.
    """
    try:
        result = _parallel_join_cost(
            blocks_s, blocks_t, num_processors,
            name_s, name_t,
            distribution_s, distribution_t, join_field,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compose_costs(steps_json: str) -> str:
    """
    Compose costs from multiple chained operations (e.g. Select-Join, Select-Sort).

    Args:
        steps_json: JSON array of step results, each with "elapsed" and "total" keys.
                    Example:
                    [
                      {"operation": "select", "elapsed": "1500 * t_d", "total": "15000 * t_d"},
                      {"operation": "join",   "elapsed": "3 * 16500 * t_d", "total": "3 * 16500 * t_d"}
                    ]

    Returns combined Elapsed and Total as symbolic strings.
    """
    try:
        steps  = json.loads(steps_json)
        result = _compose_costs(steps)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [
    compute_table_blocks,
    extract_schema_from_text,
    parse_schema,
    decide_select_algorithm,
    select_cost,
    decide_sort_algorithm,
    sort_cost,
    join_cost,
    compose_costs,
]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  (decision logic for the agent)
# ══════════════════════════════════════════════════════════════

def build_system_prompt(lang: str = "ru") -> str:
    if lang == "ru":
        lang_rule = "Always respond in Russian, regardless of input language."
    else:
        lang_rule = "Always respond in English, regardless of input language."

    return (
        "You are a parallel database systems expert specializing in query cost analysis.\n\n"
        "0. BLOCK COUNT — check this BEFORE any cost calculation.\n"
        "   For each table, determine block count by priority:\n"
        "   a) Block count given directly → use as-is, skip compute_table_blocks.\n"
        "   b) Block count NOT given, but record_count + cell_size + block_size ARE given\n"
        "      → call compute_table_blocks FIRST (one call per table).\n"
        "      Example: R(a,b,c) 100 records, cell = 20 bytes, block = 40 bytes\n"
        "        compute_table_blocks(record_count=100, num_attributes=3,\n"
        "                             cell_size_bytes=20, block_size_bytes=40, table_name='R')\n"
        "        → block_count = 150  (use this in all subsequent calls)\n"
        "   NEVER compute block counts yourself. NEVER use record_count as block_count.\n\n"
        "1. THEN call extract_schema_from_text(task_text) passing the FULL raw task text.\n"
        "   Then call parse_schema with the JSON it returns.\n"
        "   If parse_schema returns block_count = None for a table, call compute_table_blocks.\n"
        "   Never construct the schema JSON yourself — always use extract_schema_from_text.\n"
        "2. Identify which atomic operations the query needs and their order.\n"
        "3. For each Select:\n"
        "   a. Call decide_select_algorithm with question type + distribution.\n"
        "   b. Call select_cost with the chosen algorithm.\n"
        "4. For each Sort: call decide_sort_algorithm + sort_cost.\n"
        "5. For each Join: call join_cost with blocks, num_processors, distributions, join_field.\n"
        "   REGULAR JOIN (default — distributions differ or field != join field):\n"
        "     Smaller table = outer (broadcast). bs_out = ceil(B_out/p), bs_in = ceil(B_in/p)\n"
        "     Step 1 [send]:    bs_out * t_d + (p-1) * bs_out * t_s\n"
        "     Step 2 [receive]: (p-1) * bs_out * (t_s + t_d)\n"
        "     Step 3 [join]:    3 * (B_out + bs_in) * t_d\n"
        "     Elapsed = Step1 + Step2 + Step3,  Total = p * Elapsed\n"
        "   PARALLEL (HASH) JOIN (same hash/range field on both tables = join field):\n"
        "     Elapsed = 3 * (bs_R + bs_S) * t_d,  Total = p * Elapsed\n"
        "   The tool picks the algorithm and the cheaper join ordering automatically.\n"
        "6. If multiple operations: call compose_costs to combine.\n"
        "7. Present the final answer:\n"
        "   \u2022 Relational Algebra expression\n"
        "   \u2022 Algorithm choice + reason for each step\n"
        "   \u2022 Elapsed and Total in symbolic form\n\n"
        f"LANGUAGE RULE: {lang_rule}\n\n"
        "\u2550\u2550\u2550 STRICT OUTPUT FORMAT \u2550\u2550\u2550\n\n"
        "PLAIN TEXT ONLY. No LaTeX, no MathJax, no markup.\n"
        "Forbidden: \\[ \\] \\( \\) \\pi \\sigma \\bowtie \\Big \\frac \\times $...$$\n"
        "Use * for multiplication, ^ for exponents.\n\n"
        "RELATIONAL ALGEBRA \u2014 use only Unicode symbols (no backslash commands):\n"
        "  Select:  \u03c3(condition)(Table)\n"
        "  Project: \u03c0(fields)(Table)\n"
        "  Join:    Table1 \u22c8 Table2   or   Table1 \u22c8(condition) Table2\n"
        "  Example: \u03c0(cid)( \u03c3(price > 100)(Products) \u22c8(pid) \u03c3(50 \u2264 qty \u2264 100)(Orders) )\n\n"
        "After computing costs, present the answer in this structure:\n\n"
        "1. Relational Algebra expression (using \u03c3 \u03c0 \u22c8 \u2014 no LaTeX).\n\n"
        "2. For EACH table whose block count was computed (not given directly), show:\n"
        "   Block count for <TableName>:\n"
        "     Row size   : <n> attributes \u00d7 <c> bytes = <r> bytes\n"
        "     Table size : <N> records \u00d7 <r> bytes = <T> bytes\n"
        "     Block count: ceil(<T> / <b>) = <B> blocks\n\n"
        "3. For EACH operation, a block \u2014 format depends on type:\n\n"
        "   Select / Sort:\n"
        "   Step N \u2014 <operation name>\n"
        "   Operation:     <RA expression>\n"
        "   Algorithm:     <alg2 / alg3 / alg1 / alg2-sort>\n"
        "   Reason:        <one-line reason from decide_* tool>\n"
        "   Total blocks:  <B>\n"
        "   Blocks/server: <B/p = bs>\n"
        "   Elapsed:       <copy from tool EXACTLY>\n"
        "   Total:         <copy from tool EXACTLY>\n\n"
        "   Join (ALWAYS show all 3 steps):\n"
        "   Step N \u2014 Join\n"
        "   Operation:        <RA expression>\n"
        "   Algorithm:        broadcast join\n"
        "   B_S:              <left table blocks>   B_T: <right table blocks>\n"
        "   p:                <num_processors>\n"
        "   Step 1 [send]:    <copy step1 from tool EXACTLY>\n"
        "   Step 2 [receive]: <copy step2 from tool EXACTLY>\n"
        "   Step 3 [join]:    <copy step3 from tool EXACTLY>\n"
        "   Elapsed:          <copy elapsed from tool EXACTLY>\n"
        "   Total:            <copy total from tool EXACTLY>\n\n"
        "4. If multiple operations, call compose_costs, then show:\n"
        "   Elapsed = <combined \u2014 copy EXACTLY>\n"
        "   Total   = <combined \u2014 copy EXACTLY>\n\n"
        "NEVER skip per-step Elapsed/Total. NEVER compute numbers yourself \u2014 always call tools.\n"
        "NEVER collapse (A + B) into a single number in the output.\n"
        "If you are unsure which tool to call next, re-read the WORKFLOW section above."
    )


# ══════════════════════════════════════════════════════════════
#  BUILD AGENT
# ══════════════════════════════════════════════════════════════

_agent_lang: str = "en"

def set_agent_lang(lang: str):
    global _agent_lang
    _agent_lang = lang


def build_agent(llm=None):
    """
    Build the parallel query agent.
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

    def call_llm(state: QueryAgentState):
        ctx_note = ""
        db_ctx = state.get("db_context") or {}
        if db_ctx:
            ctx_note = f"\n\nCurrent DB context (sizes in blocks):\n{json.dumps(db_ctx, indent=2)}"

        messages = [SystemMessage(content=build_system_prompt(_agent_lang) + ctx_note)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: QueryAgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(QueryAgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()

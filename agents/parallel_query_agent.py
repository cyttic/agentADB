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
    select_broadcast_join_cost as _select_broadcast_join_cost,
    range_broadcast_join_cost as _range_broadcast_join_cost,
    hash_shuffle_join_cost    as _hash_shuffle_join_cost,
    selective_broadcast_join_cost as _selective_broadcast_join_cost,
    analyze_join_conditions   as _analyze_join_conditions,
    simplify_cost_expr        as _simplify_cost_expr,
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
def select_broadcast_join_cost(
    blocks_a:       int,
    blocks_b:       int,
    num_processors: int,
    name_a:         str = "A",
    name_b:         str = "B",
    sel_a:          str = "",
    sel_b:          str = "",
    join_field:     str = "",
    project_cost:   int = 0,
) -> str:
    """
    Cost of a query  π(...)( σ_a(A) ⋈ σ_b(B) )  using the select-then-broadcast
    algorithm on round-robin data. USE THIS for any Join that is preceded by
    Select filters (the common "find X who bought Y" style task).

    The SMALLER table is broadcast (outer); the LARGER stays local (inner).
    Each server reads its full local partition, filters it, broadcasts the
    filtered outer table, then joins locally. Costs are returned per step.

    SELECTIVITY (sel_a / sel_b) — pass ONE of:
      • a numeric fraction string for a UNIFORM attribute, computed as
        (number of matching values) / (number of distinct values),
        e.g. "1/2" for "50 < quantity <= 100" when quantity is uniform over 1..100.
      • a SYMBOLIC variable name when the attribute's distribution is UNKNOWN,
        e.g. "Sp" for "price > 100" with no price distribution given. The symbol
        propagates into Elapsed and Total.
      • "" (empty) when that table has no pre-join filter.

    Args:
        blocks_a / blocks_b:  Total block counts of the two tables (already resolved).
        num_processors:       Number of servers p.
        name_a / name_b:      Table names for display.
        sel_a / sel_b:        Selectivity spec for each table's pre-join Select (see above).
        join_field:           The natural-join field (e.g. "pid").
        project_cost:         Cost of the final projection in blocks (default 0).

    Returns JSON with per-step costs, Elapsed and Total (selectivity-aware), and
    a ready-to-present explanation. Copy step/Elapsed/Total values EXACTLY.
    """
    try:
        result = _select_broadcast_join_cost(
            blocks_a, blocks_b, num_processors,
            name_a, name_b,
            sel_a or None, sel_b or None,
            join_field, project_cost,
        )
        print(result["explanation"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def range_broadcast_join_cost(
    blocks_outer:      int,
    blocks_inner:      int,
    num_processors:    int,
    active_processors: int,
    name_outer:        str = "A",
    name_inner:        str = "B",
    sel_outer:         str = "",
    join_field:        str = "",
    project_cost:      int = 0,
) -> str:
    """
    Cost of a Join where the LARGER table is RANGE-partitioned on exactly the
    field its WHERE predicate filters. USE THIS instead of select_broadcast_join_cost
    when the inner table's selection is satisfied by the range partitioning itself.

    Why this case is special:
      - The matching tuples are co-located on a contiguous set of `active_processors`
        (k) servers, and EVERY tuple on those servers matches → NO Select on the
        inner table, and NO selectivity factor for it.
      - Only those k servers receive the broadcast, join, and project.
      - Therefore Total is NOT p * Elapsed — the tool sums the real per-server work.

    How to get active_processors (k):
      If the field is uniform with D distinct values over p servers (D/p per server)
      and the predicate matches a fraction s of the range, then k = round(s * p).
      Example: quantity uniform 1..100, predicate 50<quantity<=100 matches 1/2 →
      k = 1/2 * 10 = 5 servers (the top 5 partitions).

    Args:
        blocks_outer:      block count of the smaller (broadcast) table.
        blocks_inner:      block count of the larger (range-partitioned) table.
        num_processors:    total servers p.
        active_processors: k, servers holding the matching range.
        name_outer/inner:  table names for display.
        sel_outer:         selectivity of the OUTER table's filter — numeric fraction,
                           a symbolic name like "Sp", or "" if it has no filter.
        join_field:        the natural-join field.
        project_cost:      projection cost in blocks (default 0).

    Returns JSON with per-step costs, Elapsed, and Total. Copy values EXACTLY.
    """
    try:
        result = _range_broadcast_join_cost(
            blocks_outer, blocks_inner, num_processors, active_processors,
            name_outer, name_inner,
            sel_outer or None, join_field, project_cost,
        )
        print(result["explanation"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def hash_shuffle_join_cost(
    blocks_redistributed: int,
    blocks_local:         int,
    num_processors:       int,
    name_redistributed:   str = "R",
    name_local:           str = "S",
    sel_redistributed:    str = "",
    sel_local:            str = "",
    join_field:           str = "",
    project_cost:         int = 0,
) -> str:
    """
    Cost of a Join when ONE table is already HASH-partitioned on the join field
    and the other is not. USE THIS when the larger table is distributed by
    hash(join_field): re-hash (redistribute) the OTHER table on the same field so
    matching rows co-locate, then join locally — NO broadcast.

    How it differs from the broadcast tools:
      - The redistributed table is shuffled, not broadcast: each server keeps 1/p
        of its filtered rows and ships (p-1)/p away (so send = receive = (p-1)/p
        * selectivity * bs). Each server writes only its own hash bucket.
      - All p servers do identical work, so Total = p * Elapsed.

    Args:
        blocks_redistributed: blocks of the table to be re-hashed (e.g. Products, round-robin).
        blocks_local:         blocks of the table already hash(join_field)-partitioned (e.g. Orders).
        num_processors:       servers p.
        name_redistributed/name_local: table names for display.
        sel_redistributed:    selectivity of the re-hashed table's filter — numeric
                              fraction, a symbolic name like "Sp", or "" for none.
        sel_local:            selectivity of the local table's filter.
        join_field:           the hash/join field (e.g. "pid").
        project_cost:         projection cost in blocks (default 0).

    Returns JSON with an 'idea' description, per-step costs, Elapsed, and Total.
    Copy the idea, steps, Elapsed and Total EXACTLY.
    """
    try:
        result = _hash_shuffle_join_cost(
            blocks_redistributed, blocks_local, num_processors,
            name_redistributed, name_local,
            sel_redistributed or None, sel_local or None,
            join_field, project_cost,
        )
        print(result["explanation"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def selective_broadcast_join_cost(
    blocks_filtered_table: int,
    blocks_other_table:    int,
    num_processors:        int,
    result_blocks:         int,
    name_filtered:         str = "A",
    name_other:            str = "B",
    sel_other:             str = "",
    join_field:            str = "",
    project_cost:          int = 0,
) -> str:
    """
    Join where a HIGHLY SELECTIVE filter (e.g. a point predicate pid=650, possibly
    AND a narrow range) shrinks one table to a tiny result, which is broadcast and
    joined with the other (usually unfiltered) table read locally. USE THIS when the
    σ result is so small it is a handful of blocks (down to 1), instead of a fraction.

    Computing result_blocks (do this before calling):
      result_blocks = max(1, ceil(s * blocks_filtered_table)), where s is the COMBINED
      selectivity of the filter. For an AND of independent uniform predicates multiply
      the selectivities, e.g. pid=650 -> 1/1000, quantity>=91 (10 of 100 values) ->
      1/10, so s = 1/10000.

    Args:
        blocks_filtered_table: blocks of the table being filtered + broadcast (e.g. Orders).
        blocks_other_table:    blocks of the table joined locally (e.g. Customers).
        num_processors:        servers p.
        result_blocks:         block count of the filtered result (>=1).
        name_filtered/name_other: table names for display.
        sel_other:             selectivity of the OTHER table's filter, or "" if it is
                               unfiltered (then it is read directly inside the join — no
                               separate read/select/write steps).
        join_field:            the join field (e.g. "cid").
        project_cost:          projection cost in blocks (default 0).

    Returns JSON with an 'idea', per-step costs, Elapsed and Total. Total = p * Elapsed.
    """
    try:
        result = _selective_broadcast_join_cost(
            blocks_filtered_table, blocks_other_table, num_processors, result_blocks,
            name_filtered, name_other,
            sel_other or None, join_field, project_cost,
        )
        print(result["explanation"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def analyze_join_conditions(spec_json: str) -> str:
    """
    STEP 0 for any Join — analyse the task conditions and derive the data-movement
    optimization BEFORE computing costs. Returns the key facts, the recommended
    cost tool, and a plain-language 'core_idea' to present to the user.

    It detects: partition pruning (range-partitioned on the filter field → matching
    rows on k servers, no second select), co-location (both hash on the join field →
    local join), shuffle (one hash on the join field → redistribute the other),
    unfiltered-inner broadcast (selective_broadcast), and broadcast-smaller (round-robin).

    Input — a JSON string with EXACTLY two tables (only the ones the query needs):
    {
      "num_processors": 10,
      "join_field": "pid",
      "tables": [
        {"name": "Orders",   "blocks": 2000000, "distribution": "range(quantity)",
         "filter_field": "quantity", "filter_type": "range", "selectivity": "1/2"},
        {"name": "Products", "blocks": 15000,   "distribution": "round_robin",
         "filter_field": "price",    "filter_type": "range", "selectivity": "Sp"}
      ]
    }
    Per table: distribution is "round_robin" | "hash(<f>)" | "range(<f>)";
    filter_type is "point" | "range" | "scan" | "none"; selectivity is a fraction
    string ("1/2"), a symbolic name ("Sp"), or "" if the table has no pre-join filter.

    Returns JSON: {facts: [...], recommended_algorithm, core_idea, ...hints}.
    Use recommended_algorithm to choose the matching cost tool, and present core_idea.
    """
    try:
        result = _analyze_join_conditions(spec_json)
        if "core_idea" in result:
            print("[ANALYSIS] " + result["core_idea"])
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def simplify_cost(expression: str) -> str:
    """
    Collect a symbolic cost expression into its shortest form:
      (.. coefficients ..) * t_d + (.. coefficients ..) * t_s
    with every number in scientific notation. Symbolic selectivity variables
    (e.g. Sp) are preserved as coefficients.

    USE THIS to shorten any long unreduced Elapsed/Total before presenting it,
    e.g. the step1+step2+step3 sum returned by join_cost, or a compose_costs result.

    Example:
      in : "(2 * 10^3 * t_d + 9 * 2 * 10^3 * t_s) + (9 * 2 * 10^3 * (t_s + t_d))
             + (3 * (2 * 10^4 + 10^5) * t_d)"
      out: "(3.8 * 10^5) * t_d + (3.6 * 10^4) * t_s"

    Args:
        expression: the cost expression (uses *, ^, +, parentheses, t_d, t_s, symbols).

    Returns the collected scientific-form string.
    """
    print(f"[TOOL] simplify_cost(expression='{expression[:70]}...')")
    return _simplify_cost_expr(expression)


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
    select_broadcast_join_cost,
    range_broadcast_join_cost,
    hash_shuffle_join_cost,
    selective_broadcast_join_cost,
    analyze_join_conditions,
    simplify_cost,
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
        "══ PRE-SELECTED RA (human-in-the-loop) ══\n"
        "If the user message starts with [SELECTED RA]:, that line contains the Relational\n"
        "Algebra expression chosen by the user from multiple proposals.\n"
        "In that case:\n"
        "  - Use that RA expression EXACTLY as written.\n"
        "  - Do NOT generate, propose, or modify the RA.\n"
        "  - Proceed directly to schema extraction and cost computation.\n\n"
        "0. BLOCK COUNT — resolve for every table BEFORE any cost call.\n"
        "   Priority:\n"
        "   a) Block count stated directly in the task → use as-is.\n"
        "   b) Not stated → call compute_table_blocks (one call per table).\n"
        "      compute_table_blocks returns a JSON with field 'block_count'.\n"
        "      Save that value. It replaces record_count everywhere below.\n\n"
        "   !! HARD RULE — BLOCK COUNT PROPAGATION !!\n"
        "   The value returned by compute_table_blocks as 'block_count' is the ONLY\n"
        "   number you may pass to select_cost, sort_cost, and join_cost for that table.\n"
        "   NEVER pass record_count to any cost tool.\n"
        "   Concrete example:\n"
        "     compute_table_blocks(record_count=1000000, num_attributes=4,\n"
        "                          cell_size_bytes=10, block_size_bytes=2000)\n"
        "     → returns block_count = 20000\n"
        "     select_cost(block_count=20000, ...)        ← correct  (20000, NOT 1000000)\n"
        "     join_cost(blocks_s=20000, ...)             ← correct  (20000, NOT 1000000)\n"
        "     select_cost(block_count=1000000, ...)      ← WRONG — this is record_count\n\n"
        "1. Call extract_schema_from_text(task_text) with the FULL raw task text.\n"
        "   Then call parse_schema with the JSON it returns.\n"
        "   If parse_schema returns block_count = None for a table, call compute_table_blocks.\n"
        "   Never construct the schema JSON yourself — always use extract_schema_from_text.\n"
        "2. Identify which atomic operations the query needs and their order.\n"
        "   ══ STEP 0 — ANALYZE CONDITIONS, THEN STATE THE CORE IDEA (do this FIRST) ══\n"
        "   Before any cost tool, REASON about the task's facts and how they let you\n"
        "   move less data. Concretely:\n"
        "     a) Which tables are truly needed? Drop any table whose attributes are not\n"
        "        required — e.g. if the result key (cid) already lives in Orders, do NOT\n"
        "        join Customers. Use foreign keys to see what is reachable.\n"
        "     b) For each needed table note: distribution (round-robin / hash(f) / range(f)),\n"
        "        the predicate field + type (point/range/none), and its selectivity\n"
        "        (uniform: matching/distinct; unknown distribution: a symbol like Sp).\n"
        "     c) Call analyze_join_conditions with the two joined tables. It returns the\n"
        "        FACTS, the recommended cost tool, and a core_idea. Examples of facts it\n"
        "        finds: 'Orders is range-partitioned on quantity and the predicate is on\n"
        "        quantity, so all rows on the matching servers already qualify -> NO second\n"
        "        select'; 'both tables hash(pid) -> already co-located, local join'.\n"
        "     d) PRESENT a short 'Core idea:' paragraph (copy/paraphrase core_idea) that\n"
        "        explains, in words, the optimization you are exploiting — citing the\n"
        "        specific facts from the task. THEN follow recommended_algorithm to pick\n"
        "        the cost tool below and compute.\n"
        "   The point: the algorithm is a CONSEQUENCE of the conditions — never jump to a\n"
        "   formula before explaining why the data layout makes it valid.\n\n"
        "   !! If the query is a JOIN preceded by Select filters (the typical\n"
        "      'find all X who bought Y costing > N in quantity between A and B'\n"
        "      task), DO NOT use join_cost. Use select_broadcast_join_cost — it\n"
        "      models read-full / write-filtered / broadcast / local-join in one call.\n"
        "      Only tables actually needed for the answer participate (e.g. if the\n"
        "      foreign key you must return already lives in one of the joined tables,\n"
        "      do NOT drag in a third table).\n\n"
        "   ══ SELECTIVITY (needed by select_broadcast_join_cost) ══\n"
        "   For each pre-join Select, determine a selectivity s in [0,1]:\n"
        "     • UNIFORM attribute with known distinct values D and a range/point\n"
        "       predicate matching m of them →  s = m / D, passed as a fraction\n"
        "       string, e.g. '50 < quantity <= 100' over 1..100 (D=100, m=50) → '1/2'.\n"
        "     • Attribute whose distribution is NOT given (e.g. price) → introduce a\n"
        "       SYMBOLIC selectivity variable and pass its NAME, e.g. 'Sp'. It will\n"
        "       remain in Elapsed and Total. Name it after the field (Sp price, Sq qty).\n"
        "     • No filter on that table → pass '' (empty).\n"
        "   State every selectivity (and the meaning of each symbol) before the steps.\n\n"
        "   ══ RANGE PARTITIONING ON THE FILTER FIELD — use range_broadcast_join_cost ══\n"
        "   If the LARGER (inner) table is RANGE-partitioned on EXACTLY the field its\n"
        "   predicate filters, the partitioning already does the selection:\n"
        "     - The matching tuples are co-located on a contiguous set of k 'active'\n"
        "       servers, and ALL tuples there match → NO Select on the inner table\n"
        "       and NO selectivity factor for it.\n"
        "     - k = round(s * p), where s is the inner predicate's selectivity\n"
        "       (uniform field). E.g. quantity uniform 1..100, '50<quantity<=100' →\n"
        "       s = 1/2, p = 10 → k = 5 active servers (the top 5 partitions).\n"
        "     - Only those k servers receive the broadcast, join, and project, so\n"
        "       Total is NOT p * Elapsed — the tool sums the real per-server work.\n"
        "   In this case call range_broadcast_join_cost(blocks_outer, blocks_inner,\n"
        "     num_processors, active_processors=k, name_outer, name_inner, sel_outer,\n"
        "     join_field) — the OUTER table is the smaller broadcast table (still\n"
        "     filtered with its own selectivity, e.g. Sp). Otherwise (round-robin or\n"
        "     hash on the inner filter field) use select_broadcast_join_cost.\n\n"
        "   ══ HASH PARTITIONING ON THE JOIN FIELD — use hash_shuffle_join_cost ══\n"
        "   If one table is distributed by hash(<join_field>) (e.g. Orders by hash(pid)\n"
        "   and the join is on pid), DO NOT broadcast. Re-hash (redistribute) the OTHER\n"
        "   table on the same field so matching rows co-locate, then join locally:\n"
        "     - The redistributed table is shuffled: each server keeps 1/p and ships\n"
        "       (p-1)/p away → send = receive = (p-1)/p * selectivity * bs (e.g. 0.9).\n"
        "     - Each server writes only its own hash bucket (selectivity * bs).\n"
        "     - All p servers do identical work, so Total = p * Elapsed.\n"
        "   Call hash_shuffle_join_cost(blocks_redistributed, blocks_local,\n"
        "     num_processors, name_redistributed, name_local, sel_redistributed,\n"
        "     sel_local, join_field). The 'local' table is the one already\n"
        "     hash(join_field)-partitioned; the 'redistributed' one is the other.\n\n"
        "   ══ HIGHLY SELECTIVE FILTER -> use selective_broadcast_join_cost ══\n"
        "   If one table has a very selective filter (a point predicate like pid=650,\n"
        "   optionally AND a narrow range) so its σ result is a tiny handful of blocks,\n"
        "   read+filter it locally, broadcast the tiny result, and join with the other\n"
        "   (usually unfiltered) table read locally. First compute the result size:\n"
        "     combined selectivity s = product of the predicate selectivities\n"
        "       (uniform: matching_values/distinct_values; e.g. pid=650 -> 1/1000,\n"
        "        quantity>=91 -> 10/100 = 1/10, so s = 1/10000),\n"
        "     result_blocks = max(1, ceil(s * blocks_of_filtered_table)).\n"
        "   Then call selective_broadcast_join_cost(blocks_filtered_table,\n"
        "     blocks_other_table, num_processors, result_blocks, name_filtered,\n"
        "     name_other, sel_other='' , join_field). Pass sel_other='' when the other\n"
        "     table has no filter (it is read directly inside the join — no extra steps).\n\n"
        "   JOIN ALGORITHM CHOICE — summary:\n"
        "     very selective filter -> tiny broadcast result -> selective_broadcast_join_cost\n"
        "     inner range-partitioned on its filter field   -> range_broadcast_join_cost\n"
        "     a table hash-partitioned on the JOIN field     -> hash_shuffle_join_cost\n"
        "     otherwise (round-robin), Join + Selects        -> select_broadcast_join_cost\n"
        "     bare Join, no pre-join filter                  -> join_cost\n\n"
        "   ALWAYS open a Join solution with a one/two-sentence 'Idea:' line describing\n"
        "   the chosen algorithm (the tools return an 'idea' field — copy or paraphrase it).\n\n"
        "3. For each Select:\n"
        "   a. Call decide_select_algorithm with question type + distribution.\n"
        "   b. Call select_cost with the chosen algorithm.\n"
        "4. For each Sort: call decide_sort_algorithm + sort_cost.\n"
        "5. For a JOIN WITH pre-join Selects: call select_broadcast_join_cost(\n"
        "     blocks_a, blocks_b, num_processors, name_a, name_b, sel_a, sel_b, join_field).\n"
        "   It returns 8 per-server steps, plus Elapsed and Total already collected\n"
        "   into linear form over the selectivity symbols. Copy steps/Elapsed/Total\n"
        "   EXACTLY — never recompute or expand the coefficients yourself.\n"
        "   For a BARE JOIN with no pre-join filter: call join_cost instead —\n"
        "   join_cost algorithms:\n"
        "   REGULAR JOIN (default — distributions differ or field != join field):\n"
        "     outer = smaller table (broadcast),  inner = larger table (stays local)\n"
        "     bs_out = ceil(B_out/p),  bs_in = ceil(B_in/p)\n"
        "     Step 1 [send]:    bs_out * t_d + (p-1) * bs_out * t_s\n"
        "     Step 2 [receive]: (p-1) * bs_out * (t_s + t_d)\n"
        "     Step 3 [join]:    3 * (B_out + bs_in) * t_d\n"
        "       B_out = FULL outer table (every server has all of it after step 2)\n"
        "       bs_in = B_in / p = inner table PER SERVER — NOT the full B_in\n"
        "       !! WRONG: 3*(B_out + B_in)*t_d — never use B_in total in step 3 !!\n"
        "     Elapsed = Step1 + Step2 + Step3,  Total = p * Elapsed\n"
        "   PARALLEL (HASH) JOIN (same hash/range field on both tables = join field):\n"
        "     Elapsed = 3 * (bs_R + bs_S) * t_d,  Total = p * Elapsed\n"
        "   The tool picks algorithm and cheaper ordering automatically.\n"
        "   ALWAYS copy step1/step2/step3/elapsed/total EXACTLY from the tool result.\n"
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
        "═══ SCIENTIFIC NOTATION (MANDATORY) ═══\n"
        "EVERY number greater than 10 — in the Relational Algebra, block counts,\n"
        "byte sizes, and all cost expressions — MUST be written in scientific style\n"
        "  a * 10^k   with the mantissa a in the range [1, 10).\n"
        "Examples:  1000 -> 10^3,  1500 -> 1.5 * 10^3,  20000 -> 2 * 10^4,\n"
        "           16500 -> 1.65 * 10^4,  40 -> 4 * 10^1.\n"
        "Numbers 10 or smaller stay as plain digits (e.g. p = 10, the factor 3).\n"
        "The tools already emit this form — copy it EXACTLY and NEVER expand a value\n"
        "like 2 * 10^4 back into 20000. Any number you write yourself must follow the\n"
        "same rule.\n\n"
        "RELATIONAL ALGEBRA \u2014 use only Unicode symbols (no backslash commands):\n"
        "  Select:  \u03c3(condition)(Table)\n"
        "  Project: \u03c0(fields)(Table)\n"
        "  Join:    Table1 \u22c8 Table2   or   Table1 \u22c8(condition) Table2\n"
        "  Example: \u03c0(cid)( \u03c3(price > 100)(Products) \u22c8(pid) \u03c3(50 \u2264 qty \u2264 100)(Orders) )\n\n"
        "After computing costs, present the answer in this structure:\n\n"
        "0. Core idea: 1-3 sentences naming the optimization and the task facts that\n"
        "   justify it (from analyze_join_conditions), BEFORE the formulas.\n\n"
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
        "   Join (ALWAYS show all 3 steps, copy from tool \u2014 NEVER rewrite formulas):\n"
        "   Step N \u2014 Join\n"
        "   Operation:        <RA expression>\n"
        "   Algorithm:        regular join / parallel hash join\n"
        "   Outer table:      <name> \u2014 <B_out> blocks total, <bs_out> blocks/server\n"
        "   Inner table:      <name> \u2014 <B_in> blocks total, <bs_in> blocks/server\n"
        "   p:                <num_processors>\n"
        "   Step 1 [send]:    <copy step1 from tool EXACTLY>\n"
        "   Step 2 [receive]: <copy step2 from tool EXACTLY>\n"
        "   Step 3 [join]:    <copy step3 from tool EXACTLY>\n"
        "     (step3 = 3*(B_out + bs_in)*t_d \u2014 B_out full, bs_in per-server)\n"
        "   Elapsed:          <copy elapsed from tool EXACTLY>\n"
        "                   = <copy elapsed_collected \u2014 the short scientific form>\n"
        "   Total:            <copy total from tool EXACTLY>\n"
        "                   = <copy total_collected \u2014 the short scientific form>\n\n"
        "   Select-then-Broadcast Join (output of select_broadcast_join_cost):\n"
        "   First list block counts and the selectivity of every filter, e.g.:\n"
        "     Selectivity (Orders, 50 < quantity \u2264 100, uniform over 1..100): 1/2\n"
        "     Selectivity (Products, price > 100, distribution unknown): Sp \u2208 [0,1]\n"
        "   Then the algorithm as 8 numbered per-server steps, copied EXACTLY:\n"
        "     1) read + select <outer>           <copy>\n"
        "     2) send filtered <outer> to (p-1)  <copy>\n"
        "     3) receive from (p-1)              <copy>\n"
        "     4) write full broadcast <outer>    <copy>\n"
        "     5) read + select <inner>           <copy>\n"
        "     6) write filtered <inner>          <copy>\n"
        "     7) natural join                    <copy>\n"
        "     8) projection                      <copy>\n"
        "   Then:\n"
        "     E = <copy elapsed EXACTLY>\n"
        "     T = <copy total EXACTLY>\n"
        "   NEVER drop the symbolic selectivity (e.g. Sp) and NEVER expand the\n"
        "   collected coefficients back to long digit strings.\n\n"
        "   Range-Partitioned Broadcast Join (output of range_broadcast_join_cost):\n"
        "   Same as above but FIRST state the k active servers and why (e.g. 'quantity\n"
        "   range-partitioned, 50<quantity≤100 over 1..100 → top 5 of 10 servers').\n"
        "   The send step has TWO parts (non-active→k, active→k-1). The inner table is\n"
        "   NOT filtered (the partition did it). Show both E and T, and explicitly note\n"
        "   T ≠ p·E here (only k servers join). Copy E and T EXACTLY from the tool.\n\n"
        "4. If multiple operations, call compose_costs, then show:\n"
        "   Elapsed = <combined \u2014 copy EXACTLY>\n"
        "   Total   = <combined \u2014 copy EXACTLY>\n\n"
        "NEVER skip per-step Elapsed/Total. NEVER compute numbers yourself \u2014 always call tools.\n"
        "NEVER collapse (A + B) into a single number in the output.\n"
        "ALWAYS finish Elapsed and Total with their COLLECTED short form: use the\n"
        "  tool's elapsed_collected / total_collected fields, or call simplify_cost on\n"
        "  any expression you built (e.g. a compose_costs result). The collected form is\n"
        "  the linear scientific form  (..) * t_d + (..) * t_s  \u2014 keep selectivity\n"
        "  symbols, do NOT reduce it to one number.\n"
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

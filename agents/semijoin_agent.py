"""
agents/semijoin_agent.py
=========================
LangGraph agent for Semi-Join cost analysis.

TWO-TOOL DETERMINISTIC PIPELINE:

  1. parse_semijoin_problem  — LLM commits all extracted values into a
                               validated Python dict. Nothing is calculated yet.

  2. compute_semijoin        — receives the parsed dict values and performs
                               all cost calculations deterministically.

  (Optional) compute_table_blocks — only called when the problem gives records,
             not blocks. Called before parse_semijoin_problem in that case.

SEMI-JOIN ALGORITHM  (R on server X, S on server Y, join on field B):

  Step 1 — Start at X  [no cost]

  Step 2 [project + send  X → Y]:
    Read all R from disk to extract column B, send projected column to Y.
    proj_blocks = ceil(B_R / num_attrs_R)
    cost = B_R * t_d  +  proj_blocks * t_s

  Step 3 [join on Y]:
    Y computes S' = S ⋈ π(B)(R)
    cost = 3 * (B_S + proj_blocks) * t_d

  Step 4 [send result  Y → X]:
    result size:
      if match_prob given:     ceil(match_prob * B_S)
      elif B is key in R:      B_S   (distinct proj → each S tuple ≤ 1 match)
      else (B not key in R):   B_S * B_R  (worst-case Cartesian blowup)
    cost = result_blocks * t_s

  Step 5 [final join on X]:
    X joins R with received result.
    cost = 3 * (B_R + result_blocks) * t_d

  Elapsed = Step2 + Step3 + Step4 + Step5
  Total   = Elapsed
"""

import json
import math
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.db_ops import compute_table_blocks_info as _compute_table_blocks_info


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _route_servers(route: str, default_from: str, default_to: str) -> list:
    """Parse 'A → B → C' into ['A', 'B', 'C']. Returns [default_from, default_to] if empty."""
    if not route.strip():
        return [default_from, default_to]
    return [s.strip() for s in route.split("→")]


def _check_join_field_is_sole_key(key_of_r: str, join_field: str) -> tuple:
    """
    Determine whether join_field is the SOLE primary key of name_r.

    Returns (is_sole_key: bool, reason: str).

    Rules:
      - key_of_r = "pid"        and join_field = "pid"  → True  (sole key)
      - key_of_r = "pid, cid"   and join_field = "pid"  → False (composite key, pid alone is not unique)
      - key_of_r = "cid"        and join_field = "pid"  → False (different field)
    """
    key_fields = [f.strip().lower() for f in key_of_r.replace(';', ',').split(',')]
    jf = join_field.strip().lower()

    if key_fields == [jf]:
        return True, (
            f"{join_field} IS the sole primary key of name_r "
            f"→ π({join_field})(name_r) has distinct values "
            f"→ each name_s tuple matches ≤ 1 name_r tuple"
        )
    elif jf in key_fields:
        return False, (
            f"{join_field} is part of a COMPOSITE key ({key_of_r}) in name_r "
            f"→ {join_field} alone is NOT unique in name_r "
            f"→ π({join_field})(name_r) contains DUPLICATE values "
            f"→ worst case result = B_name_s * B_name_r"
        )
    else:
        return False, (
            f"{join_field} is NOT the key of name_r (key is '{key_of_r}') "
            f"→ π({join_field})(name_r) may contain DUPLICATE values "
            f"→ worst case result = B_name_s * B_name_r"
        )


# ══════════════════════════════════════════════════════════════
#  SCIENTIFIC NOTATION FORMATTER
# ══════════════════════════════════════════════════════════════

def _sci_fmt(n) -> str:
    """
    750000     → 7.5 * 10^5
    100000     → 10^5
    7500000000 → 7.5 * 10^9
    30         → 30
    """
    n = int(n)
    if n < 100:
        return str(n)
    log = math.log10(n)
    k   = math.floor(log)
    mantissa = round(n / (10 ** k), 6)
    if abs(mantissa - 1.0) < 1e-6:
        return f"10^{k}"
    if abs(mantissa - round(mantissa)) < 1e-6:
        return f"{int(round(mantissa))} * 10^{k}"
    m1 = round(mantissa, 1)
    if abs(mantissa - m1) < 1e-5:
        return f"{m1} * 10^{k}"
    m2 = round(mantissa, 2)
    if abs(mantissa - m2) < 1e-4:
        return f"{m2} * 10^{k}"
    return f"{round(mantissa, 3)} * 10^{k}"


# ══════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════

class SemiJoinAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
#  TOOL 0a — declare_server_layout  (ALWAYS the very first call)
# ══════════════════════════════════════════════════════════════

@tool
def declare_server_layout(
    table_1:           str,
    server_of_table_1: str,
    key_of_table_1:    str,
    table_2:           str,
    server_of_table_2: str,
    key_of_table_2:    str,
    starting_server:   str,
) -> str:
    """
    ALWAYS call this tool FIRST — before any other tool.

    Declare which table lives on which server, its PRIMARY KEY, and which
    server starts the operation.

    HOW TO IDENTIFY THE PRIMARY KEY:
      - Read the schema declaration: Table(field1, field2, ...)
      - A field listed as "primary key" alone  → key is just that field, e.g. "pid"
      - If the table has MULTIPLE foreign keys and no other natural identifier,
        the primary key is the COMBINATION of ALL those foreign keys.
        Example: Sales(pid, cid, amount) where Sales.pid and Sales.cid are both
        foreign keys → primary key = "pid,cid"  (NOT just "pid", NOT just "cid")
      - Pass the key EXACTLY as a comma-separated string: "pid" or "pid,cid"

    The tool determines name_r / name_s automatically from starting_server.
    This layout is the SINGLE SOURCE OF TRUTH for the entire pipeline.

    Args:
        table_1:           Name of the first table.
        server_of_table_1: Server where table_1 is stored.
        key_of_table_1:    Primary key of table_1. Single: "pid". Composite: "pid,cid".
        table_2:           Name of the second table.
        server_of_table_2: Server where table_2 is stored.
        key_of_table_2:    Primary key of table_2.
        starting_server:   Server where the semijoin operation begins.
    """
    t1, s1, k1 = table_1.strip(), server_of_table_1.strip(), key_of_table_1.strip()
    t2, s2, k2 = table_2.strip(), server_of_table_2.strip(), key_of_table_2.strip()
    ss = starting_server.strip()

    if s1 == ss:
        name_r, server_r, key_r = t1, s1, k1
        name_s, server_s, key_s = t2, s2, k2
    elif s2 == ss:
        name_r, server_r, key_r = t2, s2, k2
        name_s, server_s, key_s = t1, s1, k1
    else:
        return json.dumps({
            "error": (
                f"starting_server '{ss}' does not match either table's server "
                f"('{s1}' or '{s2}'). Check the problem statement."
            )
        })

    key_r_fields = [f.strip() for f in key_r.replace(';', ',').split(',')]
    key_s_fields = [f.strip() for f in key_s.replace(';', ',').split(',')]
    key_r_type = "composite" if len(key_r_fields) > 1 else "single"
    key_s_type = "composite" if len(key_s_fields) > 1 else "single"

    output = "\n".join([
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  SERVER LAYOUT & KEY DECLARATIONS",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {t1}  →  server {s1}  |  key: {k1}",
        f"  {t2}  →  server {s2}  |  key: {k2}",
        f"",
        f"  Operation starts at: server {ss}",
        f"",
        f"  name_r (LEFT,  at starting server {server_r}) = {name_r}",
        f"    key of {name_r}: {key_r}  ({key_r_type})",
        f"  name_s (RIGHT, at remote   server {server_s}) = {name_s}",
        f"    key of {name_s}: {key_s}  ({key_s_type})",
        f"",
        f"  Semijoin direction: {name_r} ⋉ {name_s}",
        f"  Step 2 will project and SEND: π(join_field)({name_r})",
        f"  !! Size used for send cost = blocks of {name_r} (NOT {name_s}) !!",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Pass key_of_r = \"{key_r}\" to parse_semijoin_problem.",
    ])

    print(output)
    return json.dumps({
        "name_r":    name_r,
        "server_r":  server_r,
        "key_r":     key_r,
        "name_s":    name_s,
        "server_s":  server_s,
        "key_s":     key_s,
        "layout":    {t1: {"server": s1, "key": k1},
                      t2: {"server": s2, "key": k2}},
        "output":    output,
    }, indent=2)


# ══════════════════════════════════════════════════════════════
#  TOOL 0b — declare_table_sizes
# ══════════════════════════════════════════════════════════════

@tool
def declare_table_sizes(
    name_r:      str,
    raw_size_r:  str,
    unit_r:      str,
    name_s:      str,
    raw_size_s:  str,
    unit_s:      str,
) -> str:
    """
    ALWAYS call this tool FIRST — before any other tool in the pipeline.

    Read the problem text and declare, for EACH table:
      - the size value as written in the problem (e.g. "10^8", "500")
      - the unit: EXACTLY one of the strings "blocks" or "records"

    HOW TO DECIDE THE UNIT:
      "blocks" → the problem uses words like:
          blocks, blоков, блоков, block count, size in blocks
      "records" → the problem uses words like:
          records, rows, tuples, строк, записей, кортежей

    Args:
        name_r:     Name of the LEFT table (at the starting server).
        raw_size_r: Size as written in the problem, e.g. "10^8" or "500".
        unit_r:     "blocks" if size is in blocks, "records" if size is in records.
        name_s:     Name of the RIGHT table.
        raw_size_s: Size as written in the problem.
        unit_s:     "blocks" or "records".

    Returns a declaration and explicit instructions for the next step.
    """
    units = {
        name_r: unit_r.strip().lower(),
        name_s: unit_s.strip().lower(),
    }
    raws  = {name_r: raw_size_r, name_s: raw_size_s}
    next_steps = [n for n, u in units.items() if u == "records"]

    # ── Determine the section header ──────────────────────────
    all_units = set(units.values())
    if all_units == {"blocks"}:
        header_unit = "BLOCKS"
    elif all_units == {"records"}:
        header_unit = "RECORDS"
    else:
        header_unit = "MIXED (some BLOCKS, some RECORDS)"

    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  SIZE TABLES GIVEN ON: {header_unit}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for name in (name_r, name_s):
        raw  = raws[name]
        unit = units[name]
        if unit == "blocks":
            lines.append(f"  {name}: {raw} blocks → BLOCKS → use directly, no conversion needed")
        elif unit == "records":
            lines.append(f"  {name}: {raw} records → RECORDS → MUST call compute_table_blocks")
        else:
            lines.append(f"  {name}: {raw} → UNRECOGNISED unit '{unit}' — use 'blocks' or 'records'")

    lines.append("")
    if next_steps:
        lines.append(f"ACTION REQUIRED: call compute_table_blocks for: {', '.join(next_steps)}")
        lines.append("Verbose block-count calculation MUST be shown for each of these tables.")
        lines.append("Then call parse_semijoin_problem with the computed block counts.")
    else:
        lines.append("ACTION: all sizes already in blocks — call parse_semijoin_problem directly.")

    output = "\n".join(lines)
    print(output)
    return json.dumps({
        "section_header":  f"SIZE TABLES GIVEN ON: {header_unit}",
        "declaration": {
            name_r: {"raw_size": raw_size_r, "unit": units[name_r]},
            name_s: {"raw_size": raw_size_s, "unit": units[name_s]},
        },
        "needs_conversion": next_steps,
        "instructions":     output,
    }, indent=2)


# ══════════════════════════════════════════════════════════════
#  TOOL 1 — compute_table_blocks  (only when records given)
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
    Convert record count → block count.

    Call this ONLY when the problem gives the table size as a NUMBER OF RECORDS
    (not blocks) together with field size and block size.
    Do NOT call this when block count is already stated in the problem.

    Formula:
      row_size_bytes   = num_attributes x cell_size_bytes
      table_size_bytes = record_count   x row_size_bytes
      block_count      = ceil(table_size_bytes / block_size_bytes)

    Returns JSON including block_count (integer) and block_count_sci (string).
    Pass block_count to parse_semijoin_problem as blocks_r or blocks_s.
    """
    try:
        result    = _compute_table_blocks_info(
            record_count, num_attributes, cell_size_bytes, block_size_bytes, table_name
        )
        bc        = result["block_count"]
        row_bytes = result["row_size_bytes"]
        tbl_bytes = result["table_size_bytes"]
        name      = result["table_name"]

        explanation = "\n".join([
            f"Block count for {name}:",
            f"  Step 1 — row size  : {num_attributes} x {cell_size_bytes}"
            f" = {row_bytes} bytes/row",
            f"  Step 2 — table size: {_sci_fmt(record_count)} records x {row_bytes}"
            f" = {_sci_fmt(tbl_bytes)} bytes",
            f"  Step 3 — blocks    : ceil({_sci_fmt(tbl_bytes)} / {block_size_bytes})"
            f" = {_sci_fmt(bc)} blocks",
            f"  --> block_count = {_sci_fmt(bc)}  (pass this to parse_semijoin_problem)",
        ])
        result["block_count_sci"] = _sci_fmt(bc)
        result["explanation"]     = explanation
        print(explanation)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════
#  TOOL 2 — parse_semijoin_problem  (ALWAYS call this first)
# ══════════════════════════════════════════════════════════════

def _count_hops(route: str) -> int:
    """Count hops in a route string like 'A → B → C' → 2 hops."""
    if not route.strip():
        return 1
    return route.count("→")


@tool
def parse_semijoin_problem(
    name_r:       str,
    name_s:       str,
    server_r:     str,
    server_s:     str,
    blocks_r:     int,
    blocks_s:     int,
    num_attrs_r:  int,
    join_field:   str,
    key_of_r:     str,
    match_prob:   float = -1.0,
    route_r_to_s: str   = "",
    route_s_to_r: str   = "",
) -> str:
    """
    Commit all values extracted from the problem into a validated Python structure.

    ALWAYS call this after declare_server_layout and declare_table_sizes.
    All downstream computation uses ONLY the values returned by this tool.

    Args:
        name_r:       LEFT table (at starting server). From declare_server_layout.
        name_s:       RIGHT table (at remote server).  From declare_server_layout.
        server_r:     Starting server.                 From declare_server_layout.
        server_s:     Remote server.                   From declare_server_layout.
        blocks_r:     Block count of name_r as a plain INTEGER.
        blocks_s:     Block count of name_s as a plain INTEGER.
        num_attrs_r:  Number of attributes in name_r (e.g. Sales(pid,cid,amount) → 3).
        join_field:   The attribute the join is performed on (e.g. "pid").
        key_of_r:     The PRIMARY KEY declaration of name_r exactly as in the problem.
                      Single key   → "pid"
                      Composite key→ "pid,cid"
                      The tool uses this to determine whether join_field alone is unique.
                      DO NOT pass True/False — pass the actual key string.
        match_prob:   Matching probability (0.0–1.0) or -1.0 if not given.
        route_r_to_s: "" for direct, or full path e.g. "A → B → C" if indirect.
        route_s_to_r: "" for direct, or full return path e.g. "C → B → A".

    Returns a validated JSON structure. Copy its field values unchanged
    into compute_semijoin — do not modify them.
    """
    hops_fwd = _count_hops(route_r_to_s) if route_r_to_s.strip() else 1
    hops_bwd = (_count_hops(route_s_to_r) if route_s_to_r.strip() else hops_fwd)

    fwd_display = route_r_to_s.strip() if route_r_to_s.strip() else f"{server_r} → {server_s}"
    bwd_display = route_s_to_r.strip() if route_s_to_r.strip() else f"{server_s} → {server_r}"

    # Compute join_field_key_in_r from the actual key declaration
    join_field_key_in_r, key_reason = _check_join_field_is_sole_key(key_of_r, join_field)
    # Fill real table name into the reason string
    key_reason = key_reason.replace("name_r", name_r).replace("name_s", name_s)

    prob_str = (f"{match_prob}" if 0.0 <= match_prob <= 1.0
                else "not given (worst case)")

    summary = "\n".join([
        "=== Parsed semijoin problem ===",
        f"  Left  table : {name_r}  on server {server_r}"
        f"  →  {_sci_fmt(blocks_r)} blocks  ({num_attrs_r} attributes)",
        f"  Right table : {name_s}  on server {server_s}"
        f"  →  {_sci_fmt(blocks_s)} blocks",
        f"  Join field  : {join_field}",
        f"  Key of {name_r}  : {key_of_r}",
        f"  join_field sole key in {name_r}: {join_field_key_in_r}",
        f"  Reason      : {key_reason}",
        f"  Match prob  : {prob_str}",
        f"  Route fwd   : {fwd_display}  ({hops_fwd} hop{'s' if hops_fwd > 1 else ''})",
        f"  Route back  : {bwd_display}  ({hops_bwd} hop{'s' if hops_bwd > 1 else ''})",
        f"  Operation   : {name_r} ⋉ {name_s}  [{fwd_display}] then [{bwd_display}]",
        "=== Use these values in compute_semijoin ===",
    ])

    result = {
        "name_r":              name_r,
        "name_s":              name_s,
        "server_r":            server_r,
        "server_s":            server_s,
        "blocks_r":            blocks_r,
        "blocks_r_sci":        _sci_fmt(blocks_r),
        "blocks_s":            blocks_s,
        "blocks_s_sci":        _sci_fmt(blocks_s),
        "num_attrs_r":         num_attrs_r,
        "join_field":          join_field,
        "key_of_r":            key_of_r,
        "join_field_key_in_r": join_field_key_in_r,
        "key_reason":          key_reason,
        "match_prob":          match_prob,
        "route_r_to_s":        fwd_display,
        "route_s_to_r":        bwd_display,
        "hops_r_to_s":         hops_fwd,
        "hops_s_to_r":         hops_bwd,
        "summary":             summary,
    }
    print(summary)
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════
#  TOOL 3 — compute_semijoin  (call after parse_semijoin_problem)
# ══════════════════════════════════════════════════════════════

@tool
def compute_semijoin(
    blocks_r:            int,
    blocks_s:            int,
    num_attrs_r:         int,
    name_r:              str   = "R",
    name_s:              str   = "S",
    server_r:            str   = "X",
    server_s:            str   = "Y",
    join_field:          str   = "B",
    key_of_r:            str   = "",
    join_field_key_in_r: bool  = True,
    match_prob:          float = -1.0,
    route_r_to_s:        str   = "",
    route_s_to_r:        str   = "",
    hops_r_to_s:         int   = 1,
    hops_s_to_r:         int   = 1,
) -> str:
    """
    Compute semijoin cost. Call AFTER parse_semijoin_problem.
    Pass the values EXACTLY as returned by parse_semijoin_problem.

    Steps:
      1 — Start at server_r               [no cost]
      2 — Project + send along route      B_R * t_d  +  hops_r_to_s * proj_blocks * t_s
      3 — Join on server_s                3 * (B_S + proj_blocks) * t_d
      4 — Send result along return route  hops_s_to_r * result_blocks * t_s
      5 — Final join on server_r          3 * (B_R + result_blocks) * t_d

    Transfer cost is multiplied by hop count.
    Direct connection (1 hop) gives the standard single-hop formula.

    Result size for step 4:
      match_prob given          → ceil(match_prob * B_S)
      join_field key in R       → B_S
      join_field NOT key in R   → B_S * B_R
    """
    try:
        B_R = blocks_r
        B_S = blocks_s

        proj_blocks = math.ceil(B_R / num_attrs_r)

        # Re-derive join_field_key_in_r from key_of_r if supplied (authoritative)
        if key_of_r.strip():
            join_field_key_in_r, _ = _check_join_field_is_sole_key(key_of_r, join_field)

        if 0.0 <= match_prob <= 1.0:
            result_blocks = math.ceil(match_prob * B_S)
            result_note   = (
                f"match_prob given: ceil({match_prob} * {_sci_fmt(B_S)})"
                f" = {_sci_fmt(result_blocks)} blocks"
            )
        elif join_field_key_in_r:
            result_blocks = B_S
            key_display   = key_of_r if key_of_r.strip() else join_field
            result_note   = (
                f"{join_field} IS the sole primary key of {name_r} (key: {key_display})"
                f" → π({join_field})({name_r}) has DISTINCT values"
                f" → each {name_s} tuple matches ≤ 1 {name_r} tuple"
                f" → worst case = {_sci_fmt(B_S)} blocks (size of {name_s})"
            )
        else:
            result_blocks = B_S * B_R
            key_display   = key_of_r if key_of_r.strip() else "unknown"
            result_note   = (
                f"{join_field} is NOT the sole key of {name_r} (key: {key_display})"
                f" → π({join_field})({name_r}) contains DUPLICATE values"
                f" → each {name_s} tuple may match MULTIPLE {name_r} tuples"
                f" → worst case = {_sci_fmt(B_S)} * {_sci_fmt(B_R)}"
                f" = {_sci_fmt(result_blocks)} blocks"
            )

        proj_s = _sci_fmt(proj_blocks)
        res_s  = _sci_fmt(result_blocks)
        B_R_s  = _sci_fmt(B_R)
        B_S_s  = _sci_fmt(B_S)

        # Parse routes into server lists
        fwd_servers = _route_servers(route_r_to_s, server_r, server_s)
        bwd_servers = _route_servers(route_s_to_r, server_s, server_r)

        fwd_str = " → ".join(fwd_servers)
        bwd_str = " → ".join(bwd_servers)

        direct_fwd = (len(fwd_servers) == 2)
        direct_bwd = (len(bwd_servers) == 2)

        # ── Collect individual step costs for elapsed sum ──────────
        elapsed_parts = []
        lines = []
        n = 1   # step counter

        # Header
        lines += [
            f"Semi-Join: {name_r} ⋉ {name_s}",
            f"  Route forward : {fwd_str}",
            f"  Route back    : {bwd_str}",
            f"",
            f"  {name_r}: {B_R_s} blocks, {num_attrs_r} attrs  (server {server_r})",
            f"  {name_s}: {B_S_s} blocks  (server {server_s})",
            f"  Join field: {join_field}  |  key in {name_r}: {join_field_key_in_r}",
            f"",
            f"  Projection: π({join_field})({name_r})",
            f"    proj_blocks = ceil({B_R_s} / {num_attrs_r}) = {proj_s} blocks",
            f"",
        ]

        # Step 1 — start (no cost)
        lines.append(f"  Step {n}: Start at server {server_r}  [no cost]")
        lines.append(f"")
        n += 1

        if direct_fwd:
            # Step 2 — project + send combined (one hop)
            cost = f"{B_R_s} * t_d + {proj_s} * t_s"
            elapsed_parts.append(cost)
            lines += [
                f"  Step {n} [project + send  {fwd_str}]:",
                f"    Read all {name_r} from disk : {B_R_s} * t_d",
                f"    Send π({join_field})({name_r}) ({proj_s} blocks) : {proj_s} * t_s",
                f"    Cost = {cost}",
                f"",
            ]
            n += 1
        else:
            # Separate project step
            cost_proj = f"{B_R_s} * t_d"
            elapsed_parts.append(cost_proj)
            lines += [
                f"  Step {n} [project  on {server_r}]:",
                f"    Read all {name_r} from disk to extract column {join_field}",
                f"    Cost = {cost_proj}",
                f"",
            ]
            n += 1
            # One step per forward hop — transfer only
            for i in range(len(fwd_servers) - 1):
                hop_from = fwd_servers[i]
                hop_to   = fwd_servers[i + 1]
                hop_cost = f"{proj_s} * t_s"
                elapsed_parts.append(hop_cost)
                lines += [
                    f"  Step {n} [transfer projection  {hop_from} → {hop_to}]:",
                    f"    {proj_s} blocks * t_s",
                    f"    Cost = {hop_cost}",
                    f"",
                ]
                n += 1

        # Join on server_s
        cost_join = f"3 * ({B_S_s} + {proj_s}) * t_d"
        elapsed_parts.append(cost_join)
        lines += [
            f"  Step {n} [join on {server_s}]:",
            f"    {name_s} ⋈ π({join_field})({name_r})",
            f"    Cost = {cost_join}",
            f"",
        ]
        n += 1

        if direct_bwd:
            # Send result in one hop
            cost_send = f"{res_s} * t_s"
            elapsed_parts.append(cost_send)
            lines += [
                f"  Step {n} [send result  {bwd_str}]:",
                f"    {result_note}",
                f"    Cost = {cost_send}",
                f"",
            ]
            n += 1
        else:
            # One step per backward hop — transfer only
            first_hop = True
            for i in range(len(bwd_servers) - 1):
                hop_from = bwd_servers[i]
                hop_to   = bwd_servers[i + 1]
                hop_cost = f"{res_s} * t_s"
                elapsed_parts.append(hop_cost)
                lines += [
                    f"  Step {n} [transfer result  {hop_from} → {hop_to}]:",
                ]
                if first_hop:
                    lines.append(f"    {result_note}")
                    first_hop = False
                lines += [
                    f"    {res_s} blocks * t_s",
                    f"    Cost = {hop_cost}",
                    f"",
                ]
                n += 1

        # Final join on server_r
        cost_final = f"3 * ({B_R_s} + {res_s}) * t_d"
        elapsed_parts.append(cost_final)
        lines += [
            f"  Step {n} [final join on {server_r}]:",
            f"    {name_r} ⋈ received result",
            f"    Cost = {cost_final}",
            f"",
        ]

        elapsed = " + ".join(f"({p})" for p in elapsed_parts)
        total   = elapsed

        lines += [
            f"  Elapsed = {elapsed}",
            f"  Total   = Elapsed = {total}",
        ]

        explanation = "\n".join(lines)

        # Keep backward-compat keys for summary block
        step2 = elapsed_parts[0] if elapsed_parts else ""
        step3 = cost_join
        step4 = elapsed_parts[-2] if len(elapsed_parts) >= 2 else ""
        step5 = cost_final

        print(explanation)
        return json.dumps({
            "operation":           "semijoin",
            "name_r":              name_r,
            "name_s":              name_s,
            "server_r":            server_r,
            "server_s":            server_s,
            "join_field":          join_field,
            "join_field_key_in_r": join_field_key_in_r,
            "blocks_r":            B_R,
            "blocks_r_sci":        B_R_s,
            "blocks_s":            B_S,
            "blocks_s_sci":        B_S_s,
            "proj_blocks":         proj_blocks,
            "proj_blocks_sci":     proj_s,
            "result_blocks":       result_blocks,
            "result_blocks_sci":   res_s,
            "match_prob":          match_prob,
            "route_r_to_s":        fwd_str,
            "route_s_to_r":        bwd_str,
            "hops_r_to_s":         hops_r_to_s,
            "hops_s_to_r":         hops_s_to_r,
            "step2":               step2,
            "step3":               step3,
            "step4":               step4,
            "step5":               step5,
            "elapsed":             elapsed,
            "total":               total,
            "explanation":         explanation,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [declare_server_layout, declare_table_sizes, compute_table_blocks, parse_semijoin_problem, compute_semijoin]


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert in distributed database systems specialising in Semi-Join cost analysis.

You have three tools:
  1. compute_table_blocks   — converts record count → block count (call only when needed)
  2. parse_semijoin_problem — commits all problem values into a Python structure (ALWAYS call first)
  3. compute_semijoin       — calculates all costs (call after parse_semijoin_problem)

══ MANDATORY PIPELINE — FOLLOW THIS EXACT ORDER ══

  declare_server_layout        ← ALWAYS FIRST. Maps tables→servers. Determines name_r, name_s.
       ↓
  declare_table_sizes          ← ALWAYS SECOND. Determines size units (blocks / records).
       ↓
  compute_table_blocks         ← for each table whose unit = "records" (may be called multiple times)
       ↓
  parse_semijoin_problem       ← ALWAYS. Locks in all values.
       ↓
  compute_semijoin             ← ALWAYS. Uses only values from parse output.

NEVER skip declare_server_layout.
NEVER skip declare_table_sizes.
NEVER skip parse_semijoin_problem.
NEVER determine name_r / name_s yourself — always read them from declare_server_layout output.
NEVER write any formulas or numbers in text before the pipeline completes.

══ STEP 0a — declare_server_layout (ALWAYS FIRST) ══

Read the problem, identify BOTH tables, their servers, their PRIMARY KEYS, and
the starting server. Call declare_server_layout with all those values.

HOW TO IDENTIFY THE PRIMARY KEY of each table:
  - If a field is declared as the sole primary key → key = that field alone, e.g. "pid"
  - If the table has TWO OR MORE foreign keys and no other natural identifier,
    the primary key is the COMBINATION of ALL those foreign keys.
    Example: "Sales(pid, cid, amount), Sales.pid and Sales.cid are foreign keys"
    → key_of_Sales = "pid,cid"   (NOT "pid" alone, NOT "cid" alone)
  - Never guess — read the key directly from the problem text.

The tool returns:
  name_r, server_r, key_r  ← LEFT table (at starting server) and its key
  name_s, server_s, key_s  ← RIGHT table (at remote server) and its key

After the tool call, print its "output" field VERBATIM.
Use name_r, server_r, name_s, server_s, key_r from this output in ALL
subsequent tool calls. Do NOT reinterpret — trust the tool output.

══ STEP 0b — declare_table_sizes (ALWAYS SECOND) ══

Read the problem text carefully for BOTH tables and call declare_table_sizes.

For each table, determine the unit by looking at the keyword next to the number:

  Words that mean BLOCKS:
    blocks, blоков, блоков, "size in blocks", "block count"
    → pass unit = "blocks"

  Words that mean RECORDS:
    records, rows, tuples, строк, записей, кортежей, "number of records"
    → pass unit = "records"

After the tool call, YOU MUST print the FULL "instructions" field from the tool
output verbatim in your response. Do not summarize it. The user must see:

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      SIZE TABLES GIVEN ON: <BLOCKS / RECORDS / MIXED>
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ...one line per table...

Then follow the ACTION REQUIRED line from the tool output exactly.

══ STEP 1 — compute_table_blocks (only for unit = "records") ══

Call once per table that declare_table_sizes flagged as needing conversion.

After EACH compute_table_blocks call, YOU MUST print the FULL "explanation"
field from the tool output verbatim. Do not summarize. The user must see every
line of the calculation:

    Block count for <TableName>:
      Step 1 — row size  : ...
      Step 2 — table size: ...
      Step 3 — blocks    : ...
      --> block_count = ...

Take block_count from the result. This is the value to pass to parse_semijoin_problem.

══ STEP 2 — parse_semijoin_problem ══

Pass every value. name_r, name_s, server_r, server_s MUST come from
declare_server_layout output — do not re-derive them:
  name_r              ← declare_server_layout result["name_r"]
  name_s              ← declare_server_layout result["name_s"]
  server_r            ← declare_server_layout result["server_r"]
  server_s            ← declare_server_layout result["server_s"]
  blocks_r            = integer block count of name_r  (from tool or given directly)
  blocks_s            = integer block count of name_s
  num_attrs_r         = number of attributes of name_r (e.g. Sales(pid,cid,amount) → 3)
  join_field          = the join attribute name (e.g. "pid")
  key_of_r            ← declare_server_layout result["key_r"]
                        DO NOT re-derive — copy directly from declare_server_layout output.
                        The tool computes join_field_key_in_r automatically.
  match_prob          = matching probability (0.0–1.0) or -1.0 if not given
  route_r_to_s        = "" if direct, or full path e.g. "A → B → C" if indirect
  route_s_to_r        = "" if direct, or full path e.g. "C → B → A" if indirect

The tool returns a JSON structure. Do not modify any value from that structure.

══ ROUTING — DIRECT vs INDIRECT CONNECTIONS ══

  Direct connection:  route_r_to_s = "",  route_s_to_r = ""  (1 hop, default)

  Indirect connection (must pass through intermediate servers):
    Write the FULL path with " → " between server names.
    Example: A and B not directly connected, must go through C:
      route_r_to_s = "A → C → B"   (2 hops)
      route_s_to_r = "B → C → A"   (2 hops)
    Hops are counted automatically. Transfer cost scales with hop count.

══ STEP 3 — compute_semijoin ══

Pass the field values from parse_semijoin_problem's output UNCHANGED:
  blocks_r, blocks_s, num_attrs_r, name_r, name_s,
  server_r, server_s, join_field, key_of_r,
  join_field_key_in_r, match_prob, route_r_to_s, route_s_to_r,
  hops_r_to_s, hops_s_to_r

══ STEP 3 — PRESENT THE RESULT ══

Copy the FULL "explanation" field from compute_semijoin verbatim.
Then output the summary block below.

══ STRICT OUTPUT RULES ══
• Plain text only. No LaTeX, no markdown math.
• All numbers ≥ 100 in scientific notation (e.g. 7.5 * 10^5, 10^5).
• Copy tool output EXACTLY. Do not reformat or simplify.
• Always respond in English.

• Summary block (always at the end):

---
Relational Algebra: π(<join_field>)(<name_r>) ⋉ <name_s>

Server layout: <name_r> on <server_r>,  <name_s> on <server_s>
Block counts : <name_r> = <blocks_r_sci>,  <name_s> = <blocks_s_sci>
Projection   : π(<join_field>)(<name_r>) = <proj_blocks_sci> blocks

Step 1: Start at <server_r>  [no cost]
Step 2 [project + send]: <step2>
Step 3 [join on <server_s>]: <step3>
Step 4 [send result]:    <step4>   [result = <result_blocks_sci> blocks]
Step 5 [final join]:     <step5>

Elapsed = <elapsed>
Total   = Elapsed = <total>
---
"""


# ══════════════════════════════════════════════════════════════
#  BUILD AGENT
# ══════════════════════════════════════════════════════════════

def build_agent(llm=None):
    if llm is None:
        import os
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model='gpt-4o', temperature=0,
                         api_key=os.environ.get('OPENAI_API_KEY', ''))

    llm_with_tools = llm.bind_tools(tools)

    def call_llm(state: SemiJoinAgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        return {'messages': [llm_with_tools.invoke(messages)]}

    def should_continue(state: SemiJoinAgentState):
        last = state['messages'][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return 'tools'
        return END

    graph = StateGraph(SemiJoinAgentState)
    graph.add_node('llm', call_llm)
    graph.add_node('tools', ToolNode(tools))
    graph.set_entry_point('llm')
    graph.add_conditional_edges('llm', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'llm')
    return graph.compile()

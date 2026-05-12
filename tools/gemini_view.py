"""
View-Serializability Checker — full algorithm

Steps:
  1. Is the precedence graph acyclic?
     → YES: conflict-serial → VIEW-SERIAL; topological sort gives serial order.
  2. Cycle + no blind writes?
     → NOT VIEW-SERIALIZABLE (view-serial ⟺ conflict-serial).
  3. Cycle + blind writes?
     → Enumerate all n! serial permutations, check three view conditions:
        (a) Initial read  — who reads the initial value
        (b) Reads-from    — who reads whose written value
        (c) Final write   — who writes each object last

Input format:
    ('r'|'w', transaction_id, object_name)
"""

from collections import defaultdict, deque
from itertools import permutations

def view_signature(schedule):
    """
    Compute the view-signature of a schedule from three view conditions:
      - initial_reads : frozenset {(tid, obj)} — who reads the initial value
      - reads_from    : frozenset {(writer, reader, obj)} — who reads whose value
      - final_writes  : frozenset {(tid, obj)} — who writes each object last
    """
    last_writer = {}          # obj -> tid of last writer
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

    final_writes = frozenset(last_writer.items())   # {(obj, tid)} → reorder to (tid, obj)
    final_writes = frozenset((tid, obj) for obj, tid in last_writer.items())

    return (
        frozenset(initial_reads),
        frozenset(reads_from),
        final_writes,
    )


def build_serial(transactions_order, ops_by_tid):
    """Build a serial schedule: concatenate operations in the given transaction order."""
    result = []
    for tid in transactions_order:
        result.extend(ops_by_tid[tid])
    return result


def find_view_equivalent_serial(schedule, transactions):
    """
    Enumerate all n! serial permutations of transactions.
    Returns (True, serial_schedule, order) if a view-equivalent serial is found,
    otherwise (False, None, None).
    """
    sig = view_signature(schedule)

    ops_by_tid = defaultdict(list)
    for op in schedule:
        ops_by_tid[op[1]].append(op)

    for perm in permutations(transactions):
        serial = build_serial(perm, ops_by_tid)
        if view_signature(serial) == sig:
            return True, serial, list(perm)

    return False, None, None


def analyze(schedule):
    """
    Returns a dict with the full schedule analysis.
    """
    transactions = sorted({op[1] for op in schedule})

    # --- 1. Reads-from и blind writes ---
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

    has_blind_writes = bool(blind_writes)

    # --- 2. Граф предшествования ---
    edges = set()
    n = len(schedule)

    conflict_pairs = []
    for i in range(n):
        t1, op1, obj1 = schedule[i][1], schedule[i][0], schedule[i][2]
        for j in range(i + 1, n):
            t2, op2, obj2 = schedule[j][1], schedule[j][0], schedule[j][2]
            if obj1 != obj2 or t1 == t2:
                continue
            if op1 == 'w' or op2 == 'w':
                if (t1, t2) not in edges:
                    conflict_pairs.append((schedule[i], schedule[j], t1, t2))
                edges.add((t1, t2))

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)

    # --- 3. Топологическая сортировка (алгоритм Кана) ---
    # Возвращает (order, has_cycle)
    # order — список транзакций в топологическом порядке (пуст при цикле)
    def topo_sort():
        in_degree = {t: 0 for t in transactions}
        for a, b in edges:
            in_degree[b] += 1

        queue = deque(t for t in transactions if in_degree[t] == 0)
        order = []

        while queue:
            v = queue.popleft()
            order.append(v)
            for u in sorted(adj[v]):       # sorted for determinism
                in_degree[u] -= 1
                if in_degree[u] == 0:
                    queue.append(u)

        cycle = len(order) != len(transactions)
        return order, cycle

    topo_order, cycle_found = topo_sort()

    # --- 4. Восстановление эквивалентного серийного расписания ---
    # Берём операции каждой транзакции в том порядке, как они идут
    # в исходном расписании, и конкатенируем по topo_order.
    serial_schedule = None
    if not cycle_found:
        ops_by_tid = defaultdict(list)
        for op in schedule:
            ops_by_tid[op[1]].append(op)
        serial_schedule = []
        for tid in topo_order:
            serial_schedule.extend(ops_by_tid[tid])

    # --- 5. Применяем правила ---
    serial_order = topo_order  # may be overridden below

    if not cycle_found:
        verdict = "VIEW-SERIALIZABLE"
        reason = "Precedence graph is acyclic → conflict-serial → view-serial"

    elif not has_blind_writes:
        verdict = "NOT VIEW-SERIALIZABLE"
        reason = ("Precedence graph has a cycle and no blind writes → "
                  "view-serial ⟺ conflict-serial → definitively NOT view-serial")

    else:
        # Cycle + blind writes → enumerate all n! permutations
        found, serial_schedule, serial_order = find_view_equivalent_serial(
            schedule, transactions
        )
        if found:
            verdict = "VIEW-SERIALIZABLE"
            reason = ("Precedence graph has a cycle + blind writes → "
                      "full enumeration found a view-equivalent serial schedule")
        else:
            verdict = "NOT VIEW-SERIALIZABLE"
            reason = ("Precedence graph has a cycle + blind writes → "
                      "full enumeration found no view-equivalent serial schedule")

    return {
        "verdict": verdict,
        "reason": reason,
        "transactions": transactions,
        "edges": sorted(edges),
        "cycle": cycle_found,
        "blind_writes": sorted(blind_writes),
        "reads_from": reads_from,
        "serial_order": serial_order,
        "serial_schedule": serial_schedule,
        "conflict_pairs": conflict_pairs,
    }


# ── ANSI colour helpers ──────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
MAGENTA= "\033[35m"
DIM    = "\033[2m"

def h(text, *codes):
    """Wrap text in ANSI codes."""
    return "".join(codes) + str(text) + RESET

def print_report(schedule):
    print(h("=" * 60, DIM))
    print(h("SCHEDULE:", BOLD, CYAN))
    for i, op in enumerate(schedule):
        kind = "read " if op[0] == 'r' else "write"
        color = YELLOW if op[0] == 'r' else MAGENTA
        print(f"  {i+1}. {h('T'+str(op[1]), BOLD)}: {h(kind+'('+op[2]+')', color)}")

    r = analyze(schedule)
    print()
    print(h("CONFLICT ANALYSIS:", BOLD, CYAN))
    if r['conflict_pairs']:
        for op1, op2, t1, t2 in r['conflict_pairs']:
            kind1 = h("read " if op1[0] == 'r' else "write", YELLOW if op1[0] == 'r' else MAGENTA)
            kind2 = h("read " if op2[0] == 'r' else "write", YELLOW if op2[0] == 'r' else MAGENTA)
            print(
                f"  Conflict: {h('T'+str(t1), BOLD)}:{kind1}({op1[2]})  ✕  "
                f"{h('T'+str(t2), BOLD)}:{kind2}({op2[2]})"
                f"  →  edge {h('T'+str(t1)+'→T'+str(t2), BOLD, CYAN)}"
            )
    else:
        print(f"  {h('(no conflicts)', DIM)}")
    print()
    print(h("ANALYSIS:", BOLD, CYAN))
    print(f"  Transactions : {['T'+str(t) for t in r['transactions']]}")
    print(f"  Graph edges  : {['T'+str(a)+'→T'+str(b) for a,b in r['edges']] or '(none)'}")
    cycle_val = h('Yes', RED, BOLD) if r['cycle'] else h('No', GREEN, BOLD)
    print(f"  Cycle        : {cycle_val}")
    bw = ', '.join(h('T'+str(t), YELLOW, BOLD) for t in r['blind_writes'])
    print(f"  Blind writes : {bw or '(none)'}")
    print(f"  Reads-from   : {[f'T{w}→T{rd}({o})' for w,rd,o in r['reads_from']] or '(none)'}")
    print()

    if r['verdict'].startswith("VIEW-SERIAL"):
        verdict_str = h(r['verdict'], GREEN, BOLD)
    else:
        verdict_str = h(r['verdict'], RED, BOLD)
    print(f"{h('VERDICT', BOLD)} : {verdict_str}")
    print(f"{h('REASON ', BOLD)} : {r['reason']}")

    if r['serial_schedule'] is not None:
        s = r['serial_schedule']
        print()
        order_str = h(' → '.join('T'+str(t) for t in r['serial_order']), GREEN, BOLD)
        print(f"  {h('Equivalent serial order:', BOLD, GREEN)} {order_str}")
        print()
        print(h("SERIAL SCHEDULE:", BOLD, CYAN))
        for i, op in enumerate(s):
            kind = "read " if op[0] == 'r' else "write"
            color = YELLOW if op[0] == 'r' else MAGENTA
            print(f"  {i+1}. {h('T'+str(op[1]), BOLD)}: {h(kind+'('+op[2]+')', color)}")

        rs = analyze(s)
        print()
        print(h("ANALYSIS OF SERIAL SCHEDULE:", BOLD, CYAN))
        print(f"  Transactions : {['T'+str(t) for t in rs['transactions']]}")
        print(f"  Graph edges  : {['T'+str(a)+'→T'+str(b) for a,b in rs['edges']] or '(none)'}")
        cycle_val2 = h('Yes', RED, BOLD) if rs['cycle'] else h('No', GREEN, BOLD)
        print(f"  Cycle        : {cycle_val2}")
        bw2 = ', '.join(h('T'+str(t), YELLOW, BOLD) for t in rs['blind_writes'])
        print(f"  Blind writes : {bw2 or '(none)'}")
        print(f"  Reads-from   : {[f'T{w}→T{rd}({o})' for w,rd,o in rs['reads_from']] or '(none)'}")
        print()
        print(f"  {h('View conditions (must match original):', BOLD)}")
        sig_orig = view_signature(schedule)
        sig_ser  = view_signature(s)
        labels = ["Initial reads", "Reads-from   ", "Final writes "]
        for label, o, sv in zip(labels, sig_orig, sig_ser):
            if o == sv:
                mark = h("[OK]", GREEN, BOLD)
            else:
                mark = h("[MISMATCH]", RED, BOLD)
            print(f"    {mark} {h(label, BOLD)}: {sorted(o)}")

    print(h("=" * 60, DIM))


# ---------- examples ----------

if __name__ == "__main__":

    # Example 1
    schedule1 = [
        ('r', 2, 'B'),
        ('w', 2, 'A'),
        ('r', 1, 'A'),
        ('r', 3, 'A'),
        ('w', 1, 'B'),
        ('w', 2, 'B'),
        ('w', 3, 'B'),
    ]

    # Example 2: view-serial but NOT conflict-serial (blind write saves it)
    schedule2 = [
        ('w', 1, 'A'),
        ('w', 2, 'A'),
        ('r', 3, 'A'),
        ('w', 2, 'B'),
        ('w', 3, 'B'),
    ]

    # Example 3: definitively NOT view-serial (cycle + no blind writes)
    schedule3 = [
        ('r', 1, 'A'),
        ('w', 2, 'A'),
        ('r', 2, 'B'),
        ('w', 1, 'B'),
    ]

    for s in [schedule1, schedule2, schedule3]:
        print_report(s)
        print()
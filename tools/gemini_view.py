"""
View-Serializability Checker — полный алгоритм

Этапы:
  1. Граф предшествования ацикличен?
     → ДА: conflict-serial → VIEW-SERIAL, топосорт даёт серийный порядок.
  2. Цикл + нет blind writes?
     → NOT VIEW-SERIALIZABLE (view-serial ⟺ conflict-serial).
  3. Цикл + есть blind writes?
     → Перебор всех n! серийных перестановок, проверка трёх view-условий:
        (a) Initial read  — кто читает начальное значение
        (b) Reads-from    — кто чьё значение читает
        (c) Final write   — кто последним пишет каждый объект

Формат входных данных:
    ('r'|'w', transaction_id, object_name)
"""

from collections import defaultdict, deque
from itertools import permutations

def view_signature(schedule):
    """
    Вычисляет «отпечаток» расписания из трёх view-условий:
      - initial_reads : frozenset {(tid, obj)} — кто читает начальное значение
      - reads_from    : frozenset {(writer, reader, obj)} — кто чьё значение читает
      - final_writes  : frozenset {(tid, obj)} — кто последним пишет каждый объект
    """
    last_writer = {}          # obj -> tid последней записи
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

    final_writes = frozenset(last_writer.items())   # {(obj, tid)} → удобнее (tid,obj)
    final_writes = frozenset((tid, obj) for obj, tid in last_writer.items())

    return (
        frozenset(initial_reads),
        frozenset(reads_from),
        final_writes,
    )


def build_serial(transactions_order, ops_by_tid):
    """Строит серийное расписание: конкатенация операций по заданному порядку."""
    result = []
    for tid in transactions_order:
        result.extend(ops_by_tid[tid])
    return result


def find_view_equivalent_serial(schedule, transactions):
    """
    Перебирает все n! серийных перестановок транзакций.
    Возвращает (True, serial_schedule, order) если найден view-эквивалент,
    иначе (False, None, None).
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
    Возвращает dict с полным анализом расписания.
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
            for u in sorted(adj[v]):       # sorted для детерминизма
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
    serial_order = topo_order  # может быть переопределён ниже

    if not cycle_found:
        verdict = "VIEW-SERIALIZABLE"
        reason = "Граф предшествования ацикличен → conflict-serial → view-serial"

    elif not has_blind_writes:
        verdict = "NOT VIEW-SERIALIZABLE"
        reason = ("Граф имеет цикл и нет blind writes → "
                  "view-serial ⟺ conflict-serial → точно НЕ view-serial")

    else:
        # Цикл + blind writes → полный перебор n! с проверкой view-условий
        found, serial_schedule, serial_order = find_view_equivalent_serial(
            schedule, transactions
        )
        if found:
            verdict = "VIEW-SERIALIZABLE"
            reason = (f"Граф имеет цикл + есть blind writes → "
                      f"полный перебор нашёл view-эквивалентное серийное расписание")
        else:
            verdict = "NOT VIEW-SERIALIZABLE"
            reason = ("Граф имеет цикл + есть blind writes → "
                      "полный перебор не нашёл ни одного view-эквивалентного серийного расписания")

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
    print(h("РАСПИСАНИЕ:", BOLD, CYAN))
    for i, op in enumerate(schedule):
        kind = "read " if op[0] == 'r' else "write"
        color = YELLOW if op[0] == 'r' else MAGENTA
        print(f"  {i+1}. {h('T'+str(op[1]), BOLD)}: {h(kind+'('+op[2]+')', color)}")

    r = analyze(schedule)
    print()
    print(h("АНАЛИЗ:", BOLD, CYAN))
    print(f"  Транзакции   : {['T'+str(t) for t in r['transactions']]}")
    print(f"  Рёбра графа  : {['T'+str(a)+'→T'+str(b) for a,b in r['edges']] or '(нет)'}")
    cycle_val = h('Да', RED, BOLD) if r['cycle'] else h('Нет', GREEN, BOLD)
    print(f"  Цикл в графе : {cycle_val}")
    bw = ', '.join(h('T'+str(t), YELLOW, BOLD) for t in r['blind_writes'])
    print(f"  Blind writes : {bw or '(нет)'}")
    print(f"  Reads-from   : {[f'T{w}→T{rd}({o})' for w,rd,o in r['reads_from']] or '(нет)'}")
    print()

    if r['verdict'].startswith("VIEW-SERIAL"):
        verdict_str = h(r['verdict'], GREEN, BOLD)
    else:
        verdict_str = h(r['verdict'], RED, BOLD)
    print(f"{h('ВЕРДИКТ', BOLD)} : {verdict_str}")
    print(f"{h('ПРИЧИНА', BOLD)} : {r['reason']}")

    if r['serial_schedule'] is not None:
        s = r['serial_schedule']
        print()
        order_str = h(' → '.join('T'+str(t) for t in r['serial_order']), GREEN, BOLD)
        print(f"  {h('Эквивалентный серийный порядок:', BOLD, GREEN)} {order_str}")
        print()
        print(h("СЕРИЙНОЕ РАСПИСАНИЕ:", BOLD, CYAN))
        for i, op in enumerate(s):
            kind = "read " if op[0] == 'r' else "write"
            color = YELLOW if op[0] == 'r' else MAGENTA
            print(f"  {i+1}. {h('T'+str(op[1]), BOLD)}: {h(kind+'('+op[2]+')', color)}")

        rs = analyze(s)
        print()
        print(h("АНАЛИЗ СЕРИЙНОГО РАСПИСАНИЯ:", BOLD, CYAN))
        print(f"  Транзакции   : {['T'+str(t) for t in rs['transactions']]}")
        print(f"  Рёбра графа  : {['T'+str(a)+'→T'+str(b) for a,b in rs['edges']] or '(нет)'}")
        cycle_val2 = h('Да', RED, BOLD) if rs['cycle'] else h('Нет', GREEN, BOLD)
        print(f"  Цикл в графе : {cycle_val2}")
        bw2 = ', '.join(h('T'+str(t), YELLOW, BOLD) for t in rs['blind_writes'])
        print(f"  Blind writes : {bw2 or '(нет)'}")
        print(f"  Reads-from   : {[f'T{w}→T{rd}({o})' for w,rd,o in rs['reads_from']] or '(нет)'}")
        print()
        print(f"  {h('View-условия (должны совпадать с исходным):', BOLD)}")
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


# ---------- примеры ----------

if __name__ == "__main__":

    # Пример из вопроса
    schedule1 = [
        ('r', 2, 'B'),
        ('w', 2, 'A'),
        ('r', 1, 'A'),
        ('r', 3, 'A'),
        ('w', 1, 'B'),
        ('w', 2, 'B'),
        ('w', 3, 'B'),
    ]

    # Классический view-serial но НЕ conflict-serial (blind write спасает)
    # T1: w(A); T2: w(A), w(B); T3: r(A), w(B)
    # view-equiv T1 T2 T3 если T3 читает начальное A и T2 делает last write B
    schedule2 = [
        ('w', 1, 'A'),
        ('w', 2, 'A'),
        ('r', 3, 'A'),
        ('w', 2, 'B'),
        ('w', 3, 'B'),
    ]

    # Точно НЕ view-serial: цикл + нет blind writes
    schedule3 = [
        ('r', 1, 'A'),
        ('w', 2, 'A'),
        ('r', 2, 'B'),
        ('w', 1, 'B'),
    ]

    for s in [schedule1, schedule2, schedule3]:
        print_report(s)
        print()
import itertools
from collections import defaultdict


# =========================================================
# BASIC UTILITIES
# =========================================================

def get_transactions(schedule):
    return sorted(set(op[1] for op in schedule))


# =========================================================
# CONFLICT CHECKING
# =========================================================

def is_conflict(op1, op2):
    return op1[2] == op2[2] and (op1[0] == 'w' or op2[0] == 'w')


def build_conflict_graph(schedule):
    graph = defaultdict(set)

    for i in range(len(schedule)):
        for j in range(i + 1, len(schedule)):
            o1, o2 = schedule[i], schedule[j]

            if o1[1] != o2[1] and is_conflict(o1, o2):
                graph[o1[1]].add(o2[1])

    return graph


def has_cycle(graph):
    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        stack.add(node)

        for nxt in graph[node]:
            if dfs(nxt):
                return True

        stack.remove(node)
        return False

    return any(dfs(n) for n in graph)


# =========================================================
# BLIND WRITES
# =========================================================

def has_blind_writes(schedule):
    reads = defaultdict(set)

    for typ, t, item in schedule:
        if typ == 'r':
            reads[t].add(item)

    for typ, t, item in schedule:
        if typ == 'w':
            if item not in reads[t]:
                return True

    return False


# =========================================================
# VIEW DEPENDENCIES
# =========================================================

def compute_reads_from(schedule):
    last_write = {}
    rf = []

    for typ, t, item in schedule:
        if typ == 'r':
            rf.append((t, item, last_write.get(item, "INITIAL")))
        elif typ == 'w':
            last_write[item] = t

    return rf


def compute_final_writes(schedule):
    fw = {}
    for typ, t, item in schedule:
        if typ == 'w':
            fw[item] = t
    return fw


def build_serial(schedule, order):
    grouped = defaultdict(list)

    for op in schedule:
        grouped[op[1]].append(op)

    result = []
    for t in order:
        result.extend(grouped[t])

    return result


def equivalent(s1, s2):
    return (compute_reads_from(s1) == compute_reads_from(s2) and
            compute_final_writes(s1) == compute_final_writes(s2))


# =========================================================
# VIEW EXPLANATION ENGINE
# =========================================================

def explain_view_equivalence(original, serial, order):
    print("\n==============================")
    print("VIEW SERIALIZABILITY PROOF")
    print("==============================")

    print(f"\nEquivalent serial order: {order}")

    orig_rf = compute_reads_from(original)
    ser_rf = compute_reads_from(serial)

    orig_fw = compute_final_writes(original)
    ser_fw = compute_final_writes(serial)

    print("\n--- READS-FROM ---")

    print("\nOriginal:")
    for t, item, src in orig_rf:
        print(f"  T{t} reads {item} from {src}")

    print("\nSerial:")
    for t, item, src in ser_rf:
        print(f"  T{t} reads {item} from {src}")

    print("\nResult:")
    if orig_rf == ser_rf:
        print("✔ Reads-from MATCH")
    else:
        print("❌ Reads-from MISMATCH")

    print("\n--- FINAL WRITES ---")

    print("\nOriginal:")
    for k, v in orig_fw.items():
        print(f"  {k} → T{v}")

    print("\nSerial:")
    for k, v in ser_fw.items():
        print(f"  {k} → T{v}")

    print("\nResult:")
    if orig_fw == ser_fw:
        print("✔ Final writes MATCH")
    else:
        print("❌ Final writes MISMATCH")

    print("\n--- CONCLUSION ---")
    if orig_rf == ser_rf and orig_fw == ser_fw:
        print("✔ VIEW-EQUIVALENT")
    else:
        print("❌ NOT view-equivalent")


# =========================================================
# BRUTE FORCE VIEW CHECK
# =========================================================

def is_view_serializable(schedule):
    txs = get_transactions(schedule)

    print("=== ORIGINAL SCHEDULE ===")
    for op in schedule:
        print(op)

    print("\n=== TRYING SERIAL ORDERS ===")

    for perm in itertools.permutations(txs):
        serial = build_serial(schedule, perm)

        print(f"Trying: {perm}")

        if equivalent(schedule, serial):
            print("\n✅ MATCH FOUND")
            print(f"View-serializable order: {perm}")

            explain_view_equivalence(schedule, serial, perm)

            return True, perm

    print("\n❌ NOT view-serializable")
    return False, None


# =========================================================
# HEURISTIC CHECK (optional but useful)
# =========================================================

def analyze(schedule):
    print("\n==============================")
    print("HEURISTIC ANALYSIS")
    print("==============================")

    conflict_graph = build_conflict_graph(schedule)
    conflict_cycle = has_cycle(conflict_graph)
    blind = has_blind_writes(schedule)

    print(f"Conflict cycle: {conflict_cycle}")
    print(f"Blind writes: {blind}")

    if not conflict_cycle:
        print("✔ Conflict-serializable → VIEW-serializable")

    elif not blind:
        print("✘ No blind writes + cycle → NOT view-serializable")

    else:
        print("⚠ Inconclusive → need brute force")

# -------------------------
# EXAMPLE
#from lesson
schedule =  [
    ('r', 2, 'B'),
    ('w', 2, 'A'),
    ('r', 1, 'A'),
    ('r', 3, 'A'),
    ('w', 1, 'B'),
    ('w', 2, 'B'),
    ('w', 3, 'B')
]

#2.a
schedule =  [
    ('w', 4, 'B'),
    ('r', 2, 'B'),
    ('r', 1, 'B'),
    ('w', 1, 'A'),
    ('w', 2, 'B'),
    ('r', 3, 'A'),
    ('w', 3, 'C'),
    ('r', 1, 'C'),
    ('w', 4, 'A'),
    ('r', 2, 'A'),
    ('w', 2, 'C'),
    ('r', 4, 'C'),
    ('w', 3, 'B'),
]

#2.b
schedule =  [
    ('w', 1, 'A'),
    ('w', 2, 'A'),
    ('w', 3, 'B'),
    ('w', 4, 'B'),
    ('r', 1, 'B'),
    ('r', 2, 'B'),
    ('r', 3, 'A'),
    ('r', 4, 'A')
]


analyze(schedule)
is_view_serializable(schedule)
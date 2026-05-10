from collections import defaultdict

# ── ANSI colour helpers ──────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
DIM     = "\033[2m"

def h(text, *codes):
    """Wrap text in ANSI codes."""
    return "".join(codes) + str(text) + RESET


# ── Core logic ───────────────────────────────────────────

def is_conflict(op1, op2):
    """Conflict if same data item and at least one is write."""
    return op1[2] == op2[2] and (op1[0] == 'w' or op2[0] == 'w')


def build_precedence_graph(schedule):
    graph = defaultdict(set)
    explanation = []
    n = len(schedule)

    for i in range(n):
        op1 = schedule[i]
        for j in range(i + 1, n):
            op2 = schedule[j]
            if op1[1] != op2[1]:   # different transactions
                if is_conflict(op1, op2):
                    t1, t2 = op1[1], op2[1]
                    graph[t1].add(t2)
                    explanation.append((op1, op2, t1, t2))

    return graph, explanation


def has_cycle(graph):
    visited  = set()
    rec_stack = set()

    def dfs(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        rec_stack.remove(node)
        return False

    for node in list(graph):
        if dfs(node):
            return True
    return False


def find_cycle_path(graph, transactions):
    """Return one cycle as a list of transaction ids, or None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color  = {t: WHITE for t in transactions}
    parent = {}
    result = []

    def dfs(v):
        color[v] = GRAY
        for u in graph[v]:
            if color[u] == GRAY:
                # reconstruct cycle
                path = [u, v]
                cur = v
                while cur != u and cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                result.extend(reversed(path))
                return True
            if color[u] == WHITE:
                parent[u] = v
                if dfs(u):
                    return True
        color[v] = BLACK
        return False

    for t in transactions:
        if color[t] == WHITE:
            if dfs(t):
                return result
    return None


# ── Main public function ─────────────────────────────────

def analyze_schedule(schedule):
    transactions = sorted({op[1] for op in schedule})
    graph, explanation = build_precedence_graph(schedule)
    cycle_found = has_cycle(graph)
    cycle_path  = find_cycle_path(graph, transactions) if cycle_found else None

    # Collect all edges
    edges = sorted({(t1, t2) for _, _, t1, t2 in explanation})

    # ── Print schedule ───────────────────────────────────
    print(h("=" * 60, DIM))
    print(h("SCHEDULE:", BOLD, CYAN))
    for i, op in enumerate(schedule):
        kind  = "read " if op[0] == 'r' else "write"
        color = YELLOW if op[0] == 'r' else MAGENTA
        print(f"  {i+1}. {h('T'+str(op[1]), BOLD)}: {h(kind+'('+op[2]+')', color)}")

    # ── Print conflict analysis ──────────────────────────
    print()
    print(h("CONFLICT ANALYSIS:", BOLD, CYAN))
    if explanation:
        for op1, op2, t1, t2 in explanation:
            kind1 = h("read " if op1[0]=='r' else "write", YELLOW if op1[0]=='r' else MAGENTA)
            kind2 = h("read " if op2[0]=='r' else "write", YELLOW if op2[0]=='r' else MAGENTA)
            print(
                f"  Conflict: {h('T'+str(t1), BOLD)}:{kind1}({op1[2]})  ✕  "
                f"{h('T'+str(t2), BOLD)}:{kind2}({op2[2]})"
                f"  →  edge {h('T'+str(t1)+'→T'+str(t2), BOLD, CYAN)}"
            )
    else:
        print(f"  {h('(no conflicts)', DIM)}")

    # ── Print precedence graph ───────────────────────────
    print()
    print(h("PRECEDENCE GRAPH:", BOLD, CYAN))
    if edges:
        for t1, t2 in edges:
            print(f"  {h('T'+str(t1), BOLD)} → {h('T'+str(t2), BOLD)}")
    else:
        print(f"  {h('(no edges)', DIM)}")

    cycle_val = h('Yes', RED, BOLD) if cycle_found else h('No', GREEN, BOLD)
    print(f"  Cycle in graph : {cycle_val}")
    if cycle_path:
        path_str = " → ".join(h('T'+str(t), BOLD, RED) for t in cycle_path)
        print(f"  Cycle path     : {path_str}")

    # ── Verdict ──────────────────────────────────────────
    print()
    if cycle_found:
        verdict_str = h("NOT CONFLICT-SERIALIZABLE", RED, BOLD)
        reason      = "Precedence graph contains a cycle → schedule is NOT conflict-serializable"
    else:
        verdict_str = h("CONFLICT-SERIALIZABLE", GREEN, BOLD)
        reason      = "Precedence graph is acyclic → schedule IS conflict-serializable"

        # Topological order
        from collections import deque
        in_degree = {t: 0 for t in transactions}
        for a, b in edges:
            in_degree[b] += 1
        queue = deque(t for t in transactions if in_degree[t] == 0)
        topo  = []
        adj   = defaultdict(set, {t1: graph[t1] for t1 in graph})
        while queue:
            v = queue.popleft()
            topo.append(v)
            for u in sorted(adj[v]):
                in_degree[u] -= 1
                if in_degree[u] == 0:
                    queue.append(u)

        order_str = h(' → '.join('T'+str(t) for t in topo), GREEN, BOLD)
        print(f"  {h('Equivalent serial order:', BOLD, GREEN)} {order_str}")

    print(f"{h('VERDICT', BOLD)} : {verdict_str}")
    print(f"{h('REASON ', BOLD)} : {reason}")
    print(h("=" * 60, DIM))


# ── Example usage ────────────────────────────────────────

if __name__ == "__main__":
    schedule1 = [
        ('r', 2, 'B'),
        ('w', 2, 'A'),
        ('r', 1, 'A'),
        ('r', 3, 'A'),
        ('w', 1, 'B'),
        ('w', 2, 'B'),
        ('w', 3, 'B'),
    ]

    schedule2 = [
        ('w', 1, 'A'),
        ('w', 2, 'A'),
        ('w', 3, 'B'),
        ('w', 4, 'B'),
        ('r', 1, 'B'),
        ('r', 2, 'B'),
        ('r', 3, 'A'),
        ('r', 4, 'A'),
    ]

    for s in [schedule1, schedule2]:
        analyze_schedule(s)
        print()
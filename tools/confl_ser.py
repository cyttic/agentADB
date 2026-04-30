from collections import defaultdict

def is_conflict(op1, op2):
    # conflict if same data item and at least one is write
    return op1[2] == op2[2] and (op1[0] == 'w' or op2[0] == 'w')


def build_precedence_graph(schedule):
    graph = defaultdict(set)
    explanation = []

    n = len(schedule)

    for i in range(n):
        op1 = schedule[i]
        for j in range(i + 1, n):
            op2 = schedule[j]

            # different transactions
            if op1[1] != op2[1]:
                if is_conflict(op1, op2):
                    t1 = op1[1]
                    t2 = op2[1]

                    graph[t1].add(t2)

                    explanation.append(
                        f"Conflict between {op1} and {op2} → edge T{t1} → T{t2}"
                    )

    return graph, explanation


def has_cycle(graph):
    visited = set()
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

    for node in graph:
        if dfs(node):
            return True

    return False


def analyze_schedule(schedule):
    graph, explanation = build_precedence_graph(schedule)

    print("=== Conflict Analysis ===")
    for line in explanation:
        print(line)

    print("\n=== Precedence Graph ===")
    for k, v in graph.items():
        for dest in v:
            print(f"T{k} → T{dest}")

    if has_cycle(graph):
        print("\n❌ Result: NOT conflict-serializable (cycle detected)")
    else:
        print("\n✅ Result: Conflict-serializable (no cycles)")


# -------------------------
# Example usage:

schedule = [
    ('r', 2, 'B'),
    ('w', 2, 'A'),
    ('r', 1, 'A'),
    ('r', 3, 'A'),
    ('w', 1, 'B'),
    ('w', 2, 'B'),
    ('w', 3, 'B'),
]

schedule =  [
    ('r', 2, 'B'),
    ('w', 2, 'A'),
    ('r', 1, 'A'),
    ('r', 3, 'A'),
    ('w', 1, 'B'),
    ('w', 2, 'B'),
    ('w', 3, 'B')
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

analyze_schedule(schedule)
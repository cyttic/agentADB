import itertools

def get_view_properties(schedule):
    """
    Extracts the three properties needed for view equivalence:
    1. Initial Reads
    2. Write-Read Dependencies
    3. Final Writes
    """
    transactions = sorted(list(set(op[1] for op in schedule)))
    data_items = sorted(list(set(op[2] for op in schedule)))
    
    initial_reads = {} # (item) -> trans_id
    write_read_deps = set() # (writer, reader, item)
    final_writes = {} # (item) -> trans_id
    
    # Track which transaction wrote to an item last as we iterate
    last_writer = {item: None for item in data_items}
    read_items = set()
    
    for op_type, trans_id, item in schedule:
        if op_type == 'r':
            # Rule 1: Initial Read
            if last_writer[item] is None and item not in read_items:
                initial_reads[item] = trans_id
            
            # Rule 2: Write-Read Dependency
            if last_writer[item] is not None:
                write_read_deps.add((last_writer[item], trans_id, item))
            
            read_items.add(item)
            
        elif op_type == 'w':
            # Rule 3: Final Write (will be overwritten until the end)
            last_writer[item] = trans_id
            final_writes[item] = trans_id
            
    return initial_reads, write_read_deps, final_writes

def is_view_serializable(schedule):
    # 1. Identify all unique transactions
    trans_ids = sorted(list(set(op[1] for op in schedule)))
    
    # 2. Get properties of the original schedule
    orig_props = get_view_properties(schedule)
    
    # 3. Generate all serial permutations
    # A serial schedule executes all operations of one transaction, then the next.
    for perm in itertools.permutations(trans_ids):
        serial_schedule = []
        for t_id in perm:
            # Extract all ops for this transaction in their original order
            t_ops = [op for op in schedule if op[1] == t_id]
            serial_schedule.extend(t_ops)
            
        # 4. Check if this serial permutation is view-equivalent
        if get_view_properties(serial_schedule) == orig_props:
            return True, perm
            
    return False, None

# --- Testing the provided schedule ---
schedule = [
    ('w', 1, 'A'),
    ('w', 2, 'A'),
    ('w', 3, 'B'),
    ('w', 4, 'B'),
    ('r', 1, 'B'),
    ('r', 2, 'B'),
    ('r', 3, 'A'),
    ('r', 4, 'A')
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

is_serializable, order = is_view_serializable(schedule)

if is_serializable:
    print(f"✅ The schedule is View-Serializable!")
    print(f"Equivalent Serial Order: {order}")
else:
    print("❌ The schedule is NOT View-Serializable.")
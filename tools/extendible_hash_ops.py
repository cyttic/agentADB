"""
tools/extendible_hash_ops.py
============================
Deterministic Extendible Hashing index builder for the EXTHASH agent.

Input (as the task gives it):
  - bucket capacity      : max keys a single bucket can hold
  - hash function        : h(k) = (mult*k + add) mod  (default mult=1, add=0)
  - a list of keys (numbers) inserted one by one, in order

Algorithm (exactly as specified by the task — note: MOST-significant bits):
  Each key k is hashed to h(k); h(k) is written in decimal and then in binary
  on `width` bits, where width = number of bits needed for the modulus
  (h(k) = k mod 16  ->  values 0..15  ->  4-bit keys).

  The directory is indexed by the `global depth` LEFTMOST bits of h(k).
  Each bucket carries a `local depth`. To insert k:
    * route it through the directory using the top `global depth` bits,
    * if the target bucket has room, store k,
    * if the bucket is full:
        - if local depth == global depth  -> double the directory (global +1),
        - split the bucket (local depth +1), redistributing its keys by the
          next bit from the left,
        - retry the insert.

  After every key a Mermaid diagram of the current directory + buckets is drawn,
  so the structure can be watched as it rearranges step by step.
"""

import json
import string


def _hashfn(k: int, mod: int, mult: int, add: int) -> int:
    return (mult * k + add) % mod


def _bits(h: int, width: int) -> str:
    return format(h, f'0{width}b')


def _prefix_str(h: int, width: int, depth: int) -> str:
    """The top `depth` bits of h, as a string ('' when depth == 0)."""
    if depth <= 0:
        return ''
    return _bits(h, width)[:depth]


def _bucket_name(i: int) -> str:
    """A, B, ... Z, A1, B1, ... for bucket #i (display label only)."""
    letter = string.ascii_uppercase[i % 26]
    suffix = i // 26
    return letter if suffix == 0 else f'{letter}{suffix}'


class _Bucket:
    __slots__ = ('local_depth', 'items', 'id')

    def __init__(self, local_depth, bucket_id):
        self.local_depth = local_depth
        self.items = []
        self.id = bucket_id


def _mermaid(directory, buckets, gd, width) -> str:
    """
    Render the current structure as a Mermaid flowchart, in the requested layout:
      - the DIRECTORY is one rectangle: the global depth on top, then every
        directory prefix stacked in a column underneath;
      - each BUCKET is a rectangle holding only its keys (stacked in a column);
        its LOCAL DEPTH is shown as a decimal number OUTSIDE the rectangle, as
        the label of the subgraph that wraps the bucket;
      - one arrow per directory slot points to the bucket it references.

    Labels use only digits, bit-strings and <br/> — no braces or parentheses —
    so Mermaid always parses and draws real rectangles.
    """
    if gd == 0:
        dir_lines = ['d = 0', '0']
    else:
        dir_lines = [f'd = {gd}'] + [_bits(i, gd) for i in range(len(directory))]
    dir_label = '<br/>'.join(dir_lines)

    lines = ['```mermaid', 'flowchart LR', f'    DIR["{dir_label}"]']

    # bucket nodes (in first-appearance order within the directory, then any others)
    seen = []
    for b in directory:
        if b not in seen:
            seen.append(b)
    for bk in buckets:
        if bk not in seen:
            seen.append(bk)

    node_id = {}
    for bk in seen:
        nid = f'BK{bk.id}'
        node_id[bk] = nid
        body = [str(x) for x in bk.items] or ['empty']
        # the value box holds ONLY the keys; the local depth (decimal) is the
        # label of the wrapping subgraph, so it sits OUTSIDE the rectangle
        lines.append(f'    subgraph sg{bk.id}["{bk.local_depth}"]')
        lines.append(f'        {nid}["{"<br/>".join(body)}"]')
        lines.append(f'    end')

    for i, bk in enumerate(directory):
        label = _bits(i, gd) if gd > 0 else ''
        if label:
            lines.append(f'    DIR -->|"{label}"| {node_id[bk]}')
        else:
            lines.append(f'    DIR --> {node_id[bk]}')

    lines.append('```')
    return '\n'.join(lines)


def build_extendible_hash(spec_json: str) -> str:
    """
    Build an extendible hashing index step by step and return the full worked
    report (textual trace + a Mermaid diagram after every key).

    spec_json : JSON object with keys
        capacity : int            bucket capacity (max keys per bucket)
        mod      : int            modulus M of the hash h(k) = (mult*k+add) mod M
        mult     : optional int   default 1
        add      : optional int   default 0
        numbers  : [int, ...]     the keys to insert, in order
    """
    data = json.loads(spec_json)
    if not isinstance(data, dict):
        raise ValueError('spec_json must be a JSON object')

    capacity = int(data['capacity'])
    mod = int(data['mod'])
    mult = int(data.get('mult', 1))
    add = int(data.get('add', 0))
    numbers = [int(x) for x in data.get('numbers', [])]

    if capacity < 1:
        raise ValueError('capacity must be >= 1')
    if mod < 2:
        raise ValueError('mod must be >= 2')
    if not numbers:
        raise ValueError('need at least one key in "numbers"')

    width = max(1, (mod - 1).bit_length())   # bits to represent 0..mod-1

    # ── initial state: global depth 0, a single empty bucket ──
    next_id = 0
    first = _Bucket(local_depth=0, bucket_id=next_id); next_id += 1
    buckets = [first]
    directory = [first]      # length 2^gd, holds bucket references
    gd = 0

    fn = lambda k: _hashfn(k, mod, mult, add)

    # header
    fn_str = 'k mod ' + str(mod)
    if mult != 1 or add != 0:
        fn_str = (f'{mult}*k' if mult != 1 else 'k') + (f' + {add}' if add else '') + f' mod {mod}'
    out = []
    out.append('EXTENDIBLE HASHING')
    out.append(f'Hash function : h(k) = {fn_str}   ({width}-bit keys)')
    out.append(f'Bucket capacity: {capacity}')
    out.append(f'Keys (in order): {", ".join(str(k) for k in numbers)}')
    out.append('')
    out.append('Directory is indexed by the LEFTMOST (most-significant) bits of h(k).')
    out.append('')

    MAX_SPLITS = width + 2   # guard against non-terminating splits (duplicate hashes)

    for step, k in enumerate(numbers, 1):
        h = fn(k)
        bits = _bits(h, width)
        pfx = _prefix_str(h, width, gd)
        out.append(f'==================== Step {step}: insert {k} ====================')
        out.append(f'h({k}) = {k} mod {mod} = {h} => {bits}'
                   + (f'   (prefix "{pfx}")' if pfx else '   (global depth d=0, no prefix yet)'))

        splits = 0
        while True:
            slot = h >> (width - gd) if gd > 0 else 0
            bucket = directory[slot]
            slot_lbl = _bits(slot, gd) if gd > 0 else '(root)'

            if len(bucket.items) < capacity:
                bucket.items.append(k)
                out.append(f'  -> slot "{slot_lbl}" -> bucket {_bucket_name(bucket.id)}: '
                           f'room available, store {k}.   '
                           f'{_bucket_name(bucket.id)} = {{{", ".join(str(x) for x in bucket.items)}}}')
                break

            # bucket full -> report, then split
            out.append(f'  -> slot "{slot_lbl}" -> bucket {_bucket_name(bucket.id)} = '
                       f'{{{", ".join(str(x) for x in bucket.items)}}}: '
                       f'FULL (capacity {capacity}), no room for {k}.')

            splits += 1
            if bucket.local_depth >= width or splits > MAX_SPLITS:
                out.append(f'  !! cannot split further (local depth {bucket.local_depth} = key width '
                           f'{width}); keys collide on the same {width}-bit hash. Storing {k} as overflow.')
                bucket.items.append(k)
                break

            if bucket.local_depth == gd:
                gd += 1
                directory = [directory[i >> 1] for i in range(2 ** gd)]
                out.append(f'     local depth ({bucket.local_depth}) == global depth -> '
                           f'double directory: d = {gd - 1} -> {gd}')

            # split `bucket` into itself (new bit = 0) and a new bucket (new bit = 1)
            L = bucket.local_depth + 1
            bucket.local_depth = L
            new_b = _Bucket(local_depth=L, bucket_id=next_id); next_id += 1
            buckets.append(new_b)

            moved = bucket.items
            bucket.items = []
            for it in moved:
                top_L = fn(it) >> (width - L)
                (new_b if (top_L & 1) else bucket).items.append(it)

            # repoint the directory slots that referenced the old bucket
            for i in range(len(directory)):
                if directory[i] is bucket:
                    if (i >> (gd - L)) & 1:
                        directory[i] = new_b

            out.append(f'     split bucket {_bucket_name(bucket.id)} (local depth -> {L}) '
                       f'on bit #{L} from the left:')
            for it in moved:
                p = _prefix_str(fn(it), width, L)
                tgt = new_b if (fn(it) >> (width - L)) & 1 else bucket
                out.append(f'        {it} = {_bits(fn(it), width)} -> prefix "{p}" '
                           f'-> bucket {_bucket_name(tgt.id)}')
            out.append(f'     retry insert {k} ...')
            # loop back to re-route k with the new structure

        out.append('')
        out.append(_mermaid(directory, buckets, gd, width))
        out.append('')

    # ── final summary ──
    out.append('==================== Final index ====================')
    out.append(f'Global depth d = {gd}   (directory has {len(directory)} slots)')
    shown = set()
    for bk in directory:
        if bk.id in shown:
            continue
        shown.add(bk.id)
        out.append(f'  bucket {_bucket_name(bk.id)} (local depth {bk.local_depth}): '
                   f'{{{", ".join(str(x) for x in bk.items)}}}')
    out.append('')
    out.append(_mermaid(directory, buckets, gd, width))
    return '\n'.join(out)

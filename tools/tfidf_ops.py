"""
tools/tfidf_ops.py
===================
Deterministic TF-IDF computation for the TF-IDF agent.

Input table layout (as the task gives it):
  - rows    = documents
  - columns = words / terms
  - cell    = n(d, t), the raw count of term t in document d
  - last column n(d) = number of words in document d
  - last row    N(t) = number of documents that contain term t

Formulas (exactly as specified by the task — the LLM never computes these):
  TF(d, t)            = log10( 1 + n(d, t) / n(d) )       → one per (doc, word)
  IDF(t)              = 1 / N(t)                           → one per word
  TF-IDF(document)    = Σ_t  TF(d, t) * IDF(t)             → one per document
  answer              = the document with the maximum TF-IDF

n(d) and N(t) are taken from the table when supplied; otherwise n(d) is the row
sum and N(t) is the document frequency (number of non-zero cells in a column).
"""

import json
import math

_DECIMALS = 4


def _r(v: float) -> str:
    return f'{v:.{_DECIMALS}f}'


def _parse(matrix_json: str):
    data = json.loads(matrix_json)
    if not isinstance(data, dict):
        raise ValueError('matrix_json must be a JSON object')

    documents = [str(d) for d in data.get('documents', [])]
    words = [str(w) for w in data.get('words', [])]
    counts = data.get('counts', [])

    if not documents or not words:
        raise ValueError('need at least one document and one word')
    if len(counts) != len(documents):
        raise ValueError(f'counts has {len(counts)} rows but there are {len(documents)} documents')
    for r, row in enumerate(counts):
        if len(row) != len(words):
            raise ValueError(f'row {r} ({documents[r]}) has {len(row)} cells but there are {len(words)} words')
    counts = [[float(x) for x in row] for row in counts]

    n_d_given = data.get('n_d')
    if n_d_given is not None:
        if len(n_d_given) != len(documents):
            raise ValueError('n_d length must match number of documents')
        n_d = [float(x) for x in n_d_given]
    else:
        n_d = [sum(row) for row in counts]

    N_t_given = data.get('N_t')
    if N_t_given is not None:
        if len(N_t_given) != len(words):
            raise ValueError('N_t length must match number of words')
        N_t = [float(x) for x in N_t_given]
    else:
        N_t = [sum(1 for row in counts if row[j] > 0) for j in range(len(words))]

    return documents, words, counts, n_d, N_t, (n_d_given is not None), (N_t_given is not None)


def _full_table_block(words, documents, counts, n_d, N_t) -> list[str]:
    """
    Echo the full input table with n(d) as the last column and N(t) as the last
    row, each in its proper place (a (docs+1)×(words+1) grid). The bottom-right
    corner cell (N(t) ∩ n(d)) is left blank.
    """
    col_labels = list(words) + ['n(d)']
    row_labels = list(documents) + ['N(t)']

    body = []
    for i in range(len(documents)):
        body.append([f'{counts[i][j]:g}' for j in range(len(words))] + [f'{n_d[i]:g}'])
    body.append([f'{N_t[j]:g}' for j in range(len(words))] + [''])  # N(t) row, blank corner

    row_hdr_w = max(len(l) for l in row_labels)
    col_w = {c: max(len(col_labels[c]), *(len(body[r][c]) for r in range(len(body))))
             for c in range(len(col_labels))}

    def line(label, cells):
        return label.ljust(row_hdr_w) + ' | ' + ' | '.join(
            cells[c].rjust(col_w[c]) for c in range(len(col_labels)))

    sep = '-' * row_hdr_w + '-+-' + '-+-'.join('-' * col_w[c] for c in range(len(col_labels)))
    out = ['Input table (last column n(d), last row N(t)):',
           line('', col_labels), sep]
    for r in range(len(documents)):
        out.append(line(row_labels[r], body[r]))
    out.append(sep)
    out.append(line(row_labels[-1], body[-1]))
    return out


def compute_tfidf(matrix_json: str) -> str:
    """
    Compute TF, IDF and per-document TF-IDF and return the full worked report.

    matrix_json : JSON object with keys
        documents : ["d1","d2",...]                 row labels
        words     : ["w1","w2",...]                 column labels
        counts    : [[n(d,t) ...] per document]     raw term counts
        n_d       : optional [n(d) ...]             last column (else row sums)
        N_t       : optional [N(t) ...]             last row    (else doc-frequency)
    """
    documents, words, counts, n_d, N_t, n_d_given, N_t_given = _parse(matrix_json)
    N = len(documents)

    # ── TF(d,t) = log10(1 + n(d,t)/n(d)) ──
    tf = {}
    for i in range(N):
        for j in range(len(words)):
            ratio = counts[i][j] / n_d[i] if n_d[i] else 0.0
            tf[(i, j)] = math.log10(1.0 + ratio)

    # ── IDF(t) = 1/N(t) ──
    idf = [1.0 / N_t[j] if N_t[j] else 0.0 for j in range(len(words))]

    out = []
    out.append('TF-IDF')
    out.append(f'Documents (N = {N}): {", ".join(documents)}')
    out.append(f'Words: {", ".join(words)}')
    out.append(f'n(d) source: {"given in table" if n_d_given else "computed as row sum"}'
               f'   |   N(t) source: {"given in table" if N_t_given else "computed as document frequency"}')
    out.append('')

    out += _full_table_block(words, documents, counts, n_d, N_t)
    out.append('')

    # ── Step 1: TF, one line per (document, word) ──
    out.append('Step 1 — TF(d,t) = log10(1 + n(d,t)/n(d)):')
    for i in range(N):
        for j in range(len(words)):
            ndt, nd = counts[i][j], n_d[i]
            ratio = ndt / nd if nd else 0.0
            out.append(f'  TF({documents[i]},{words[j]}) = log10(1 + {ndt:g}/{nd:g})'
                       f' = log10({1.0 + ratio:.4f}) = {_r(tf[(i, j)])}')
    out.append('')

    # ── Step 2: IDF, one line per word ──
    out.append('Step 2 — IDF(t) = 1/N(t):')
    for j in range(len(words)):
        if N_t[j] > 0:
            out.append(f'  IDF({words[j]}) = 1/{N_t[j]:g} = {_r(idf[j])}')
        else:
            out.append(f'  IDF({words[j]}) = 0 (term appears in no document)')
    out.append('')

    # ── Step 3: TF-IDF per document = Σ_t TF(d,t)*IDF(t) ──
    out.append('Step 3 — TF-IDF(document) = Σ_t TF(d,t)*IDF(t):')
    score = {}
    for i in range(N):
        products = [tf[(i, j)] * idf[j] for j in range(len(words))]
        score[i] = sum(products)
        terms = ' + '.join(f'{_r(tf[(i, j)])}*{_r(idf[j])}' for j in range(len(words)))
        nums = ' + '.join(_r(p) for p in products)
        out.append(f'  TF-IDF({documents[i]}) = {terms}')
        out.append(f'{" " * (len(documents[i]) + 12)}= {nums} = {_r(score[i])}')
    out.append('')

    # ── Maximum ──
    best = max(score.values())
    winners = [documents[i] for i in range(N) if abs(score[i] - best) < 0.5 * 10 ** (-_DECIMALS)]
    if len(winners) == 1:
        out.append(f'Maximum TF-IDF: {winners[0]} (TF-IDF = {_r(best)})')
    else:
        out.append(f'Maximum TF-IDF (tie): {", ".join(winners)} (TF-IDF = {_r(best)})')
    return '\n'.join(out)

"""Candidate-viewpoint subset construction for the |C| sensitivity study.

The planner always evaluates the pretrained 98-logit view head; a subset is
applied by masking the logits before the argmax, so no retraining is involved
and the only thing that changes is which viewpoints the robot is allowed to
visit. Viewpoints are indexed as cam_idx = row * N_COL + col on the 7x14 grid.

Spec grammar (value of --view_subset):

    all                 all 98 viewpoints (default; identical to the paper runs)
    uniform:RxC         evenly spaced R rows and C cols        -> R*C views
    narrow:RxC          R rows x C *contiguous, centered* cols -> R*C views,
                        i.e. the same budget over a reduced horizontal baseline
    random:K[:SEED]     K viewpoints drawn uniformly without replacement

Examples:
    uniform:4x6   -> 24 views spread over the full grid
    uniform:7x7   -> 49 views spread over the full grid
    narrow:7x7    -> 49 views, all 7 rows but only the middle 7 of 14 columns
    random:49:3   -> 49 random views, seed 3
"""
from __future__ import annotations

import numpy as np

N_ROW, N_COL = 7, 14
N_CAM = N_ROW * N_COL


def _evenly_spaced(n_total: int, n_take: int) -> np.ndarray:
    if not 1 <= n_take <= n_total:
        raise ValueError(f'cannot take {n_take} of {n_total}')
    return np.linspace(0, n_total - 1, n_take).round().astype(int)


def _centered_block(n_total: int, n_take: int) -> np.ndarray:
    if not 1 <= n_take <= n_total:
        raise ValueError(f'cannot take {n_take} of {n_total}')
    start = (n_total - n_take) // 2
    return np.arange(start, start + n_take)


def parse_view_subset(spec: str) -> np.ndarray:
    """Return sorted cam indices allowed by `spec`. 'all' -> arange(98)."""
    spec = (spec or 'all').strip()
    if spec in ('', 'all', 'full', str(N_CAM)):
        return np.arange(N_CAM)

    kind, _, rest = spec.partition(':')
    kind = kind.lower()

    if kind in ('uniform', 'narrow'):
        rows_s, _, cols_s = rest.lower().partition('x')
        n_rows, n_cols = int(rows_s), int(cols_s)
        rows = _evenly_spaced(N_ROW, n_rows)
        cols = (_evenly_spaced(N_COL, n_cols) if kind == 'uniform'
                else _centered_block(N_COL, n_cols))
        idx = np.array([r * N_COL + c for r in rows for c in cols], dtype=int)

    elif kind == 'random':
        parts = rest.split(':')
        k = int(parts[0])
        seed = int(parts[1]) if len(parts) > 1 else 0
        idx = np.sort(np.random.default_rng(seed).choice(
            N_CAM, size=k, replace=False))

    else:
        raise ValueError(f'unknown --view_subset spec: {spec!r}')

    return np.sort(np.unique(idx))


def describe(spec: str) -> str:
    idx = parse_view_subset(spec)
    rows = sorted({int(i) // N_COL for i in idx})
    cols = sorted({int(i) % N_COL for i in idx})
    return (f'{spec}: |C|={len(idx)}  rows={rows}  cols={cols}')


if __name__ == '__main__':
    for s in ['all', 'uniform:4x6', 'uniform:7x7', 'narrow:7x7',
              'uniform:3x4', 'random:49:0']:
        print(describe(s))

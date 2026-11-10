"""
Module for Box-Behnken response surface design generation.

Designs from Table 4 of Box & Behnken (1960), "Some new three level designs
for the study of quantitative variables", Technometrics 2(4), pp. 455-475.
"""

from itertools import combinations, product

import numpy as np

from .utils import BLOCK_COLUMN, _assert_no_mixture, _to_dataframe, _validate_factors

# Blocks of the tabulated designs, in the row order of the paper: each tuple
# lists the factors (1-based) that run a full two-level factorial at +/-1 while
# the others are held at 0. For 11 factors each block runs the half fraction of
# the 2^5 with x1 x2 x3 x4 x5 = +1 (16 runs instead of 32).
# fmt: off
_BBD_BLOCKS = {
    3: [(1, 2), (1, 3), (2, 3)],
    4: [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)],
    5: [(1, 2), (3, 4), (2, 5), (1, 3), (4, 5), (2, 3), (1, 4), (3, 5), (1, 5), (2, 4)],
    6: [(1, 2, 4), (2, 3, 5), (3, 4, 6), (1, 4, 5), (2, 5, 6), (1, 3, 6)],
    7: [(4, 5, 6), (1, 6, 7), (2, 5, 7), (1, 2, 4), (3, 4, 7), (1, 3, 5), (2, 3, 6)],
    9: [(1, 4, 7), (2, 5, 8), (3, 6, 9), (1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 5, 9), (3, 4, 8), (2, 6, 7),
        (1, 6, 8), (2, 4, 9), (3, 5, 7), (1, 4, 7), (2, 5, 8), (3, 6, 9)],
    10: [(2, 6, 7, 10), (1, 2, 5, 10), (2, 3, 7, 8), (2, 4, 6, 9), (1, 8, 9, 10),
         (3, 4, 5, 10), (1, 4, 7, 8), (3, 5, 7, 9), (1, 3, 6, 9), (4, 5, 6, 8)],
    11: [(3, 7, 8, 9, 11), (1, 4, 8, 9, 10), (2, 5, 9, 10, 11), (1, 3, 6, 10, 11), (1, 2, 4, 7, 11),
         (1, 2, 3, 5, 8), (2, 3, 4, 6, 9), (3, 4, 5, 7, 10), (4, 5, 6, 8, 11), (1, 5, 6, 7, 9),
         (2, 6, 7, 8, 10)],
    12: [(1, 2, 5, 7), (2, 3, 6, 8), (3, 4, 7, 9), (4, 5, 8, 10), (5, 6, 9, 11), (6, 7, 10, 12),
         (1, 7, 8, 11), (2, 8, 9, 12), (1, 3, 9, 10), (2, 4, 10, 11), (3, 5, 11, 12), (1, 4, 6, 12)],
    16: [(1, 2, 6, 9), (3, 4, 8, 11), (5, 10, 13, 14), (7, 12, 15, 16),
         (2, 3, 7, 10), (1, 4, 5, 12), (6, 11, 14, 15), (8, 9, 13, 16),
         (2, 5, 6, 13), (4, 7, 8, 15), (1, 9, 10, 14), (3, 11, 12, 16),
         (3, 6, 7, 14), (1, 5, 8, 16), (2, 10, 11, 15), (4, 9, 12, 13),
         (1, 3, 13, 15), (2, 4, 14, 16), (5, 7, 9, 11), (6, 8, 10, 12),
         (4, 6, 10, 16), (3, 5, 9, 15), (1, 7, 11, 13), (2, 8, 12, 14)],
}
# fmt: on

# Number of center points of the tabulated designs.
_BBD_CENTER = {3: 3, 4: 3, 5: 6, 6: 6, 7: 6, 9: 10, 10: 10, 11: 12, 12: 12, 16: 12}

# Number of factors whose tabulated blocks run a half fraction (x1 ... xm = +1).
_BBD_HALF_FRACTION = {11}

# Orthogonal blockings of the tabulated designs, {n_factors: {n_blocks: (n_groups, split)}}.
# The rows of _BBD_BLOCKS are split into n_groups consecutive groups of equal size
# (the dashed sections of Table 4). With split, each group is split further into
# the runs where the product of the nonzero coded values is +1 and those where it
# is -1. The center points are divided equally over the blocks.
# fmt: off
_BBD_BLOCKINGS = {
    4: {3: (3, False)},
    5: {2: (2, False)},
    6: {2: (1, True)},
    7: {2: (1, True)},
    9: {5: (5, False), 10: (5, True)},
    10: {2: (1, True)},
    12: {2: (1, True)},
    16: {6: (6, False), 12: (6, True)},
}
# fmt: on


def _block_runs(k, block, half_fraction):
    """Runs of one block: a two-level factorial in the block's factors, others at 0."""
    signs = np.array(list(product([-1.0, 1.0], repeat=len(block))), dtype=np.float64)
    if half_fraction:
        signs = signs[np.prod(signs, axis=1) > 0]
    runs = np.zeros((len(signs), k), dtype=np.float64)
    runs[:, [i - 1 for i in block]] = signs
    return runs


def box_behnken(factors, n_center=None, n_blocks=1):
    """
    Generate a Box-Behnken response surface design.

    The design is built from blocks of factors: within each block the
    factors run a two-level factorial at ``min``/``max`` while all other
    factors are held at mid-range, and center points are appended.
    Every factor takes the three levels ``min``, mid-range and ``max``;
    explicit factor ``levels`` are not used.  Requires at least 3
    continuous factors.

    For 3, 4, 5, 6, 7, 9, 10, 11, 12 and 16 factors this is the tabulated
    design of Box & Behnken (1960), with the blocks in the order of the
    paper: pairs of factors for 3 to 5 factors, balanced incomplete
    blocks of 3 to 5 factors for more (e.g. 48 instead of 60 non-center
    runs for 6 factors).  For 11 factors each block runs a half fraction
    of the 2^5.  For other numbers of factors there is no tabulated
    design and every pair of factors forms a block instead: still a
    valid spherical three-level design for the full quadratic model,
    with more runs.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors (at least 3).
    n_center : int or None
        Number of center points; per block in a blocked design.  ``None``
        (default) uses the value of the tabulated design: 3 for 3 or 4
        factors, 6 for 5 to 7, 10 for 9 or 10, and 12 for 11, 12 or 16,
        divided equally over the blocks in a blocked design.  Other
        numbers of factors require an explicit *n_center*.
    n_blocks : int
        Number of blocks.  Default 1 (no blocks).  The tabulated designs
        can be run in the orthogonal blocks given in the paper: 3 blocks
        for 4 factors, 2 for 5, 6, 7, 10 and 12 factors, 5 or 10 for 9
        factors, and 6 or 12 for 16 factors.  The design then gets a
        ``'Block'`` column numbered from 1.

    Returns
    -------
    Y : pd.DataFrame
        The Box-Behnken design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 3, f"Box-Behnken designs require at least 3 factors (got {k})"

    _assert_no_mixture(factors, "Box-Behnken design")
    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for a Box-Behnken design"

    options = _BBD_BLOCKINGS.get(k, {})
    assert n_blocks == 1 or n_blocks in options, (
        f"A Box-Behnken design with {k} factors can only be run in "
        + (" or ".join(str(nb) for nb in [1, *options]) + " blocks" if options else "1 block")
        + f" (got {n_blocks})"
    )

    if n_center is None:
        assert k in _BBD_CENTER, (
            f"No tabulated default number of center points for a Box-Behnken design with {k} factors; "
            f"specify n_center explicitly"
        )
        n_center = _BBD_CENTER[k] // n_blocks
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"

    rows = _BBD_BLOCKS.get(k, [(i + 1, j + 1) for i, j in combinations(range(k), 2)])
    half_fraction = k in _BBD_HALF_FRACTION
    center = np.zeros((n_center, k), dtype=np.float64)

    if n_blocks == 1:
        design = np.vstack([_block_runs(k, row, half_fraction) for row in rows] + [center])
        return _to_dataframe(design, factors)

    n_groups, split = options[n_blocks]
    per_group = len(rows) // n_groups
    groups = []
    for g in range(n_groups):
        runs = np.vstack([_block_runs(k, row, half_fraction) for row in rows[g * per_group : (g + 1) * per_group]])
        if split:
            sign = np.prod(np.where(runs != 0, runs, 1), axis=1)
            groups += [runs[sign > 0], runs[sign < 0]]
        else:
            groups.append(runs)

    parts, blocks = [], []
    for b, runs in enumerate(groups):
        parts.append(np.vstack([runs, center]))
        blocks.append(np.full(len(runs) + len(center), b + 1, dtype=np.int64))
    Y = _to_dataframe(np.vstack(parts), factors)
    Y[BLOCK_COLUMN] = np.concatenate(blocks)
    return Y

"""
Module for Plackett-Burman screening design generation.

Generators sourced from Plackett & Burman (1946), "The Design of
Optimum Multifactorial Experiments", Biometrika 33(4), pp. 305-25.
"""

import numpy as np

from .utils import _to_dataframe, _two_level_normalized, _validate_factors

# fmt: off
_PB_GENERATORS = {
    12: [
        1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1,
    ],
    20: [
        1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1, 1, 1,-1,
    ],
    24: [
        1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1,-1,-1,
    ],
    36: [
       -1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,
       -1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1,
    ],
    # N=40 uses Sylvester doubling of PB20, see _pb40_matrix().
    44: [
        1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1,
       -1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,
        1,-1, 1,-1, 1, 1,-1,
    ],
    48: [
        1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,
        1,-1,-1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,
        1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,
    ],
}

_PB28_INIT_ROWS = [
    [ 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1],
    [ 1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1, 1, 1,-1],
    [-1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1],
    [-1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1],
    [-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1],
    [-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1],
    [ 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1],
    [ 1, 1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1],
    [ 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1],
]
# fmt: on


def _hadamard(n):
    """Sylvester construction of a 2^n Hadamard matrix."""
    H = np.array([[1]], dtype=np.float64)
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    return H


def _pb40_matrix():
    """
    Build the 40 x 39 Plackett-Burman matrix via Sylvester doubling
    of the PB20 design (40 = 2 x 20).
    """
    gen = np.array(_PB_GENERATORS[20], dtype=np.float64)
    M = np.zeros((20, 19), dtype=np.float64)
    for i in range(19):
        M[i] = np.roll(gen, i)
    M[19] = -1.0

    ones_col = np.ones((20, 1), dtype=np.float64)
    H20 = np.hstack([ones_col, M])
    H40 = np.block([[H20, H20], [H20, -H20]])
    return H40[:, 1:]


def _pb28_matrix():
    """
    Build the 28 x 27 Plackett-Burman matrix via the Paley block
    construction.

    The 9 initial rows are split into three 9-column blocks (B1, B2,
    B3) which are cyclically permuted to fill the 27 x 27 core, then
    a 28th all-minus row is appended.
    """
    init = np.array(_PB28_INIT_ROWS, dtype=np.float64)
    B1 = init[:, :9]
    B2 = init[:, 9:18]
    B3 = init[:, 18:27]

    block_rows = [
        np.hstack([B1, B2, B3]),
        np.hstack([B3, B1, B2]),
        np.hstack([B2, B3, B1]),
    ]
    core = np.vstack(block_rows)
    last_row = -np.ones((1, 27), dtype=np.float64)
    return np.vstack([core, last_row])


def _pb_matrix(N):
    """
    Build the N x (N-1) Plackett-Burman matrix for a given run count.
    N must be a multiple of 4.
    """
    if N == 28:
        return _pb28_matrix()

    if N == 40:
        return _pb40_matrix()

    if N & (N - 1) == 0 and N >= 4:
        n = int(np.log2(N))
        H = _hadamard(n)
        return H[:, 1:]

    if N in _PB_GENERATORS:
        gen = np.array(_PB_GENERATORS[N], dtype=np.float64)
        rows = np.zeros((N, N - 1), dtype=np.float64)
        for i in range(N - 1):
            rows[i] = np.roll(gen, i)
        rows[N - 1] = -1.0
        return rows

    special = {28, 40}
    supported = sorted(set(special | set(_PB_GENERATORS.keys()) | {2**i for i in range(2, 10)}))
    raise ValueError(
        f"No Plackett-Burman design available for N={N}. "
        f"Supported sizes: powers of 2 (4, 8, 16, 32, ...) and "
        f"{sorted(n for n in supported if n & (n - 1) != 0)}."
    )


def plackett_burman(factors):
    """
    Generate a Plackett-Burman screening design.

    All factors must be two-level (continuous with default or two explicit
    levels, or categorical with exactly two levels).  The design size N is
    the smallest multiple of 4 that is >= k + 1, where k is the number of
    factors.

    Supported design sizes: powers of 2 (4, 8, 16, 32, ...) and
    12, 20, 24, 28, 36, 40, 44, 48.

    Parameters
    ----------
    factors : list(Factor)
        Two-level factors.

    Returns
    -------
    Y : pd.DataFrame
        The Plackett-Burman design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)

    for f in factors:
        nlev = (2 if f.levels is None else len(f.levels)) if f.is_continuous else len(f.levels)
        assert nlev == 2, f"Factor {f.name} must have exactly 2 levels for a Plackett-Burman design (has {nlev})"

    special = {28, 40}
    N_candidates = sorted(set(special | set(_PB_GENERATORS.keys()) | {2**i for i in range(2, 10)}))
    N = next((n for n in N_candidates if n >= k + 1), None)
    if N is None:
        raise ValueError(f"Too many factors ({k}) for available Plackett-Burman sizes (max {max(N_candidates) - 1})")

    mat = _pb_matrix(N)
    design = mat[:, :k]

    normalized = _two_level_normalized(design, factors)
    return _to_dataframe(normalized, factors)

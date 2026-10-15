"""
Module for Plackett-Burman screening design generation.

Generators sourced from Plackett & Burman (1946), "The Design of
Optimum Multifactorial Experiments", Biometrika 33(4), pp. 305-25.
Sizes 52 and 60 from Paley, R. E. A. C. (1933), "On orthogonal matrices",
Journal of Mathematics and Physics 12, pp. 311-320.
"""

import numpy as np

from ._paley import _paley_hadamard
from .utils import (
    _assert_no_categorical_center,
    _center_points,
    _levels,
    _to_dataframe,
    _two_level_normalized,
    _validate_factors,
)

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

# All supported run sizes N, in increasing order
_SUPPORTED_SIZES = sorted(set(_PB_GENERATORS) | {28, 40, 52, 56, 60} | {2**i for i in range(2, 10)})


def _hadamard(n):
    """Sylvester construction of a 2^n Hadamard matrix."""
    H = np.array([[1]], dtype=np.float64)
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    return H


def _sylvester_doubling(mat):
    """
    Double an N x (N-1) Plackett-Burman matrix to a 2N x (2N-1) one: with
    H = [1, mat] the corresponding Hadamard matrix, ``[[H, H], [H, -H]]``
    without its first column (e.g. 40 = 2 x 20 and 56 = 2 x 28).
    """
    H = np.hstack([np.ones((mat.shape[0], 1), dtype=np.float64), mat])
    return np.block([[H, H], [H, -H]])[:, 1:]


def _from_hadamard(H):
    """Plackett-Burman matrix from a Hadamard matrix: rows signed to a first column of +1, then dropped."""
    return (H * H[:, :1])[:, 1:]


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

    if N in (40, 56):
        return _sylvester_doubling(_pb_matrix(N // 2))

    if N == 52:
        # Paley II with q = 25 = 5^2: 52 = 2 (25 + 1)
        return _from_hadamard(_paley_hadamard(25))

    if N == 60:
        # Paley I with q = 59: 60 = 59 + 1
        return _from_hadamard(_paley_hadamard(59))

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

    raise ValueError(
        f"No Plackett-Burman design available for N={N}. "
        f"Supported sizes: powers of 2 (4, 8, 16, 32, ...) and "
        f"{sorted(n for n in _SUPPORTED_SIZES if n & (n - 1) != 0)}."
    )


def plackett_burman(factors, n_center=0):
    """
    Generate a Plackett-Burman screening design.

    All factors must be two-level (continuous without explicit levels,
    which default to 2 levels at ``min``/``max``, continuous with two
    explicit levels, or categorical with exactly two levels).  The design
    size N is the smallest supported size that is >= k + 1, where k is
    the number of factors.

    Supported design sizes: every multiple of 4 up to 64, then the
    powers of 2 (128, 256, 512).  Up to 63 factors N is therefore the
    smallest multiple of 4 that is >= k + 1; 64 to 127 factors jump to
    N = 128.  The sizes come from the cyclic generators of Plackett &
    Burman (12, 20, 24, 36, 44, 48), the Sylvester construction (powers
    of 2), Sylvester doubling (40 = 2 x 20, 56 = 2 x 28), the Paley
    block construction (28), and the Paley constructions of Hadamard
    matrices (52 with q = 25, 60 with q = 59; Paley, 1933).

    Parameters
    ----------
    factors : list(Factor)
        Two-level factors.
    n_center : int
        Number of center points to append (default 0, as for a
        screening design), with each factor at the middle of its two
        levels.  Not allowed with categorical factors; add such center
        runs manually.

    Returns
    -------
    Y : pd.DataFrame
        The Plackett-Burman design (denormalized).
    """
    _validate_factors(factors)
    _assert_no_categorical_center(factors, n_center, "Plackett-Burman design")
    k = len(factors)

    for f in factors:
        nlev = len(_levels(f, 2))
        assert nlev == 2, f"Factor {f.name} must have exactly 2 levels for a Plackett-Burman design (has {nlev})"

    N = next((n for n in _SUPPORTED_SIZES if n >= k + 1), None)
    if N is None:
        raise ValueError(f"Too many factors ({k}) for available Plackett-Burman sizes (max {_SUPPORTED_SIZES[-1] - 1})")

    mat = _pb_matrix(N)
    design = mat[:, :k]

    normalized = _two_level_normalized(design, factors)
    normalized = np.vstack([normalized, _center_points(factors, 2, n_center)])
    return _to_dataframe(normalized, factors)

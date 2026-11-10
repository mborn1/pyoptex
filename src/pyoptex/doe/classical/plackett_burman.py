"""
Module for Plackett-Burman screening design generation.

Generators sourced from Plackett & Burman (1946), "The Design of
Optimum Multifactorial Experiments", Biometrika 33(4), pp. 305-25.
Sizes 52 and 60 from Paley, R. E. A. C. (1933), "On orthogonal matrices",
Journal of Mathematics and Physics 12, pp. 311-320.
"""

import numpy as np

from ._min_aberration import _MA_GENERATORS
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
    Double an N x (N-1) Plackett-Burman matrix with columns h_j to a
    2N x (2N-1) one (e.g. 40 = 2 x 20 and 56 = 2 x 28): the columns of the
    Hadamard matrix ``[[H, H], [H, -H]]`` with H = [1, mat], ordered as the
    N-1 mirror-image columns (h_j; -h_j), then (1; -1), then the N-1
    columns (h_j; h_j).

    The rows are the original design followed by its mirror image. For up to
    N-1 factors the design is therefore the fold-over of the N-run design:
    main effects are orthogonal to all two-factor interactions (every product
    of three columns sums to 0, since each run x appears together with -x),
    and the first N runs are the original design, so it can be run in two
    stages. Taking the (h_j; h_j) columns first would instead give the N-run
    design run twice.
    """
    n = mat.shape[0]
    mirror = np.vstack([mat, -mat])
    sign = np.concatenate([np.ones(n), -np.ones(n)])[:, None]
    repeat = np.vstack([mat, mat])
    return np.hstack([mirror, sign, repeat])


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
        # Paley II with q = 25 = 5^2: 52 = 2 (25 + 1). A consequence of this construction is that
        # with k <= 24 factors runs 26 and 52 coincide (one duplicated run, harmless).
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


def _power_of_two_error(k, n_runs, default_n):
    """Error for a power-of-2 n_runs above the default size: a regular design should be used instead."""
    m = 1 << k.bit_length()  # smallest power of 2 >= k + 1
    if m < n_runs:
        why = (
            f"the first {k} columns of the {n_runs}-run Sylvester design are the {m}-run design "
            f"replicated {n_runs // m} times"
        )
    else:
        why = f"the first {k} columns of the {n_runs}-run Sylvester design form a regular resolution III fraction"

    if 2**k == n_runs:
        use = f"use the full {2**k}-run factorial of the {k} factors instead: full_factorial(factors)"
    elif (k, n_runs) in _MA_GENERATORS:
        use = (
            f"use the minimum aberration design for {k} factors in {n_runs} runs instead: "
            f"fractional_factorial(factors, min_aberration_generators(factors, n_runs={n_runs}))"
        )
    elif 2**k < n_runs:
        use = f"use full_factorial(factors) for all {2**k} runs of the {k} factors, replicated {n_runs // 2**k} times"
    else:
        use = (
            f"no catalogued minimum aberration design exists for {k} factors in {n_runs} runs; use the default "
            f"n_runs=None ({default_n} runs) or a supported size that is not a power of 2"
        )
    return ValueError(
        f"n_runs={n_runs} is a power of 2 above the default size ({default_n}) for {k} factors: {why}; {use}"
    )


def plackett_burman(factors, n_runs=None, n_center=0):
    """
    Generate a Plackett-Burman screening design.

    All factors must be two-level (continuous without explicit levels,
    which default to 2 levels at ``min``/``max``, continuous with two
    explicit levels, or categorical with exactly two levels).  By default
    the design size N is the smallest supported size that is >= k + 1,
    where k is the number of factors.

    Supported design sizes: every multiple of 4 up to 64, then the
    powers of 2 (128, 256, 512).  Up to 63 factors N is therefore the
    smallest multiple of 4 that is >= k + 1; 64 to 127 factors jump to
    N = 128.  The sizes come from the cyclic generators of Plackett &
    Burman (12, 20, 24, 36, 44, 48), the Sylvester construction (powers
    of 2), Sylvester doubling (40 = 2 x 20, 56 = 2 x 28), the Paley
    block construction (28), and the Paley constructions of Hadamard
    matrices (52 with q = 25, 60 with q = 59; Paley, 1933).

    **Column choice.**  The first k columns of the N-run design are used,
    as in the published Plackett & Burman tables.  Typical uses are:

    * k close to N - 1, for screening main effects in as few runs as
      possible;
    * more runs than factors (*n_runs*), where the unused columns serve
      as dummy factors to estimate the error, and the nonregular sizes
      (12, 20, 24, ...) allow interactions among a few active factors to
      be estimated thanks to partial aliasing: e.g. any 3 columns of the
      12-run design contain a full 2^3 factorial (Lin & Draper, 1992,
      "Projection properties of Plackett and Burman designs",
      Technometrics 34(4), pp. 423-428).

    For the nonregular sizes the choice of columns affects the amount of
    partial aliasing between main effects and two-factor interactions.
    The 40- and 56-run designs with k <= N/2 - 1 factors are the fold-over
    of the 20- and 28-run designs (the original runs followed by their
    mirror image), so main effects are orthogonal to all two-factor
    interactions.
    For power-of-2 sizes the design is a regular resolution III fraction;
    for 4 factors in 8 runs, or for more runs with a power of 2, use
    :py:func:`fractional_factorial <pyoptex.doe.classical.fractional_factorial>`
    with :py:func:`min_aberration_generators
    <pyoptex.doe.classical.min_aberration_generators>` instead, which has
    a higher resolution (e.g. resolution IV for 4 factors in 8 runs).

    Parameters
    ----------
    factors : list(Factor)
        Two-level factors.
    n_runs : int or None
        Number of runs N.  ``None`` (default) uses the smallest supported
        size that is >= k + 1.  Otherwise N must be a supported size and
        >= k + 1.  A power of 2 larger than the default size is rejected,
        since a regular fractional factorial is then better (see above).
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

    default_n = next((n for n in _SUPPORTED_SIZES if n >= k + 1), None)
    if default_n is None:
        raise ValueError(f"Too many factors ({k}) for available Plackett-Burman sizes (max {_SUPPORTED_SIZES[-1] - 1})")

    if n_runs is None:
        N = default_n
    else:
        if n_runs not in _SUPPORTED_SIZES or n_runs < k + 1:
            raise ValueError(
                f"n_runs={n_runs} is not possible for {k} factors: it must be at least {k + 1} and one of the "
                f"supported sizes {_SUPPORTED_SIZES}"
            )
        if n_runs & (n_runs - 1) == 0 and n_runs > default_n:
            raise _power_of_two_error(k, n_runs, default_n)
        N = n_runs

    design = _pb_matrix(N)[:, :k]

    normalized = _two_level_normalized(design, factors)
    normalized = np.vstack([normalized, _center_points(factors, 2, n_center)])
    return _to_dataframe(normalized, factors)

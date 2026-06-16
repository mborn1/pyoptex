"""
Module for Box-Behnken response surface design generation.
"""

from itertools import combinations, product

import numpy as np

from .utils import _to_dataframe, _validate_factors


def box_behnken(factors, n_center=3):
    """
    Generate a Box-Behnken response surface design.

    Each pair of continuous factors is varied at +/-1 (a full 2^2 block)
    while the remaining factors are held at their center (0).  Center
    points are appended.  Requires at least 3 continuous factors.

    This all-pairs construction coincides with the classical (tabulated)
    Box-Behnken design for 3, 4 and 5 factors.  For 6 or more factors it
    is still a valid spherical three-level design, but uses every factor
    pair rather than the smaller balanced-incomplete-block subset of the
    textbook design, so it has more runs than the minimal tabulated BBD.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors (at least 3).
    n_center : int
        Number of center points.

    Returns
    -------
    Y : pd.DataFrame
        The Box-Behnken design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 3, f"Box-Behnken designs require at least 3 factors (got {k})"

    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for a Box-Behnken design"

    rows = []
    for i, j in combinations(range(k), 2):
        for si, sj in product([-1.0, 1.0], repeat=2):
            row = np.zeros(k, dtype=np.float64)
            row[i] = si
            row[j] = sj
            rows.append(row)

    for _ in range(n_center):
        rows.append(np.zeros(k, dtype=np.float64))

    design = np.array(rows, dtype=np.float64)
    return _to_dataframe(design, factors)

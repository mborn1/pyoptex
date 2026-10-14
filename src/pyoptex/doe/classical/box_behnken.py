"""
Module for Box-Behnken response surface design generation.
"""

from itertools import combinations, product

import numpy as np

from .utils import _to_dataframe, _validate_factors

# Number of center points of the tabulated designs in Box & Behnken (1960),
# "Some new three level designs for the study of quantitative variables",
# Technometrics 2(4), keyed by the number of factors.
_BBD_CENTER = {3: 3, 4: 3, 5: 6, 6: 6, 7: 6}


def box_behnken(factors, n_center=None):
    """
    Generate a Box-Behnken response surface design.

    Each pair of continuous factors is varied at +/-1 (a full 2^2 block)
    while the remaining factors are held at their center (0).  Center
    points are appended.  Requires at least 3 continuous factors.
    Every factor takes the fixed levels ``min``, mid-range and ``max``;
    explicit factor ``levels`` are not used.

    This all-pairs construction coincides with the classical (tabulated)
    Box-Behnken design for 3, 4 and 5 factors.  For 6 or more factors it
    is still a valid spherical three-level design, but uses every factor
    pair rather than the smaller balanced-incomplete-block subset of the
    textbook design, so it has more runs than the minimal tabulated BBD.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors (at least 3).
    n_center : int or None
        Number of center points.  ``None`` (default) uses the value of
        the tabulated design in Box & Behnken (1960): 3 for 3 or 4
        factors and 6 for 5 to 7 factors.  Other numbers of factors
        require an explicit *n_center*.

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

    if n_center is None:
        assert k in _BBD_CENTER, (
            f"No tabulated default number of center points for a Box-Behnken design with {k} factors; "
            f"specify n_center explicitly"
        )
        n_center = _BBD_CENTER[k]
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"

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

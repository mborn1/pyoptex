"""
Module for full factorial design generation.
"""

from itertools import product

import numpy as np

from .utils import _center_points, _levels, _to_dataframe, _validate_factors


def full_factorial(factors, n_levels=2, n_center=0):
    """
    Generate a full factorial design using all factor levels.

    Continuous factors use their explicit ``levels`` when specified,
    otherwise *n_levels* equally spaced levels between ``min`` and
    ``max`` (default 2, i.e. a classical 2^k design). Categorical
    factors enumerate all their levels.

    Parameters
    ----------
    factors : list(Factor)
        The list of factors.
    n_levels : int
        Number of levels for continuous factors without explicit
        ``levels``. Use 3 for a 3^k design.
    n_center : int
        Number of center points to append (default 0, as for a
        screening design). Center points set each continuous factor to
        the middle of its level range and are repeated for every
        combination of categorical levels: with a 3-level categorical
        factor, ``n_center=2`` appends 6 center runs. Requires at least
        one continuous factor.

    Returns
    -------
    Y : pd.DataFrame
        The full factorial design (denormalized).
    """
    _validate_factors(factors)
    assert n_levels >= 2, f"n_levels must be at least 2 (got {n_levels})"

    design = np.array(list(product(*(_levels(f, n_levels) for f in factors))), dtype=np.float64)
    design = np.vstack([design, _center_points(factors, n_levels, n_center)])

    return _to_dataframe(design, factors)

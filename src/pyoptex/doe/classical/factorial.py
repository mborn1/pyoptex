"""
Module for full factorial design generation.
"""

from itertools import product

import numpy as np

from .utils import _block_scheme, _blocked_dataframe, _center_points, _levels, _to_dataframe, _validate_factors


def full_factorial(factors, n_levels=2, n_center=0, n_blocks=1):
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
        screening design); per block in a blocked design. Center points
        set each continuous factor to the middle of its level range and
        are repeated for every combination of categorical levels: with a
        3-level categorical factor, ``n_center=2`` appends 6 center runs.
        Requires at least one continuous factor.
    n_blocks : int
        Number of blocks (a power of 2), for two-level designs. The
        blocks confound the highest-order interactions possible with
        the block effect (minimum aberration blocking, e.g. ABC for a
        2^3 in 2 blocks). The design gets a ``'Block'`` column numbered
        from 1, and the confounded effects are listed in
        ``Y.attrs['confounded_with_blocks']``. Default 1 (no blocks).

    Returns
    -------
    Y : pd.DataFrame
        The full factorial design (denormalized).
    """
    _validate_factors(factors)
    assert n_levels >= 2, f"n_levels must be at least 2 (got {n_levels})"
    assert n_blocks >= 1, f"n_blocks must be at least 1 (got {n_blocks})"

    levels = [_levels(f, n_levels) for f in factors]
    design = np.array(list(product(*levels)), dtype=np.float64)
    center = _center_points(factors, n_levels, n_center)

    if n_blocks == 1:
        return _to_dataframe(np.vstack([design, center]), factors)

    for f, lv in zip(factors, levels):
        assert len(lv) == 2, f"Blocking requires two-level factors, but factor {f.name} has {len(lv)} levels"
    coded = np.array(list(product([-1, 1], repeat=len(factors))), dtype=np.int8)
    labels, confounded = _block_scheme(coded, list(range(len(factors))), n_blocks)
    return _blocked_dataframe(design, labels, center, factors, confounded)

"""
Module for supersaturated screening design generation.

Construction from Lin (1993), "A new class of supersaturated designs",
Technometrics 35(1), pp. 28-31.
"""

import numpy as np

from .plackett_burman import _pb_matrix
from .utils import _levels, _to_dataframe, _two_level_normalized, _validate_factors

# Plackett-Burman sizes whose half fractions give valid supersaturated designs.
# The Sylvester (powers of 2) and doubled (N=40) matrices are excluded: there the
# product of two columns is another column, so the half fraction would contain
# identical columns.
_SSD_SIZES = (12, 20, 24, 28, 36, 44, 48)


def _ssd_matrix(N):
    """
    The (N/2) x (N-2) supersaturated design of Lin (1993): the rows of the
    N-run Plackett-Burman design where the first (branching) column is +1,
    without that column.
    """
    mat = _pb_matrix(N)
    return mat[mat[:, 0] > 0, 1:]


def supersaturated(factors):
    """
    Generate a two-level supersaturated screening design.

    A supersaturated design has fewer runs than factors, so not all
    main effects can be estimated at once.  It is only useful under
    effect sparsity: when few of the factors are active, they can be
    identified by e.g. stepwise selection.

    The design is the construction of Lin (1993): take the N-run
    Plackett-Burman design, keep the N/2 rows where one (branching)
    column is +1, and drop that column.  This gives N/2 runs for up to
    N - 2 factors, with every column balanced.  N is the smallest
    supported size with N - 2 >= k, where k is the number of factors.

    Supported design sizes N: 12, 20, 24, 28, 36, 44, 48, i.e. 6, 10, 12,
    14, 18, 22 or 24 runs for up to 10, 18, 22, 26, 34, 42 or 46 factors.

    Parameters
    ----------
    factors : list(Factor)
        Two-level factors (continuous without explicit levels, continuous
        with two explicit levels, or categorical with two levels).

    Returns
    -------
    Y : pd.DataFrame
        The supersaturated design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)

    for f in factors:
        nlev = len(_levels(f, 2))
        assert nlev == 2, f"Factor {f.name} must have exactly 2 levels for a supersaturated design (has {nlev})"

    N = next((n for n in _SSD_SIZES if n - 2 >= k), None)
    if N is None:
        raise ValueError(f"Too many factors ({k}) for available supersaturated designs (max {_SSD_SIZES[-1] - 2})")
    assert k >= N // 2, (
        f"{k} factors with an intercept can be estimated from {N // 2} runs, so the design would not be "
        f"supersaturated; use plackett_burman instead"
    )

    design = _ssd_matrix(N)[:, :k]
    return _to_dataframe(_two_level_normalized(design, factors), factors)

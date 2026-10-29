"""
Module for Doehlert (uniform shell) design generation.

Design from Doehlert (1970), "Uniform shell designs", Journal of the Royal
Statistical Society, Series C 19(3), pp. 231-239. Scaling to the factor
ranges as in Ferreira et al. (2004), "Doehlert matrix: a chemometric tool
for analytical chemistry - review", Talanta 63(4), pp. 1061-1067.
"""

import numpy as np

from .utils import _assert_no_mixture, _to_dataframe, _validate_factors


def _regular_simplex(k):
    """
    Vertices of a regular simplex with unit edges in k dimensions: vertex j
    is the centroid of the previous vertices plus ``sqrt((j + 1) / (2j))``
    along axis j.
    """
    V = np.zeros((k + 1, k), dtype=np.float64)
    for j in range(1, k + 1):
        V[j] = V[:j].mean(axis=0)
        V[j, j - 1] = np.sqrt((j + 1) / (2 * j))
    return V


def doehlert(factors, n_center=1):
    """
    Generate a Doehlert (uniform shell) design.

    The center point plus all ``k(k + 1)`` pairwise differences between
    the vertices of a regular simplex: ``k^2 + k + 1`` runs, uniformly
    spread on a sphere around the center, which support a full quadratic
    model.  The factors get different numbers of levels: 5 for the first
    factor, 3 for the last and 7 for every factor in between (e.g. 5, 7
    and 3 for three factors), so order the factors accordingly.  A
    common choice is to put the factor that is hardest to vary last.

    Each factor's coded extremes are mapped to its ``min`` and ``max``
    (Ferreira et al., 2004), so the levels are equally spaced within the
    factor range; explicit factor ``levels`` are not used.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors (at least 2).
    n_center : int
        Number of center runs (default 1, the run count of the original
        design).

    Returns
    -------
    Y : pd.DataFrame
        The Doehlert design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 2, f"Doehlert designs require at least 2 factors (got {k})"
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"

    _assert_no_mixture(factors, "Doehlert design")
    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for a Doehlert design"

    V = _regular_simplex(k)
    shell = np.array([V[i] - V[j] for i in range(k + 1) for j in range(k + 1) if i != j], dtype=np.float64)
    shell /= np.max(np.abs(shell), axis=0)

    design = np.vstack([np.zeros((n_center, k), dtype=np.float64), shell])
    return _to_dataframe(design, factors)

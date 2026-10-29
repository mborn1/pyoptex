"""
Module for Latin Hypercube Sampling design generation.
"""

import numpy as np
from scipy.stats import qmc

from .utils import _qmc_engine, _space_filling_dataframe, _validate_space_filling_factors


def _maximin_lhs(n, k, iterations):
    """Best of *iterations* random Latin hypercubes in [0, 1]^k under the maximin criterion."""
    best_design = None
    best_min_dist = -np.inf

    for _ in range(iterations):
        design = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            perm = np.random.permutation(n)
            low = perm / n
            high = (perm + 1) / n
            design[:, j] = np.random.uniform(low, high)

        dists = np.sum((design[:, np.newaxis, :] - design[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_design = design

    return best_design


def latin_hypercube(factors, n, iterations=100, method="maximin"):
    """
    Generate a Latin Hypercube design.

    Each factor's range is split into *n* strata with one run per
    stratum; explicit factor ``levels`` are not used.  The strata are
    combined according to *method*:

    * ``'maximin'`` -- draws *iterations* random Latin hypercubes and
      returns the one with the largest minimum inter-point Euclidean
      distance (maximin criterion).
    * ``'discrepancy'`` -- a single Latin hypercube optimized for low
      centered discrepancy by
      :py:class:`scipy.stats.qmc.LatinHypercube` with
      ``optimization='random-cd'``, which gives a more uniform spread
      than the random search.  *iterations* is not used.

    Both methods draw from numpy's global random generator, so
    :py:func:`set_seed <pyoptex._seed.set_seed>` makes them reproducible.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    n : int
        Number of runs (samples).
    iterations : int
        Number of random LHS candidates to evaluate (``'maximin'`` only).
    method : str
        ``'maximin'`` (default) or ``'discrepancy'``.

    Returns
    -------
    Y : pd.DataFrame
        The Latin Hypercube design (denormalized).
    """
    _validate_space_filling_factors(factors, "Latin Hypercube sampling")
    k = len(factors)
    assert n >= 2, f"Latin Hypercube requires at least 2 runs (got {n})"
    assert iterations >= 1, f"iterations must be at least 1 (got {iterations})"

    if method == "maximin":
        unit = _maximin_lhs(n, k, iterations)
    elif method == "discrepancy":
        unit = _qmc_engine(qmc.LatinHypercube, k, optimization="random-cd").random(n)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'maximin' or 'discrepancy'.")

    return _space_filling_dataframe(unit, factors)

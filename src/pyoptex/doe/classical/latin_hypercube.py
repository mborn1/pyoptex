"""
Module for Latin Hypercube Sampling design generation.
"""

import numpy as np

from .utils import _to_dataframe, _validate_factors


def latin_hypercube(factors, n, iterations=100):
    """
    Generate a maximin Latin Hypercube design.

    Creates *iterations* random Latin Hypercube samples and returns the
    one with the largest minimum inter-point Euclidean distance
    (maximin criterion).

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    n : int
        Number of runs (samples).
    iterations : int
        Number of random LHS candidates to evaluate.

    Returns
    -------
    Y : pd.DataFrame
        The Latin Hypercube design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)
    assert n >= 2, f"Latin Hypercube requires at least 2 runs (got {n})"
    assert iterations >= 1, f"iterations must be at least 1 (got {iterations})"

    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for Latin Hypercube sampling"

    best_design = None
    best_min_dist = -np.inf

    for _ in range(iterations):
        design = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            perm = np.random.permutation(n)
            low = perm / n
            high = (perm + 1) / n
            uniform = np.random.uniform(low, high)
            design[:, j] = 2.0 * uniform - 1.0

        dists = np.sum((design[:, np.newaxis, :] - design[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_design = design

    return _to_dataframe(best_design, factors)

"""
Module containing all the generic initialization functions
"""

from math import lgamma

import numpy as np

from ._init_cy import init_single_unconstrained_cython_impl

# Candidate runs rejected in a row before the constraints are considered to leave no (or almost no) design region
MAX_REJECTED_RUNS = 10_000_000


def init_single_unconstrained(colstart, coords, run, effect_types):
    """
    Initializes a run at random. There are three possibilities:
    'continuous' which is random between (-1, 1), categorical which is
    a random level, or from coords which selects a random coordinate from
    `coords`.

    Parameters
    ----------
    colstart : np.array(1d)
        The starting column of each factor.
    coords : list(np.array(2d) or None)
        The coordinates to sample from.
    run : np.array(2d)
        Output buffer of the function. Also returned at the end.
    effect_types : np.array(1d)
        The type of each effect in case no coordinates are specified.

    Returns
    -------
    run : np.array(2d)
        The randomly sampled run.
    """
    return init_single_unconstrained_cython_impl(colstart, coords, np.ascontiguousarray(run), effect_types)


def init_design_region(
    colstart: np.ndarray, coords: list, run: np.ndarray, effect_types: np.ndarray, factors: list
) -> np.ndarray:
    """
    Samples runs from the complete design region: as
    :py:func:`init_single_unconstrained`, but with the mixture components
    drawn uniformly from the mixture region (each between its lowest and
    highest level, the implied last component at least 0). The narrowest
    components are proposed uniformly within their bounds and the others
    uniformly from the simplex of what those leave, with the split that gives
    the smallest proposal (all in the box, or all in the simplex, are the two
    extremes). Proposals outside the region are rejected, and in a mixed split
    a proposal is kept with probability (left / rest)^m, with m the number of
    simplex components: without that step the samples are not uniform.

    Parameters
    ----------
    colstart : np.array(1d)
        The start column of each factor.
    coords : list(np.array(2d) or None)
        The coordinates of each factor, see :py:func:`init_single_unconstrained`.
    run : np.array(2d)
        Output buffer of the function. Also returned at the end.
    effect_types : np.array(1d)
        The type of each effect in case no coordinates are specified.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.FactorMixin>`)
        The factors, to find the mixture components and their levels.

    Returns
    -------
    run : np.array(2d)
        The randomly sampled runs.
    """
    run = np.asarray(init_single_unconstrained(colstart, coords, run, effect_types))
    mixture = [i for i, f in enumerate(factors) if f.is_mixture]
    if not mixture:
        return run

    # The region: lower <= x <= upper and sum(x) <= 1 (the implied component at least 0)
    lower = np.array([np.min(factors[i].coords_) for i in mixture], dtype=np.float64)
    rest = 1 - float(np.sum(lower))
    if rest < -1e-6:  # The tolerance eps of the mixture constraint
        raise ValueError(f"The lowest levels of the mixture components sum to {1 - rest:.10g} > 1")
    rest = max(rest, 0)
    upper = np.minimum([np.max(factors[i].coords_) for i in mixture], lower + rest)

    # Proposal: the j narrowest components uniformly within their bounds, the others uniformly from the simplex
    # of what those leave, with j such that the proposal is smallest (j = k is the box, j = 0 the simplex)
    k, n = len(mixture), len(run)
    if rest == 0:
        run[:, colstart[mixture]] = lower
        return run
    order = np.argsort(upper - lower)
    widths = (upper - lower)[order]
    fixed = int(np.sum(widths == 0))  # Components with a single level always go in the box part
    log_volumes = [
        np.sum(np.log(widths[fixed:j])) + (k - j) * np.log(rest) - lgamma(k - j + 1) for j in range(fixed, k + 1)
    ]
    j = fixed + int(np.argmin(log_volumes))
    box, simplex, m = np.sort(order[:j]), np.sort(order[j:]), k - j

    # Rejection sampling
    x = np.empty((0, k))
    while len(x) < n:
        candidates = np.empty((n, k))
        candidates[:, box] = lower[box] + (upper - lower)[box] * np.random.rand(n, j)
        left = np.maximum(1 - np.sum(candidates[:, box], axis=1) - np.sum(lower[simplex]), 0)
        if m > 0:  # No Dirichlet draws for the pure box, which would use up random numbers
            candidates[:, simplex] = lower[simplex] + left[:, None] * np.random.dirichlet(np.ones(m + 1), n)[:, :m]
        keep = (np.sum(candidates, axis=1) <= 1 + 1e-6) & np.all(candidates <= upper, axis=1)
        if 0 < j < k:
            # The simplex part is denser where less is left: keep it with probability (left / rest)^m to stay uniform
            keep &= np.random.rand(n) < (left / rest) ** m
        x = np.concatenate((x, candidates[keep]))
    run[:, colstart[mixture]] = x[:n]
    return run


def full_factorial(colstart, coords, Y=None):
    """
    Generates a full factorial design.

    Parameters
    ----------
    colstart : np.array(1d)
        The starting columns of each factor
    coords : list(np.array(2d))
        The list of possible coordinates for each factor.
    Y : np.array(2d) or None
        The output array for the full factorial design.

    Returns
    -------
    Y : np.array(2d)
        The full factorial design.
    """
    # Initialize Y
    if Y is None:
        n = np.prod([coords[i].shape[0] for i in range(len(coords))])
        Y = np.zeros((n, colstart[-1]), dtype=np.float64)

    # Create the full factorial matrix
    tile = 1
    rep = len(Y)
    for i in range(colstart.size - 1):
        rep = int(rep / coords[i].shape[0])
        Y[:, colstart[i] : colstart[i + 1]] = np.tile(np.repeat(coords[i], rep, axis=0), (tile, 1))
        tile *= coords[i].shape[0]

    return Y

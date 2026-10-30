"""
Module containing all the generic initialization functions
"""

from math import factorial

import numpy as np

from ._init_cy import init_single_unconstrained_cython_impl


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


def init_design_region(colstart, coords, run, effect_types, factors):
    """
    Samples runs from the complete design region: as
    :py:func:`init_single_unconstrained`, but with the mixture components
    drawn uniformly from the mixture region (each between its lowest and
    highest level, the implied last component at least 0). The mixtures are
    proposed from the box of the bounds or from the simplex above the lowest
    levels, whichever is smaller, and rejected if outside the region.

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
    lower = np.array([np.min(factors[i].levels) for i in mixture], dtype=np.float64)
    rest = 1 - np.sum(lower)
    if rest < -1e-6:  # The tolerance eps of the mixture constraint
        raise ValueError(f"The lowest levels of the mixture components sum to {1 - rest:.10g} > 1")
    rest = max(rest, 0)
    upper = np.minimum([np.max(factors[i].levels) for i in mixture], lower + rest)

    # Rejection sampling from the smaller proposal
    k, n = len(mixture), len(run)
    box = np.prod(upper - lower) <= rest**k / factorial(k)
    x = np.empty((0, k))
    while len(x) < n:
        if box:
            candidates = lower + (upper - lower) * np.random.rand(n, k)
            x = np.concatenate((x, candidates[np.sum(candidates, axis=1) <= 1 + 1e-6]))
        else:
            candidates = lower + rest * np.random.dirichlet(np.ones(k + 1), n)[:, :k]
            x = np.concatenate((x, candidates[np.all(candidates <= upper, axis=1)]))
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

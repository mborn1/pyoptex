"""
Module with the helpers of the space-filling criteria (coverage and maximin), shared by the
fixed structure and the cost optimal designs.
"""

import numpy as np

from ...utils.design import decode_design


def coverage_features(Yenc, factors, effect_types, coords):
    """
    The coordinates in which the space-filling criteria measure distances, from an encoded design.
    Per factor: a continuous factor is its normalized value divided by 2, so that the range
    from its minimum to its maximum counts as 1; a mixture component is its value divided by
    the range of its levels; a categorical factor is the one-hot encoding of its level, times
    1/√2, so that the same level is at distance 0 and another level at distance 1.

    Parameters
    ----------
    Yenc : np.array(2d)
        The encoded design.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.FactorMixin>`)
        The factors of the design.
    effect_types : np.array(1d)
        The effect types of the factors (1 for continuous, the number of levels for categorical).
    coords : list(np.array(2d))
        The encoding of each factor.

    Returns
    -------
    features : np.array(2d)
        The coordinates of each run, one row per run.
    """
    Y = decode_design(Yenc, effect_types, coords)
    columns = []
    for i, f in enumerate(factors):
        if f.is_categorical:
            columns.append(np.eye(len(f.levels))[Y[:, i].astype(np.int64)] / np.sqrt(2))
        elif f.is_mixture:
            levels = np.asarray(f.levels, dtype=np.float64)
            columns.append(Y[:, i : i + 1] / (levels.max() - levels.min()))
        else:
            columns.append(Y[:, i : i + 1] / 2)
    return np.hstack(columns).astype(np.float64)


def _squared_distances(a, b):
    """The squared distances between the rows of a and the rows of b, in one matrix product, clipped at 0."""
    d2 = np.sum(np.square(a), axis=1)[:, np.newaxis] + np.sum(np.square(b), axis=1)[np.newaxis, :] - 2 * a @ b.T
    return np.maximum(d2, 0)


def coverage(runs, samples, p):
    """
    The coverage of the samples by the runs: the negative power mean (with power `p`) of the
    distance from each sample to its nearest run, so that larger is better. The score is
    always below 0, and -inf without runs.

    Parameters
    ----------
    runs : np.array(2d)
        The coordinates of the runs, as from
        :py:func:`coverage_features <pyoptex.doe.utils.coverage.coverage_features>`.
    samples : np.array(2d)
        The coordinates of the samples of the region, in the same columns.
    p : float
        The power of the mean.

    Returns
    -------
    coverage : float
        The coverage, -(mean over the samples of d^p)^(1/p).
    """
    if len(runs) == 0:
        return -np.inf
    d = np.sqrt(_squared_distances(samples, runs).min(axis=1))
    dmax = d.max()
    if dmax == 0:
        # CODEX's relative acceptance needs a metric with one sign, so a fully covered region scores just below 0
        return -1e-12
    return -float(dmax * np.mean((d / dmax) ** p) ** (1 / p))


def maximin(runs, p):
    """
    The maximin criterion of Morris and Mitchell: -φp, with φp = (sum over all pairs of runs of d^-p)^(1/p),
    so that larger is better. For a large p, φp tends to one over the smallest distance between two runs.
    Distances are floored at 1e-6. The score is -inf with fewer than two runs.

    Parameters
    ----------
    runs : np.array(2d)
        The coordinates of the runs, as from
        :py:func:`coverage_features <pyoptex.doe.utils.coverage.coverage_features>`.
    p : float
        The power.

    Returns
    -------
    maximin : float
        The negative of φp.
    """
    if len(runs) < 2:
        return -np.inf
    i, j = np.triu_indices(len(runs), 1)
    d = np.maximum(np.sqrt(_squared_distances(runs, runs)[i, j]), 1e-6)
    dmin = d.min()
    # Scaled by the smallest distance, so that every ratio is at most 1 (no overflow for a large p)
    return -float(np.sum((dmin / d) ** p) ** (1 / p) / dmin)

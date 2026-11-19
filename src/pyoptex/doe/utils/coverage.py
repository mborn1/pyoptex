"""
Module with the helpers of the coverage criterion, shared by the fixed structure
and the cost optimal designs.
"""

import numpy as np

from ...utils.design import decode_design


def coverage_features(Yenc, params):
    """
    The coordinates in which the coverage criterion measures distances, from an encoded design.
    Per factor: a continuous factor is its normalized value divided by 2, so that the range
    from its minimum to its maximum counts as 1; a mixture component is its value divided by
    the range of its levels; a categorical factor is the one-hot encoding of its level, times
    1/√2, so that the same level is at distance 0 and another level at distance 1.

    Parameters
    ----------
    Yenc : np.array(2d)
        The encoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The parameters, for the factors and their encoding.

    Returns
    -------
    features : np.array(2d)
        The coordinates of each run, one row per run.
    """
    Y = decode_design(Yenc, params.effect_types, params.coords)
    columns = []
    for i, f in enumerate(params.factors):
        if f.is_categorical:
            columns.append(np.eye(len(f.levels))[Y[:, i].astype(np.int64)] / np.sqrt(2))
        elif f.is_mixture:
            levels = np.asarray(f.levels, dtype=np.float64)
            columns.append(Y[:, i : i + 1] / (levels.max() - levels.min()))
        else:
            columns.append(Y[:, i : i + 1] / 2)
    return np.hstack(columns).astype(np.float64)


def coverage(runs, samples, p):
    """
    The coverage of the samples by the runs: the negative power mean (with power `p`) of the
    distance from each sample to its nearest run, so that larger is better.

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
    # The squared distances from every sample to every run, in one matrix product
    d2 = (
        np.sum(np.square(samples), axis=1)[:, np.newaxis]
        + np.sum(np.square(runs), axis=1)[np.newaxis, :]
        - 2 * samples @ runs.T
    )
    d = np.sqrt(np.maximum(d2.min(axis=1), 0))
    return -float(np.power(np.mean(np.power(d, p)), 1 / p))

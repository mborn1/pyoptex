"""
Module with the helpers of the space-filling criteria (coverage and maximin), shared by the
fixed structure and the cost optimal designs.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist, pdist

if TYPE_CHECKING:
    from ..cost_optimal.utils import Factor as CostOptimalFactor
    from ..fixed_structure.utils import Factor as FixedStructureFactor


def distance_map(
    factors: Sequence["FixedStructureFactor | CostOptimalFactor"], coords: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    The affine map from an encoded design to the coordinates in which the space-filling criteria
    measure distances: `(Yenc - offset) @ W`, with W block-diagonal per factor.
    Per factor: a continuous factor is its normalized value divided by 2, so that the range
    from its minimum to its maximum counts as 1; a mixture component is its value divided by
    the range of its levels; a categorical factor with k levels gets k coordinates, in which
    every two levels are at distance 1.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.utils.factor.FactorMixin>`)
        The factors of the design.
    coords : list(np.array(2d))
        The encoding of each factor, one row per level.

    Returns
    -------
    offset : np.array(1d)
        The offset, one value per column of the encoded design.
    W : np.array(2d)
        The map, one row per column of the encoded design.
    """
    offsets, blocks = [], []
    for f, C in zip(factors, coords, strict=True):
        C = np.asarray(C, dtype=np.float64)
        if f.is_categorical:
            # Centred one-hot over sqrt(2): every two levels at distance 1, whatever the encoding
            k = len(C)
            Cc = C - C.mean(axis=0)
            target = (np.eye(k) - 1 / k) / np.sqrt(2)
            T = np.linalg.lstsq(Cc, target, rcond=None)[0]
            assert np.allclose(Cc @ T, target), (
                f"The encoding of categorical factor {f.name} does not allow equal distances between its levels"
            )
            offsets.append(C.mean(axis=0))
            blocks.append(T)
        elif f.is_mixture:
            offsets.append(np.zeros(1))
            blocks.append(np.array([[1 / (C.max() - C.min())]]))
        else:
            offsets.append(np.zeros(1))
            blocks.append(np.array([[1 / 2]]))
    return np.concatenate(offsets), block_diag(*blocks)


def coverage(runs: np.ndarray, samples: np.ndarray, p: float) -> float:
    """
    The coverage of the samples by the runs: the negative power mean (with power `p`) of the
    distance from each sample to its nearest run, so that larger is better. The score is
    always below 0, and -inf without runs.

    Parameters
    ----------
    runs : np.array(2d)
        The coordinates of the runs, as from
        :py:func:`distance_map <pyoptex.doe.utils.space_filling.distance_map>`.
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
    d = cdist(samples, runs).min(axis=1)
    dmax = d.max()
    if dmax == 0:
        # CODEX's relative acceptance needs a metric with one sign, so a fully covered region scores just below 0
        return -1e-12
    return -float(dmax * np.mean((d / dmax) ** p) ** (1 / p))


def maximin(runs: np.ndarray, p: float, keep: np.ndarray | None = None) -> float:
    """
    The maximin criterion of Morris and Mitchell: -φp, with φp = (sum over all pairs of runs of d^-p)^(1/p),
    so that larger is better. For a large p, φp tends to one over the smallest distance between two runs.
    Distances are floored at 1e-6. The score is -inf without any pair.

    Pairs of two runs that the optimization cannot change (such as prior runs) are left out (`keep`):
    they cannot change, and a replicate among them would hide every other distance. Every prior run still counts separately in its distances to the new runs,
    so a replicated prior run pushes new runs away more strongly.

    Parameters
    ----------
    runs : np.array(2d)
        The coordinates of the runs, as from
        :py:func:`distance_map <pyoptex.doe.utils.space_filling.distance_map>`.
    p : float
        The power.
    keep : np.array(1d) or None
        Which pairs of runs to keep, in the order of `scipy.spatial.distance.pdist` (the pairs (i, j)
        with i < j, row by row). None keeps every pair.

    Returns
    -------
    maximin : float
        The negative of φp.
    """
    if len(runs) < 2:
        return -np.inf
    d = pdist(runs)
    if keep is not None:
        d = d[keep]
    d = np.maximum(d, 1e-6)
    if len(d) == 0:
        return -np.inf
    dmin = d.min()
    # Scaled by the smallest distance, so that every ratio is at most 1 (no overflow for a large p)
    return -float(np.sum((dmin / d) ** p) ** (1 / p) / dmin)

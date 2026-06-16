"""
Module for Central Composite Design (CCD) generation.
"""

from itertools import product

import numpy as np

from .utils import _to_dataframe, _validate_factors


def central_composite(factors, alpha="rotatable", n_center=1):
    """
    Generate a Central Composite Design (CCD).

    The design consists of a 2^k factorial portion at +/-1, 2k axial
    (star) points at +/-alpha, and center points.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    alpha : str or float
        Axial distance.

        * ``'rotatable'`` -- alpha = (2^k)^(1/4) for rotatability.
        * ``'face-centered'`` -- alpha = 1 (all points within factor range).
        * ``'inscribed'`` -- rotatable alpha, with design scaled so that
          axial points lie at +/-1 (factor extremes) and factorial points
          are pulled inward.
        * *float* -- use the given value directly.
    n_center : int
        Number of center points.

    Returns
    -------
    Y : pd.DataFrame
        The CCD (denormalized).  For ``alpha='rotatable'`` with alpha > 1
        the axial points extend beyond the factor's ``[min, max]`` range.
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 2, f"Central Composite Design requires at least 2 factors (got {k})"

    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for a CCD"

    nf = 2**k
    inscribed = False

    if isinstance(alpha, str):
        if alpha == "rotatable":
            alpha_val = nf**0.25
        elif alpha == "face-centered":
            alpha_val = 1.0
        elif alpha == "inscribed":
            alpha_val = nf**0.25
            inscribed = True
        else:
            raise ValueError(
                f"Unknown alpha type '{alpha}'. Use 'rotatable', 'face-centered', 'inscribed', or a float."
            )
    else:
        alpha_val = float(alpha)

    assert alpha_val > 0, f"alpha must be strictly positive (got {alpha_val})"

    factorial = np.array(list(product([-1.0, 1.0], repeat=k)), dtype=np.float64)

    axial = np.zeros((2 * k, k), dtype=np.float64)
    for i in range(k):
        axial[2 * i, i] = alpha_val
        axial[2 * i + 1, i] = -alpha_val

    center = np.zeros((n_center, k), dtype=np.float64)

    design = np.vstack([factorial, axial, center])

    if inscribed:
        design /= alpha_val

    return _to_dataframe(design, factors)

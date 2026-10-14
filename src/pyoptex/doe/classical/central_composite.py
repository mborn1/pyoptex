"""
Module for Central Composite Design (CCD) generation.
"""

import numpy as np

from ...utils.factor import Factor
from .fractional_factorial import fractional_factorial
from .utils import _to_dataframe, _validate_factors

# Number of center points for uniform precision of a rotatable CCD, keyed by
# (number of factors, number of cube runs). Box & Hunter (1957), "Multi-factor
# experimental designs for exploring response surfaces", Ann. Math. Statist.
# 28(1); tabulated in Montgomery, "Design and Analysis of Experiments".
_CCD_CENTER = {
    (2, 4): 5,
    (3, 8): 6,
    (4, 16): 7,
    (5, 32): 10,
    (5, 16): 6,
    (6, 64): 15,
    (6, 32): 9,
    (7, 128): 21,
    (7, 64): 14,
}


def _cube_points(factors, cube):
    """Coded +/-1 factorial (cube) portion of the CCD."""
    k = len(factors)
    if isinstance(cube, str):
        if cube == "full":
            generators = {}
        elif cube == "half":
            assert k >= 5, (
                f"A half-fraction cube needs at least 5 factors (got {k}); with fewer factors it has "
                f"resolution below V and aliases two-factor interactions"
            )
            names = [str(f.name) for f in factors]
            generators = {names[-1]: names[:-1]}
        else:
            raise ValueError(f"Unknown cube type '{cube}'. Use 'full', 'half', or a dict of generators.")
    else:
        generators = cube

    coded = [Factor(str(f.name)) for f in factors]
    return fractional_factorial(coded, generators).to_numpy(dtype=np.float64)


def central_composite(factors, alpha="rotatable", n_center=None, cube="full"):
    """
    Generate a Central Composite Design (CCD).

    The design consists of a two-level factorial (cube) portion at +/-1,
    2k axial (star) points at +/-alpha, and center points.  Every factor
    takes these fixed coded levels (five, or three when face-centered);
    explicit factor ``levels`` are not used.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    alpha : str or float
        Axial distance.

        * ``'rotatable'`` -- alpha = (n_cube)^(1/4) for rotatability,
          where n_cube is the number of cube runs.
        * ``'face-centered'`` -- alpha = 1 (all points within factor range).
        * ``'inscribed'`` -- rotatable alpha, with design scaled so that
          axial points lie at +/-1 (factor extremes) and factorial points
          are pulled inward.
        * *float* -- use the given value directly.
    n_center : int or None
        Number of center points.  ``None`` (default) uses the textbook
        value for uniform precision of Box & Hunter (1957): 5, 6, 7, 10,
        15 and 21 for a full cube with 2 to 7 factors, and 6, 9 and 14
        for a half-fraction cube with 5 to 7 factors.  Other designs
        have no tabulated default and require an explicit *n_center*.
    cube : str or dict
        Factorial portion.

        * ``'full'`` -- the full 2^k factorial (default).
        * ``'half'`` -- the resolution-k half fraction with the last
          factor generated as the product of all others (e.g.
          ``E = ABCD`` for 5 factors).  Requires at least 5 factors,
          so that the cube has resolution V or higher.
        * *dict* -- generators for a custom fraction, as in
          :py:func:`fractional_factorial <pyoptex.doe.classical.fractional_factorial>`.
          The fraction should have resolution V or higher to estimate
          the full quadratic model.

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

    factorial = _cube_points(factors, cube)
    nf = len(factorial)
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

    if n_center is None:
        assert (k, nf) in _CCD_CENTER, (
            f"No tabulated default number of center points for a CCD with {k} factors and {nf} cube runs; "
            f"specify n_center explicitly"
        )
        n_center = _CCD_CENTER[(k, nf)]
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"

    axial = np.zeros((2 * k, k), dtype=np.float64)
    for i in range(k):
        axial[2 * i, i] = alpha_val
        axial[2 * i + 1, i] = -alpha_val

    center = np.zeros((n_center, k), dtype=np.float64)

    design = np.vstack([factorial, axial, center])

    if inscribed:
        design /= alpha_val

    return _to_dataframe(design, factors)

"""
Module for Central Composite Design (CCD) generation.
"""

import numpy as np

from ...utils.factor import Factor
from .fractional_factorial import fractional_factorial
from .utils import BLOCK_COLUMN, _to_dataframe, _validate_factors

# Number of center points for uniform precision of a rotatable CCD, keyed by
# (number of factors, number of cube runs). Box, G. E. P. and Hunter, J. S. (1957),
# "Multi-factor experimental designs for exploring response surfaces", Annals of
# Mathematical Statistics 28(1), pp. 195-241, Table 3, p. 227.
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
    (8, 256): 28,
    (8, 128): 20,
    (8, 64): 13,
}

# Rotatable CCDs that block orthogonally, keyed by (number of factors, number of
# cube runs): (number of cube blocks, center points per cube block, center points
# in the axial block). Box & Hunter (1957), Table 4, p. 233.
#
# The printed table has two misprints, which do not affect these counts but are
# noted here so the derived values are not "corrected" back to the print:
# * 7 factors, full cube: the two alpha values are swapped. The orthogonal-blocking
#   alpha is 3.3333 (from these center points) and the rotatable alpha is
#   128^(1/4) = 3.3636, not the other way around.
# * 7 factors, half cube: the printed total of 80 runs should be 90
#   (8 x (8 + 1) cube runs plus 14 + 4 axial runs).
_CCD_BLOCKED = {
    (2, 4): (1, 3, 3),
    (3, 8): (2, 2, 2),
    (4, 16): (2, 2, 2),
    (5, 32): (4, 2, 4),
    (5, 16): (1, 6, 1),
    (6, 64): (8, 1, 6),
    (6, 32): (2, 4, 2),
    (7, 128): (16, 1, 11),
    (7, 64): (8, 1, 4),
}


def _cube_generators(factors, cube):
    """Generators of the factorial (cube) portion of the CCD."""
    k = len(factors)
    if not isinstance(cube, str):
        return cube
    if cube == "full":
        return {}
    if cube == "half":
        assert k >= 5, (
            f"A half-fraction cube needs at least 5 factors (got {k}); with fewer factors it has "
            f"resolution below V and aliases two-factor interactions"
        )
        names = [str(f.name) for f in factors]
        return {names[-1]: names[:-1]}
    raise ValueError(f"Unknown cube type '{cube}'. Use 'full', 'half', or a dict of generators.")


def central_composite(factors, alpha=None, n_center=None, cube="full", n_blocks=1):
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
    alpha : str, float or None
        Axial distance.  ``None`` (default) uses ``'rotatable'`` for an
        unblocked design and ``'orthogonal'`` for a blocked one.

        * ``'rotatable'`` -- alpha = (n_cube)^(1/4) for rotatability,
          where n_cube is the number of cube runs.
        * ``'face-centered'`` -- alpha = 1 (all points within factor range).
        * ``'inscribed'`` -- rotatable alpha, with design scaled so that
          axial points lie at +/-1 (factor extremes) and factorial points
          are pulled inward.
        * ``'orthogonal'`` -- for blocked designs, the alpha for which
          the blocks are orthogonal to the full quadratic model:
          ``alpha^2 = n_cube (2k + n_a) / (2 (n_cube + n_c))`` with
          ``n_c`` the center points in all cube blocks and ``n_a`` those
          in the axial block.  For the tabulated designs (see
          *n_blocks*) this is also (nearly) rotatable.
        * *float* -- use the given value directly.
    n_center : int, tuple(int, int) or None
        Number of center points.  ``None`` (default) uses the textbook
        values of Box & Hunter (1957).  Without blocks, the values for
        uniform precision (Table 3): 5, 6, 7, 10, 15, 21 and 28 for a full
        cube with 2 to 8 factors, 6, 9, 14 and 20 for a half-fraction
        cube with 5 to 8 factors, and 13 for a quarter-fraction cube
        with 8 factors.  With blocks, the values of the tabulated orthogonally
        blocked designs (see *n_blocks*).  Other designs have no
        tabulated default and require an explicit *n_center*.  In a
        blocked design an int gives the center points per block, and a
        tuple ``(per cube block, axial block)`` sets them separately.
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
    n_blocks : int
        Number of blocks.  Default 1 (no blocks).  Otherwise the axial
        points form the last block and the cube is split into
        ``n_blocks - 1`` blocks (a power of 2) by minimum aberration
        blocking, as in
        :py:func:`fractional_factorial <pyoptex.doe.classical.fractional_factorial>`.
        The design gets a ``'Block'`` column numbered from 1.  The
        tabulated designs of Box & Hunter (1957, Table 4), with default
        center points, have 2 blocks for 2 factors, 3 for 3 and 4
        factors, 5 for 5 factors, 9 for 6 factors, 17 for 7 factors, and
        with a half cube 2 for 5 factors, 3 for 6 factors and 9 for 7
        factors.  The default ``alpha='orthogonal'`` makes the blocks
        orthogonal to the full quadratic model.

    Returns
    -------
    Y : pd.DataFrame
        The CCD (denormalized).  For ``alpha='rotatable'`` with alpha > 1
        the axial points extend beyond the factor's ``[min, max]`` range.
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 2, f"Central Composite Design requires at least 2 factors (got {k})"
    assert n_blocks >= 1, f"n_blocks must be at least 1 (got {n_blocks})"

    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for a CCD"

    # Cube portion, in n_blocks - 1 blocks when blocked
    coded = [Factor(str(f.name)) for f in factors]
    n_cube_blocks = max(n_blocks - 1, 1)
    cube_df = fractional_factorial(coded, _cube_generators(factors, cube), n_blocks=n_cube_blocks)
    names = [str(f.name) for f in factors]
    factorial = cube_df[names].to_numpy(dtype=np.float64)
    cube_labels = cube_df[BLOCK_COLUMN].to_numpy() - 1 if n_cube_blocks > 1 else np.zeros(len(factorial), dtype=int)
    nf = len(factorial)

    # Center points: per cube block and in the axial block
    if n_center is None:
        if n_blocks == 1:
            assert (k, nf) in _CCD_CENTER, (
                f"No tabulated default number of center points for a CCD with {k} factors and {nf} cube runs; "
                f"specify n_center explicitly"
            )
            n_center = _CCD_CENTER[(k, nf)]
        else:
            assert _CCD_BLOCKED.get((k, nf), (None,))[0] == n_cube_blocks, (
                f"No tabulated default number of center points for a blocked CCD with {k} factors, {nf} cube "
                f"runs and {n_blocks} blocks; specify n_center explicitly"
            )
            n_center = _CCD_BLOCKED[(k, nf)][1:]
    n_cube_center, n_axial_center = n_center if isinstance(n_center, tuple) else (n_center, n_center)
    assert n_cube_center >= 0 and n_axial_center >= 0, f"n_center must be non-negative (got {n_center})"
    assert n_blocks > 1 or n_cube_center == n_axial_center, "Separate center points require a blocked design"

    if alpha is None:
        alpha = "orthogonal" if n_blocks > 1 else "rotatable"
    if isinstance(alpha, str):
        inscribed = alpha == "inscribed"
        if alpha in ("rotatable", "inscribed"):
            alpha_val = nf**0.25
        elif alpha == "face-centered":
            alpha_val = 1.0
        elif alpha == "orthogonal":
            assert n_blocks > 1, "alpha='orthogonal' requires a blocked design (n_blocks > 1)"
            n_c = n_cube_center * n_cube_blocks
            alpha_val = np.sqrt(nf * (2 * k + n_axial_center) / (2 * (nf + n_c)))
        else:
            raise ValueError(
                f"Unknown alpha type '{alpha}'. Use 'rotatable', 'face-centered', 'inscribed', 'orthogonal', "
                f"or a float."
            )
    else:
        inscribed = False
        alpha_val = float(alpha)

    assert alpha_val > 0, f"alpha must be strictly positive (got {alpha_val})"

    axial = np.zeros((2 * k, k), dtype=np.float64)
    for i in range(k):
        axial[2 * i, i] = alpha_val
        axial[2 * i + 1, i] = -alpha_val

    if n_blocks == 1:
        design = np.vstack([factorial, axial, np.zeros((n_cube_center, k), dtype=np.float64)])
        blocks = None
    else:
        parts, blocks = [], []
        for b in range(n_cube_blocks):
            runs = np.vstack([factorial[cube_labels == b], np.zeros((n_cube_center, k), dtype=np.float64)])
            parts.append(runs)
            blocks += [b + 1] * len(runs)
        parts.append(np.vstack([axial, np.zeros((n_axial_center, k), dtype=np.float64)]))
        blocks += [n_blocks] * (2 * k + n_axial_center)
        design = np.vstack(parts)

    if inscribed:
        design /= alpha_val

    Y = _to_dataframe(design, factors)
    if blocks is not None:
        Y[BLOCK_COLUMN] = np.array(blocks, dtype=np.int64)
        if n_cube_blocks > 1:
            Y.attrs["confounded_with_blocks"] = cube_df.attrs["confounded_with_blocks"]
    return Y

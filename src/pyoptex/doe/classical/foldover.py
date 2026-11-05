"""
Module for the fold-over of two-level designs.
"""

import numpy as np
import pandas as pd

from .utils import BLOCK_COLUMN, _validate_factors


def foldover(Y, factors, fold=None):
    """
    Append the fold-over (mirror image) of a design.

    The fold-over is the classical follow-up experiment after a
    resolution III fractional factorial or Plackett-Burman screening
    design: the original runs are repeated with the signs of some or all
    factors switched, and the combined design dealiases effects that
    were confounded in the first stage.

    * ``fold=None`` -- full fold-over: every factor is switched.  This
      frees all main effects from two-factor interactions (a resolution
      III design becomes resolution IV).
    * ``fold=["A"]`` -- single-factor fold-over: only A is switched.
      This frees A and all its two-factor interactions from the other
      main effects and two-factor interactions.

    A switched continuous factor is reflected about the middle of its
    range, ``x -> min + max - x`` (so ``min`` and ``max`` swap and the
    mid-range stays, e.g. for center points), where the range is the
    factor's explicit ``levels`` if given, else ``[min, max]``.  A
    switched categorical factor must have two levels, which swap.
    Factors that are not switched are copied unchanged.

    The runs of the two stages form blocks: a ``'Block'`` column is added
    with 1 for the original runs and 2 for the mirror image, or, if the
    design already has blocks, the mirror runs get new block numbers
    after the existing ones.  Other columns that are not factors (e.g. a
    measured response) are kept for the original runs and left empty
    for the mirror runs.

    References: Box, Hunter & Hunter (2005), "Statistics for
    Experimenters", 2nd ed., Wiley; Montgomery, "Design and Analysis of
    Experiments".

    Parameters
    ----------
    Y : pd.DataFrame
        The design (denormalized).
    factors : list(Factor)
        The factors of the design.
    fold : list(str) or None
        Names of the factors to switch.  ``None`` (default) switches all
        factors.

    Returns
    -------
    Y : pd.DataFrame
        The original runs followed by the mirror image, with a new index
        and a ``'Block'`` column.
    """
    assert isinstance(Y, pd.DataFrame), "Y must be a DataFrame"
    _validate_factors(factors)
    names = [str(f.name) for f in factors]
    for name in names:
        assert name in Y.columns, f"Factor {name} is not a column of the design"
    fold = names if fold is None else [str(n) for n in fold]
    assert len(fold) > 0, "Specify at least one factor to fold"
    for name in fold:
        assert name in names, f"Factor {name} in fold is not one of the factors"

    mirror = Y.copy()
    for f in factors:
        name = str(f.name)
        if name not in fold:
            continue
        assert not f.is_mixture, f"Mixture factor {name} cannot be folded"
        if f.is_continuous:
            lo, hi = (min(f.levels), max(f.levels)) if f.levels is not None else (f.min, f.max)
            mirror[name] = lo + hi - Y[name]
        else:
            assert len(f.levels) == 2, (
                f"Categorical factor {name} has {len(f.levels)} levels; only two-level factors can be folded"
            )
            swap = {f.levels[0]: f.levels[1], f.levels[1]: f.levels[0]}
            mirror[name] = Y[name].map(swap)

    other = [c for c in Y.columns if c not in names and c != BLOCK_COLUMN]
    mirror[other] = np.nan

    if BLOCK_COLUMN in Y.columns:
        original_blocks = Y[BLOCK_COLUMN].to_numpy()
        mirror[BLOCK_COLUMN] = original_blocks + original_blocks.max()
        result = pd.concat([Y, mirror], ignore_index=True)
    else:
        result = pd.concat([Y, mirror], ignore_index=True)
        result[BLOCK_COLUMN] = np.repeat(np.array([1, 2], dtype=np.int64), len(Y))
    result.attrs = {}
    return result

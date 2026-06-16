"""
Shared helpers for classical design generation.
"""

import numpy as np
import pandas as pd

from ...utils.factor import FactorMixin


def _validate_factors(factors):
    """Ensure factors is a non-empty list of FactorMixin instances."""
    assert len(factors) > 0, "At least one factor must be provided"
    for i, f in enumerate(factors):
        assert isinstance(f, FactorMixin), f"Factor {i} is not a valid Factor instance"


def _factor_encoding_params(factors):
    """
    Extract the encoding parameters from a list of factors.

    Returns the ``effect_types``, ``coords``, and ``colstart`` arrays
    required by the encoded design utilities (``full_factorial``,
    ``encode_design``, ``decode_design``).

    Parameters
    ----------
    factors : list(Factor)
        The list of factors.

    Returns
    -------
    effect_types : np.ndarray
        1 for continuous, ``len(levels)`` for categorical.
    coords : list(np.ndarray)
        Per-factor encoded coordinates (from ``Factor.coords_``).
    colstart : np.ndarray
        Column-start indices in the encoded design matrix.
    """
    effect_types = np.array(
        [1 if f.is_continuous else len(f.levels) for f in factors],
        dtype=np.int64,
    )
    coords = [f.coords_ for f in factors]
    colstart = np.concatenate(
        ([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))),
        dtype=np.int64,
    )
    return effect_types, coords, colstart


def _to_dataframe(design, factors):
    """
    Convert a normalized design matrix to a denormalized DataFrame.

    Parameters
    ----------
    design : np.ndarray
        Normalized design matrix (nruns x nfactors). Continuous values
        in [-1, 1], categorical values as integer level indices.
    factors : list(Factor)
        The factor definitions.

    Returns
    -------
    Y : pd.DataFrame
        Denormalized, decoded design.
    """
    col_names = [str(f.name) for f in factors]
    Y = pd.DataFrame(design, columns=col_names)
    for f in factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])
    return Y


def _two_level_normalized(design, factors):
    """
    Map a +/-1 coded two-level design to the normalized values expected
    by :py:meth:`denormalize <pyoptex.utils.factor.FactorMixin.denormalize>`.

    * Continuous factors take their two coded levels: the normalized
      ``levels`` when explicitly specified (so e.g. ``levels=[10, 20]``
      is honored even with the default ``min``/``max``), otherwise the
      coded extremes ``-1``/``+1`` (which denormalize to ``min``/``max``).
    * Categorical factors take their level indices ``{0, 1}``.

    Parameters
    ----------
    design : np.ndarray
        The +/-1 coded design matrix (nruns x nfactors).
    factors : list(Factor)
        The two-level factor definitions.

    Returns
    -------
    normalized : np.ndarray
        Normalized design matrix ready for ``_to_dataframe``.
    """
    normalized = np.empty_like(design, dtype=np.float64)
    for j, f in enumerate(factors):
        col = design[:, j]
        if f.is_continuous:
            if f.levels is not None:
                lo = float(f.normalize(np.array([f.levels[0]], dtype=np.float64))[0])
                hi = float(f.normalize(np.array([f.levels[1]], dtype=np.float64))[0])
            else:
                lo, hi = -1.0, 1.0
            normalized[:, j] = np.where(col < 0, lo, hi)
        else:
            normalized[:, j] = (col + 1) / 2
    return normalized

"""
Module for full factorial design generation.
"""

import numpy as np

from ...utils.design import decode_design
from ..utils.init import full_factorial as _full_factorial_encoded
from .utils import _factor_encoding_params, _to_dataframe, _validate_factors


def full_factorial(factors, n_center=0):
    """
    Generate a full factorial design using all factor levels.

    For continuous factors the levels used are the factor's ``coords_``:
    the ``levels`` attribute when specified, otherwise ``[-1, 0, 1]``
    (three-level). For categorical factors all levels are enumerated.

    Parameters
    ----------
    factors : list(Factor)
        The list of factors.
    n_center : int
        Number of center points to append. Center points set each
        continuous factor to 0 (mid-range) and each categorical
        factor to its first level.

    Returns
    -------
    Y : pd.DataFrame
        The full factorial design (denormalized).
    """
    _validate_factors(factors)

    effect_types, coords, colstart = _factor_encoding_params(factors)

    Y_enc = _full_factorial_encoded(colstart, coords)
    Y_dec = decode_design(Y_enc, effect_types, coords=coords)

    if n_center > 0:
        center = np.zeros((n_center, len(factors)), dtype=np.float64)
        Y_dec = np.vstack([Y_dec, center])

    return _to_dataframe(Y_dec, factors)

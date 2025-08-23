"""
Module for the generic coordinate-exchange algorithm.
"""

import numpy as np

from ..._profile import profile
from .validation import validate_state
from .utils import State
from ._optimize_cy import _optimize_cython_impl


@profile
def optimize(params, max_it=10000, validate=False, eps=1e-4):
    """
    Optimize a model iteratively using the coordinate-exchange algorithm.
    Only specific groups at each level are updated to allow design augmentation.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The parameters of the design generation.
    max_it : int
        The maximum number of iterations to prevent potential infinite loops.
    validate : bool
        Whether to validate the update formulas at each step. This is used
        to debug.
    eps : float
        A relative increase of at least epsilon is required to accept the change.

    Returns
    -------
    Y : np.array(2d)
        The generated design
    state : :py:class:`State <pyoptex.doe.fixed_structure.utils.State>`
        The state according to the generated design.
    """
    # Call the Cython implementation
    # It handles initialization, looping, and Python callbacks internally
    Y, X, metric = _optimize_cython_impl(params, max_it, eps)

    # Construct the final state object
    state = State(Y, X, metric)

    # Optional final validation (can be done here)
    if validate:
        validate_state(state, params)

    # Return the design and the final state
    return Y, state

"""
Module for the generic coordinate-exchange algorithm.
"""

import numpy as np

from ..._profile import profile
from .validation import validate_state
from .utils import State


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
    # Initialize a design
    _, (Y, X) = params.fn.init(params)

    # Initialization
    params.fn.metric.init(Y, X, params)
    metric = params.fn.metric.call(Y, X, params)
    state = State(Y, X, metric)
    if validate:
        validate_state(state, params)

    # Make sure we are not stuck in finite loop
    for it in range(max_it):
        # Start with updated false
        updated = False

        # Loop over all factors
        for i in range(params.effect_types.size):

            # Extract factor level parameters
            level = params.effect_levels[i]

            # Loop over all run-groups
            for grp in params.grps[i]:

                # Generate coordinates
                possible_coords = params.coords[i]
                cols = slice(params.colstart[i], params.colstart[i+1])
                if level == 0:
                    runs = np.array([grp])
                else:
                    runs = np.flatnonzero(params.Zs[level-1] == grp)

                # Extract current coordinate (as best)
                Ycoord = np.copy(state.Y[runs[0], cols])
                Xrows = np.copy(state.X[runs])
                co = Ycoord

                # Loop over possible new coordinates
                for new_coord in possible_coords:

                    # Short-circuit original coordinates
                    if np.any(new_coord != co):

                        # Check validity of new coordinates
                        state.Y[runs, cols] = new_coord

                        # Validate whether to check the coordinate
                        if not np.any(params.fn.constraints(state.Y[runs])):
                            # Update the X
                            state.X[runs] = params.fn.Y2X(state.Y[runs])

                            # Check if the update is accepted
                            new_metric = params.fn.metric.call(state.Y, state.X, params)
                            up = new_metric - state.metric

                            # New best design
                            if ((state.metric == 0 or np.isinf(state.metric)) and up > 0) or up / np.abs(state.metric) > eps:
                                # Store the best coordinates
                                Ycoord = new_coord
                                Xrows = np.copy(state.X[runs])
                                metric = new_metric
                                state = State(state.Y, state.X, metric)

                                # Validate the state
                                if validate:
                                    validate_state(state, params)

                                # Set update
                                updated = True

                    # Set the correct coordinates
                    state.Y[runs, cols] = Ycoord
                    state.X[runs] = Xrows

                # Validate the state
                if validate:
                    validate_state(state, params)
             
        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    validate_state(state, params)
    return Y, state

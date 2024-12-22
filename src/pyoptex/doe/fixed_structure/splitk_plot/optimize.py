"""
Module for the split^k-plot coordinate-exchange algorithm.
"""

import numpy as np
import warnings

from ...._profile import profile
from ..validation import validate_state
from ..utils import State
from .utils import Update


@profile
def optimize(params, max_it=10000, validate=False, eps=1e-4):
    """
    Optimize a model iteratively using the coordinate-exchange algorithm.
    Only specific groups at each level are updated to allow design augmentation.

    Parameters
    ----------
    params : `Parametes <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
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
    state : `State <pyoptex.doe.fixed_structure.utils.State>`
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
            jmp = params.thetas[level]

            # Loop over all run-groups
            for grp in params.grps[i]:

                # Generate coordinates
                possible_coords = params.coords[i]
                cols = slice(params.colstart[i], params.colstart[i+1])
                runs = slice(grp*jmp, (grp+1)*jmp)

                # Extract current coordinate (as best)
                Ycoord = np.copy(state.Y[runs.start, cols])
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
                            update = Update(level, grp, runs, cols, new_coord, Ycoord, Xrows, state.metric)
                            up = params.fn.metric.update(state.Y, state.X, params, update)

                            # New best design
                            if ((state.metric == 0 or np.isinf(state.metric)) and up > 0) or up / np.abs(state.metric) > eps:
                                # Mark the metric as accepted
                                params.fn.metric.accepted(state.Y, state.X, params, update)

                                # Store the best coordinates
                                Ycoord = new_coord
                                Xrows = np.copy(state.X[runs])
                                if np.isinf(up):
                                    metric = params.fn.metric.call(state.Y, state.X, params)
                                else:
                                    metric = state.metric + up
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
            
        # Recompute metric for numerical stability
        old_metric = state.metric
        state = state._replace(metric=params.fn.metric.call(state.Y, state.X, params))
        if ((state.metric == 0 and old_metric > 0) or (np.isinf(state.metric) and ~np.isinf(state.metric))) and params.compute_update:
            warnings.warn('Update formulas are very unstable for this problem, try rerunning without update formulas', RuntimeWarning)
            
        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    validate_state(state, params)
    return Y, state

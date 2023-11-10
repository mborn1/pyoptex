import numba
import numpy as np

from .utils import Update, State
from .formulas import compute_update_UD
from .init import initialize_feasible
from .validation import validate_state, validate_UD

def optimize(params, max_it=10000, validate=False):
    """
    Optimize a model iteratively using the coordinate exchange algorithm.
    Only specific groups at each level are updated to allow design augmentation.

    Parameters
    ----------

    Returns
    -------
    """
    # Initialize a design
    _, (Y, X) = initialize_feasible(params)

    # Initialization
    params.fn.metric.init(Y, X, params)
    state = State(Y, X, params.fn.metric.call(Y, X))

    # Make sure we are not stuck in finite loop
    for it in range(max_it):
        # Start with updated false
        updated = False

        # Loop over all factors
        for i in range(params.effect_types.size):

            # Extract factor level parameters
            level = params.effect_levels[i]
            cat_lvl = params.effect_types[i]
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
                            # Compute new X
                            Xi_star = params.Y2X(state.Y[runs])

                            # Compute updates
                            U, D = compute_update_UD(
                                level, grp, Xi_star, state.X,
                                params.plot_sizes, params.c, params.thetas, params.thetas_inv
                            )

                            # Check for validation
                            if validate:
                                validate_UD(U, D, Xi_star, runs, state, params)

                            # Update the X
                            state.X[runs] = Xi_star

                            # Check if the update is accepted
                            update = Update(level, grp, runs, cols, new_coord, Ycoord, state.metric, U, D)
                            up = params.fn.metric.update(state.Y, state.X, update)

                            # New best design
                            if up > 0:
                                # Mark the metric as accepted
                                params.fn.metric.accepted(state.Y, state.X, update)

                                # Store the best coordinates
                                Ycoord = new_coord
                                Xrows = np.copy(state.X[runs])
                                state = State(state.Y, state.X, state.metric + up)

                                # Validate the state
                                if validate:
                                    validate_state(state, params)

                                # Set update
                                updated = True
                            else:
                                # Reset the current coordinates
                                state.Y[runs, cols] = Ycoord
                                state.X[runs] = Xrows
            
        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    return Y, state

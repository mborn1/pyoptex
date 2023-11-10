import numba
import numpy as np

from .utils import Update
from .formulas import compute_update_UD
from .init import initialize_feasible

def optimize(params, max_it=10000):
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
    params.fn.metric.init(params, Y, X)

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
                Ycoord = np.copy(Y[runs.start, cols])
                Xrows = np.copy(X[runs])
                co = Ycoord

                # Loop over possible new coordinates
                for new_coord in possible_coords:

                    # Short-circuit original coordinates
                    if np.any(new_coord != co):

                        # Check validity of new coordinates
                        Y[runs, cols] = new_coord

                        # Validate whether to check the coordinate
                        if np.any(params.fn.constraints(Y[runs])):
                            # Compute new X
                            Xi_star = params.Y2X(Y[runs])

                            # Compute updates
                            UD = [compute_update_UD(
                                level, grp, Xi_star, X, 
                                params.plot_sizes, params.c, params.thetas, params.thetas_inv
                            ) for c in params.c]
                            U = np.array([UD[i][0] for i in range(len(UD))])
                            D = np.array([UD[i][1] for i in range(len(UD))])

                            # Update the X
                            X[runs] = Xi_star

                            # Check if the update is accepted
                            update = Update(level, grp, runs, cols, new_coord, U, D)
                            accept = params.fn.metric.update(Y, X, update)

                            # New best design
                            if accept:
                                # Store the best coordinates
                                Ycoord = new_coord
                                Xrows = np.copy(X[runs])

                                # Set update
                                updated = True
                
                # Set the current coordinates
                Y[runs, cols] = Ycoord
                X[runs] = Xrows
        
        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    return Y, X

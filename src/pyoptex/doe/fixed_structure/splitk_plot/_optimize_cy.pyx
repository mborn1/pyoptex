# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
Module for the split^k-plot coordinate-exchange algorithm.
"""

# cython: boundscheck=False, wraparound=False
# import numpy as np
# cimport numpy as np
# cimport cython
# import warnings

# from ...._profile import profile
# from ..validation import validate_state


import numpy as np
cimport numpy as cnp
from libc.math cimport isinf, abs

# Import State for type checking, though we'll handle its components
from ..utils import State
from .utils import Update

# Ensure NumPy C API is initialized (important for Cython extensions using NumPy)
cnp.import_array()

cdef bint cython_equal(const double[::1] a, const double[::1] b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t idx
    for idx in range(n):
        if a[idx] != b[idx]:
            return False
    return True

cdef bint cython_any(const unsigned char[::1] arr, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i
    for i in range(n):
        if arr[i]:
            return True
    return False

cpdef _optimize_cython_impl(object params, int max_it, double eps) noexcept:
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
    # Unpack the parameters
    cdef long long[::1] effect_types = params.effect_types
    cdef long long[::1] effect_levels = params.effect_levels
    cdef long long[::1] thetas = params.thetas
    cdef long long[::1] colstart = params.colstart
    cdef list grps = params.grps
    cdef list coords = params.coords
    cdef object fn = params.fn

    ###############################################

    # Initialize a design
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] Y
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] X
    _, (Y, X) = fn.init(params)

    # Use memoryviews for efficient access in loops
    cdef double[:, ::1] Y_view = Y
    cdef double[:, ::1] X_view = X

    ###############################################

    # Initialize the metric
    fn.metric.init(Y, X, params)
    cdef double metric = fn.metric.call(Y, X, params)

    ###############################################

    # Loop variables
    cdef Py_ssize_t n_factors = effect_types.shape[0]
    cdef Py_ssize_t factor_idx, i, j, k
    cdef Py_ssize_t col_start, col_end, nb_cols, row_start, row_end, nb_rows
    cdef bint updated
    cdef int it, run_idx
    cdef object update
    cdef long long grp, level, jmp
    cdef double up

    cdef long long[::1] grp_view
    cdef double[::1] Ycoord = np.zeros(Y_view.shape[1], dtype=np.double)
    cdef double[::1] co = np.zeros(Y_view.shape[1], dtype=np.double)
    cdef double[::1] new_coord

    cdef double[:, ::1] possible_coords, new_X_rows
    cdef double[:,::1] Xrows = np.zeros((X_view.shape[0], X_view.shape[1]), dtype=np.double)

    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] new_Y_rows

    for it in range(max_it):
        # Initialize the updated flag
        updated = False

        # Iterate over the effect types
        for factor_idx in range(n_factors):
            # Extract the level and jump and group view
            level = effect_levels[factor_idx]
            jmp = thetas[level]
            grp_view = grps[factor_idx]
            possible_coords = coords[factor_idx]

            # Iterate over the groups
            for j in range(grp_view.shape[0]):
                # Extract the group
                grp = grp_view[j]

                # Extract the columns
                col_start = colstart[factor_idx]
                col_end = colstart[factor_idx+1]
                nb_cols = col_end - col_start

                # Extract the runs
                row_start = grp*jmp
                row_end = (grp+1)*jmp
                nb_rows = row_end - row_start

                # Extract the coordinates and X rows
                Ycoord[:nb_cols] = Y_view[row_start, col_start:col_end]
                Xrows[:nb_rows] = X_view[row_start:row_end, :]

                # Copy the coordinates to the comparison array
                co[:nb_cols] = Ycoord[:nb_cols]

                # Iterate over the possible coordinates
                for k in range(possible_coords.shape[0]):
                    new_coord = possible_coords[k]

                    # Check if a new coordinate
                    if not cython_equal(new_coord, co, nb_cols):
                        # Update Y_view
                        for run_idx in range(nb_rows):
                            Y_view[row_start+run_idx, col_start:col_end] = new_coord

                        # Get the affected runs
                        new_Y_rows = Y[row_start:row_end, :]

                        # Check if the constraints are satisfied (Call Python constraint function)
                        if not cython_any(fn.constraints(new_Y_rows), nb_rows):

                            # Update the X coordinates
                            new_X_rows = fn.Y2X(new_Y_rows)
                            X_view[row_start:row_end, :] = new_X_rows

                            # Compute the update
                            update = Update(
                                level, grp, row_start, row_end, col_start, col_end, 
                                new_coord, Ycoord, Xrows[:nb_rows], metric
                            )
                            up = fn.metric.update(Y, X, params, update)

                            # Check if the update is accepted
                            if ((metric == 0 or isinf(metric)) and up > 0) or up / abs(metric) > eps:
                                # Update the metric state
                                fn.metric.accepted(Y, X, params, update)

                                # Update the coordinates
                                Ycoord[:nb_cols] = new_coord
                                Xrows[:nb_rows] = X_view[row_start:row_end, :]

                                # Update the metric
                                if isinf(up):
                                    metric = fn.metric.call(Y, X, params)
                                else:
                                    metric = metric + up

                                # Update the updated flag
                                updated = True

                    # Set the correct coordinates
                    for run_idx in range(nb_rows):
                        Y_view[row_start+run_idx, col_start:col_end] = Ycoord[:nb_cols]
                        X_view[row_start+run_idx, :] = Xrows[run_idx, :]

        ###############################################


        # Recompute the final metric
        metric = fn.metric.call(Y, X, params)

        # # Check if the update formulas are unstable
        # if ((state.metric == 0 and old_metric > 0) or (np.isinf(state.metric) and ~np.isinf(state.metric))) and params.compute_update:
        #     warnings.warn('Update formulas are very unstable for this problem, try rerunning without update formulas', RuntimeWarning)

        # Check if the update is accepted
        if not updated:
            break

    # Return the design and state
    return Y, X, metric
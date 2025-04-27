# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# To compile this file (from the project root directory): python setup.py build_ext --inplace

import numpy as np
cimport numpy as np

# Import State for type checking, though we'll handle its components
from .utils import State

# Ensure NumPy C API is initialized (important for Cython extensions using NumPy)
np.import_array()

# Define the Cython implementation function
cpdef _optimize_cython_impl(object params, int max_it, bint validate, double eps):
    """Cython implementation of the core coordinate-exchange logic."""
    # Type declarations for parameters accessed frequently
    cdef np.ndarray[np.intp_t, ndim=1, mode='c'] effect_types = params.effect_types
    cdef np.ndarray[np.intp_t, ndim=1, mode='c'] effect_levels = params.effect_levels
    cdef object grps = params.grps
    cdef object coords = params.coords
    cdef np.ndarray[np.intp_t, ndim=1, mode='c'] colstart = params.colstart
    cdef np.ndarray[np.intp_t, ndim=2, mode='c'] Zs = params.Zs
    cdef object fn = params.fn # Keep fn as a Python object for callbacks

    ###############################################

    # Initialize a design (Call Python function)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] Y
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] X
    _, (Y, X) = fn.init(params)

    # Use memoryviews for efficient access in loops
    cdef double[:, ::1] Y_view = Y
    cdef double[:, ::1] X_view = X

    ###############################################

    # Initialization (Call Python methods on fn.metric)
    fn.metric.init(Y, X, params)
    cdef double current_metric = fn.metric.call(Y, X, params)

    ###############################################

    # Loop variables
    cdef int it, i, level, grp_val, nruns
    cdef Py_ssize_t j, k, run_idx, col_idx, slice_len
    cdef bint updated = False
    cdef double new_metric, up

    # Temporary arrays/views needed inside loops
    cdef long[::1] runs = np.zeros(Y.shape[0], dtype=np.int64)
    cdef long[:] grp_runs
    cdef double[::1] Ycoord = np.zeros(Y.shape[1], dtype=np.double)
    cdef double[::1] co = np.zeros(Y.shape[1], dtype=np.double)
    cdef double[:, :] Xrows = np.zeros((X.shape[0], X.shape[1]), dtype=np.double)
    cdef double[:, :] new_X_rows
    cdef double[:, :] possible_coords_arr

    # Make sure we are not stuck in infinite loop
    for it in range(max_it):
        updated = False # Reset update flag for this iteration

        # Loop over all factors
        for i in range(effect_types.shape[0]):
            level = effect_levels[i]
            possible_coords_arr = coords[i]
            slice_len = colstart[i+1] - colstart[i]

            # Loop over all run-groups for this factor
            for grp_val in grps[i]:
                # Determine runs affected by this group: TODO: precompute
                if level == 0:
                    nruns = 1
                    runs[0] = grp_val
                else:
                    grp_runs = np.flatnonzero(Zs[level-1] == grp_val)
                    nruns = grp_runs.shape[0]
                    runs[:nruns] = grp_runs

                # Take from the first run in the group
                Ycoord[:slice_len] = Y_view[runs[0], colstart[i]:colstart[i+1]]
                co[:slice_len] = Ycoord[:slice_len]
                for run_idx in range(nruns):
                    Xrows[runs[run_idx]] = X_view[runs[run_idx]]

                # Loop over possible new coordinates
                for new_coord in possible_coords_arr:

                    # Short-circuit original coordinates
                    if not np.array_equal(new_coord, co):
                        
                        # Update Y_view
                        for run_idx in range(nruns):
                            Y_view[runs[run_idx], colstart[i]:colstart[i+1]] = new_coord

                        # Validate whether to check the coordinate
                        if not np.any(fn.constraints(Y[runs[:nruns]])):
                            
                            # Update X_view
                            new_X_rows = fn.Y2X(Y[runs[:nruns]])
                            for run_idx in range(nruns):
                                X_view[runs[run_idx]] = new_X_rows[run_idx]

                            # Check if the update is accepted (Call Python metric function)
                            new_metric = fn.metric.call(Y, X, params)
                            up = new_metric - current_metric

                            # New best design
                            if ((current_metric == 0 or np.isinf(current_metric)) and up > 0) or up / np.abs(current_metric) > eps:
                                # Store the best coordinates found so far
                                Ycoord[:slice_len] = new_coord
                                for run_idx in range(nruns):
                                    Xrows[runs[run_idx]] = X_view[runs[run_idx]]
                                current_metric = new_metric
                                updated = True

                # Set the correct coordinates
                for run_idx in range(nruns):
                    Y_view[runs[run_idx], colstart[i]:colstart[i+1]] = Ycoord[:slice_len]
                    X_view[runs[run_idx]] = Xrows[run_idx]

        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    # Return the final state components
    return Y, X, current_metric

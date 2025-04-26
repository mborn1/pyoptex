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
    cdef np.ndarray effect_types = params.effect_types
    cdef np.ndarray effect_levels = params.effect_levels
    cdef object grps = params.grps
    cdef object coords = params.coords
    cdef np.ndarray colstart = params.colstart
    cdef np.ndarray Zs = params.Zs
    cdef object fn = params.fn # Keep fn as a Python object for callbacks

    # Initialize a design (Call Python function)
    _, (Y_init, X_init) = fn.init(params)

    # Ensure arrays are C-contiguous and have correct types if possible
    # Use float64 for numerical stability and calculations
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] Y = np.ascontiguousarray(Y_init, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] X = np.ascontiguousarray(X_init, dtype=np.double)

    # Use memoryviews for efficient access in loops
    cdef double[:, ::1] Y_view = Y
    cdef double[:, ::1] X_view = X

    # Initialization (Call Python methods on fn.metric)
    fn.metric.init(Y, X, params)
    cdef double current_metric = fn.metric.call(Y, X, params)

    # Loop variables
    cdef int it, i, level, grp_val
    cdef Py_ssize_t j, k, run_idx, col_idx, slice_len
    cdef bint updated = False
    cdef double new_metric, up

    # Temporary arrays/views needed inside loops
    cdef np.ndarray[np.intp_t, ndim=1] runs
    cdef np.ndarray[np.double_t, ndim=1] Ycoord, co
    cdef np.ndarray[np.double_t, ndim=2] Xrows
    cdef np.ndarray possible_coords_arr

    # Make sure we are not stuck in finite loop
    for it in range(max_it):
        updated = False # Reset update flag for this iteration

        # Loop over all factors
        for i in range(effect_types.shape[0]):
            level = effect_levels[i]
            possible_coords_arr = coords[i]
            slice_len = colstart[i+1] - colstart[i]

            # Loop over all run-groups for this factor
            for grp_val in grps[i]:
                # Determine runs affected by this group
                if level == 0:
                    runs = np.array([grp_val], dtype=np.intp)
                else:
                    runs = np.flatnonzero(Zs[level-1] == grp_val).astype(np.intp)
                
                # No runs in this group, skip
                if runs.shape[0] == 0:
                    continue

                # Take from the first run in the group
                Ycoord = np.array(Y_view[runs[0], colstart[i]:colstart[i+1]], copy=True, dtype=np.double)
                co = Ycoord.copy()
                Xrows = np.ascontiguousarray([X_view[run_idx, :] for run_idx in runs], dtype=np.double)
                
                # Loop over possible new coordinates
                for new_coord in possible_coords_arr:

                    # Short-circuit original coordinates
                    if np.any(new_coord != co):
                        
                        # Update Y_view
                        for run_idx in range(runs.shape[0]):
                           for k in range(slice_len):
                               Y_view[runs[run_idx], colstart[i] + k] = new_coord[k]

                        # Validate whether to check the coordinate
                        if not np.any(fn.constraints(Y[runs])):
                            
                            # Update X_view
                            new_X_rows = np.ascontiguousarray(fn.Y2X(Y[runs]), dtype=np.double)
                            for idx, run_idx in enumerate(runs):
                                for col_idx in range(X_view.shape[1]):
                                    X_view[run_idx, col_idx] = new_X_rows[idx, col_idx]

                            # Check if the update is accepted (Call Python metric function)
                            new_metric = fn.metric.call(Y, X, params)
                            up = new_metric - current_metric

                            # New best design
                            if ((current_metric == 0 or np.isinf(current_metric)) and up > 0) or up / np.abs(current_metric) > eps:
                                # Store the best coordinates found so far
                                Ycoord = new_coord.copy()
                                Xrows = np.ascontiguousarray([X_view[run_idx, :] for run_idx in runs], dtype=np.double)
                                current_metric = new_metric
                                updated = True

                # Set the correct coordinates
                for run_idx in range(runs.shape[0]):
                    for k in range(slice_len):
                        Y_view[runs[run_idx], colstart[i] + k] = Ycoord[k]
                for idx, run_idx in enumerate(runs):
                    for col_idx in range(X_view.shape[1]):
                        X_view[run_idx, col_idx] = Xrows[idx, col_idx]

        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    # Return the final state components
    return Y, X, current_metric

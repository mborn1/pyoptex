# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

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

    # Initialization (Call Python methods on fn.metric)
    fn.metric.init(Y, X, params)
    cdef double current_metric = fn.metric.call(Y, X, params)

    # Use memoryviews for efficient access in loops
    cdef double[:, ::1] Y_view = Y
    cdef double[:, ::1] X_view = X

    # Loop variables
    cdef int it, i, level, grp_val
    cdef Py_ssize_t j, k, run_idx
    cdef Py_ssize_t col_idx # For nested element-wise assignment
    cdef bint updated = False
    cdef double new_metric, up

    # Temporary arrays/views needed inside loops
    cdef np.ndarray[np.intp_t, ndim=1] runs
    cdef np.ndarray[np.double_t, ndim=1] Ycoord, new_coord, co
    cdef np.ndarray[np.double_t, ndim=2] Xrows
    cdef np.ndarray possible_coords_arr
    cdef Py_ssize_t slice_len # For element-wise assignment loops (k already declared above)

    # Make sure we are not stuck in finite loop
    for it in range(max_it):
        updated = False # Reset update flag for this iteration

        # Loop over all factors
        for i in range(effect_types.shape[0]):
            level = effect_levels[i]
            possible_coords_arr = coords[i] # This is likely a list/tuple of arrays

            # Loop over all run-groups for this factor
            for grp_val in grps[i]:
                # Determine runs affected by this group
                if level == 0:
                    runs = np.array([grp_val], dtype=np.intp) # Need dtype for indexing
                else:
                    # Ensure Zs[level-1] is a NumPy array for efficient comparison
                    Zs_level = np.asarray(Zs[level-1])
                    runs = np.flatnonzero(Zs_level == grp_val).astype(np.intp)
                
                if runs.shape[0] == 0:
                    continue # No runs in this group, skip

                # Extract current coordinate (as best), ensure it's a C-contiguous double array
                # Take from the first run in the group
                Ycoord = np.array(Y_view[runs[0], colstart[i]:colstart[i+1]], copy=True, dtype=np.double)
                # Keep track of the best coordinates found so far for this (factor, group)
                co = Ycoord.copy()
                # Also copy the corresponding X rows
                # Avoid advanced indexing on memoryview: iterate through 'runs'
                Xrows = np.ascontiguousarray([X_view[run_idx, :] for run_idx in runs], dtype=np.double)
                
                # Loop over possible new coordinates
                for new_coord_obj in possible_coords_arr:
                    # Ensure new_coord is a C-contiguous double array
                    new_coord = np.ascontiguousarray(new_coord_obj, dtype=np.double)

                    # Short-circuit original coordinates
                    # Using np.any for potentially multi-dimensional check
                    if np.any(new_coord != co):
                        
                        # Temporarily update Y view for constraint/metric check
                        # Iterate through runs to update Y_view efficiently
                        for run_idx in range(runs.shape[0]):
                           # Element-wise assignment to memoryview slice
                           slice_len = colstart[i+1] - colstart[i]
                           for k in range(slice_len):
                               Y_view[runs[run_idx], colstart[i] + k] = new_coord[k]
                           # Y_view[runs[run_idx], colstart[i]:colstart[i+1]] = new_coord # Original line causing error

                        # Check validity of new coordinates (Call Python constraint function)
                        # Pass only the relevant slice/rows of Y
                        # fn.constraints expects a 2D array (n_runs_in_group, n_cols_in_slice)
                        if not np.any(fn.constraints(Y[runs])): # Pass the updated Y slice
                            # Update the X view (Call Python Y2X function)
                            # Pass relevant Y rows, get back relevant X rows
                            # Calculate new X rows first
                            new_X_rows = np.ascontiguousarray(fn.Y2X(Y[runs]), dtype=np.double)
                            # Element-wise assignment for X_view
                            for idx, run_idx in enumerate(runs):
                                # Iterate through columns for explicit assignment
                                for col_idx in range(X_view.shape[1]):
                                    X_view[run_idx, col_idx] = new_X_rows[idx, col_idx]
                                # X_view[run_idx, :] = new_X_rows[idx, :] # Original loop line causing error
                            # X_view[runs, :] = np.ascontiguousarray(fn.Y2X(Y[runs]), dtype=np.double) # Original error line

                            # Check if the update is accepted (Call Python metric function)
                            new_metric = fn.metric.call(Y, X, params)
                            up = new_metric - current_metric

                            # New best design criteria
                            # Handle potential division by zero or infinite metric
                            accept_update = False
                            if current_metric == 0 or np.isinf(current_metric):
                                if up > 0:
                                    accept_update = True
                            elif abs(current_metric) > 1e-15: # Avoid division by near-zero
                                if (up / abs(current_metric)) > eps:
                                    accept_update = True
                            elif up > eps: # If metric is near zero, require absolute improvement
                                accept_update = True

                            if accept_update:
                                # Store the best coordinates found so far
                                Ycoord = new_coord.copy() # Store the array itself
                                Xrows = np.ascontiguousarray([X_view[run_idx, :] for run_idx in runs], dtype=np.double)
                                current_metric = new_metric

                                # # Validation (Optional - potentially slow)
                                # if validate:
                                #     # Reconstruct temporary state for validation if needed
                                #     temp_state = State(Y, X, current_metric)
                                #     # Need to import/call validate_state (might require Python call)
                                #     # Consider validating only outside the loop or in the Python wrapper
                                
                                updated = True

                # Reset Y and X views to the state corresponding to the best coordinate (Ycoord)
                # This ensures the state reflects the accepted coordinate before moving to the next group/factor
                for run_idx in range(runs.shape[0]):
                    # Element-wise assignment to memoryview slice
                    slice_len = colstart[i+1] - colstart[i]
                    for k in range(slice_len):
                        Y_view[runs[run_idx], colstart[i] + k] = Ycoord[k]
                    # Y_view[runs[run_idx], colstart[i]:colstart[i+1]] = Ycoord # Original similar line
                # Element-wise assignment for X_view reset
                for idx, run_idx in enumerate(runs):
                    # Iterate through columns for explicit assignment
                    for col_idx in range(X_view.shape[1]):
                        X_view[run_idx, col_idx] = Xrows[idx, col_idx]
                    # X_view[run_idx, :] = Xrows[idx, :] # Original loop line, replaced
                # X_view[runs, :] = Xrows # Original similar line, replaced with loop

            # # Optional validation after each factor
            # if validate:
            #    pass # Similar considerations as above

        # Stop if nothing updated for an entire iteration
        if not updated:
            break

    # # Final validation (Optional - do in Python wrapper)
    # if validate:
    #    pass 

    # Return the final state components
    return Y, X, current_metric

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# To compile this file (from the project root directory): python setup.py build_ext --inplace

import numpy as np
cimport numpy as cnp
from libc.math cimport isinf, abs

# Import State for type checking, though we'll handle its components
from .utils import State

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

# Define the Cython implementation function
cpdef _optimize_cython_impl(object params, int max_it, bint validate, double eps):
    """Cython implementation of the core coordinate-exchange logic."""
    # Type declarations for parameters accessed frequently
    cdef long[::1] effect_types = params.effect_types
    cdef long[::1] effect_levels = params.effect_levels
    cdef list grps = params.grps
    cdef list coords = params.coords
    cdef long[::1] colstart = params.colstart
    cdef long[:,::1] Zs = params.Zs
    cdef object fn = params.fn # Keep fn as a Python object for callbacks
    cdef list grp_runs = params.grp_runs

    ###############################################

    # Initialize a design (Call Python function)
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] Y
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] X
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
    cdef int it, h, i, j, level, grp_val, nruns
    cdef Py_ssize_t k, run_idx, slice_len
    cdef bint updated = False
    cdef double new_metric, up

    # Temporary arrays/views needed inside loops
    cdef long[::1] grp_view
    cdef long[::1] runs
    cdef double[::1] Ycoord = np.zeros(Y_view.shape[1], dtype=np.double)
    cdef double[::1] co = np.zeros(Y_view.shape[1], dtype=np.double)
    cdef double[:,::1] Xrows = np.zeros((X_view.shape[0], X_view.shape[1]), dtype=np.double)
    cdef double[:,::1] new_X_rows
    cdef double[:,::1] possible_coords_arr
    cdef double[::1] new_coord
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] new_Y_rows

    ###############################################

    # Make sure we are not stuck in infinite loop
    for it in range(max_it):
        # Reset update flag for this iteration
        updated = False 

        # Loop over all factors
        for i in range(effect_types.shape[0]):
            level = effect_levels[i]
            possible_coords_arr = coords[i]
            slice_len = colstart[i+1] - colstart[i]
            grp_view = grps[i]

            # Loop over all run-groups for this factor
            for j in range(grp_view.shape[0]):
                grp_val = grp_view[j]

                # Determine runs affected by this group
                runs = grp_runs[i][j]
                nruns = runs.shape[0]

                # Store the coordinate and affected rows
                Ycoord[:slice_len] = Y_view[runs[0], colstart[i]:colstart[i+1]]
                co[:slice_len] = Ycoord[:slice_len]
                for run_idx in range(nruns):
                    Xrows[run_idx] = X_view[runs[run_idx]]
                
                # Loop over possible new coordinates
                for h in range(possible_coords_arr.shape[0]):
                    new_coord = possible_coords_arr[h]

                    # Short-circuit original coordinates
                    if not cython_equal(new_coord, co, slice_len):
                        # Update Y_view
                        for run_idx in range(nruns):
                            Y_view[runs[run_idx], colstart[i]:colstart[i+1]] = new_coord

                        # Get the affected runs
                        new_Y_rows = Y[runs[:nruns], :]

                        # Validate whether to check the coordinate
                        if not cython_any(fn.constraints(new_Y_rows), nruns):
                            
                            # Update X_view
                            new_X_rows = fn.Y2X(new_Y_rows)
                            for run_idx in range(nruns):
                                X_view[runs[run_idx]] = new_X_rows[run_idx]

                            # Check if the update is accepted (Call Python metric function)
                            new_metric = fn.metric.call(Y, X, params)
                            up = new_metric - current_metric

                            # New best design
                            if ((current_metric == 0 or isinf(current_metric)) and up > 0) or up / abs(current_metric) > eps:
                                # Store the best coordinates found so far
                                Ycoord[:slice_len] = new_coord
                                for run_idx in range(nruns):
                                    Xrows[run_idx] = X_view[runs[run_idx]]
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

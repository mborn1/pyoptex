# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cython
import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef bint cython_equal(const double[::1] a, const double[::1] b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t idx
    for idx in range(n):
        if a[idx] != b[idx]:
            return False
    return True

cpdef obs_var_cy(
        double[:, ::1] Yenc,
        long long[::1] colstart,
        double[::1] ratios,
        unsigned char[::1] grouped_cols
    ):
    """
    Cython implementation to compute the observation matrix from the design.
    """
    # Initialize variables
    cdef Py_ssize_t n_runs = Yenc.shape[0]
    cdef Py_ssize_t n_factors = colstart.size - 1

    # Initialize output
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] V = np.eye(n_runs, dtype=np.double)
    cdef double[:, ::1] V_view = V
    
    # Loop variables
    cdef Py_ssize_t i, j, k
    cdef long long start_col, end_col, prev_border

    # Loop over the factors
    for i in range(n_factors):
        # Check if the factor is grouped
        if grouped_cols[i]:
            # Extract the start and end columns
            start_col = colstart[i]
            end_col = colstart[i+1]
            prev_border = 0

            # Determine the borders of each group
            for j in range(1, n_runs):
                if not cython_equal(
                    Yenc[j, start_col:end_col], 
                    Yenc[j-1, start_col:end_col], 
                    end_col - start_col
                ):
                    V[prev_border:j, prev_border:j] += ratios[i]
                    prev_border = j

            # Set the last block
            if prev_border < n_runs:
                V[prev_border:n_runs, prev_border:n_runs] += ratios[i]

    return V
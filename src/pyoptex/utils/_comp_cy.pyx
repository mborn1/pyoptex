# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cnp.import_array()

cpdef outer_integral_cython_impl(const double[:, ::1] arr):
    cdef Py_ssize_t i, j, k, nrows, ncols
    cdef cnp.ndarray[cnp.double_t, ndim=2] out
    
    nrows = arr.shape[0]
    ncols = arr.shape[1]
    out = np.zeros((ncols, ncols), dtype=np.float64)

    cdef double[:, ::1] out_view = out

    for i in range(nrows):
        for j in range(ncols):
            for k in range(ncols):
                out_view[j, k] += arr[i, j] * arr[i, k]
      
    return out / nrows

cpdef int2bool_cython_impl(const long long[:, ::1] arr, long long n, long long size, long long arr_size):
    # Create the output array
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] out = np.zeros((n, size), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] out_view = out

    # Convert to boolean
    for i in range(n):
        for j in range(arr_size):
            out_view[i, arr[i, j]] = True

    return out

cpdef choice_bool(const unsigned char[:, ::1] valids, int axis=0):
    """
    For each row in valids, chooses a random index of the true
    elements in that row. For example, if valids
    is [[True, False], [True, True]], the first element of out
    must be 0 as there is no other options, the second element
    has a 50% chance to be zero, and a 50% chance to be one.
    If all elements are False, -1 is returned.

    Parameters
    ----------
    valids : np.array(2d)
        A 2d-boolean matrix.

    Returns
    -------
    out : np.array(1d)
        An integer array with the randomly chosen indices.
    """
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t n = valids.shape[0]
    cdef Py_ssize_t m = valids.shape[1]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out
    cdef cnp.int64_t[::1] out_view
    cdef cnp.ndarray[cnp.int64_t, ndim=1] idx
    cdef cnp.int64_t[::1] idx_view

    # Initialize the output array
    if axis == 0:
        out = np.zeros(n, dtype=np.int64)
        out_view = out
        idx = np.empty(m, dtype=np.int64)
        idx_view = idx

        for i in range(n):
            # Check which indices are viable
            k = 0
            for j in range(m):
                if valids[i, j]:
                    idx_view[k] = j
                    k += 1

            # Choose a random index
            if k > 0:
                out_view[i] = np.random.choice(idx[:k])
            else:
                out_view[i] = -1

    else:
        out = np.zeros(m, dtype=np.int64)
        out_view = out
        idx = np.empty(n, dtype=np.int64)
        idx_view = idx

        for i in range(m):
            # Check which indices are viable
            k = 0
            for j in range(n):
                if valids[j, i]:
                    idx_view[k] = j
                    k += 1

            # Choose a random index
            if k > 0:
                out_view[i] = np.random.choice(idx[:k])
            else:
                out_view[i] = -1
    
    return out

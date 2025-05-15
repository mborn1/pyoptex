# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cnp.import_array()

def outer_integral_cython_impl(const double[:, ::1] arr not None):
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
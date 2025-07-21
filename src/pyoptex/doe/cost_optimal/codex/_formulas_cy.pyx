# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cython
import numpy as np
cimport numpy as cnp

cnp.import_array()

cpdef inv_PpD(double[:,:,::1] P, double[::1] ratios):
    """
    Part of update formulas, see article for information.
    """
    cdef Py_ssize_t i
    cdef long long nb_c = P.shape[0]

    for i in range(nb_c):
        P[i, 0, 0] += 1/ratios[i]
        P[i, 1, 1] += 1/ratios[i]

    cdef cnp.ndarray[cnp.double_t, ndim=3, mode='c'] out = np.empty((P.shape[0], P.shape[1], P.shape[2]))
    for i in range(nb_c):
        out[i] = np.linalg.inv(P[i])

    return out

cpdef _group_update_vinv(
        cnp.ndarray[cnp.double_t, ndim=3, mode='c'] Vinv,
        long long[::1] Zi,
        long long row_start,
        long long row_end,
        long long group_from,
        long long group_to,
        double[::1] ratios
    ):
    """
    Part of update formulas, see article for information.
    """
    # Initialize variables
    cdef Py_ssize_t i
    cdef long long nb_runs = Zi.shape[0]
    cdef long long nb_c = Vinv.shape[0]
    cdef double[:, :, ::1] Vinv_view = Vinv

    # Initialize T arrays
    cdef long long[::1] T_from_before = cython.view.array(
        shape=(row_start + 1,),
        itemsize=sizeof(long long),
        format='q'
    )
    cdef long long[::1] T_not_from_between = cython.view.array(
        shape=(row_end - row_start + 1,),
        itemsize=sizeof(long long),
        format='q'
    )
    cdef long long[::1] T_from_after = cython.view.array(
        shape=(nb_runs - row_end + 1,),
        itemsize=sizeof(long long),
        format='q'
    )
    cdef long long[::1] T_to_before = cython.view.array(
        shape=(row_start + 1,),
        itemsize=sizeof(long long),
        format='q'
    )
    cdef long long[::1] T_to_between = cython.view.array(
        shape=(row_end - row_start + 1,),
        itemsize=sizeof(long long),
        format='q'
    )
    cdef long long[::1] T_to_after = cython.view.array(
        shape=(nb_runs - row_end + 1,),
        itemsize=sizeof(long long),
        format='q'
    )

    # Determine the T arrays
    cdef long long nb_T_from_before = 0
    cdef long long nb_T_not_from_between = 0
    cdef long long nb_T_from_after = 0
    cdef long long nb_T_to_before = 0
    cdef long long nb_T_to_between = 0
    cdef long long nb_T_to_after = 0
    for i in range(row_start):
        if Zi[i] == group_from:
            T_from_before[nb_T_from_before] = i
            nb_T_from_before += 1
        if Zi[i] == group_to:
            T_to_before[nb_T_to_before] = i
            nb_T_to_before += 1
    for i in range(row_start, row_end):
        if Zi[i] != group_from:
            T_not_from_between[nb_T_not_from_between] = i
            nb_T_not_from_between += 1
        if Zi[i] == group_to:
            T_to_between[nb_T_to_between] = i
            nb_T_to_between += 1
    for i in range(row_end, nb_runs):
        if Zi[i] == group_from:
            T_from_after[nb_T_from_after] = i
            nb_T_from_after += 1
        if Zi[i] == group_to:
            T_to_after[nb_T_to_after] = i
            nb_T_to_after += 1

    # Compute VR submatrix
    cdef cnp.ndarray[cnp.double_t, ndim=3, mode='c'] VR = np.zeros((nb_c, nb_runs, 2))
    cdef double[:, :, ::1] VR_view = VR
    for j in range(nb_c):
        for k in range(nb_runs):
            for i in range(row_start, row_end):
                VR_view[j, k, 0] += Vinv_view[j, k, i]
    for j in range(nb_c):
        for k in range(nb_runs):
            for i in range(nb_T_to_before):
                VR_view[j, k, 1] += Vinv_view[j, k, T_to_before[i]]
            for i in range(nb_T_to_after):
                VR_view[j, k, 1] += Vinv_view[j, k, T_to_after[i]]
            for i in range(nb_T_from_before):
                VR_view[j, k, 1] -= Vinv_view[j, k, T_from_before[i]]
            for i in range(nb_T_from_after):
                VR_view[j, k, 1] -= Vinv_view[j, k, T_from_after[i]]

    # Compute SV submatrix
    cdef cnp.ndarray[cnp.double_t, ndim=3, mode='c'] SV = np.empty((nb_c, 2, nb_runs))
    cdef double[:, :, ::1] SV_view = SV
    for j in range(nb_c):
        for k in range(nb_runs):
            SV_view[j, 0, k] = VR_view[j, k, 1] / 2
            for i in range(nb_T_not_from_between):
                SV_view[j, 0, k] += Vinv_view[j, k, T_not_from_between[i]]
            for i in range(nb_T_to_between):
                SV_view[j, 0, k] += Vinv_view[j, k, T_to_between[i]]
            SV_view[j, 0, k] *= 2
    SV_view[:, 1, :] = VR_view[:, :, 0]

    # Compute P submatrix
    cdef cnp.ndarray[cnp.double_t, ndim=3, mode='c'] P = np.zeros((nb_c, 2, 2))
    cdef double[:, :, ::1] P_view = P
    for j in range(nb_c):
        for i in range(row_start, row_end):
            P_view[j, 0, 0] += SV_view[j, 0, i]
            P_view[j, 1, 0] += SV_view[j, 1, i]
    for j in range(nb_c):
        for i in range(nb_T_to_before):
            P_view[j, 0, 1] += SV_view[j, 0, T_to_before[i]]
            P_view[j, 1, 1] += SV_view[j, 1, T_to_before[i]]
        for i in range(nb_T_to_after):
            P_view[j, 0, 1] += SV_view[j, 0, T_to_after[i]]
            P_view[j, 1, 1] += SV_view[j, 1, T_to_after[i]]
        for i in range(nb_T_from_before):
            P_view[j, 0, 1] -= SV_view[j, 0, T_from_before[i]]
            P_view[j, 1, 1] -= SV_view[j, 1, T_from_before[i]]
        for i in range(nb_T_from_after):
            P_view[j, 0, 1] -= SV_view[j, 0, T_from_after[i]]
            P_view[j, 1, 1] -= SV_view[j, 1, T_from_after[i]]

    # Perform the update
    PpDinv = inv_PpD(P, ratios)
    for i in range(len(Vinv)):
        Vinv[i] -= VR[i] @ (PpDinv[i] @ SV[i])

    return Vinv

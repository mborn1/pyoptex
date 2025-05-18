# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from numpy.random import PCG64
cimport numpy as cnp

cnp.import_array()

cdef bint cython_any(const unsigned char[::1] arr, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i
    for i in range(n):
        if arr[i]:
            return True
    return False
cdef bint cython_all_idx(const unsigned char[::1] arr, const int[::1] idx, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i
    for i in range(n):
        if not arr[idx[i]]:
            return False
    return True

cpdef __init_unconstrained(long long[::1] effect_types,
                         long long[::1] effect_levels,
                         list grps,
                         list coords,
                         long long[:, ::1] Zs,
                         cnp.ndarray[double, ndim=2] Y,
                         bint complete=False):
    """
    This function generates a random design without 
    considering any design constraints.

    .. note::
        The resulting design matrix `Y` is not encoded.

    Parameters
    ----------
    effect_types : np.array(1d)
        The effect types of each factor, representing 
        a 1 for a continuous factor and the number of 
        levels for a categorical factor.
    effect_levels : np.array(1d)
        The level of each factor.
    grps : list of np.array(1d)
        The groups for each factor to initialize.
    coords : list of np.array(2d)
        The coordinates for each factor to use.
    Zs : np.array(2d)
        Every grouping vector Z stacked vertically.
    Y : np.array(2d)
        The design matrix to be initialized. May contain the
        some fixed settings if not optimizing all groups.
        This matrix should not be encoded.
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.

    Returns
    -------
    Y : np.array(2d)
        The initialized design matrix.
    """
    cdef int i, j, n_replicates, ngrps
    cdef long long typ, level
    cdef long long[::1] lgrps, Z
    cdef double[::1] r, choices
    cdef double[:, ::1] Y_view = Y
    
    if complete:
        for i in range(effect_types.shape[0]):
            # Extract parameters
            typ = effect_types[i]
            lgrps = grps[i]
            ngrps = lgrps.shape[0]

            if typ == 1:
                # Continuous factor
                r = np.random.rand(ngrps) * 2.0 - 1.0
            else:
                # Discrete factor
                choices = np.arange(typ, dtype=np.float64)
                if typ >= ngrps:
                    r = np.random.choice(choices, ngrps, replace=False)
                else:
                    n_replicates = ngrps // choices.shape[0]
                    r = np.random.permutation(
                        np.concatenate((
                            np.repeat(choices, n_replicates), 
                            np.random.choice(choices, ngrps - choices.shape[0] * n_replicates)
                        ))
                    )

            # Fill the design
            level = effect_levels[i]
            if level == 0:
                # Every run is separate
                for j in range(ngrps):
                    Y_view[lgrps[j], i] = r[j]
            else:
                # Make sure to match the groups
                Z = Zs[level-1]
                for j in range(ngrps):
                    for k in range(Y_view.shape[0]):
                        if Z[k] == lgrps[j]:
                            Y_view[k, i] = r[j]

    else:
        for i in range(effect_types.shape[0]):
            # Extract parameters
            typ = effect_types[i]
            lgrps = grps[i]
            ngrps = lgrps.shape[0]

            # Determine the choices
            if typ == 1:
                choices = coords[i].flatten()
            else:
                choices = np.arange(coords[i].shape[0], dtype=np.float64)

            # Pick from the choices and try to have all of them atleast once
            if choices.shape[0] >= ngrps:
                r = np.random.choice(choices, ngrps, replace=False)
            else:
                n_replicates = ngrps // choices.shape[0]
                r = np.random.permutation(np.concatenate((
                    np.repeat(choices, n_replicates), 
                    np.random.choice(choices, ngrps - choices.shape[0] * n_replicates)
                )))

            # Fill the design
            level = effect_levels[i]
            if level == 0:
                # Every run is separate
                for j in range(ngrps):
                    Y_view[lgrps[j], i] = r[j]
            else:
                # Make sure to match the groups
                Z = Zs[level-1]
                for j in range(ngrps):
                    for k in range(Y_view.shape[0]):
                        if Z[k] == lgrps[j]:
                            Y_view[k, i] = r[j]

    return Y

def __correct_constraints(long long[::1] effect_types not None,
                          long long[::1] effect_levels not None,
                          list grps not None,
                          list coords not None,
                          object constraints not None,
                          long long[:, ::1] Zs not None,
                          cnp.ndarray[double, ndim=2] Y not None,
                          bint complete=False):

    # Extract parameters
    cdef int n_factors = effect_types.shape[0]
    cdef double[:, ::1] Y_view = Y

    # Determine which runs are invalid
    cdef unsigned char[::1] invalid_run = constraints(Y)

    # Determine in which order to correct the constraints
    cdef int[::1] zidx = np.zeros(Zs.shape[0] + 1, dtype=np.int32)
    zidx[:Zs.shape[0]] = np.argsort(np.array([len(np.unique(zi)) for zi in Zs], dtype=np.int32)) + 1

    # Loop variables
    cdef int[::1] runs = np.zeros(Y_view.shape[0], dtype=np.int32)
    cdef long long[::1] Z = np.zeros(Y_view.shape[0], dtype=np.int64)
    cdef unsigned char[::1] invalid_run_selected = np.zeros(Y_view.shape[0], dtype=np.bool_)
    cdef unsigned char[::1] permitted_to_optimize = np.zeros(Y_view.shape[0], dtype=np.bool_)
    cdef cnp.ndarray[double, ndim=2] Y_selected = np.zeros((Y_view.shape[0], Y_view.shape[1]), dtype=np.float64)
    cdef double[:, ::1] Y_selected_view = Y_selected

    cdef int i, j, k, l, level, nruns, ngrps
    cdef long long[::1] lgrps, unique_vals, grp
    cdef list grps_
    cdef bint c
    
    for j in range(zidx.shape[0]):
        # Retrieve the level of this factor
        level = zidx[j]

        # Initialize permitted to optimize to false
        if level == 0:
            ngrps = Y_view.shape[0]
            permitted_to_optimize[:] = False
        else:
            ngrps = np.max(Zs[level-1])+1
            permitted_to_optimize[:ngrps] = False

        # Check which ones are permitted for this level
        for i in range(n_factors):
            if effect_levels[i] == level:
                lgrps = grps[i]
                for k in range(lgrps.shape[0]):
                    permitted_to_optimize[lgrps[k]] = True

        # Loop over all groups
        for i in range(ngrps):
            # Check if permitted to be optimized
            if permitted_to_optimize[i]:

                # Determine which runs belong to that group
                if level == 0:
                    runs[0] = i
                    nruns = 1
                else:
                    nruns = 0
                    for k in range(Zs[level-1].shape[0]):
                        if Zs[level-1][k] == i:
                            runs[nruns] = k
                            nruns += 1
                
                # Check if all invalid
                if cython_all_idx(invalid_run, runs, nruns):
                    # Specify which groups to regenerate
                    grps_ = [np.empty((0,), dtype=np.int64)] * n_factors
                    for k in range(n_factors):
                        # Only regenerate levels that are not yet optimized
                        if effect_levels[k] in zidx[j:]:
                            if effect_levels[k] == 0:
                                # Take every runs that we can optimize
                                grp = np.array([co for co in runs[:nruns] if co in grps[k]], dtype=np.int64)
                            else:
                                # Detect unique group values
                                for l in range(nruns):
                                    Z[l] = Zs[effect_levels[k]-1][runs[l]]
                                unique_vals = np.unique(Z[:nruns])

                                # Take every unique group value that we can optimize
                                grp = np.array([co for co in unique_vals if co in grps[k]], dtype=np.int64)
                            grps_[k] = grp

                    # Regenerate until no longer all invalid
                    c = True
                    while c:
                        # Regenerate the unconstrained design
                        Y = __init_unconstrained(effect_types, effect_levels, grps_, 
                                                 coords, Zs, Y, complete)

                        # Validate if any of the runs are invalid
                        for k in range(nruns):
                            Y_selected_view[k] = Y_view[runs[k]]
                        c = cython_any(constraints(Y_selected[:nruns]), nruns)

                    # Update the runs
                    invalid_run = constraints(Y)

    return Y

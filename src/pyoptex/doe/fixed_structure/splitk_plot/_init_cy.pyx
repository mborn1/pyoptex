# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

cnp.import_array()

cpdef __init_unconstrained(const long long[::1] effect_types,
                            const long long[::1] effect_levels,
                            list grps,
                            const long long[::1] thetas,
                            list coords,
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
    grps : list
        The groups for each factor to initialize.
    thetas : np.array(1d)
        The array of thetas.
        thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    coords : list(np.array(2d))
        The coordinates for each factor to use.
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
    cdef Py_ssize_t i
    cdef long long typ, level, size, ngrps, n_replicates, j
    cdef long long[::1] lgrps
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
            size = thetas[level]
            for j in range(ngrps):
                Y_view[lgrps[j]*size: (lgrps[j]+1)*size, i] = r[j]

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
            size = thetas[level]
            for j in range(ngrps):
                Y_view[lgrps[j]*size: (lgrps[j]+1)*size, i] = r[j]

    return Y





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

def __correct_constraints(const long long[::1] effect_types not None,
                          const long long[::1] effect_levels not None,
                          list grps not None,
                          const long long[::1] thetas,
                          list coords not None,
                          const long long[::1] plot_sizes,
                          object constraints not None,
                          cnp.ndarray[double, ndim=2] Y not None,
                          bint complete=False):

    # Extract parameters
    cdef int n_factors = effect_types.shape[0]
    cdef double[:, ::1] Y_view = Y

    # Determine which runs are invalid
    cdef unsigned char[::1] invalid_run = constraints(Y)


    # # Store aggregated values of invalid run per level
    # level_all_invalid = [invalid_run]
    # for i in range(plot_sizes.size - 1):
    #     all_invalid = numba_all_axis1(level_all_invalid[i].reshape(-1, plot_sizes[i]))
    #     level_all_invalid.append(all_invalid)

    # Loop variables
    cdef long long i, j, level, jmp
    cdef long long ngrps = 1

    for i in range(plot_sizes.size - 1, -1, -1):
        level = effect_levels[i]
        jmp = thetas[level]
        ngrps *= plot_sizes[i]
        all_invalid = ... # Compute aggregation of invalid runs

        for j in range(ngrps):
            # Specify which groups to regenerate
            # grps_ = [np.empty((0,), dtype=np.int64)] * n_factors
            # for k in range(n_factors):
            #     # Only regenerate levels that are not yet optimized
            #     if effect_levels[k] in zidx[j:]:
            #         if effect_levels[k] == 0:
            #             # Take every runs that we can optimize
            #             grp = np.array([co for co in runs[:nruns] if co in grps[k]], dtype=np.int64)
            #         else:
            #             # Detect unique group values
            #             for l in range(nruns):
            #                 Z[l] = Zs[effect_levels[k]-1][runs[l]]
            #             unique_vals = np.unique(Z[:nruns])

            #             # Take every unique group value that we can optimize
            #             grp = np.array([co for co in unique_vals if co in grps[k]], dtype=np.int64)
            #         grps_[k] = grp


        

    # for level, all_invalid in zip(range(plot_sizes.size - 1, -1, -1), level_all_invalid[::-1]):
    #     # Define the jump
    #     jmp = thetas[level]

    #     ##################################################
    #     # SELECT ENTIRELY INVALID BLOCKS
    #     ##################################################
    #     # Loop over all groups in the level
    #     for grp in np.where(all_invalid)[0]:
    #         # Specify which groups to update
    #         grps_ = [
    #             np.array([
    #                 g for g in grps[col] 
    #                 if g >= grp*jmp/thetas[l] and g < (grp+1)*jmp/thetas[l]
    #             ], dtype=np.int64) 
    #             if l < level else (
    #                 np.arange(grp, grp+1, dtype=np.int64) 
    #                 if (l == level and grp in grps[col])
    #                 else np.arange(0, dtype=np.int64)
    #             )
    #             for col, l in enumerate(effect_levels)
    #         ]

    #         ##################################################
    #         # REGENERATE BLOCK
    #         ##################################################
    #         # Loop until no longer all invalid
    #         while all_invalid[grp]:
    #             # Adjust the design
    #             Y = __init_unconstrained(effect_types, effect_levels, grps_, 
    #                                      thetas, coords, Y, complete)
    #             # Validate the constraints
    #             c = constraints(Y[grp*jmp:(grp+1)*jmp])
    #             # Update all invalid
    #             for l in range(level):
    #                 level_all_invalid[l][grp*int(jmp/thetas[l]):(grp+1)*int(jmp/thetas[l])] = c
    #                 c = numba_all_axis1(c.reshape(-1, plot_sizes[l]))
    #             all_invalid[grp] = c[0]



    return Y

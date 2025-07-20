# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

cdef void _cython_all_axis1(
        unsigned char[::1] out, const unsigned char[::1] arr, 
        Py_ssize_t dim0, Py_ssize_t dim1
    ) noexcept:
    cdef Py_ssize_t i, j
    cdef bint all_true

    for i in range(dim0):
        all_true = True
        for j in range(dim1):
            if not arr[i*dim1 + j]:
                all_true = False
                break
        out[i] = all_true

cdef bint cython_all(const unsigned char[::1] arr, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i
    for i in range(n):
        if not arr[i]:
            return False
    return True


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



def __correct_constraints(const long long[::1] effect_types not None,
                          const long long[::1] effect_levels not None,
                          list grps not None,
                          const long long[::1] thetas not None,
                          list coords not None,
                          const long long[::1] plot_sizes not None,
                          object constraints not None,
                          cnp.ndarray[double, ndim=2] Y not None,
                          bint complete=False):
    cdef Py_ssize_t i, j, k

    # Determine which runs are invalid
    cdef unsigned char[::1] invalid_run = np.ascontiguousarray(constraints(Y), dtype=np.uint8)

    # Determine the total number of runs
    cdef long long nb_runs = thetas[plot_sizes.shape[0]]
    
    # Loop variables
    cdef cnp.ndarray[long long] empty_grps = np.arange(0, dtype=np.int64)
    cdef cnp.ndarray[long long] filtered_grps = np.empty(nb_runs, dtype=np.int64)
    cdef long long[::1] filtered_grps_view = filtered_grps
    cdef unsigned char[::1] all_invalid = cython.view.array(
        shape=(nb_runs,), 
        itemsize=cython.sizeof(cython.uchar), 
        format='B'
    )
    cdef Py_ssize_t nb_plots, plot_idx, factor_idx
    cdef list grps_list = [None] * effect_types.shape[0]
    cdef long long[::1] grps_factor
    cdef long long lvl_factor, grp_start, grp_end, jmp
    cdef bint is_in_grps, is_invalid
    cdef unsigned char[::1] c_res    

    # Loop over all levels
    for i in range(plot_sizes.shape[0] - 1, -1, -1):
        # Determine the jump size
        jmp = thetas[i]
        nb_plots = nb_runs // thetas[i]

        # Determine which plots are invalid
        _cython_all_axis1(all_invalid, invalid_run, nb_plots, thetas[i])

        # Loop over all plots
        for plot_idx in range(nb_plots):
            # Check if invalid
            if all_invalid[plot_idx]:

                # Specify which groups to update
                for factor_idx in range(effect_types.shape[0]):
                    lvl_factor = effect_levels[factor_idx]
                    grps_factor = grps[factor_idx]

                    # Check if the level is lower than the current level    
                    if lvl_factor < i:
                        # Determine the start and end of the groups
                        grp_start = plot_idx * jmp // thetas[lvl_factor]
                        grp_end = (plot_idx + 1) * jmp // thetas[lvl_factor]

                        # Filter the groups
                        k = 0
                        for j in range(grps_factor.shape[0]):
                            if grps_factor[j] >= grp_start and grps_factor[j] < grp_end:
                                filtered_grps_view[k] = grps_factor[j]
                                k += 1

                        # Add the filtered groups to the list
                        grps_list[factor_idx] = filtered_grps[:k].copy()
                    else:
                        if lvl_factor == i:
                            # Check if the group is in the list
                            is_in_grps = False
                            for j in range(grps_factor.shape[0]):
                                if grps_factor[j] == plot_idx:
                                    is_in_grps = True
                                    break

                            # Add the group to the list
                            if is_in_grps:
                                grps_list[factor_idx] = np.array([plot_idx], dtype=np.int64)
                            else:
                                grps_list[factor_idx] = empty_grps
                        else:
                            # Add the empty groups to the list
                            grps_list[factor_idx] = empty_grps
            
                # Loop until no longer all invalid
                is_invalid = True
                while is_invalid:
                    # Adjust the design
                    __init_unconstrained(effect_types, effect_levels, grps_list, 
                                         thetas, coords, Y, complete)
                    
                    # Validate the constraints
                    invalid_run[plot_idx*jmp:(plot_idx+1)*jmp] = \
                            np.ascontiguousarray(constraints(Y[plot_idx*jmp:(plot_idx+1)*jmp]), dtype=np.uint8)

                    # Check if the constraints are invalid
                    is_invalid = cython_all(invalid_run[plot_idx*jmp:(plot_idx+1)*jmp], jmp)

    return Y

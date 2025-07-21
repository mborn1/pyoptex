# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

# Ensure NumPy C API is initialized (important for Cython extensions using NumPy)
cnp.import_array()

cpdef compute_update_UD(
        long long level, long long grp, double[:,::1] Xi_old, double[:,::1] X, 
        long long[::1] plot_sizes, double[:,::1] c, long long[::1] thetas, double[::1] thetas_inv
    ) noexcept:
    """
    Compute the update to the information matrix after making
    a single coordinate adjustment. This update is expressed
    in the form: :math:`M^* = M + U^T D U`. D is a diagonal
    matrix in this case.

    Parameters
    ----------
    level: int
        The stratum at which the update occurs (0 for the lowest).
    grp : int
        The group within this stratum for which the update occurs.
    Xi_old : np.array(2d)
        The old runs after the update.
    X : np.array(2d)
        The new design matrix X (after the update).
    plot_sizes : np.array(1d)
        The size of each stratum b_i.
    c : np.array(2d)
        The coefficients c (every row specifies one set of a priori variance ratios). 
        The second dimension is added for Bayesian approaches.
    thetas : np.array(1d)
        The array of thetas.
        thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    thetas_inv : np.array(1d)
        The array of 1/thetas.
        thetas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/thetas[1:])))

    Returns
    -------
    U : np.array(2d)
        The U-matrix of the update
    D : np.array(2d)
        The set of diagonal matrices corresponding to the c parameter. To first row of
        D specifies a diagonal matrix corresponding to the first row of c.
    """
    cdef Py_ssize_t i, j, k, l
    # First runs
    cdef long long jmp = thetas[level]
    cdef long long run_start = grp * jmp
    cdef long long run_end = run_start + jmp
    cdef long long nb_runs = run_end - run_start
    cdef long long nb_factors = Xi_old.shape[1]
    cdef long long nb_c = c.shape[0]
    
    # Extract new X section
    cdef double[:,::1] Xi_star = X[run_start:run_end, :]

    # Initialize U and D
    cdef long long star_offset = <long long>((<double>nb_runs) * (1.0 + thetas_inv[level])) + (plot_sizes.size - level - 1)
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] U = np.zeros((2*star_offset, nb_factors), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode='c'] D = np.zeros((nb_c, 2*star_offset), dtype=np.float64)
    cdef double[:,::1] U_view = U
    cdef double[:,::1] D_view = D

    # Store level-0 results
    U_view[:nb_runs, :] = Xi_old
    U_view[star_offset: star_offset + nb_runs, :] = Xi_star
    D_view[:, :nb_runs] = -1.0
    D_view[:, star_offset: star_offset + nb_runs] = 1.0
    cdef long long co = nb_runs

    # Reshape offsets
    cdef long long dim0 = plot_sizes[0]
    cdef long long dim1 = <long long>(nb_runs // dim0)
    cdef long long ustart = 0

    # Loop before (= summations)
    if level != 0:
        # Loop over all levels before the current level
        for i in range(1, level):
            for k in range(dim1):
                # Aggregate into U_view
                U_view[co + k, :] = U_view[ustart + k*dim0, :]
                U_view[star_offset + co + k, :] = U_view[star_offset + ustart + k*dim0, :]
                for j in range(1, dim0):
                    for l in range(nb_factors):
                        U_view[co + k, l] += U_view[ustart + k*dim0 + j, l]
                        U_view[star_offset + co + k, l] += U_view[star_offset + ustart + k*dim0 + j, l]

            # Store D_view
            for j in range(nb_c):
                for k in range(dim1):
                    D_view[j, co + k] = -c[j, i-1]
                    D_view[j, star_offset + co + k] = c[j, i-1]

            # Update the variables for the next iteration
            ustart = co
            co += dim1
            dim0 = plot_sizes[i]
            dim1 = <long long>(dim1 // dim0)
        
        # Sum the level-section
        U_view[co, :] = U_view[ustart, :]
        U_view[star_offset+co, :] = U_view[star_offset+ustart, :]
        for j in range(1, dim0):
            for l in range(nb_factors):
                U_view[co, l] += U_view[ustart + j, l]
                U_view[star_offset+co, l] += U_view[star_offset+ustart + j, l]
        for j in range(nb_c):
            D_view[j, co] = -c[j, level-1]
            D_view[j, star_offset+co] = c[j, level-1]

        ustart = co
        co += 1
    

    # Loop after (= updates)
    for j in range(level, plot_sizes.size - 1):
        # Adjust group one level higher
        jmp *= plot_sizes[j]
        grp = grp // plot_sizes[j]

        # Compute U_view
        U_view[star_offset+co, :] = X[grp*jmp, :]
        for k in range(1, jmp):
            for l in range(nb_factors):
                U_view[star_offset+co, l] += X[grp*jmp+k, l]
        for k in range(nb_factors):
            U_view[co, k] = U_view[star_offset+co, k] - U_view[star_offset+ustart, k] + U_view[ustart, k]

        # Store D_view
        for j in range(nb_c):
            D_view[j, co] = -c[j, j]
            D_view[j, star_offset+co] = c[j, j]

        # Update the variables for the next iteration
        ustart = co
        co += 1

    # Return values
    return U, D
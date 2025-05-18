# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

# Initialize NumPy C-API (important for Cython modules using NumPy)
cnp.import_array()

def init_single_unconstrained_cython_impl(
        const long long[::1] colstart not None,
        list coords,
        double[:, ::1] run,
        const long long[::1] effect_types not None,
    ):
    """
    Cython implementation of <pyoptex.doe.utils.init.init_single_unconstrained>.
    """
    cdef Py_ssize_t i, j
    cdef long long factor_type, start_col, end_col
    cdef long long[::1] lvls
    cdef double[:, ::1] choices
    cdef double[::1] random_values

    cdef long long n_factors = colstart.size - 1
    cdef Py_ssize_t n_runs = run.shape[0]

    if coords is None:
        # Continuous sampling
        for i in range(n_factors):
            # Extract parameters
            factor_type = effect_types[i]
            start_col = colstart[i]
            end_col = colstart[i+1]
            
            if factor_type == 1: 
                # Continuous factor
                random_values = np.random.rand(n_runs) * 2.0 - 1.0
                run[:, start_col] = random_values
            else: 
                # Categorical factor
                lvls = np.random.randint(factor_type, size=n_runs, dtype=np.int64)              
                for j in range(n_runs):
                    if lvls[j] == factor_type - 1:
                        run[j, start_col:end_col] = -1.0
                    else:
                        run[j, start_col + lvls[j]] = 1.0
    else:
        for i in range(n_factors):
            # Extract parameters
            factor_type = effect_types[i]
            start_col = colstart[i]
            end_col = colstart[i+1]

            # Determine the choices
            choices = coords[i]

            # Generate random sampling
            lvls = np.random.randint(choices.shape[0], size=n_runs, dtype=np.int64)
            for j in range(n_runs):
                run[j, start_col:end_col] = choices[lvls[j]]
    
    return run


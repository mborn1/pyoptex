import numpy as np
import pandas as pd

def cov_time_trend(ntime, nruns, model, col_name='time'):
    assert nruns % ntime == 0, 'Number of runs should be divisable by the number of time changes'

    # Create the time array
    time_array = np.repeat(np.linspace(-1, 1, ntime), nruns//ntime).reshape(-1, 1)

    # Create the additional terms
    model = pd.DataFrame([np.concatenate((np.zeros(model.shape[1]), [1]))], columns=[*model.columns, col_name])

    # Create the covariate
    cov = (time_array, model, {col_name: 1})
    return cov

def cov_double_time_trend(
        ntime_outer, ntime_inner, nruns, model,
        col_name_outer='time_outer', col_name_inner='time_inner'
    ):
    assert nruns % ntime_outer == 0, 'Number of runs should be divisable by the number of time changes'
    assert (nruns//ntime_outer) % ntime_inner == 0, 'Number of runs within one outer timestep should be divisable by the number of inner time changes'

    # Create the time array
    time_array_outer = np.repeat(np.linspace(-1, 1, ntime_outer), nruns//ntime_outer)
    time_array_inner = np.tile(
        np.repeat(np.linspace(-1, 1, ntime_inner), (nruns//ntime_outer)//ntime_inner),
        ntime_outer
    )
    time_array = np.stack((time_array_outer, time_array_inner)).T

    # Create the additional terms
    terms = np.array([[1, 0], [0, 1]])
    model = pd.DataFrame(
        np.concatenate((np.zeros((terms.shape[0], model.shape[1])), terms), axis=1), 
        columns=[*model.columns, col_name_outer, col_name_inner]
    )

    # Create the covariate
    cov = (time_array, model, {col_name_outer: 1, col_name_inner: 1})
    return cov

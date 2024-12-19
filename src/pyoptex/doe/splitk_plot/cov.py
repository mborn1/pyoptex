import numpy as np
import pandas as pd

def no_cov(Y, X, random=False, subset=None):
    return Y, X

def cov_time_trend(ntime, nruns):
    """
    Covariance function to account for time trends.
    The entire design is divided in `ntime` equidistant 
    sections and each section has a value of time ranging
    from -1 to 1.

    Parameters
    ----------
    ntime : int
        The total number of distinct time points.
    nruns : int
        The total number of runs in the design.
    
    Returns
    -------
    t : np.array(2d)
        The array of time values.
    model : pd.Dataframe
        The additional model terms.
    type: dict
        Dictionary mapping the `col_name` to a continuous variable. 
    """
    assert nruns % ntime == 0, 'Number of runs should be divisable by the number of time changes'

    # Create the time array
    time_array = np.repeat(np.linspace(-1, 1, ntime), nruns//ntime).reshape(-1, 1)

    def _cov(Y, X, random=False, subset=slice(None, None)):
        # Extract time
        if random:
            t = np.expand_dims(np.random.rand(Y.shape[0]) * 2 - 1, 1)
        else:
            t = time_array[subset]

        # Augment Y and X
        Y = np.concatenate((Y, t), axis=1)
        X = np.concatenate((X, t), axis=1)
        return Y, X

    return _cov

def cov_double_time_trend(ntime_outer, ntime_inner, nruns):
    """
    Covariance function to account for a double time trend.
    This is defined by a global time trend divided in `ntime_outer`
    sections, where each section has its own time trend consisting of
    `ntime_inner` sections. Each section contains a constant time value
    ranging from -1 to 1.

    Parameters
    ----------
    ntime_outer : int
        The total number of global time sections.
    ntime_inner : int
        The total number of time sections per outer section, i.e.,
        the number of nested sections.
    nruns : int
        The total number of runs in the design.
    model : np.array(2d)
        The current model.
    col_name_outer : str
        The name of the new outer time column. Defaults to 'time_outer'.
    col_name_inner : str
        The name of the new inner time column. Defaults to 'time_inner'.
    
    Returns
    -------
    t : np.array(2d)
        The array of time values.
    model : pd.Dataframe
        The additional model terms.
    type: dict
        Dictionary mapping the `col_name_outer` and `col_name_inner` to a continuous variable. 
    """
    assert nruns % ntime_outer == 0, 'Number of runs should be divisable by the number of time changes'
    assert (nruns//ntime_outer) % ntime_inner == 0, 'Number of runs within one outer timestep should be divisable by the number of inner time changes'

    # Create the time array
    time_array_outer = np.repeat(np.linspace(-1, 1, ntime_outer), nruns//ntime_outer)
    time_array_inner = np.tile(
        np.repeat(np.linspace(-1, 1, ntime_inner), (nruns//ntime_outer)//ntime_inner),
        ntime_outer
    )
    time_array = np.stack((time_array_outer, time_array_inner)).T

    def _cov(Y, X, random=False, subset=slice(None, None)):
        # Extract time
        if random:
            t = np.random.rand(Y.shape[0], 2) * 2 - 1
        else:
            t = time_array[subset]

        # Augment Y and X
        Y = np.concatenate((Y, t), axis=1)
        X = np.concatenate((X, t), axis=1)
        return Y, X
    
    return _cov

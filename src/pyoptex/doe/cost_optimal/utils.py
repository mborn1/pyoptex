import numba
import numpy as np
from collections import namedtuple

from ..constraints import no_constraints
from ...utils.numba import numba_any_axis1, numba_diff_axis0

FunctionSet = namedtuple('FunctionSet', 'init sample cost metric temp accept restart insert remove constraints', defaults=(None,)*9 + (no_constraints,))
Parameters = namedtuple('Parameters', 'fn colstart coords ratios effect_types grouped_cols prior Y2X')
State = namedtuple('State', 'Y X Zs Vinv metric cost_Y costs, max_cost')

def obs_var_Zs(Yenc, colstart, grouped_cols=None):
    """
    Create the grouping matrices (1D array) for each of the factors that are
    supposed to be grouped. Runs are in the same group as long as the factor
    did not change as this is generally how it happens in engineering practices.

    Parameters
    ----------
    Yenc : np.array(2d)
        The categorically encoded design matrix
    colstart : np.array(1d)
        The start column of each factor
    grouped_cols : np.array(1d)
        A boolean array indicating whether the factor is grouped or not

    Returns
    -------
    Zs : tuple(np.array(1d) or None)
        A tuple of grouping matrices or None if the factor is not grouped
    """
    grouped_cols = grouped_cols if grouped_cols is not None else np.ones(colstart.size - 1, dtype=np.bool_)
    Zs = [None] * (colstart.size - 1)
    for i in range(colstart.size - 1):
        if grouped_cols[i]:
            borders = np.concatenate((
                np.array([0]), 
                np.where(numba_any_axis1(numba_diff_axis0(Yenc[:, colstart[i]:colstart[i+1]]) != 0))[0] + 1, 
                np.array([len(Yenc)])
            ))
            grp = np.repeat(np.arange(len(borders)-1), np.diff(borders))
            Zs[i] = grp
        else:
            Zs[i] = None
    return tuple(Zs)

@numba.njit
def obs_var(Yenc, colstart, ratios=None, grouped_cols=None):
    """
    Directly computes the observation matrix from the design. Is similar to
    :py:func:`obs_var_Zs` followed by :py:func:`obs_var_from_Zs`.

    Parameters
    ----------
    Yenc : np.array(2d)
        The categorically encoded design matrix
    colstart : np.array(1d)
        The start column of each factor
    ratios : np.array(1d)
        The variance ratios of the different groups compared to the variance of epsilon.
    grouped_cols : np.array(1d)
        A boolean array indicating whether the factor is grouped or not

    Returns
    -------
    V : np.array(2d)
        The observation covariance matrix.
    """
    grouped_cols = grouped_cols if grouped_cols is not None else np.ones(colstart.size - 1, dtype=np.bool_)

    V = np.eye(len(Yenc))
    if ratios is None:
        ratios = np.ones(colstart.size - 1)

    for i in range(colstart.size - 1):
        if grouped_cols[i]:
            borders = np.concatenate((
                np.array([0]), 
                np.where(numba_any_axis1(numba_diff_axis0(Yenc[:, colstart[i]:colstart[i+1]]) != 0))[0] + 1, 
                np.array([len(Yenc)])
            ))
            grp = np.repeat(np.arange(len(borders)-1), np.diff(borders))
            Z = np.eye(len(borders)-1)[grp]
            V += ratios[i] * Z @ Z.T
    
    return V


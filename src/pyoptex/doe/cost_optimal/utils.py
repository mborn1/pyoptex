"""
Module for all utility functions of the cost optimal designs
"""

from collections import namedtuple
import numpy as np

from ...utils.factor import FactorMixin
from ..constraints import no_constraints
from ._utils_cy import obs_var_cy

FunctionSet = namedtuple('FunctionSet', 'Y2X init cost metric constraints', defaults=(None,)*4 + (no_constraints,))
Parameters = namedtuple('Parameters', 'fn factors colstart coords ratios effect_types grouped_cols prior stats use_formulas')
State = namedtuple('State', 'Y X Zs Vinv metric cost_Y costs max_cost')
__Factor__ = namedtuple('__Factor__', 'name grouped ratio type min max levels coords', 
                        defaults=(None, True, 1, 'cont', -1, 1, None, None))
class Factor(FactorMixin, __Factor__):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        # Create the object
        self = super(Factor, cls).__new__(cls, *args, **kwargs)
        self = self.validate()

        # Validate object
        if isinstance(self.ratio, tuple) or isinstance(self.ratio, list) or isinstance(self.ratio, np.ndarray):
            assert all(r >= 0 for r in self.ratio), f'Variance ratio of factor {self.name} must be larger than or equal to zero, but is {self.ratio}'
        else:
            assert self.ratio >= 0, f'Variance ratio of factor {self.name} must be larger than or equal to zero, but is {self.ratio}'
        
        return self


def obs_var_Zs(Yenc, colstart, grouped_cols=None):
    """
    Create the grouping matrices (1D array) for each of the factors that are
    supposed to be grouped. Runs are in the same group as long as the factor
    did not change as this is generally how it happens in engineering practices.

    Parameters
    ----------
    Yenc : np.array(2d)
        The categorically encoded design matrix.
    colstart : np.array(1d)
        The start column of each factor.
    grouped_cols : np.array(1d)
        A boolean array indicating whether the factor is grouped or not.

    Returns
    -------
    Zs : tuple(np.array(1d) or None)
        A tuple of grouping matrices or None if the factor is not grouped.
    """
    # Determines the grouped columns
    grouped_cols = grouped_cols if grouped_cols is not None\
                     else np.ones(colstart.size - 1, dtype=np.bool_)
    
    # Initializes the grouping matrices
    Zs = [None] * (colstart.size - 1)

    # Computes each grouping matrix
    for i in range(colstart.size - 1):
        # Check if grouped
        if grouped_cols[i]:
            # Determine the borders of the groups
            borders = np.concatenate((
                np.array([0]), 
                np.where(np.any(np.diff(Yenc[:, colstart[i]:colstart[i+1]], axis=0) != 0, axis=1))[0] + 1, 
                np.array([len(Yenc)])
            ))

            # Determine the groups
            grp = np.repeat(np.arange(len(borders)-1), np.diff(borders))
            Zs[i] = grp

    return tuple(Zs)

def obs_var(Yenc, colstart, ratios=None, grouped_cols=None):
    """
    Directly computes the observation matrix from the design. Is similar to
    :py:func:`obs_var_Zs <pyoptex.doe.cost_optimal.utils.obs_var_Zs>` 
    followed by :py:func:`obs_var_from_Zs <pyoptex.utils.design.obs_var_from_Zs>`.

    Parameters
    ----------
    Yenc : np.array(2d)
        The categorically encoded design matrix.
    colstart : np.array(1d)
        The start column of each factor.
    ratios : np.array(1d)
        The variance ratios of the different groups compared to the variance of 
        the random errors.
    grouped_cols : np.array(1d)
        A boolean array indicating whether the factor is grouped or not.

    Returns
    -------
    V : np.array(2d)
        The observation covariance matrix.
    """
    # Determines the grouped columns and ensure correct type
    if grouped_cols is None:
        grouped_cols = np.ones(colstart.size - 1, dtype=np.uint8)
    else:
        grouped_cols = np.asarray(grouped_cols, dtype=np.uint8)

    # Initializes the variance ratios and ensure correct type
    if ratios is None:
        ratios = np.ones(colstart.size - 1, dtype=np.double)
    else:
        ratios = np.asarray(ratios, dtype=np.double)
    
    return obs_var_cy(Yenc, colstart, ratios, grouped_cols)

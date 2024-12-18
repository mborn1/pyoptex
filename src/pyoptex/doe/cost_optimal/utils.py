import numba
import numpy as np
import pandas as pd
from functools import cached_property
from collections import namedtuple

from ..constraints import no_constraints
from ...utils.numba import numba_any_axis1, numba_diff_axis0
from ..utils.design import create_default_coords, encode_design

FunctionSet = namedtuple('FunctionSet', 'init sample cost metric temp accept restart insert remove constraints', defaults=(None,)*9 + (no_constraints,))
Parameters = namedtuple('Parameters', 'fn colstart coords ratios effect_types grouped_cols prior Y2X stats use_formulas')
State = namedtuple('State', 'Y X Zs Vinv metric cost_Y costs max_cost')
__Factor__ = namedtuple('__Factor__', 'name grouped ratio type min max levels coords', 
                        defaults=(None, True, 1, 'cont', -1, 1, None, None))
class Factor(__Factor__):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):

        # Create the object
        self = super(Factor, cls).__new__(cls, *args, **kwargs)

        # Validate the object creation
        assert self.type in ['cont', 'continuous', 'cat', 'categorical'], f'The type of factor {self.name} must be either continuous or categorical, but is {self.type}'
        if isinstance(self.ratio, tuple) or isinstance(self.ratio, list) or isinstance(self.ratio, np.ndarray):
            assert all(r >= 0 for r in self.ratio), f'Variance ratio of factor {self.name} must be larger than or equal to zero, but is {self.ratio}'
        else:
            assert self.ratio >= 0, f'Variance ratio of factor {self.name} must be larger than or equal to zero, but is {self.ratio}'
        if self.is_continuous:
            assert isinstance(self.min, float) or isinstance(self.min, int), f'Factor {self.name} must have an integer or float minimum, but is {self.min}'
            assert isinstance(self.max, float) or isinstance(self.max, int), f'Factor {self.name} must have an integer or float maximum, but is {self.max}'        
            assert self.min < self.max, f'Factor {self.name} must have a lower minimum than maximum, but is {self.min} vs. {self.max}'
            assert self.coords is None, f'Cannot specify coordinates for continuous factors, but factor {self.name} has {self.coords}. Please specify the levels'
            assert self.levels is None or len(self.levels) >= 2, f'A continuous factor must have at least two levels when specified, but factor {self.name} has {len(self.levels)}'
        else:
            assert len(self.levels) >= 2, f'A categorical factor must have at least 2 levels, but factor {self.name} has {len(self.levels)}'
            if self.coords is not None:
                coords = np.array(self.coords)
                assert len(coords.shape) == 2, f'Factor {self.name} requires a 2d array as coordinates, but has {len(coords.shape)} dimensions'
                assert coords.shape[0] == len(self.levels), f'Factor {self.name} requires one encoding for every level, but has {len(self.levels)} levels and {coords.shape[0]} encodings'
                assert coords.shape[1] == len(self.levels) - 1, f'Factor {self.name} has N levels and requires N-1 dummy columns, but has {len(self.levels)} levels and {coords.shape[1]} dummy columns'
                assert np.linalg.matrix_rank(coords) == coords.shape[1], f'Factor {self.name} does not have a valid (full rank) encoding'

        return self

    @property
    def mean(self):
        return (self.min + self.max) / 2

    @property
    def scale(self):
        return (self.max - self.min) / 2

    @property
    def is_continuous(self):
        return self.type.lower() in ['cont', 'continuous']

    @property 
    def is_categorical(self):
        return not self.is_continuous

    @property
    def coords_(self):
        if self.coords is None:
            if self.is_continuous:
                if self.levels is not None:
                    coord = self.normalize(np.array(self.levels))
                else:
                    coord = create_default_coords(1)
            else:
                coord = create_default_coords(len(self.levels))
                coord = encode_design(coord, np.array([len(self.levels)]))
        else:
            coord = np.array(self.coords).astype(np.float64)
        return coord

    def normalize(self, data):
        if self.is_continuous:
            return (data - self.mean) / self.scale
        else:
            m = {lname: i for i, lname in enumerate(self.levels)}
            if isinstance(data, str):
                x = m[data]
            else:
                x = pd.Series(data).map(m)
                if isinstance(data, np.ndarray):
                    x = x.to_numpy()
            return x

    def denormalize(self, data):
        if self.is_continuous:
            return data * self.scale + self.mean
        else:
            m = {i: lname for i, lname in enumerate(self.levels)}
            if isinstance(data, int) or isinstance(data, float):
                x = m[int(data)]
            else:
                x = pd.Series(data).astype(int).map(m)
                if isinstance(data, np.ndarray):
                    x = x.to_numpy()
            return x


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


"""
Module for all utility functions of the generic coordinate-exchange algorithm
"""

from collections import namedtuple
import numpy as np

from ...utils.factor import FactorMixin

FunctionSet = namedtuple('FunctionSet', 'metric Y2X constraints constraintso init')
Parameters = namedtuple('Parameters', 'fn factors nruns effect_types effect_levels grps ratios coords prior colstart Zs Vinv')
State = namedtuple('State', 'Y X metric')

__RandomEffect__ = namedtuple('__RandomEffect__', 'Z ratio', defaults=(None, 1))
class RandomEffect(__RandomEffect__):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        self = super(RandomEffect, cls).__new__(cls, *args, **kwargs)
        # Assert Z
        assert self.Z is not None and len(self.Z) > 0, f'Z must be an array with at least one element, but is {self.Z}' 
        
        # Assert type of Z
        Z = np.array(self.Z)
        max_Z = np.max(Z)
        assert len(Z.shape) == 1, f'Z must be a 1-D array'
        assert np.issubdtype(Z.dtype, np.integer), f'Z must be an integer array'
        assert np.all(np.any(np.expand_dims(Z, 0) == np.expand_dims(np.arange(max_Z), 1), axis=1)), f'The Z array must contain all integers in the interval [0, np.max(Z)]'
        
        # Assert ratios
        if isinstance(self.ratio, (tuple, list, np.ndarray)):
            assert all(r >= 0 for r in self.ratio), f'Variance ratios must be larger than or equal to zero, but is {self.ratio}'
        else:
            assert self.ratio >= 0, f'Variance ratios must be larger than or equal to zero, but is {self.ratio}'
        
        return self
    
    def __eq__(self, other):
        Zeq = np.all(self.Z == other.Z)
        ratioeq = False
        if isinstance(self.ratio, (tuple, list, np.ndarray)):
            if isinstance(other.ratio, (tuple, list, np.ndarray)):
                ratioeq = np.all(np.array(self.ratio) == np.array(other.ratio))
        else:
            if not isinstance(other.ratio, (tuple, list, np.ndarray)):
                ratioeq = self.ratio == other.ratio
        return Zeq and ratioeq

__Factor__ = namedtuple('__Factor__', 'name re type min max levels coords',
                        defaults=(None, None, 'cont', -1, 1, None, None))
class Factor(FactorMixin, __Factor__):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        self = super(Factor, cls).__new__(cls, *args, **kwargs)
        return self.validate()

"""
Module for all utility functions of the generic coordinate-exchange algorithm
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from ..utils.design import create_default_coords, encode_design

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
class Factor(__Factor__):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):

        # Create the object
        self = super(Factor, cls).__new__(cls, *args, **kwargs)

        # Validate the object creation
        assert self.type in ['cont', 'continuous', 'cat', 'categorical', 'qual', 'qualitative', 'quan', 'quantitative'], f'The type of factor {self.name} must be either continuous or categorical, but is {self.type}'
        if self.is_continuous:
            assert isinstance(self.min, (float, int)), f'Factor {self.name} must have an integer or float minimum, but is {self.min}'
            assert isinstance(self.max, (float, int)), f'Factor {self.name} must have an integer or float maximum, but is {self.max}'        
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
        return self.type.lower() in ['cont', 'continuous', 'quan', 'quantitative']

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

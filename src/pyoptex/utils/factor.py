
from collections import namedtuple
import numpy as np
import pandas as pd

from .design import create_default_coords, encode_design

__Factor__ = namedtuple('__Factor__', 'name type min max levels coords',
                        defaults=(None, 'cont', -1, 1, None, None))

class FactorMixin:
    def validate(self):
        # Check for a mixture component
        if self.type in ('mixt', 'mixture'):
            # Alter default minimum and maximum
            assert (self.min == -1 and self.max == 1), 'Cannot specify a minimum and maximum for mixture components. Use levels parameters to specify minimum and maximum consumption per run'

            # Define default coordinates as positive
            levels = self.levels if self.levels is not None \
                     else np.array([0, 0.5, 1])
            
            # Transform to a new factor
            params = self._asdict()
            params['type'] = 'cont_mixture'
            params['levels'] = levels
            return self.__class__.__new__(self.__class__, **params)

        # Validate the object creation
        assert self.type in ['cont', 'continuous', 'cont_mixture', 'cat', 'categorical', 'qual', 'qualitative', 'quan', 'quantitative'], f'The type of factor {self.name} must be either continuous, categorical or mixture, but is {self.type}'
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
        return self.type.lower() in ['cont', 'continuous', 'quan', 'quantitative', 'cont_mixture']

    @property 
    def is_categorical(self):
        return not self.is_continuous
    
    @property
    def is_mixture(self):
        return self.type.lower() in ['cont_mixture']

    @property
    def coords_(self):
        if self.coords is None:
            if self.is_continuous:
                if self.levels is not None:
                    coord = np.expand_dims(self.normalize(np.array(self.levels)), 1)
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

class Factor(FactorMixin, __Factor__):
    def __new__(cls, *args, **kwargs):
        self = super(Factor, cls).__new__(cls, *args, **kwargs)
        return self.validate()

    

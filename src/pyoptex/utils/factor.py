
from collections import namedtuple
import numpy as np
import pandas as pd

from .design import create_default_coords, encode_design

__Factor__ = namedtuple('__Factor__', 'name type min max levels coords',
                        defaults=(None, 'cont', -1, 1, None, None))

class FactorMixin:
    """
    Factor Mixin to be used whenever creating a new factor class.
    This mixin contains common validation code and functions to normalize 
    and denormalize columns based on the factors settings.

    Attributes
    ----------
    name : str
        The name of the factor
    type : str
        The type of the factor, can be 'continuous', 'cont', 'categorical',
        'cat', 'mixture', 'mixt', 'quantitative', 'quant', 'qualitative', 'qual'
    min : float
        The minimum of the factor. Used for continuous factors.
    max : float
        The maximum of the factor. Used for continuous factors.
    levels : list(float) or list(str)
        For continuous factors, this is the list of possible numerical values.
        For categorical factors, this is a list of strings for the possible 
        categories.
    coords : list(np.array(2d))
        Only possible for categorical factors. The list of encodings of each level.
    """
    def validate(self):
        """
        Validation of the settings. To be called in the __new__ constructor.
        """
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
        """
        The mean of the factor for normalization of a continuous
        factor. y = (x - mean) / scale.

        Returns
        -------
        mean : float
            The mean.
        """
        return (self.min + self.max) / 2

    @property
    def scale(self):
        """
        The scale of the factor for normalization of a continuous
        factor. y = (x - mean) / scale.

        Returns
        -------
        scale : float
            The scale.
        """
        return (self.max - self.min) / 2

    @property
    def is_continuous(self):
        """
        Checks whether the factor is continuous. Also includes
        mixture factors.

        Returns
        -------
        is_continuous : bool
            If this factor is continuous (or mixture).
        """
        return self.type.lower() in ['cont', 'continuous', 'quan', 'quantitative', 'cont_mixture']

    @property 
    def is_categorical(self):
        """
        Check wether the factor is catgorical.

        Returns
        -------
        is_categorical : bool
            If this factor is categorical.
        """
        return not self.is_continuous
    
    @property
    def is_mixture(self):
        """
        Checks wether the factor is a mixture.

        Returns
        -------
        is_mixture : bool
            If this factor is a mixture component.
        """
        return self.type.lower() in ['cont_mixture']

    @property
    def coords_(self):
        """
        Extracts the encoded coordinates for the factor. Returns
        the categorical encoding if specified, or the encoded levels for
        a continuous factor between -1 and 1. Otherwise returns the default
        coordinates which is effect encoding for categorical factors, (-1, 0, 1)
        for continuous factors and (0, 0.5, 1) for mixtures.

        Returns
        -------
        coords : list(np.array(2d))
            The encoded coordinates.
        """
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
        """
        Normalizes the data coming from this factor.
        A categorical factor is normalized to an array of numbers
        from 0 until the number of levels. A continuous factor
        is normalized to be between -1 and 1, using the min and max
        specified.

        .. note::
            This is the inverse of
            :py:func:`denormalize <pyoptex.utils.factor.FactorMixin.denormalize>`.

        Parameters
        ----------
        data : float or str or np.array(1d) or pd.Series
            The data to be normalized. A continuous factor
            requires a float, array of floats or a series of
            floats. A categorical factor requires a string
            or series of strings.
        
        Returns
        -------
        norm_data : float or int or np.array(1d) or pd.Series
            The normalized data, in the same type as the data.
            Categorical inputs become integers.
        """
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
        """
        Denormalizes the data coming from this factor.
        A categorical factor is denormalized from an array of numbers
        with values from 0 until the number of levels, to an array
        of strings. A continuous factor
        is denormalized from between -1 and 1, to between min and max.

        .. note::
            This is the inverse of
            :py:func:`normalize <pyoptex.utils.factor.FactorMixin.normalize>`.

        Parameters
        ----------
        data : float or int or np.array(1d) or pd.Series
            The data to be normalized. A continuous factor
            requires a float, array of floats or a series of
            floats. A categorical factor requires an int
            or series of ints.
        
        Returns
        -------
        denorm_data : float or str or np.array(1d) or pd.Series
            The denormalized data, in the same type as the data.
            Categorical inputs become strings.
        """
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
    """
    The base factor, also used for the analysis.
    """
    def __new__(cls, *args, **kwargs):
        self = super(Factor, cls).__new__(cls, *args, **kwargs)
        return self.validate()

    

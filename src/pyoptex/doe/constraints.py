import re
import numpy as np
import numba
from numba.typed import List
from .utils.design import encode_design
from ..utils.numba import numba_all_axis1

def parse_constraints_script(script, factors, exclude=True, eps=1e-6):
    """
    Parse a script of constraints using the factor names. It creates a constraint evaluation
    function capable which returns True if any constraints are violated.

    For example, "(`A` > 0) & (`B` < 0)" specifies that if A is larger than 0, B cannot
    be smaller than 0.

    Parameters
    ----------
    script : str
        The script to parse
    effect_types : dict
        The type of each effect mapping the effect name to 1 for continuous or higher for categorical
        with that many levels.
    eps : float
        The epsilon parameter to be used in the parsing

    Returns
    -------
    constraint_tree : func
        The constraint tree which can be used to extract a function for both normal and encoded
        design matrices using .func() and .encode() respectively.
    """
    # Extract column names
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))

    # Function to convert name to column object
    def name2col(m):
        i = col_names.index(m.group(1))
        return f'Col({i}, factors[{i}], {colstart[i]})'

    # Create the script
    script = re.sub(r'"(.*?)"', lambda m: f'Col("{m.group(1)}", None)', script)
    script = re.sub(r'`(.*?)`', name2col, script)
    script = script.replace('^', '**')
    script = 'Col('.join(x[:x.find(')')] + re.sub(r'[-\.\d]+', lambda m: f'Col({m.group(0)}, None)', x[x.find(')'):]) for x in script.split('Col('))
    if not exclude:
        script = f'~({script})'
    tree = eval(script, {'Col': Col, 'BinaryCol': BinaryCol, 'UnaryCol': UnaryCol, 'CompCol': CompCol, 
                         'factors': factors, 'eps': Col(eps, None)})
    return tree


class Col:
    CATEGORICAL_MSG = 'Can only perform comparison with categorical columns'

    def __init__(self, col, factor, colstart=0):
        self.col = col
        self.factor = factor
        self.colstart = colstart
        self.is_constant = self.factor is None
        self.is_categorical = (self.factor is not None) and (self.factor.is_categorical)

        self.pre_normalized_ = False

    ##############################################
    def __str__(self):
        return f'Y__[:,{self.col}]' if not self.is_constant else str(self.col)

    def func(self):
        return numba.njit(eval(f'lambda Y__: {str(self)}'))

    def _encode(self):
        if self.is_constant:
            return str(self.col)
        elif self.is_categorical:
            if self.pre_normalized_:
                return f'(Y__[:,{self.colstart}:{self.colstart+len(self.factor.levels)-1}])'
            else:
                raise NotImplementedError('This branch has not been implemented yet')
        else:
            if self.pre_normalized_:
                return f'(Y__[:,{self.colstart}])'
            else:
                return f'(Y__[:,{self.colstart}] * {self.factor.scale} + {self.factor.mean})'

    def encode(self):
        return numba.njit(eval(f'lambda Y__: {self._encode()}', {'numba_all_axis1': numba_all_axis1, 'np': np}))

    ##############################################

    def __validate_unary__(self):
        if self.is_categorical:
            raise ValueError(self.CATEGORICAL_MSG)

    def __validate_binary__(self, other):
        if self.is_categorical or other.is_categorical:   
            raise ValueError(self.CATEGORICAL_MSG)

    def __validate_comp__(self, other):
        if self.is_categorical:
            if not other.is_constant:
                raise ValueError(self.CATEGORICAL_MSG)
            if other.col not in self.factor.levels:
                raise ValueError(f'Categorical comparison unknown: {other.col} not in levels {self.factor.levels}')
        if other.is_categorical:
            if not self.is_constant:
                raise ValueError(self.CATEGORICAL_MSG)
            if self.col not in other.factor.levels:
                raise ValueError(f'Categorical comparison unknown: {self.col} not in levels {other.factor.levels}')

    ##############################################
    def __pos__(self):
        self.__validate_unary__()
        return UnaryCol(self, prefix='+')

    def __neg__(self):
        self.__validate_unary__()
        return UnaryCol(self, prefix='-')

    def __abs__(self):
        self.__validate_unary__()
        return UnaryCol(self, prefix='abs(', suffix=')')

    def __invert__(self):
        self.__validate_unary__()
        return UnaryCol(self, prefix='~')

    ##############################################
    def __add__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '+')

    def __sub__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '-')

    def __mul__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '*')

    def __floordiv__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '//')

    def __truediv__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '/')

    def __mod__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '%')

    def __pow__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '**')        

    def __eq__(self, other):
        self.__validate_comp__(other)
        return CompCol(self, other, '==')

    def __ne__(self, other):
        self.__validate_comp__(other)
        return CompCol(self, other, '!=')

    def __ge__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '>=')

    def __gt__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '>')

    def __le__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '<=')

    def __lt__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '<')

    def __and__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '&')

    def __or__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '|')

    def __xor__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '^')


class UnaryCol(Col):
    def __init__(self, col, prefix='', suffix=''):
        super().__init__(col, None)
        self.prefix = prefix
        self.suffix = suffix
    def __str__(self):
        return f'{self.prefix}{str(self.col)}{self.suffix}'
    def _encode(self):
        return f'{self.prefix}{self.col._encode()}{self.suffix}'


class BinaryCol(Col):
    def __init__(self, left, right, sep):
        super().__init__(left, None)
        self.col2 = right
        self.sep = sep

    def __str__(self):
        return f'({str(self.col)} {self.sep} {str(self.col2)})'
    
    def _encode(self):
        return f'({self.col._encode()} {self.sep} {self.col2._encode()})'


class CompCol(BinaryCol):
    def __str__(self):
        return f'({str(self.col)} {self.sep} {str(self.col2)})'

    def __encode__(self, col1, col2):
        assert col1.is_categorical and col2.is_constant, 'Can only compare constant and categorical column'
        if not col1.pre_normalized_:
            encoded = encode_design(np.array([[col1.factor.normalize(col2.col)]]), np.array([len(col1.factor.levels)]), 
                                    List([col1.factor.coords_]))[0]
            col2.col = f'np.array({list(encoded)})'
            col1.pre_normalized_ = True
        return f'numba_all_axis1({col1._encode()} {self.sep} {col2._encode()})'

    def _encode(self):
        if self.col.is_categorical:
            return self.__encode__(self.col, self.col2)
        elif self.col2.is_categorical:
            return self.__encode__(self.col2, self.col)
        else:
            return f'({self.col._encode()} {self.sep} {self.col2._encode()})'

no_constraints = Col('np.zeros(len(Y__), dtype=np.bool_)', None)


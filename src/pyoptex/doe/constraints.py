"""
Module containing the constraints functions.
"""

import re

import numba
import numpy as np
from numba.typed import List

from ..utils.numba import numba_all_axis1
from ..utils.design import encode_design


def parse_constraints_script(script, factors, exclude=True, eps=1e-6):
    """
    Parses a formula of constraints. The function returns an object of type
    :py:class:`Col <pyoptex.doe.constraints.Col>` which can generate a function evaluating
    either the encoded or non-encoded design matrix.
    The generated function returns True where the constraints are violated. 

    For example, "(`A` > 0) & (`B` < 0)" specifies that if A is larger than 0, B cannot
    be smaller than 0.

    Another example, "(`A` == "L1") & (`B` < 0)" specifies that if A is "L1", B cannot
    be smaller than 0.

    Use constraint.func() to retrieve a function which can evaluate a decoded design matrix,
    use constraint.encode() to retrieve a function which can evaluate an encoded design matrix.

    .. note::
        Parameter `exclude` can invert the function, specifying where the 
        constraints are upheld.

    Parameters
    ----------
    script : str
        The script to parse
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.
    exclude : bool
        Determines whether the presented formula should be excluded 
        (= return True when the constraints are violated), or not
        (= return False when the constraints are violated).
    eps : float
        The epsilon parameter to be used in the parsing. this parameter
        can be used to deal with numerical precision. For example, instead of
        "`A` > 0", use "`A` > 0 + eps".

    Returns
    -------
    constraint : :py:class:`Col <pyoptex.doe.constraints.Col>`
        The root constraint which can be converted to a function to evaluate
        either the encoded or decoded design matrix.
    """
    # Extract column names
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate((
        [0], 
        np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))
    ))

    # Function to convert name to column object
    def name2col(m):
        i = col_names.index(m.group(1))
        return f'Col({i}, factors[{i}], {colstart[i]})'

    # Create the script
    script = re.sub(r'"(.*?)"', lambda m: f'Col("{m.group(1)}", None)', script)
    script = re.sub(r'`(.*?)`', name2col, script)
    script = script.replace('^', '**')
    script = 'Col('.join(
        x[:x.find(')')] + re.sub(r'[-\.\d]+', lambda m: f'Col({m.group(0)}, None)', x[x.find(')'):]) 
        for x in script.split('Col(')
    )
    if not exclude:
        script = f'~({script})'
    tree = eval(script, {'Col': Col, 'BinaryCol': BinaryCol, 'UnaryCol': UnaryCol, 
                         'CompCol': CompCol, 'factors': factors, 'eps': Col(eps, None)})
    return tree


class Col:
    """
    The base Column object to represent a constraint. If this object
    is contained in a variable called `constraint`, use constraint.func()
    to retrieve a function which can evaluate the specified constraints
    on a decoded design matrix. Use constraint.encode() to retrieve
    a function which can evaluate the specified constraints on an encoded
    design matrix.

    Attributes
    ----------
    col : float or int or string
        Meaning depends on the value of `factor`.
    factor : None or :py:class:`Factor <pyoptex.utils.factor.Factor>`
        If None, this column represents a simple constant and `col`
        represents the constant value. Otherwise, it represents
        the column and the attribute `col` represents the index of the
        factor in the design.
    colstart : int
        The start of the column in the encoded design matrix.
    is_constant : bool
        Whether this column is a constant
    is_categorical : bool
        Whether this is a categorical column
    """
    CATEGORICAL_MSG = 'Can only perform comparison with categorical columns'

    def __init__(self, col, factor, colstart=0):
        self.col = col
        self.factor = factor
        self.colstart = colstart
        self.is_constant = self.factor is None
        self.is_categorical = (self.factor is not None) and (self.factor.is_categorical)

        self.col_encoded_ = self.col
        self.col_normalized_ = self.col
        self.pre_normalized_ = False
        self.pre_normalized_encoded_ = False

    ##############################################
    def __str__(self):
        """
        Retrieves the string representation of a function to evaluate 
        the decoded, but normalized design matrix.

        Returns
        -------
        constraint : str
            The string representation of the function.
        """
        if self.is_constant:
            return str(self.col_normalized_)
        elif self.is_categorical:
            if self.pre_normalized_:
                return f'Y__[:,{self.col}]'
            else:
                raise NotImplementedError('This branch has not been implemented yet')
        else:
            if self.pre_normalized_:
                return f'(Y__[:,{self.col}])'
            else:
                return f'(Y__[:,{self.col}] * {self.factor.scale} + {self.factor.mean})'

    def func(self):
        """
        Retrieves a function to evaluate the decoded design matrix.

        Returns
        -------
        constraint : func(Y)
            A function which returns True when the constraints are violated
            for that run. Y is a decoded design, but normalized design matrix.
        """
        return numba.njit(eval(f'lambda Y__: {str(self)}'))

    def _encode(self):
        """
        Retrieves the string representation of a function to evaluate 
        the encoded design matrix.

        Returns
        -------
        constraint : str
            The string representation of the function.
        """
        if self.is_constant:
            return str(self.col_encoded_)
        elif self.is_categorical:
            if self.pre_normalized_encoded_:
                return f'(Y__[:,{self.colstart}:{self.colstart+len(self.factor.levels)-1}])'
            else:
                raise NotImplementedError('This branch has not been implemented yet')
        else:
            if self.pre_normalized_encoded_:
                return f'(Y__[:,{self.colstart}])'
            else:
                return f'(Y__[:,{self.colstart}] * {self.factor.scale} + {self.factor.mean})'

    def encode(self):
        """
        Retrieves a function to evaluate the encoded design matrix.

        Returns
        -------
        constraint : func(Y)
            A function which returns True when the constraints are violated
            for that run. Y is a encoded design design matrix.
        """
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
    def _str(self, col1, col2):
        assert col1.is_categorical and col2.is_constant, 'Can only compare constant and categorical column'
        if not col1.pre_normalized_:
            col2.col_normalized_ = col1.factor.levels.index(col2.col)
            col1.pre_normalized_ = True
        return f'({str(col1)} {self.sep} {str(col2)})'

    def __str__(self):
        if self.col.is_categorical:
            return self._str(self.col, self.col2)
        elif self.col2.is_categorical:
            return self._str(self.col2, self.col)
        else:
            return f'({str(self.col)} {self.sep} {str(self.col2)})'

    def __encode__(self, col1, col2):
        assert col1.is_categorical and col2.is_constant, 'Can only compare constant and categorical column'
        if not col1.pre_normalized_encoded_:
            encoded = encode_design(np.array([[col1.factor.normalize(col2.col)]]), np.array([len(col1.factor.levels)]), 
                                    List([col1.factor.coords_]))[0]
            col2.col_encoded_ = f'np.array({list(encoded)})'
            col1.pre_normalized_encoded_ = True
        return f'numba_all_axis1({col1._encode()} {self.sep} {col2._encode()})'

    def _encode(self):
        if self.col.is_categorical:
            return self.__encode__(self.col, self.col2)
        elif self.col2.is_categorical:
            return self.__encode__(self.col2, self.col)
        else:
            return f'({self.col._encode()} {self.sep} {self.col2._encode()})'


"""A function always returning False"""
no_constraints = Col('np.zeros(len(Y__), dtype=np.bool_)', None)

def mixture_constraints(names, factors):
    """
    Create the mixture constraints based on the names of the
    mixture factors.
    
    .. note::
        The number of names is
        the number of mixture components minus 1. The final mixture
        component is never explicitely specified.

        E.g. if A + B + C = 1, only ('A', 'B'), ('A', 'C'), or
        ('B', 'C') is specified.

    Parameters
    ----------
    names : list(str)
        The names of the mixture components.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The factors of the experiment.
    
    Returns
    -------
    mixture_constraint : :py:class:`Col <pyoptex.doe.constraints.Col>`
        The mixture constraint.
    """
    script = ' + '.join(f'`{name}`' for name in names) + f' > 1 + eps'
    return parse_constraints_script(script, factors)

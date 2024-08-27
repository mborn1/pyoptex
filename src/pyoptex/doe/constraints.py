import re
import numpy as np
import numba


def parse_script(script, effect_types, eps=1e-6):
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
    # Determine column starts
    columns = list(effect_types.keys())
    effect_types = np.array(list(effect_types.values()))
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))

    # Create the script
    script = re.sub(r'`(.*?)`', lambda m: f'Col({columns.index(m.group(1))}, (effect_types, col_start))', script)
    script = script.replace('^', '**')
    script = re.sub(r'(?<!Col\()(-*(?=[\.\d]+)[\.\d]+)', lambda m: f'Col({m.group(1)}, (effect_types, col_start), is_constant=True)', script)
    tree = eval(script, {'Col': Col, 'BinaryCol': BinaryCol, 'UnaryCol': UnaryCol, 'CompCol': CompCol, 
                         'effect_types': effect_types, 'col_start': colstart, 'eps': Col(eps, (effect_types, colstart), is_constant=True)})
    return tree


class Col:
    CATEGORICAL_MSG = 'Can only perform comparison with categorical columns'

    def __init__(self, col, state, is_constant=False):
        self.col = col
        self.state = state
        self.is_constant = is_constant
        self.is_categorical = (not self.is_constant) and isinstance(col, int) and self.effect_types[col] > 1

    @property
    def effect_types(self):
        return self.state[0]
    
    @property
    def col_start(self):
        return self.state[1]

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
            if other.col >= self.effect_types[self.col]:
                raise ValueError('Categorical comparison outside range')
        if other.is_categorical:
            if not self.is_constant:
                raise ValueError(self.CATEGORICAL_MSG)
            if self.col >= self.effect_types[other.col]:
                raise ValueError('Categorical comparison outside range')

    ##############################################
    def __pos__(self):
        self.__validate_unary__()
        return UnaryCol(self, self.state, prefix='+')

    def __neg__(self):
        self.__validate_unary__()
        return UnaryCol(self, self.state, prefix='-')

    def __abs__(self):
        self.__validate_unary__()
        return UnaryCol(self, self.state, prefix='abs(', suffix=')')

    def __invert__(self):
        self.__validate_unary__()
        return UnaryCol(self, self.state, prefix='~')

    ##############################################
    def __add__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '+', self.state)

    def __sub__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '-', self.state)

    def __mul__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '*', self.state)

    def __floordiv__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '//', self.state)

    def __truediv__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '/', self.state)

    def __mod__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '%', self.state)

    def __pow__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '**', self.state)        

    def __eq__(self, other):
        self.__validate_comp__(other)
        return CompCol(self, other, '==', self.state)

    def __ne__(self, other):
        self.__validate_comp__(other)
        return CompCol(self, other, '!=', self.state)

    def __ge__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '>=', self.state)

    def __gt__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '>', self.state)

    def __le__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '<=', self.state)

    def __lt__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '<', self.state)

    def __and__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '&', self.state)

    def __or__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '|', self.state)

    def __xor__(self, other):
        self.__validate_binary__(other)
        return BinaryCol(self, other, '^', self.state)

    ##############################################
    def __str__(self):
        return f'Y__[:,{self.col}]' if not self.is_constant else str(self.col)

    def func(self):
        return eval(f'lambda Y__: {str(self)}')

    def _encode(self):
        return f'Y__[:,{self.col_start[self.col]}]' if not self.is_constant else str(self.col)

    def encode(self):
        return eval(f'lambda Y__: {self._encode()}')


class UnaryCol(Col):
    def __init__(self, col, state, prefix='', suffix=''):
        super().__init__(col, state)
        self.prefix = prefix
        self.suffix = suffix
    def __str__(self):
        return f'{self.prefix}{str(self.col)}{self.suffix}'
    def _encode(self):
        return f'{self.prefix}{self.col._encode()}{self.suffix}'


class BinaryCol(Col):
    def __init__(self, left, right, sep, state):
        super().__init__(left, state)
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
        # Check encoding
        if col2.col == self.effect_types[col1.col] - 1:
            # Last column
            x = ' & '.join((f'({str(Col(self.col_start[col1.col] + i, self.state))} {self.sep} -1)' for i in range(col2.col)))
            return f'({x})'
        else:
            return f'({str(Col(self.col_start[col1.col] + col2.col, self.state))} {self.sep} 1)'

    def _encode(self):
        if self.col.is_categorical:
            return self.__encode__(self.col, self.col2)
        elif self.col2.is_categorical:
            return self.__encode__(self.col2, self.col)
        else:
            return f'({self.col._encode()} {self.sep} {self.col2._encode()})'

no_constraints = Col('np.zeros(len(Y__), dtype=np.bool_)', (None, None), is_constant=True)


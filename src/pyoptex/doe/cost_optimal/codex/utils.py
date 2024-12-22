"""
Module for all utility functions of the CODEX algorithm
"""

from collections import namedtuple

from ..utils import FunctionSet as FunctionSeto

FunctionSet = namedtuple('FunctionSet', ' '.join(FunctionSeto._fields) + ' sample temp accept restart insert remove optimizers final_optimizers', defaults=tuple(FunctionSeto._field_defaults.values()) + (None,)*8)

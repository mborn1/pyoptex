#!/usr/bin/env python3

"""
Example: fractional factorial design generation.

Demonstrates a 2^(4-1) fractional factorial with the generator D = ABC.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import fractional_factorial
from pyoptex.utils import Factor

factors = [
    Factor("A", levels=[-1, 1]),
    Factor("B", levels=[-1, 1]),
    Factor("C", levels=[-1, 1]),
    Factor("D", levels=[-1, 1]),
]

Y = fractional_factorial(factors, generators={"D": ["A", "B", "C"]})
print("=== Fractional Factorial 2^(4-1), D=ABC ===")
print(f"Runs: {len(Y)}")
print(Y)

log_checkpoint("fractional_shape", list(Y.shape))

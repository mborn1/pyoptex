#!/usr/bin/env python3

"""
Example: full factorial design generation.

Demonstrates a full factorial with continuous and categorical factors.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import full_factorial
from pyoptex.utils import Factor

factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5, levels=[1, 3, 5]),
    Factor("Catalyst", type="categorical", levels=["A", "B"]),
]

Y = full_factorial(factors)
print("=== Full Factorial ===")
print(f"Runs: {len(Y)}")
print(Y)

log_checkpoint("full_factorial_shape", list(Y.shape))

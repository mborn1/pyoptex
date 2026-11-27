#!/usr/bin/env python3

"""
Example: fractional factorial design generation.

Demonstrates a 2^(4-1) fractional factorial with the generator D = ABC,
its fold-over (D = -ABC), and the L9 orthogonal array as a 3^(4-2)
design with generators C = A + B and D = 2A + B (mod 3).
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import fractional_factorial, min_aberration_generators
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

# The other half of the 2^4: running both halves is the classical fold-over
Y_fold = fractional_factorial(factors, generators={"D": ["-A", "B", "C"]})
print("=== Fold-over 2^(4-1), D=-ABC ===")
print(f"Runs: {len(Y_fold)}")
print(Y_fold)

log_checkpoint("fractional_foldover_shape", list(Y_fold.shape))

# The L9 orthogonal array: 4 three-level factors in 9 runs.
# A dict gives the coefficients: D = 2A + B (mod 3). With levels coded 1, 2, 3
# this reproduces the standard printed L9 table row for row.
factors_3 = [Factor(name, min=1, max=3) for name in "ABCD"]
Y_l9 = fractional_factorial(factors_3, generators={"C": ["A", "B"], "D": {"A": 2, "B": 1}}, n_levels=3)
print("=== L9: Fractional Factorial 3^(4-2), C=A+B, D=2A+B ===")
print(f"Runs: {len(Y_l9)}")
print(Y_l9)

log_checkpoint("fractional_l9_shape", list(Y_l9.shape))

# Generators from the catalogue of minimum aberration designs: the smallest
# design of resolution IV for 7 factors (16 runs: E = ABC, F = ABD, G = ACD)
factors_7 = [Factor(name) for name in "ABCDEFG"]
generators = min_aberration_generators(factors_7, resolution=4)
Y_ma = fractional_factorial(factors_7, generators=generators)
print("=== Minimum aberration 2^(7-3), resolution IV ===")
print(f"Generators: {generators}")
print(f"Runs: {len(Y_ma)}")

log_checkpoint("fractional_min_aberration_shape", list(Y_ma.shape))

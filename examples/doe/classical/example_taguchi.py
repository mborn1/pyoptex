#!/usr/bin/env python3

"""
Example: Taguchi orthogonal arrays.

Demonstrates an L9 for 4 three-level factors, and an L8 where factors are
assigned to columns 1, 2 and 4 so that column 3 stays free for the
interaction between the first two factors (Taguchi's interaction table:
1 x 2 = 3).
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import taguchi
from pyoptex.utils import Factor

# L9: 4 factors at 3 levels in 9 runs
factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
    Factor("Time", min=10, max=60),
    Factor("Catalyst", type="categorical", levels=["A", "B", "C"]),
]
Y = taguchi("L9", factors)
print("=== Taguchi L9 (4 three-level factors) ===")
print(f"Runs: {len(Y)}")
print(Y)

log_checkpoint("taguchi_l9_shape", list(Y.shape))

# L8: 3 two-level factors on columns 1, 2 and 4; column 3 estimates Speed x Feed
factors_2 = [Factor("Speed", min=100, max=500), Factor("Feed", min=0.05, max=0.25), Factor("Depth", min=0.5, max=2.0)]
Y8 = taguchi("L8", factors_2, columns=[1, 2, 4])
print("=== Taguchi L8, columns 1, 2 and 4 ===")
print(f"Runs: {len(Y8)}")
print(Y8)

log_checkpoint("taguchi_l8_shape", list(Y8.shape))

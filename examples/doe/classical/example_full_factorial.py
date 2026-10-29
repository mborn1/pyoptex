#!/usr/bin/env python3

"""
Example: full factorial design generation.

Demonstrates a full factorial with continuous and categorical factors.
Continuous factors without explicit levels default to two levels
(min, max); ``n_levels=3`` adds the mid-range level.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex._seed import set_seed
from pyoptex.doe.classical import full_factorial, randomize
from pyoptex.utils import Factor

set_seed(42)

factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5, levels=[1, 3, 5]),
    Factor("Catalyst", type="categorical", levels=["A", "B"]),
]

# Temperature: 2 default levels, Pressure: 3 explicit levels, Catalyst: 2 levels -> 12 runs
Y = full_factorial(factors)
print("=== Full Factorial ===")
print(f"Runs: {len(Y)}")
print(Y)

log_checkpoint("full_factorial_shape", list(Y.shape))

# Temperature now gets 3 levels (150, 175, 200); Pressure keeps its explicit levels -> 18 runs
Y3 = full_factorial(factors, n_levels=3)
print("=== Full Factorial, n_levels=3 ===")
print(f"Runs: {len(Y3)}")

log_checkpoint("full_factorial_n_levels_3_shape", list(Y3.shape))

# Add 2 center runs: these are repeated at each Catalyst level -> 4 center runs.
# Then randomize the run order; the index keeps the standard order of each run.
Y_run = randomize(full_factorial(factors, n_center=2))
print("=== Full Factorial with center points, randomized run order ===")
print(f"Runs: {len(Y_run)}")
print(Y_run)

log_checkpoint("full_factorial_randomized_shape", list(Y_run.shape))

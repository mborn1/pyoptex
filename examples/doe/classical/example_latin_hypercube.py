#!/usr/bin/env python3

"""
Example: Latin Hypercube Sampling (space-filling design).

Demonstrates an optimized Latin Hypercube design for 3 continuous
factors with 20 runs.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex._seed import set_seed
from pyoptex.doe.classical import latin_hypercube
from pyoptex.utils import Factor

set_seed(42)

factors = [
    Factor("X1", min=0, max=10),
    Factor("X2", min=0, max=10),
    Factor("X3", min=0, max=10),
]

Y = latin_hypercube(factors, n=20, iterations=50)
print("=== Latin Hypercube (3 factors, 20 runs) ===")
print(f"Runs: {len(Y)}")
print(Y)

log_checkpoint("lhs_shape", list(Y.shape))

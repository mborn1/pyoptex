#!/usr/bin/env python3

"""
Example: Plackett-Burman screening design.

Demonstrates a Plackett-Burman design for 7 factors.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import plackett_burman
from pyoptex.utils import Factor

factors = [Factor(f"X{i}", min=0, max=100) for i in range(7)]

Y = plackett_burman(factors)
print("=== Plackett-Burman (7 factors) ===")
print(f"Runs: {len(Y)}")
print(Y)

log_checkpoint("pb_shape", list(Y.shape))

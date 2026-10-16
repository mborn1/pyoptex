#!/usr/bin/env python3

"""
Example: Plackett-Burman screening design.

Demonstrates a Plackett-Burman design for 7 factors in 8 runs, and the
12-run design for 6 factors: the unused columns serve as dummy factors to
estimate the error, and any 3 columns of the 12-run design contain a full
2^3 factorial, so interactions among a few active factors can be estimated
(Lin & Draper, 1992).
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

# More runs than factors: 6 factors in the 12-run design
Y12 = plackett_burman(factors[:6], n_runs=12)
print("=== Plackett-Burman (6 factors, 12 runs) ===")
print(f"Runs: {len(Y12)}")
print(Y12)

log_checkpoint("pb_n_runs_12_shape", list(Y12.shape))

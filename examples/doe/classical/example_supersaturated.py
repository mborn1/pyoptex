#!/usr/bin/env python3

"""
Example: supersaturated screening design.

Demonstrates Lin's (1993) supersaturated design: 10 two-level factors in
only 6 runs. With fewer runs than factors, not all main effects can be
estimated at once, so this only works under effect sparsity: when few
factors are active, they can be found with e.g. stepwise selection.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

import numpy as np

from pyoptex.doe.classical import supersaturated
from pyoptex.utils import Factor

factors = [Factor(f"X{i}", min=0, max=100) for i in range(10)]

Y = supersaturated(factors)
print("=== Supersaturated design (10 factors, 6 runs) ===")
print(f"Runs: {len(Y)}")
print(Y)

# E(s^2): the average squared inner product between two coded columns.
# Zero would mean orthogonal columns, which is impossible with fewer runs than factors.
X = (Y.to_numpy(dtype=np.float64) - 50) / 50
S = X.T @ X
s = S[~np.eye(len(S), dtype=bool)]
print(f"E(s^2): {np.mean(s**2):.2f}")

log_checkpoint("ssd_shape", list(Y.shape))

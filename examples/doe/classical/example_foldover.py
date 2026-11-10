#!/usr/bin/env python3

"""
Example: fold-over of a screening design.

A Plackett-Burman design screens 11 factors in 12 runs, but its main
effects are partially aliased with two-factor interactions. Running the
fold-over (the mirror image of all runs) as a second stage frees every
main effect from the two-factor interactions. The two stages form two
blocks.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from itertools import combinations

import numpy as np

from pyoptex.doe.classical import foldover, plackett_burman
from pyoptex.utils import Factor

factors = [Factor(f"X{i}", min=0, max=10) for i in range(11)]

# Stage 1: screening in 12 runs
Y = plackett_burman(factors)
print("=== Stage 1: Plackett-Burman (11 factors, 12 runs) ===")
print(f"Runs: {len(Y)}")

# Stage 2: the mirror image of all runs
Yf = foldover(Y, factors)
print("=== Stages 1 + 2: with the fold-over ===")
print(f"Runs: {len(Yf)} in blocks {sorted(Yf['Block'].unique())}")
print(Yf)

# Main effects are now orthogonal to every two-factor interaction:
# every product of three coded columns sums to zero.
names = [str(f.name) for f in factors]
X = (Yf[names].to_numpy() - 5) / 5
worst = max(abs(np.prod(X[:, list(c)], axis=1).sum()) for c in combinations(range(11), 3))
print(f"Largest |sum of a product of three columns|: {worst:.1f}")

log_checkpoint("foldover_shape", list(Yf.shape))

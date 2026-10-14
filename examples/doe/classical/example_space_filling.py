#!/usr/bin/env python3

"""
Example: space-filling designs.

Compares a Latin hypercube, a Sobol' sequence, a Halton sequence and a
uniform design for 2 factors and 16 runs. Space-filling designs suit
computer experiments and response surfaces of unknown shape, where the
runs should cover the whole factor space rather than its corners.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

import numpy as np
from scipy.stats import qmc

from pyoptex._seed import set_seed
from pyoptex.doe.classical import halton, latin_hypercube, sobol, uniform
from pyoptex.utils import Factor

set_seed(42)

factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
]
n = 16  # a power of 2, as Sobol' sequences require

designs = {
    "Latin hypercube (maximin)": latin_hypercube(factors, n),
    "Latin hypercube (discrepancy)": latin_hypercube(factors, n, method="discrepancy"),
    "Sobol'": sobol(factors, n),
    "Halton": halton(factors, n),
    "Uniform": uniform(factors, n),
}

# Compare in the unit square: lower discrepancy = more uniform spread,
# larger minimum distance = no two runs close together.
print(f"=== Space-filling designs (2 factors, {n} runs) ===")
print(f"{'Design':<32}{'Discrepancy':>12}{'Min distance':>14}")
for name, Y in designs.items():
    unit = np.column_stack([(Y[str(f.name)] - f.min) / (f.max - f.min) for f in factors])
    dists = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=-1)
    min_dist = dists[~np.eye(n, dtype=bool)].min()
    print(f"{name:<32}{qmc.discrepancy(unit):>12.4f}{min_dist:>14.3f}")

log_checkpoint("space_filling_shapes", [list(Y.shape) for Y in designs.values()])

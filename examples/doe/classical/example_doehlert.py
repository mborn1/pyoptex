#!/usr/bin/env python3

"""
Example: Doehlert (uniform shell) design.

Demonstrates a Doehlert design for 3 factors: 13 runs spread uniformly
on a sphere around the center. The factors get 5, 7 and 3 levels in the
order they are listed, so the factor that is hardest to vary goes last.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import doehlert
from pyoptex.utils import Factor

factors = [
    Factor("pH", min=3, max=7),  # 5 levels
    Factor("Temperature", min=20, max=80),  # 7 levels: the easiest to vary
    Factor("Catalyst load", min=1, max=3),  # 3 levels: the hardest to vary
]

Y = doehlert(factors)
print("=== Doehlert design (3 factors) ===")
print(f"Runs: {len(Y)}")
print(Y.round(2))
for f in factors:
    print(f"{f.name}: {Y[str(f.name)].round(10).nunique()} levels")

log_checkpoint("doehlert_shape", list(Y.shape))

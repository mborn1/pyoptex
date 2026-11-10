#!/usr/bin/env python3

"""
Example: definitive screening design (DSD).

Demonstrates a 6-factor DSD in 13 runs and its alias matrix against the
two-factor interactions: the main effects are not biased by any active
two-factor interaction.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from itertools import combinations

import numpy as np
import pandas as pd

from pyoptex.doe.classical import definitive_screening
from pyoptex.doe.classical.evaluate import alias_matrix
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, partial_rsm_names

names = ["Temperature", "Pressure", "Time", "Speed", "Feed", "Depth"]
factors = [Factor(name, min=0, max=10) for name in names]

Y = definitive_screening(factors)
print("=== Definitive Screening Design (6 factors) ===")
print(f"Runs: {len(Y)}")
print(Y)

# Fitted model: intercept + main effects.
Y2X_primary = model2Y2X(partial_rsm_names({name: "lin" for name in names}), factors)

# Potential omitted terms: all two-factor interactions (one row per interaction).
tfi = pd.DataFrame(
    [[1 if name in pair else 0 for name in names] for pair in combinations(names, 2)],
    columns=names,
)
Y2X_potential = model2Y2X(tfi, factors)

A = alias_matrix(Y, factors, Y2X_primary, Y2X_potential)
print(f"Alias matrix shape (intercept + main effects x two-factor interactions): {A.shape}")
print(f"Largest alias of a main effect with a two-factor interaction: {np.max(np.abs(A[1:])):.2e}")

log_checkpoint("dsd_shape", list(Y.shape))

#!/usr/bin/env python3

"""
Example: evaluating a classical design.

Demonstrates the alias matrix and the estimation variances of a
2^(4-1) fractional factorial with D = ABC (resolution IV): main effects
are clear of two-factor interactions, but the two-factor interactions
are aliased in pairs (AB = CD, AC = BD, AD = BC).
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from itertools import combinations

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.classical import fractional_factorial
from pyoptex.doe.classical.evaluate import alias_matrix, design_info, estimation_variance
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

names = ["A", "B", "C", "D"]
factors = [Factor(name) for name in names]
Y = fractional_factorial(factors, generators={"D": ["A", "B", "C"]})
print("=== Fractional Factorial 2^(4-1), D=ABC ===")
print(Y)

# Model rows for the two-factor interactions (one row per interaction)
pairs = list(combinations(names, 2))
tfi = pd.DataFrame([[1 if name in pair else 0 for name in names] for pair in pairs], columns=names)
tfi_names = ["".join(pair) for pair in pairs]

# 1. Main-effects model, with the two-factor interactions as potential terms:
#    no main effect is biased by an active two-factor interaction.
Y2X_main = model2Y2X(partial_rsm_names({name: "lin" for name in names}), factors)
A = alias_matrix(Y, factors, Y2X_main, model2Y2X(tfi, factors))
print("\n=== Alias matrix: main-effects model vs two-factor interactions ===")
print(pd.DataFrame(A, index=["1", *names], columns=tfi_names).round(2))

# 2. Model with main effects and AB, AC, AD: each estimated interaction
#    absorbs the one it is aliased with (AB with CD, AC with BD, AD with BC).
model = pd.concat([partial_rsm_names({name: "lin" for name in names}), tfi.iloc[:3]], ignore_index=True)
A = alias_matrix(Y, factors, model2Y2X(model, factors), model2Y2X(tfi.iloc[3:], factors))
print("\n=== Alias matrix: main effects + AB, AC, AD vs BC, BD, CD ===")
print(pd.DataFrame(A, index=["1", *names, *tfi_names[:3]], columns=tfi_names[3:]).round(2))

# 3. Estimation variances (unit error variance): 1/n for an orthogonal design
var = estimation_variance(Y, factors, Y2X_main)
print("\n=== Estimation variances, main-effects model ===")
print(pd.Series(var, index=["1", *names]).round(4))

info = design_info(Y, factors, Y2X_main)
print(f"\nD-efficiency: {info['d_efficiency']:.2f}%")
print(f"G-efficiency: {info['g_efficiency']:.2f}%")
print(f"Average prediction variance: {info['average_prediction_variance']:.4f}")

log_checkpoint("evaluate_estimation_variance", np.asarray(var))

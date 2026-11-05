#!/usr/bin/env python3

"""
Example: a classical design next to an optimal design.

Compares a face-centered central composite design (CCD) with a pyoptex
I-optimal design with the same number of runs, the same factors and the
same full quadratic model. Both are scored with the same metrics, which
match the optimizer's own definitions (d_efficiency = 100 * Dopt / n,
average_prediction_variance = -Iopt).

Besides efficiency, the comparison covers the two arguments usually made
for classical designs:

* Confounding within the model: correlation between the estimated terms,
  summarized by the variance inflation factor (VIF, 1 = orthogonal).
* Confounding with terms outside the model: the alias matrix against the
  omitted third-order terms (ABC, A^2B, AB^2, ...).

and the power to detect each type of term.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from itertools import permutations

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.classical import central_composite
from pyoptex.doe.classical.evaluate import alias_matrix, design_info, power, variance_inflation
from pyoptex.doe.fixed_structure import Factor, create_fixed_structure_design, create_parameters, default_fn
from pyoptex.doe.fixed_structure.metric import Iopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

names = ["Temperature", "Pressure", "Time"]
factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
    Factor("Time", min=10, max=60),
]
model = partial_rsm_names({name: "quad" for name in names})
Y2X = model2Y2X(model, factors)

# Term type of every model column, from the exponents of the model rows
exponents = model[names].to_numpy()
term_type = np.select(
    [exponents.sum(axis=1) == 0, exponents.max(axis=1) == 2, exponents.sum(axis=1) == 2],
    ["intercept", "quadratic", "interaction"],
    default="main effect",
)

# Omitted third-order terms: ABC and the six A^2B-type terms. Pure cubic terms
# are left out: in any three-level design A^3 equals A, so they cannot be separated.
third_order = [[1, 1, 1]] + [
    [2 if i == a else 1 if i == b else 0 for i in range(3)] for a, b in permutations(range(3), 2)
]
Y2X_third = model2Y2X(pd.DataFrame(third_order, columns=names), factors)

# Classical: face-centered CCD with the textbook 6 center points (20 runs)
Y_ccd = central_composite(factors, alpha="face-centered")
nruns = len(Y_ccd)

# Optimal: I-optimal design with the same number of runs
fn = default_fn(factors, Iopt(), Y2X)
params = create_parameters(factors, fn, nruns)
Y_iopt, _ = create_fixed_structure_design(params, n_tries=10)

designs = {"Face-centered CCD": Y_ccd, "I-optimal": Y_iopt}
results = {}
for name, Y in designs.items():
    # Score both designs on the same random points of the design region
    set_seed(0)
    info = design_info(Y, factors, Y2X, n_samples=10000)
    vif = variance_inflation(Y, factors, Y2X)
    aliases = alias_matrix(Y, factors, Y2X, Y2X_third)[term_type != "intercept"]
    pw = power(Y, factors, Y2X, effect_size=1.0, alpha=0.05)
    results[name] = {
        "D-efficiency (%)": info["d_efficiency"],
        "G-efficiency (%)": info["g_efficiency"],
        "Avg. prediction variance": info["average_prediction_variance"],
        "Max VIF": np.nanmax(vif),
        "Max |alias| with 3rd order": np.max(np.abs(aliases)),
        "Sum of squared aliases": np.sum(aliases**2),
        "Power, main effects": pw[term_type == "main effect"].mean(),
        "Power, interactions": pw[term_type == "interaction"].mean(),
        "Power, quadratic": pw[term_type == "quadratic"].mean(),
    }

print(f"=== Classical vs optimal: {nruns} runs, full quadratic model in 3 factors ===")
print(pd.DataFrame(results).round(3))
print(
    "\nLower is better for the prediction variance, VIF and aliases; higher for the rest.\n"
    "Power: two-sided t-test at alpha = 0.05 for a coefficient equal to the error standard\n"
    "deviation (a main effect of 2 standard deviations from min to max), averaged per term type."
)

log_checkpoint("classical_vs_optimal_runs", nruns)

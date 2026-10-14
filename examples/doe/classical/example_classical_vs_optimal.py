#!/usr/bin/env python3

"""
Example: a classical design next to an optimal design.

Compares a face-centered central composite design (CCD) with a pyoptex
I-optimal design with the same number of runs, the same factors and the
same full quadratic model. Both are scored with the same metrics, which
match the optimizer's own definitions (d_efficiency = 100 * Dopt / n,
average_prediction_variance = -Iopt).
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex._seed import set_seed
from pyoptex.doe.classical import central_composite
from pyoptex.doe.classical.evaluate import design_info
from pyoptex.doe.fixed_structure import Factor, create_fixed_structure_design, create_parameters, default_fn
from pyoptex.doe.fixed_structure.metric import Iopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
    Factor("Time", min=10, max=60),
]
model = partial_rsm_names({"Temperature": "quad", "Pressure": "quad", "Time": "quad"})
Y2X = model2Y2X(model, factors)

# Classical: face-centered CCD with the textbook 6 center points (20 runs)
Y_ccd = central_composite(factors, alpha="face-centered")
nruns = len(Y_ccd)

# Optimal: I-optimal design with the same number of runs
fn = default_fn(factors, Iopt(), Y2X)
params = create_parameters(factors, fn, nruns)
Y_iopt, _ = create_fixed_structure_design(params, n_tries=10)

# Score both designs on the same random points of the design region
designs = {"Face-centered CCD": Y_ccd, "I-optimal": Y_iopt}
infos = {}
for name, Y in designs.items():
    set_seed(0)
    infos[name] = design_info(Y, factors, Y2X, n_samples=10000)

print(f"=== Classical vs optimal: {nruns} runs, full quadratic model in 3 factors ===")
print(f"{'Metric':<30}" + "".join(f"{name:>20}" for name in designs))
rows = [
    ("D-efficiency (%)", "d_efficiency", "{:.2f}"),
    ("A-efficiency (%)", "a_efficiency", "{:.2f}"),
    ("G-efficiency (%)", "g_efficiency", "{:.2f}"),
    ("Avg. prediction variance", "average_prediction_variance", "{:.4f}"),
]
for label, key, fmt in rows:
    print(f"{label:<30}" + "".join(f"{fmt.format(infos[name][key]):>20}" for name in designs))
print("Lower average prediction variance is better; the I-optimal design minimizes it.")

log_checkpoint("classical_vs_optimal_runs", nruns)

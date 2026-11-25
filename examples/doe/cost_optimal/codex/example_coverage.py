#!/usr/bin/env python3

# Normal imports
import os
import time

import numpy as np
import pandas as pd

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.metric import Coverage

# Set the seed
set_seed(42)

# Define the factors, with more levels than the default three for the continuous factors
factors = [
    Factor("A", type="continuous", grouped=False, levels=np.linspace(-1, 1, 11)),
    Factor("B", type="continuous", grouped=False, min=2, max=5, levels=np.linspace(2, 5, 11)),
    Factor("C", type="categorical", grouped=False, levels=["L1", "L2", "L3"]),
]
log_checkpoint("factor_names", [str(f.name) for f in factors])

# Define the criterion: space-filling, without a model
metric = Coverage()

# Define the existing design, whose gaps the new runs fill
prior = pd.DataFrame([[-1, 2, "L1"], [-0.6, 2.3, "L1"], [-1, 2.6, "L2"], [-0.4, 2, "L3"]], columns=["A", "B", "C"])

# Cost function: a run costs 2, so the budget allows about 8 new runs next to the 4 existing ones
transition_costs = {"A": 1, "B": 1, "C": 1}
max_cost = 24
cost_fn = parallel_worker_cost(transition_costs, factors, max_cost)

#######################################################################

# Simulation parameters
nsims = 20
nreps = 1
fn = default_fn(nsims, factors, cost_fn, metric, None)
params = create_parameters(factors, fn, prior=prior)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps, validate=True)
end_time = time.time()

log_checkpoint("Y_shape", list(Y.shape))
log_checkpoint("Y_columns", Y.columns.tolist())
log_checkpoint("Y_values", Y.values.tolist())
log_checkpoint("metric", float(state.metric))
log_checkpoint("n_experiments", len(state.Y))

#######################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, "example_coverage.csv"), index=False)
print(Y)

print("Completed optimization")
print(f"Metric: {state.metric:.3f}")
print(f"Cost: {state.cost_Y}")
print(f"Number of experiments: {len(state.Y)}")
print(f"Execution time: {end_time - start_time:.3f}")

#######################################################################

# # Generic evaluation
# from pyoptex.doe.utils.evaluate import design_heatmap
# design_heatmap(Y, factors).show()

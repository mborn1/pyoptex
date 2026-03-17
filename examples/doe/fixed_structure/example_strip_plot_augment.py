#!/usr/bin/env python3

# Python imports
import os
import time

import numpy as np
import pandas as pd

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import (
    Factor,
    RandomEffect,
    create_fixed_structure_design,
    create_parameters,
    default_fn,
)
from pyoptex.doe.fixed_structure.metric import Aopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

# Set the seed
set_seed(42)

# Augmentation: start from 15 runs (5 plots x 3 runs), add 5 runs (one per plot) to get 20 runs
n_old = 15
nruns = 20
nplots = 5

# Z: first 15 runs = plots 0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4; runs 15-18 augment plots 0-3, run 19 is in new plot 5
Z = np.concatenate([np.repeat(np.arange(nplots), 3), np.arange(nplots)]).astype(np.int64)
Z[-1] = 5
re = RandomEffect(Z)

# Define the factors
factors = [
    Factor("A", re, type="categorical", levels=["L1", "L2", "L3"]),
    Factor("B", type="continuous"),
    Factor("C", type="continuous"),
]

# Create a partial response surface model
model = partial_rsm_names(
    {
        "A": "tfi",
        "B": "quad",
        "C": "quad",
    }
)
Y2X = model2Y2X(model, factors)
log_checkpoint("factor_names", [str(f.name) for f in factors])
log_checkpoint("model_shape", list(model.shape))
log_checkpoint("model_values", model.values.tolist())

# Define the metric
metric = Aopt()

# Prior design: 15 runs, denormalized. Same plot must have same A; B,C vary per run.
# Plots 0..4 get A levels L1,L2,L3,L1,L2; B,C are simple spread over [-1,1].
prior = pd.DataFrame(
    [
        ["L1", -1.0, 0.0],
        ["L1", 0.0, 1.0],
        ["L1", 1.0, -1.0],
        ["L2", -1.0, -1.0],
        ["L2", 0.0, 0.0],
        ["L2", 1.0, 1.0],
        ["L3", -1.0, 1.0],
        ["L3", 0.0, -1.0],
        ["L3", 1.0, 0.0],
        ["L1", -0.5, 0.5],
        ["L1", 0.5, -0.5],
        ["L1", 0.0, 0.0],
        ["L2", -0.5, -0.5],
        ["L2", 0.5, 0.5],
        ["L2", 0.0, 0.0],
    ],
    columns=["A", "B", "C"],
)

#########################################################################

# Parameter initialization
n_tries = 10

# Create the set of operators
fn = default_fn(factors, metric, Y2X)
params = create_parameters(factors, fn, nruns, prior=prior)

# Create design
start_time = time.time()
Y, state = create_fixed_structure_design(params, n_tries=n_tries)
end_time = time.time()

log_checkpoint("Y_shape", list(Y.shape))
log_checkpoint("Y_columns", Y.columns.tolist())
log_checkpoint("Y_values", Y.values.tolist())
log_checkpoint("metric", float(state.metric))

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, "example_strip_plot_augment.csv"), index=False)

print("Completed optimization")
print(f"Metric: {state.metric:.3f}")
print(f"Execution time: {end_time - start_time:.3f}")
print(Y)

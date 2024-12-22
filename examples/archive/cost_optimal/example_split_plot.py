#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time

# Library imports
from pyoptex.doe.cost_optimal import create_cost_optimal_design, default_fn
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import obs_var_from_Zs
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt

np.random.seed(42)

# Define parameters
effects = {
    # Define effect type, model type, is_grouped
    'A': (1, 'tfi', False),
    'B': (1, 'tfi', False),
    'H': (1, 'tfi', False),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.array([v[2] for v in effects.values()])

#########################################################################

# Cost function
nb_plots = 8
runs_per_plot = 4
def cost_fn(Y):
    # Short-circuit
    if len(Y) == 0:
        return []

    # Determine the number of resets
    resets = np.zeros(len(Y))
    resets[1:] = (np.diff(Y[:, 0]) != 0).astype(np.int64)
    for i in range(runs_per_plot, len(resets)):
        if np.all(resets[i-(runs_per_plot-1):i] == 0):
            resets[i] = 1

    # Determine number of runs per plot
    idx = np.concatenate([[0], np.flatnonzero(resets), [len(Y)]])
    plot_costs = [None] * (len(idx) - 1)
    for i in range(len(idx)-1):
        plot_costs[i] = (np.ones(idx[i+1] - idx[i]), runs_per_plot, np.arange(idx[i], idx[i+1]))
    
    return [
        (resets, nb_plots - 1, np.arange(len(Y))),
        *plot_costs
    ]

# Split plot covariance matrix
def cov(Y, X, Zs, Vinv, costs, random=False):
    resets = costs[0][0].astype(np.int64)
    Z1 = np.cumsum(resets)
    Zs = [Z1]
    V = obs_var_from_Zs(Zs, len(Y), ratios=np.array([1.]))
    Vinv = np.expand_dims(np.linalg.inv(V), 0)
    return Y, X, Zs, Vinv

# Define the metric
metric = Dopt(cov=cov)

# Define prior
prior = None

# Define global ratios (not useful)
ratios = None

#########################################################################

# Parameter initialization
nsims = 500
nreps = 1

# Create the set of operators
fn = default_fn(nsims, cost_fn, metric)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    effect_types, fn, model=model, 
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    prior=prior, ratios=ratios,
    validate=True
)
end_time = time.time()

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'results', 'example_split_plot.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


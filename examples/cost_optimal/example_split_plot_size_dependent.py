#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time

# Library imports
from pyoptex.doe.constraints import parse_script
from pyoptex.doe.cost_optimal import create_cost_optimal_design, default_fn
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import obs_var_from_Zs

# Parse arguments
import argparse
parser = argparse.ArgumentParser(description="Perform a simulation")
parser.add_argument('--proc', type=int, help='The current process', default=0)
parser.add_argument('--py-name', type=str, help='The Python file name')
args = parser.parse_args()

# Define parameters
effects = {
    # Define effect type, model type
    'A': (1, 'tfi'),
    'B': (1, 'tfi'),
    'C': (1, 'quad'),
    'D': (1, 'quad'),
    'E': (1, 'quad'),
    'F': (1, 'quad'),
    'G': (1, 'quad'),
    'H': (1, 'tfi'),
    'I': (1, 'tfi'),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

# Constraints
constraints = parse_script(f'((`B` > 0) & (`C` < -0.4)) | ((`B` < 0) & (`C` > 0.4))', effect_types).encode()

#########################################################################

# Cost function
nb_plots = 6
runs_per_plot_low = 7
runs_per_plot_high = 14
def cost_fn(Y):
    # Short-circuit
    if len(Y) == 0:
        return []

    # Determine the number of resets
    resets = np.zeros(len(Y))
    resets[1:] = np.any(np.diff(Y[:, :2], axis=0) != 0, axis=1).astype(np.int64)
    for i in range(0, len(resets)):
        # Reset when factor does not change for a period depending on the first factor
        if (Y[i-1, 0] == -1 and i >= runs_per_plot_low and np.all(resets[i-(runs_per_plot_low-1):i] == 0))\
                or (Y[i-1, 0] == 1 and i >= runs_per_plot_high and np.all(resets[i-(runs_per_plot_high-1):i] == 0)):
            resets[i] = 1

    # Determine number of runs per plot
    idx = np.concatenate([[0], np.flatnonzero(resets), [len(Y)]])
    plot_costs = [None] * (len(idx) - 1)
    for i in range(len(idx)-1):
        if Y[idx[i], 0] == -1:
            rp = runs_per_plot_low
        else:
            rp = runs_per_plot_high

        plot_costs[i] = (np.ones(idx[i+1] - idx[i]), rp, np.arange(idx[i], idx[i+1]))
    
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

coords = [
    np.array([-1, 1]).reshape(-1, 1),
    np.array([-1, 1]).reshape(-1, 1),
    np.array([-1, -1/3, 1/3, 1]).reshape(-1, 1),
    None, None, None, None, None, None
]

#########################################################################

# Parameter initialization
nsims = 15000
nreps = 1

# Create the set of operators
from pyoptex.doe.cost_optimal.init import init_feasible
fn = default_fn(
    nsims, cost_fn, metric, 
    init=lambda p: init_feasible(p, max_tries=0, force_cost_feasible=False), 
    constraints=constraints
)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    effect_types, fn, model=model, coords=coords,
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    validate=True
)
end_time = time.time()

#########################################################################

# Write design to storage
filename = __file__ if args.py_name is None else args.py_name
root, filename = os.path.split(filename)
Y.to_csv(os.path.join(root, 'results', f'{filename[:-3]}_{args.proc}.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


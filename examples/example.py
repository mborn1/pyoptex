#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time

# Library imports
from pyoptex.doe.cost_optimal import simulate_wrapper, default_fn
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt
from pyoptex.doe.cost_optimal.cov import cov_block_cost
from pyoptex.doe.cost_optimal.cost import discount_effect_trans_cost

np.random.seed(42)

# Define parameters
effects = {
    # Define effect type, model type, is_grouped, cost
    'A': (3, 'tfi', True, 2*60),
    'E': (1, 'quad', False, 5),
    'F': (1, 'quad', False, 5),
    'G': (1, 'quad', False, 5),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.array([v[2] for v in effects.values()])
costs = np.array([c for (_, _, _, c) in effects.values()])

#########################################################################

# Cost function
max_cost = np.array([3*4*60])
base_cost = 5
cost_fn = discount_effect_trans_cost(costs, effect_types, base_cost)

# Define the metric
metric = Iopt()

# Define prior
# prior = pd.read_csv('example_design.csv')[list(effects.keys())].to_numpy()[:5]
prior = None

# Define multiple ratios
ratios = np.stack((np.ones(len(effects)) * 10, np.ones(len(effects)) * 0.1))

#########################################################################

# Parameter initialization
nsims = 100
nreps = 1

# Create the set of operators
fn = default_fn(nsims, cost_fn, metric)

# Create design
start_time = time.time()
Y, state = simulate_wrapper(
    effect_types, max_cost, fn, model=model, 
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    prior=prior, ratios=ratios,
    validate=True
)
end_time = time.time()

#########################################################################

# Write design to storage
Y.to_csv(f'example_design.csv', index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


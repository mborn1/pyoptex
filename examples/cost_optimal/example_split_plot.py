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
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt
from pyoptex.doe.cost_optimal.cov import cov_block_cost
from pyoptex.doe.cost_optimal.cost import discount_effect_trans_cost
from pyoptex.doe.cost_optimal.init import init_feasible, full_factorial

np.random.seed(42)

# Define parameters
effects = {
    # Define effect type, model type, is_grouped
    'A': (1, 'tfi', False),
    'B': (1, 'tfi', False),
    # 'C': (1, 'tfi', False),
    # 'E': (1, 'tfi', False),
    # 'F': (1, 'tfi', False),
    # 'G': (1, 'tfi', False),
    'H': (1, 'tfi', False),
    'J': (1, 'tfi', False),
    'K': (1, 'tfi', False),
    'L': (1, 'tfi', False),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.array([v[2] for v in effects.values()])

#########################################################################

# Cost function
max_cost = np.array([8] + [4]*8)
def cost_fn(Y):
    # Short-circuit
    if len(Y) == 0:
        return np.zeros(len(max_cost))

    # HTV factors
    htv = np.concatenate([[False], np.any(np.diff(Y[:, :2], axis=0) != 0, axis=1)])
    htv[::4] = True  # Every 4th is always a reset
    htx = htv.astype(np.int64)

    # Count ETV
    etv = np.zeros((len(Y), 8))
    htx_i = np.concatenate([np.flatnonzero(htv), [len(Y)]])
    for i in range(min(8, len(htx_i) - 1)):
        etv[htx_i[i]:htx_i[i+1], i] = 1

    # Combine costs
    costs = np.concatenate((htv[:, np.newaxis], etv), axis=1).T
    
    return costs

# Split plot covariance matrix
def cov(Y, X, Zs, Vinv, costs, random=False):
    Z1 = np.cumsum(np.any((np.diff(Y[:, :2], axis=0) > 0), axis=1))
    Zs = [Z1]
    V = obs_var_from_Zs(Zs, len(Y), ratios=np.array([[1.]]))
    return Y, X, Zs, Vinv, costs

# Define the metric
metric = Iopt()

# Define prior
prior = None

# Define global ratios (not useful)
ratios = None

#########################################################################

# Parameter initialization
nsims = 100
nreps = 1

def init(params):
    # Generate full factorial design
    Y = full_factorial(params.colstart, params.coords)
    X = params.Y2X(Y)

    # Drop blocks
    keep = np.ones(len(Y), dtype=np.bool_)
    for i in range((len(Y) // 4) - 1):
        keep[i*4:(i+1)*4] = False
        Xk = X[keep]
        if np.linalg.matrix_rank(Xk) != X.shape[1]:
            keep[i*4:(i+1)*4] = True
    Y = Y[keep]
    
    return Y

# Create the set of operators
fn = default_fn(nsims, cost_fn, metric)
fn = fn._replace(init=init)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
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


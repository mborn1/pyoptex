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
from pyoptex.doe.cost_optimal.metric import Aliasing
from pyoptex.doe.cost_optimal.init import init

np.random.seed(42)

# Define parameters
effects = {
    # Define effect type, model type, is_grouped
    'A': (1, 'quad'),
    'B': (1, 'quad'),
    'C': (1, 'quad'),
    'D': (1, 'quad'),
    'E': (1, 'quad'),
    'F': (1, 'quad'),
    # 'G': (1, 'quad'),
    # 'H': (1, 'quad'),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

#########################################################################

# Cost function
nruns = 30
def cost_fn(Y):
    return [(np.ones(len(Y)), nruns, np.arange(len(Y)))]

# # Define the weights
# n1, n2 = len(effects), len(model)-2*len(effects)-1
# W = np.block([
#     [ np.ones(( 1, 1)), np.ones(( 1, n1)),  np.ones(( 1, n2)), np.zeros(( 1, n1))], # Intercept
#     [ np.ones((n1, 1)), np.ones((n1, n1)),  np.ones((n1, n2)),  np.ones((n1, n1))], # Main effects
#     [ np.ones((n2, 1)), np.ones((n2, n1)), np.zeros((n2, n2)), np.zeros((n2, n1))], # TFI effects
#     [np.zeros((n1, 1)), np.ones((n1, n1)), np.zeros((n1, n2)), np.zeros((n1, n1))], # Quad effects
# ])
# W[np.arange(len(W)), np.arange(len(W))] = 0
# assert np.all(W == W.T)

# Define the weights
n1, n2 = len(effects), len(model)-2*len(effects)-1
w1, w2 = 1/((n1+1)*(n1+1)), 1/((n2+n1)*(n1+1))
W = np.block([
    [ w1 * np.ones(( 1, 1)), w1 * np.ones(( 1, n1)), w2 * np.ones(( 1, n2)), w2 * np.zeros(( 1, n1))], # Intercept
    [ w1 * np.ones((n1, 1)), w1 * np.ones((n1, n1)), w2 * np.ones((n1, n2)), w2 *  np.ones((n1, n1))], # Main effects
])
W[np.arange(len(W)), np.arange(len(W))] = 0

# Define the metric
metric = Aliasing(np.arange(len(effects)+1), np.arange(len(model)), W=W)

#########################################################################

# Parameter initialization
nsims = 2500
nreps = 1

# Create the set of operators
fn = default_fn(nsims, cost_fn, metric, init=lambda p: init(p, nruns))

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    effect_types, fn, model=model, 
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    validate=True
)
end_time = time.time()

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'results', 'example_approx_omars.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


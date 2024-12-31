#!/usr/bin/env python3

# Python imports
import time
import os
import numpy as np

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.utils.model import mixtureY2X
from pyoptex.doe.cost_optimal import Factor, cost_fn
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor('A', type='mixture', grouped=False, levels=np.arange(0, 1.0001, 0.05)),
    Factor('B', type='mixture', grouped=False, levels=np.arange(0, 1.0001, 0.05)),
]

# Create a Scheffe model
Y2X = mixtureY2X(
    factors,
    mixture_effects=(('A', 'B'), 'tfi'),
)

# Define the criterion for optimization
metric = Iopt()

# Cost function
max_cost = np.array([2.5, 4, 10])
@cost_fn(denormalize=False, decoded=False, contains_params=False)
def cost(Y):
    # Extract experiment consumption costs
    c1 = Y[:, 0]
    c2 = Y[:, 1]

    # Return each subcost
    return [
        (c1, max_cost[0], np.arange(len(Y))),
        (c2, max_cost[1], np.arange(len(Y))),
        (1 - c1 - c2, max_cost[2], np.arange(len(Y)))
    ]

#######################################################################

# Simulation parameters
nsims = 10
nreps = 1
fn = default_fn(nsims, factors, cost, metric, Y2X)
params = create_parameters(factors, fn)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_codex_design(
    params, nsims=nsims, nreps=nreps
)
end_time = time.time()

# Add the final mixture components
Y['C'] = 1 - Y.sum(axis=1)

#######################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.round(2).to_csv(os.path.join(root, f'example_mixture.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

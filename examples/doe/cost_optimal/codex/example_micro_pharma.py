#!/usr/bin/env python3

# Python imports
import time
import os
import numpy as np
import pandas as pd

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.cost_optimal import Factor, cost_fn
from pyoptex.doe.cost_optimal.metric import Dopt
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor('X1', type='continuous', grouped=False, min=-1, max=1, levels=[-1, 0, 1]),
    Factor('X2', type='continuous', grouped=False, min=6, max=36, levels=np.linspace(6, 36, 11)),
    Factor('X3', type='continuous', grouped=False, min=12, max=36, levels=np.linspace(12, 36, 9)),
]

# Create a partial response surface model
model = partial_rsm_names({
    'X1': 'quad',
    'X2': 'quad',
    'X3': 'quad',
})
Y2X = model2Y2X(model, factors)

# Define the criterion for optimization
metric = Dopt()

# Define the constraints
constraints = parse_constraints_script(
    f'(`X2` <= `X3`)', 
    factors, exclude=False
)

# Cost function
max_units = 150
@cost_fn(denormalize=False, decoded=False, contains_params=False)
def cost(Y):
    units = 2 + (Y[:, 0] + 1) * 6
    return [(units, max_units, np.arange(len(Y)))]

#######################################################################

# Simulation parameters
nsims = 10
nreps = 10
fn = default_fn(nsims, factors, cost, metric, Y2X, constraints=constraints)
params = create_parameters(factors, fn)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_codex_design(
    params, nsims=nsims, nreps=nreps
)
end_time = time.time()

#######################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, f'example_micro_pharma.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

#######################################################################

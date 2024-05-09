#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import os
import time

# Library imports
from pyoptex.doe.constraints import parse_script
from pyoptex.doe.cost_optimal import create_cost_optimal_design, default_fn
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import x2fx
from pyoptex.doe.cost_optimal.metric import Iopt

# Define parameters
effects = {
    # Define effect type, model type
    'A': (1, 'quad'),
    'B': (1, 'quad'),
    # 'C': (1, 'tfi'),
}
# Second order sheffe model = full quadratic with 1 less term and constraint on sum

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

# Define new (augmented) sheffe model
model_aug = partial_rsm_names({key: 'tfi' for key in ['A', 'B', 'C']}).iloc[1:].to_numpy()

# Cost function
max_cost = np.array([2.5, 4, 10]) # In kg
def cost_fn(Y):
    # Extract experiment consumption costs
    c1 = Y[:, 0]
    c2 = Y[:, 1]

    # Return each subcost
    return [
        (c1, max_cost[0], np.arange(len(Y))),
        (c2, max_cost[1], np.arange(len(Y))),
        (1 - c1 - c2, max_cost[2], np.arange(len(Y)))
    ]

# Define coordinates
coords = [
    np.arange(0, 1.0001, 0.05)[:, np.newaxis],
    np.arange(0, 1.0001, 0.05)[:, np.newaxis],
]

# Covariance function
def cov(Y, X, Zs, Vinv, costs, random=False):
    # Define Y
    Y = np.concatenate((Y, np.expand_dims(1-Y[:,0]-Y[:,1], axis=1)), axis=1)

    # Recompute X given the augmented model
    X = x2fx(Y, model_aug)

    return Y, X, Zs, Vinv

# Define the metric
metric = Iopt(cov=cov)

# Define constraints
constraints = parse_script(
    f'(`A` + `B` > 1) | (`A` < 0.1) | (`B` < 0.2) | (`A` > 0.4) | (`B` > 0.5)', 
    effect_types
).encode()

# Parameter initialization
nsims = 5000
nreps = 10

# Create the set of operators
fn = default_fn(nsims, cost_fn, metric, constraints=constraints)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    effect_types, fn, model=model, 
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    coords=coords,
    validate=True
)
end_time = time.time()

#########################################################################

root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'results', 'example_ingredient_stock.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

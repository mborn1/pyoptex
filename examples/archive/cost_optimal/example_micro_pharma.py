#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time

# Library imports
from pyoptex.doe.constraints import parse_script
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import obs_var_from_Zs
from pyoptex.doe.cost_optimal import create_cost_optimal_design, default_fn
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt

# np.random.seed(42)

# X2 <= X3
# encoded : 12 * X3 - 15 * X2 + 3 >= 0
# coords: (-1, 0, 1), range(6, 36+1, 3), range(12, 36+1, 3)
# 200 units: X1: 2 (-1); 8 (0); 14 (1) (discrete options)
# ?? 45 days: X2: 2-8 (-1 -> 1) (continuous option)
# D-optimal

# Define parameters
effects = {
    # Define effect type, model type, is_grouped
    'A': (1, 'quad'),
    'B': (1, 'quad'),
    'C': (1, 'quad'),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

#########################################################################

# Cost function
max_units = 150
def cost_fn(Y):
    units = 2 + (Y[:, 0] + 1) * 6
    return [(units, max_units, np.arange(len(Y)))]

# Define the metric
metric = Dopt()

# Define constraints  12 * X3 - 15 * X2 + 3 >= 0
constraints = parse_script(f'(12 * `C` - 15 * `B` + 3 < 0)', effect_types)

# Define coordinates
coords = [
    np.linspace(-1, 1, 3)[:, np.newaxis],
    np.linspace(-1, 1, 11)[:, np.newaxis],
    np.linspace(-1, 1, 9)[:, np.newaxis],
]

#########################################################################

# Parameter initialization
nsims = 5000
nreps = 1

# Create the set of operators
from pyoptex.doe.cost_optimal.remove import remove_optimal_onebyone_prevent
fn = default_fn(nsims, cost_fn, metric, constraints=constraints,
                remove=remove_optimal_onebyone_prevent)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    effect_types, fn, model=model, coords=coords,
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    validate=True
)
end_time = time.time()

# Decode the columns
# Y['B'] = (Y['B'] + 1) / 2 * 30 + 6
# Y['C'] = (Y['C'] + 1) / 2 * 24 + 12

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'results', 'example_micro_pharma.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


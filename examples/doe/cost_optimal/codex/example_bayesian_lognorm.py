#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import os
import time

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt, DoptBayesian
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.cov import cov_time_trend
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor('Ca', type='continuous'),
    Factor('Co1', type='continuous'),
    Factor('Co2', type='continuous'),
    Factor('Co3', type='continuous'),
]

# Create a partial response surface model
model = partial_rsm_names({
    'Ca': 'quad',
    'Co1': 'quad',
    'Co2': 'quad',
    'Co3': 'quad'
})
Y2X = model2Y2X(model, factors)

# Define the criterion for optimization
metric = DoptBayesian(factors)

# Cost function
max_transition_cost = 600
transition_costs = {
    'Ca': 15,
    'Co1': 15,
    'Co2': 0,
    'Co3': 0
}
execution_cost = 5
cost_fn = parallel_worker_cost(transition_costs, factors, max_transition_cost, execution_cost)

#######################################################################

# Simulation parameters
nsims = 7500
nreps = 5
fn = default_fn(nsims, factors, cost_fn, metric, Y2X)
params = create_parameters(factors, fn)
params = params._replace(ratios=metric.ratios)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_codex_design(
    params, nsims=nsims, nreps=nreps, validate=True
)
end_time = time.time()

#######################################################################

# Write design to storage
root = os.path.split(__file__)[0]
filename = f'{root}/bayesian'
with open(filename, 'w') as f:
    f.write(f'{state.metric:.3f}, {state.cost_Y}, {len(state.Y)}\n')
    f.write(f'sim_bayesian -- ' + str({'nsims': nsims, 'time': end_time - start_time, 'validate': True}))
    f.write('\n')
    np.savetxt(f, state.Y)
print(Y)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


#!/usr/bin/env python3

# Python imports
import time
import os

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor('A', type='categorical', levels=['L1', 'L2', 'L3', 'L4']),
    Factor('E', type='continuous', grouped=False),
    Factor('F', type='continuous', grouped=False, min=2, max=5),
    Factor('G', type='continuous', grouped=False),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'tfi',
    'E': 'quad',
    'F': 'quad',
    'G': 'quad'
})
Y2X = model2Y2X(model, factors)

# Define the criterion for optimization
metric = Iopt()

# Cost function
max_transition_cost = 3*4*60
transition_costs = {
    'A': 2*60,
    'E': 1,
    'F': 1,
    'G': 1
}
execution_cost = 5
cost_fn = parallel_worker_cost(
    transition_costs, factors, 
    max_transition_cost, execution_cost
)

#######################################################################

# Simulation parameters
nsims = 10
nreps = 1
fn = default_fn(nsims, factors, cost_fn, metric, Y2X)
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
Y.to_csv(os.path.join(root, f'example_cost_optimal_codex.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

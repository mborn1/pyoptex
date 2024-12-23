#!/usr/bin/env python3

# Normal imports
import time

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.metric import Dopt
from pyoptex.doe.cost_optimal.cost import scaled_parallel_worker_cost
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors: make sure factor A is the first
factors = [
    Factor('A', type='continuous', min=2, max=5),
    Factor('B', type='continuous'),
    Factor('E', type='continuous', grouped=False),
    Factor('F', type='continuous', grouped=False),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'quad',
    'B': 'quad',
    'E': 'quad',
    'F': 'quad',
})
Y2X = model2Y2X(model, factors)

# Define the criterion for optimization
metric = Dopt()

#######################################################

# Cost function
max_transition_cost = 3*4*60
transition_costs = {
    'A': (0, 0, 2*60, 2*60), # From -1 to +1, scaled in between, no base cost
    'B': (60, 60, 0, 0), # Constant transition cost
    'E': (1, 1, 0, 0), # Constant transition cost
    'F': (1, 1, 0, 0), # Constant transition cost
}
execution_cost = 5

cost_fn = scaled_parallel_worker_cost(
    transition_costs, factors, 
    max_transition_cost, execution_cost
)

#######################################################################

# Simulation parameters
nsims = 1000
nreps = 1
fn = default_fn(nsims, factors, cost_fn, metric, Y2X)
params = create_parameters(factors, fn)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_codex_design(
    params, nsims=nsims, nreps=nreps, validate=True
)
end_time = time.time()

#######################################################################

# Write design to storage
Y.to_csv(f'example_scaled.csv', index=False)
print(Y)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

#######################################################################

# Generic evaluation
from pyoptex.doe.utils.evaluate import design_heatmap
design_heatmap(Y, factors).show()

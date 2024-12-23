#!/usr/bin/env python3

# Python imports
import time
import os
import numpy as np

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.metric import Aliasing
from pyoptex.doe.cost_optimal.cost import fixed_runs_cost
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor('A', type='continuous', grouped=False),
    Factor('B', type='continuous', grouped=False),
    Factor('C', type='continuous', grouped=False),
    Factor('D', type='continuous', grouped=False),
    Factor('E', type='continuous', grouped=False),
    Factor('F', type='continuous', grouped=False),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'quad',
    'B': 'quad',
    'C': 'quad',
    'D': 'quad',
    'E': 'quad',
    'F': 'quad',
})
Y2X = model2Y2X(model, factors)

# Define the weights (equal weights on main, two-factor and quadratic aliasing)
# Minimize aliasing of main effects to full response surface design
n1, n2 = len(factors), len(model)-2*len(factors)-1
w1, w2 = 1/((n1+1)*(n1+1)), 1/((n2+n1)*(n1+1))
W = np.block([
    [ w1 * np.ones(( 1, 1)), w1 * np.ones(( 1, n1)), w2 * np.ones(( 1, n2)), w2 * np.zeros(( 1, n1))], # Intercept
    [ w1 * np.ones((n1, 1)), w1 * np.ones((n1, n1)), w2 * np.ones((n1, n2)), w2 *  np.ones((n1, n1))], # Main effects
])

# Set variances to zero (only interested in aliasing as an example)
W[np.arange(len(W)), np.arange(len(W))] = 0

# Define the metric
main_effects = np.arange(len(factors)+1) # The indices of the main effects in the model, and intercept
metric = Aliasing(effects=main_effects, alias=np.arange(len(model)), W=W)

# 30 runs maximum
nruns = 30
cost_fn = fixed_runs_cost(nruns)

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
Y.to_csv(os.path.join(root, f'example_approx_omars.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

#######################################################################

# Specific evaluation
from pyoptex.doe.cost_optimal.evaluate import plot_estimation_variance_matrix
plot_estimation_variance_matrix(Y, params, model).show()

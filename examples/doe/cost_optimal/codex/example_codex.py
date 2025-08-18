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
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.cov import cov_time_trend
from pyoptex.doe.cost_optimal.codex import (
    create_cost_optimal_codex_design, default_fn, create_parameters
)

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor('A1', type='categorical', levels=['L1', 'L2', 'L3', 'L4'], 
            coords=np.array([[-1, -1, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            ratio=[0.1, 1, 10]),
    Factor('E', type='continuous', grouped=False),
    Factor('F', type='continuous', grouped=False, 
           levels=[2, 3, 4, 5], min=2, max=5),
    Factor('G', type='continuous', grouped=False),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A1': 'tfi',
    'E': 'tfi',
    'F': 'quad',
    'G': 'tfi'
})
Y2X = model2Y2X(model, factors)

# Define the criterion for optimization
metric = Iopt(cov=cov_time_trend(time=60))

# Define the prior design for augmentation
prior = pd.DataFrame([['L1', 0, 2, 0]], columns=['A1', 'E', 'F', 'G'])

# Cost function
max_transition_cost = 3*4*60
transition_costs = {
    'A1': 2*60,
    'E': 1,
    'F': 1,
    'G': 1
}
execution_cost = 5
cost_fn = parallel_worker_cost(transition_costs, factors, max_transition_cost, execution_cost)

# Define constraints
constraints = parse_constraints_script(
    f'(`A1` == "L1") & (`E` < -0.5-0.25)', 
    factors, exclude=True
)

#######################################################################

# Simulation parameters
nsims = 10
nreps = 1
fn = default_fn(nsims, factors, cost_fn, metric, Y2X, constraints=constraints)
params = create_parameters(factors, fn, prior=prior)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_codex_design(
    params, nsims=nsims, nreps=nreps, validate=True
)
end_time = time.time()

#######################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, f'example_codex.csv'), index=False)
print(Y)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

#######################################################################

# # Generic evaluation
# from pyoptex.doe.utils.evaluate import design_heatmap, plot_correlation_map
# design_heatmap(Y, factors).show()
# plot_correlation_map(Y, factors, fn.Y2X, model=model).show()

# # Specific evaluation
# from pyoptex.doe.cost_optimal.evaluate import (
#     evaluate_metrics, plot_fraction_of_design_space, 
#     plot_estimation_variance_matrix, estimation_variance
# )
# print(evaluate_metrics(Y, params, [metric, Dopt(), Iopt(), Aopt()]))
# plot_fraction_of_design_space(Y, params).show()
# plot_estimation_variance_matrix(Y, params, model).show()
# print(estimation_variance(Y, params))


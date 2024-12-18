#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import create_cost_optimal_design, default_fn, Factor
from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt
from pyoptex.doe.cost_optimal.cov import cov_time_trend
from pyoptex.doe.cost_optimal.cost import transition_discount_cost
from pyoptex.doe.cost_optimal.init import init_feasible
from pyoptex.doe.constraints import parse_constraints_script

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor(name='A1', type='categorical', levels=['L1', 'L2', 'L3', 'L4'], 
            coords=np.array([[-1, -1, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            ratio=[0.1, 1, 10]),
    Factor(name='E', type='continuous', grouped=False),
    Factor(name='F', type='continuous', grouped=False, min=2, max=5),
    Factor(name='G', type='continuous', grouped=False),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A1': 'tfi',
    'E': 'tfi',
    'F': 'tfi',
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
cost_fn = transition_discount_cost(transition_costs, factors, max_transition_cost, execution_cost)

# Define constraints
constraints = parse_constraints_script(f'(`A1` == "L1") & (`E` < -0.5-0.25)', factors, exclude=True)

# TODO: what about covariates? -- add decorator like cost function
# TODO: introduce analysis -- reuse functions and factors
# TODO: make more helper cost functions
# TODO: compile and simplify constraints tree

#######################################################################

# Simulation parameters
nsims = 10
nreps = 1
fn = default_fn(nsims, cost_fn, metric, constraints=constraints)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    factors, fn, Y2X, nsims=nsims, nreps=nreps, prior=prior, 
    validate=True
)
end_time = time.time()

#######################################################################

# Write design to storage
Y.to_csv(f'example_design.csv', index=False)
print(Y)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')

#######################################################################

from pyoptex.doe.utils.evaluate import design_heatmap, plot_correlation_map
design_heatmap(Y, factors).show()
plot_correlation_map(Y, factors, Y2X, model=model).show()

from pyoptex.doe.cost_optimal.evaluate import evaluate_metrics, plot_fraction_of_design_space, plot_estimation_variance_matrix, estimation_variance
print(evaluate_metrics(Y, [metric, Dopt(), Iopt(), Aopt()], factors, Y2X, fn))
plot_fraction_of_design_space(Y, factors, Y2X, fn).show()
plot_estimation_variance_matrix(Y, factors, Y2X, fn, model).show()
print(estimation_variance(Y, factors, Y2X, fn))


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
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt
from pyoptex.doe.cost_optimal.cov import cov_block_cost, cov_augment_time, cov_augment_time_double
from pyoptex.doe.cost_optimal.cost import transition_discount_cost
from pyoptex.doe.cost_optimal.init import init_feasible

# Set the seed
set_seed(42)

# Define the factors
factors = [
    Factor(name='A', type='categorical', levels=['L1', 'L2', 'L3']),
    Factor(name='E', type='continuous', grouped=False),
    Factor(name='F', type='continuous', grouped=False),
    Factor(name='G', type='continuous', grouped=False),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'tfi',
    'E': 'quad',
    'F': 'quad',
    'G': 'quad'
})

# Define the criterion for optimization
metric = Iopt()

# Define the prior design for augmentation
prior = None

# Cost function
max_transition_cost = 3*4*60
transition_costs = {
    'A': 2*60,
    'E': 1,
    'F': 1,
    'G': 1
}
execution_cost = 5
cost_fn = transition_discount_cost(transition_costs, factors, max_transition_cost, execution_cost)

# TODO: fix the constraints based on new factors and encoding
# TODO: work on making evaluate more accessible (via factors)
# TODO: what about covariates?
# TODO: make more helper cost functions

# TODO: do the same for split^k-plot
# TODO: introduce analysis -- reuse functions and factors

#######################################################################

# Simulation parameters
nsims = 100
nreps = 1
fn = default_fn(nsims, cost_fn, metric)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    factors, fn, model=model, nsims=nsims, nreps=nreps, prior=prior, 
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

# # #######################################################################

# # from pyoptex.doe.cost_optimal.evaluate import evaluate_metrics, plot_fraction_of_design_space, plot_estimation_variance_matrix
# # from pyoptex.doe.utils.plot import plot_correlation_map
# # print(evaluate_metrics(Y, effect_types, cost_fn=cost_fn, model=model, grouped_cols=grouped_cols, ratios=ratios))
# # plot_fraction_of_design_space(Y, effect_types, cost_fn=cost_fn, model=model, grouped_cols=grouped_cols, ratios=ratios).show()
# # plot_estimation_variance_matrix(Y, effect_types, cost_fn=cost_fn, model=model, grouped_cols=grouped_cols, ratios=ratios).show()
# # plot_correlation_map(Y, effect_types, model=model).show()


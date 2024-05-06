#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time
from numba.typed import List

# Library imports
from pyoptex.doe.splitk_plot import create_splitk_plot_design, default_fn
from pyoptex.doe.splitk_plot.metric import Dopt, Iopt, Aopt
from pyoptex.doe.splitk_plot.cov import cov_time_trend, cov_double_time_trend
from pyoptex.doe.utils.model import partial_rsm_names

np.random.seed(42)

# Define parameters
effects = {
    # Define effect type, model type, is_grouped, cost
    'Block': (2, 'lin', 1),
    'A': (1, 'quad', 0),
    'B': (1, 'quad', 0),
    'C': (1, 'quad', 0),
    'D': (1, 'quad', 0),
}
plot_sizes = np.array([20, 2])

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
effect_levels = {key: value[2] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})

extra_terms = pd.DataFrame([
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 3, 0, 0, 0],
    [0, 4, 0, 0, 0],
], columns=list(effects.keys()))
model = pd.concat((model, extra_terms))

#########################################################################

# Define the metric
metric = Dopt()

# Define prior
# prior = (pd.read_csv('example_design.csv'), np.array([4, 6]))
# plot_sizes = np.array([5, 6])
prior = None

# Define multiple ratios
ratios = np.stack((np.ones(1) * 1e-5,))

# Covariate
# cov = cov_time_trend(plot_sizes[1], np.prod(plot_sizes), model)
# cov = cov_double_time_trend(plot_sizes[1], plot_sizes[0], np.prod(plot_sizes), model)
cov = None

coords = [
    None,
    np.array([-1, -0.65, 0, 0.65, 1]).reshape(-1, 1),
    None, None, None
]

#########################################################################

# Parameter initialization
n_tries = 1000

# Create the set of operators
fn = default_fn(metric)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(
    fn, effect_types, effect_levels, plot_sizes, ratios=ratios, cov=cov,
    model=model, prior=prior, n_tries=n_tries, validate=False, coords=coords,
)
end_time = time.time()

#########################################################################

# Write design to storage
if prior is None:
    Y.to_csv(f'example_design_saif.csv', index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)


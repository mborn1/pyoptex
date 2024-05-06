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
    # Define effect type, model type
    'A': (1, 'tfi', 1),
    'B': (1, 'tfi', 0),
    'C': (1, 'tfi', 0),
}
plot_sizes = np.array([4, 8])

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
effect_levels = {key: value[2] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})

#########################################################################

# Define the metric
metric = Dopt()

# Define prior
prior = None

# Define multiple ratios
ratios = None

#########################################################################

# Parameter initialization
n_tries = 10000

# Create the set of operators
fn = default_fn(metric)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(
    fn, effect_types, effect_levels, plot_sizes, ratios=ratios,
    model=model, prior=prior, n_tries=n_tries, validate=False
)
end_time = time.time()

#########################################################################

# Write design to storage
if prior is None:
    Y.to_csv(f'example_design_split_plot.csv', index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)


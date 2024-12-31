#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import os
import time

# Library imports
from pyoptex._seed import set_seed
from pyoptex.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.fixed_structure import Factor
from pyoptex.doe.fixed_structure.splitk_plot import (
    create_splitk_plot_design, default_fn, create_parameters, Plot
)
from pyoptex.doe.fixed_structure.splitk_plot.metric import Aopt
from pyoptex.doe.fixed_structure.splitk_plot.utils import validate_plot_sizes

# TODO: fix notebook

# Set the seed
set_seed(42)

# Define the plots
etc = Plot(level=0, size=4)
htc = Plot(level=1, size=8)
plots = [etc, htc]
nruns = np.prod([p.size for p in plots])

# Define the factors
factors = [
    Factor('A', htc, type='categorical', levels=['L1', 'L2', 'L3']),
    Factor('B', etc, type='continuous'),
    Factor('C', etc, type='continuous'),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'tfi',
    'B': 'quad',
    'C': 'quad',
})
Y2X = model2Y2X(model, factors)

# Define the metric
metric = Aopt()

# Define prior
prior = (
    pd.DataFrame([
        ['L1'], ['L1'], ['L1'], ['L1'],
        ['L2'], ['L2'], ['L2'], ['L2'],
        ['L3'], ['L3'], ['L3'], ['L3'],
        ['L2'], ['L2'], ['L2'], ['L2'],
        ['L1'], ['L1'], ['L1'], ['L1'],
        ['L3'], ['L3'], ['L3'], ['L3'],
        ['L1'], ['L1'], ['L1'], ['L1'],
        ['L2'], ['L2'], ['L2'], ['L2'],
    ], columns=['A']).assign(B=0, C=0),
    [Plot(level=0, size=4), Plot(level=1, size=8)]
)

# Define that the htc factors cannot be optimized, but all etc can
grps = [np.array([]), np.arange(nruns), np.arange(nruns)]

#########################################################################

# Validate variance components are estimable
validate_plot_sizes(factors, model)

# Parameter initialization
n_tries = 10

# Create the set of operators
fn = default_fn(factors, metric, Y2X)
params = create_parameters(factors, fn, prior=prior, grps=grps)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(params, n_tries=n_tries, validate=True)
end_time = time.time()

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'example_splitk_fixed_factor.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)

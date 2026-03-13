#!/usr/bin/env python3

# Python imports
import os
import time
import numpy as np

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.utils.model import mixtureY2X
from pyoptex.doe.fixed_structure import (
    Factor, create_fixed_structure_design, 
    create_parameters, default_fn
)
from pyoptex.doe.fixed_structure.metric import Dopt

# Set the seed
set_seed(42)

# Define the plots
nruns = 20

# Define the factors (last mixture is not explicitely specified)
factors = [
    Factor('A', type='mixture', levels=np.arange(0, 1.001, 0.05)),
    Factor('B', type='mixture', levels=np.arange(0, 1.001, 0.05)),
    Factor('C', type='mixture', levels=np.arange(0.2, 0.501, 0.05)),
]

# Create a Scheffe model
Y2X = mixtureY2X(
    factors, 
    mixture_effects=(('A', 'B', 'C'), 'tfi'), 
)
log_checkpoint("factor_names", [str(f.name) for f in factors])
log_checkpoint("nruns", nruns)

# Define the metric
metric = Dopt()

#########################################################################

# Parameter initialization
n_tries = 10

# Create the set of operators
fn = default_fn(factors, metric, Y2X)
params = create_parameters(factors, fn, nruns)

# Create design
start_time = time.time()
Y, state = create_fixed_structure_design(params, n_tries=n_tries)
end_time = time.time()

# Add the final mixture components
Y['D'] = 1 - Y.sum(axis=1)

log_checkpoint("Y_shape", list(Y.shape))
log_checkpoint("Y_columns", Y.columns.tolist())
log_checkpoint("Y_values", Y.values.tolist())
log_checkpoint("metric", float(state.metric))

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.round(2).to_csv(os.path.join(root, 'example_mixture.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)

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
from pyoptex.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.fixed_structure import Factor
from pyoptex.doe.fixed_structure.splitk_plot import (
    create_splitk_plot_design, default_fn, create_parameters, Plot
)
from pyoptex.doe.fixed_structure.splitk_plot.metric import Dopt

# Set the seed
set_seed(42)

# Define the plots
etc = Plot(level=0, size=4)
htc = Plot(level=1, size=5, ratio=0.1)
plots = [etc, htc]
nruns = np.prod([p.size for p in plots])

# Define the factors
factors = [
    Factor('A', htc, type='categorical', levels=['L1', 'L2', 'L3']),
    Factor('B', etc, type='continuous'),
    Factor('C', etc, type='continuous', min=2, max=5),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'tfi',
    'B': 'quad',
    'C': 'quad',
})
Y2X = model2Y2X(model, factors)
log_checkpoint("factor_names", [str(f.name) for f in factors])
log_checkpoint("nruns", int(nruns))
log_checkpoint("model_shape", list(model.shape))
log_checkpoint("model_values", model.values.tolist())

# Define the metric
metric = Dopt()

#########################################################################

# Parameter initialization
n_tries = 10

# Create the set of operators
fn = default_fn(factors, metric, Y2X)
params = create_parameters(factors, fn)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(params, n_tries=n_tries)
end_time = time.time()

log_checkpoint("Y_shape", list(Y.shape))
log_checkpoint("Y_columns", Y.columns.tolist())
log_checkpoint("Y_values", Y.values.tolist())
log_checkpoint("metric", float(state.metric))

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'example_splitplot_sp.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)

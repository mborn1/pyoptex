#!/usr/bin/env python3

# Normal imports
import os
import time

import numpy as np

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import Factor
from pyoptex.doe.fixed_structure.splitk_plot import Plot, create_parameters, create_splitk_plot_design, default_fn
from pyoptex.doe.fixed_structure.splitk_plot.metric import Coverage

# Set the seed
set_seed(42)

# Define the plots: 5 whole plots of 4 runs
etc = Plot(level=0, size=4)
htc = Plot(level=1, size=5)
nruns = np.prod([etc.size, htc.size])

# Define the factors, with more levels than the default three
factors = [
    Factor("A", htc, type="continuous", levels=np.linspace(-1, 1, 11)),
    Factor("B", etc, type="continuous", levels=np.linspace(-1, 1, 11)),
    Factor("C", etc, type="continuous", min=2, max=5, levels=np.linspace(2, 5, 11)),
]
log_checkpoint("factor_names", [str(f.name) for f in factors])
log_checkpoint("nruns", int(nruns))

# Define the criterion: space-filling, without a model
metric = Coverage()

#########################################################################

# Parameter initialization
n_tries = 5

# Create the set of operators
fn = default_fn(factors, metric, None)
params = create_parameters(factors, fn)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(params, n_tries=n_tries, validate=True)
end_time = time.time()

log_checkpoint("Y_shape", list(Y.shape))
log_checkpoint("Y_columns", Y.columns.tolist())
log_checkpoint("Y_values", Y.values.tolist())
log_checkpoint("metric", float(state.metric))

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, "example_splitk_coverage.csv"), index=False)

print("Completed optimization")
print(f"Metric: {state.metric:.3f}")
print(f"Execution time: {end_time - start_time:.3f}")
print(Y)

#########################################################################

# from pyoptex.doe.utils.evaluate import design_heatmap
# design_heatmap(Y, factors).show()

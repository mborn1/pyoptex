#!/usr/bin/env python3

# Python imports
import os
import time
import numpy as np

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.doe.splitk_plot import create_splitk_plot_design, default_fn, Factor, Plot
from pyoptex.doe.splitk_plot.metric import Dopt
from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X

# Set the seed
set_seed(42)

# Define the plots
etc = Plot(level=0, size=4)
htc = Plot(level=1, size=5, ratio=0.1)
plots = [etc, htc]
nruns = np.prod([p.size for p in plots])

# Define the factors
factors = [
    Factor(name='A', plot=htc, type='categorical', levels=['L1', 'L2', 'L3']),
    Factor(name='B', plot=etc, type='continuous'),
    Factor(name='C', plot=etc, type='continuous', min=2, max=5),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'tfi',
    'B': 'quad',
    'C': 'quad',
})
Y2X = model2Y2X(model, factors)

# Define the metric
metric = Dopt()

#########################################################################

# Parameter initialization
n_tries = 1000

# Create the set of operators
fn = default_fn(metric, Y2X)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(
    factors, fn, n_tries=n_tries,
)
end_time = time.time()

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'example_randomized.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)

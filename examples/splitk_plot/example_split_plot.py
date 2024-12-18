#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time
from numba.typed import List

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.splitk_plot import create_splitk_plot_design, default_fn, Factor, Plot
from pyoptex.doe.splitk_plot.metric import Dopt, Iopt, Aopt
from pyoptex.doe.splitk_plot.cov import cov_time_trend, cov_double_time_trend
from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X

# Set the seed
set_seed(42)

# Define the plots
etc = Plot(level=0, size=4, ratio=1)
htc = Plot(level=1, size=8, ratio=1)
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
metric = Dopt(cov=cov_double_time_trend(htc.size, etc.size, nruns))

# Define prior
prior = None

# TODO: test with prior and fixed grps
# TODO: test with constraints
# TODO: refactor evaluate

#########################################################################

# Parameter initialization
n_tries = 10

# Create the set of operators
fn = default_fn(metric)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(
    factors, fn, Y2X, prior=prior, grps=None, 
    n_tries=n_tries, validate=True
)
end_time = time.time()

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv('example_split_plot.csv', index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)

#########################################################################

# from pyoptex.doe.splitk_plot.evaluate import evaluate_metrics, plot_fraction_of_design_space, plot_estimation_variance_matrix
# from pyoptex.doe.utils.plot import plot_correlation_map
# print(evaluate_metrics(Y, effect_types, plot_sizes, model=model, ratios=ratios))
# plot_fraction_of_design_space(Y, effect_types, plot_sizes, model=model, ratios=ratios).show()
# plot_estimation_variance_matrix(Y, effect_types, plot_sizes, model=model, ratios=ratios).show()
# plot_correlation_map(Y, effect_types, model=model).show()



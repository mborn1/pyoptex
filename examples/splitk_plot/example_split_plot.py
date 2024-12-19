#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import os
import time

# Library imports
from pyoptex._seed import set_seed
from pyoptex.doe.splitk_plot import create_splitk_plot_design, default_fn, Factor, Plot
from pyoptex.doe.splitk_plot.metric import Dopt, Iopt, Aopt
from pyoptex.doe.splitk_plot.cov import cov_double_time_trend
from pyoptex.doe.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.splitk_plot.utils import validate_plot_sizes

# TODO: upgrade I-optimality with weighing of the matrix (also in cost_optimal)

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
metric = Aopt(cov=cov_double_time_trend(htc.size, etc.size, nruns))

# Define prior
prior = (
    pd.DataFrame([
        ['L1', 0, 2],
        ['L1', 1, 5],
        ['L2', -1, 3.5],
        ['L2', 0, 2]
    ], columns=['A', 'B', 'C']),
    [Plot(level=0, size=2), Plot(level=1, size=2)]
)

# Constraints
constraints = parse_constraints_script(f'(`A` == "L1") & (`B` < -0.5-0.25)', factors, exclude=True)

#########################################################################

# Validate variance components are estimable
validate_plot_sizes(factors, model)

# Parameter initialization
n_tries = 10

# Create the set of operators
fn = default_fn(metric, Y2X, constraints=constraints)

# Create design
start_time = time.time()
Y, state = create_splitk_plot_design(
    factors, fn, prior=prior, grps=None, 
    n_tries=n_tries, validate=False
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

from pyoptex.doe.utils.evaluate import design_heatmap, plot_correlation_map
design_heatmap(Y, factors).show()
plot_correlation_map(Y, factors, fn.Y2X, model=model).show()

from pyoptex.doe.splitk_plot.evaluate import evaluate_metrics, plot_fraction_of_design_space, plot_estimation_variance_matrix, estimation_variance
print(evaluate_metrics(Y, [metric, Dopt(), Iopt(), Aopt()], factors, fn))
plot_fraction_of_design_space(Y, factors, fn).show()
plot_estimation_variance_matrix(Y, factors, fn, model).show()
print(estimation_variance(Y, factors, fn))

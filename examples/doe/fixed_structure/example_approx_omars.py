#!/usr/bin/env python3

# Python imports
import os
import time
import numpy as np

# PyOptEx imports
from pyoptex._seed import set_seed
from pyoptex.utils.model import partial_rsm_names, model2Y2X
from pyoptex.doe.fixed_structure import (
    Factor, create_fixed_structure_design, 
    create_parameters, default_fn
)
from pyoptex.doe.fixed_structure.metric import Aliasing

# Set the seed
set_seed(42)

# Define the plots
nruns = 30

# Define the factors
factors = [
    Factor('A', type='continuous'),
    Factor('B', type='continuous'),
    Factor('C', type='continuous'),
    Factor('D', type='continuous'),
    Factor('E', type='continuous'),
    Factor('F', type='continuous'),
]

# Create a partial response surface model
model = partial_rsm_names({
    'A': 'quad',
    'B': 'quad',
    'C': 'quad',
    'D': 'quad',
    'E': 'quad',
    'F': 'quad',
})
Y2X = model2Y2X(model, factors)

# Define the weights (equal weights on main, two-factor and quadratic aliasing)
# Minimize aliasing of main effects to full response surface design
n1, n2 = len(factors), len(model)-2*len(factors)-1
w1, w2 = 1/((n1+1)*(n1+1)), 1/((n2+n1)*(n1+1))
W = np.block([
    [ w1 * np.ones(( 1, 1)), w1 * np.ones(( 1, n1)), w2 * np.ones(( 1, n2)), w2 * np.zeros(( 1, n1))], # Intercept
    [ w1 * np.ones((n1, 1)), w1 * np.ones((n1, n1)), w2 * np.ones((n1, n2)), w2 *  np.ones((n1, n1))], # Main effects
])

# Set variances to zero (only interested in aliasing as an example)
W[np.arange(len(W)), np.arange(len(W))] = 0

# Define the metric
main_effects = np.arange(len(factors)+1) # The indices of the main effects in the model, and intercept
metric = Aliasing(effects=main_effects, alias=np.arange(len(model)), W=W)

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

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'example_approx_omars.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Execution time: {end_time - start_time:.3f}')
print(Y)

#########################################################################

from pyoptex.doe.fixed_structure.evaluate import plot_estimation_variance_matrix
plot_estimation_variance_matrix(Y, params, model).show()

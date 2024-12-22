import numpy as np
import pandas as pd
import os
import plotly.express as px
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import x2fx, obs_var_from_Zs
from pyoptex.doe.cost_optimal.metric import Aliasing
from pyoptex.doe.cost_optimal.codex.wrapper import create_parameters, default_fn

# Get root folder
root = os.path.split(__file__)[0]

effects = {
    # Define effect type, model type
    'A': (1, 'quad'),
    'B': (1, 'quad'),
    'C': (1, 'quad'),
    'D': (1, 'quad'),
    'E': (1, 'quad'),
    'F': (1, 'quad'),
    # 'G': (1, 'quad'),
    # 'H': (1, 'quad'),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

# Cost function
nruns = 30
def cost_fn(Y):
    return [(np.ones(len(Y)), nruns, np.arange(len(Y)))]

# Define the weights
n1, n2 = len(effects), len(model)-2*len(effects)-1
w1, w2 = 1/((n1+1)*(n1+1)), 1/((n2+n1)*(n1+1))
W = np.block([
    [ w1 * np.ones(( 1, 1)), w1 * np.ones(( 1, n1)), w2 * np.ones(( 1, n2)), w2 * np.zeros(( 1, n1))], # Intercept
    [ w1 * np.ones((n1, 1)), w1 * np.ones((n1, n1)), w2 * np.ones((n1, n2)), w2 *  np.ones((n1, n1))], # Main effects
])
W[np.arange(len(W)), np.arange(len(W))] = 0

# Define the metric
metric = Aliasing(np.arange(len(effects)+1), np.arange(len(model)), W=W)

# Prepare the parameters
fn = default_fn(1, cost_fn, metric)
params, _ = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols)
params.fn.metric.init(params)

# Evaluate the cost model
Y = pd.read_csv(f'{root}/../cost_optimal/results/example_approx_omars.csv').to_numpy()
X = params.Y2X(Y)

# # Compute aliasing matrix
# Xeff = X[:, metric.effects]
# Xa = X[:, metric.alias]
# A = np.linalg.solve(Xeff.T @ Xeff, Xeff.T) @ Xa

# Compute correlation matrix
A = pd.DataFrame(X[1:, 1:]).corr().to_numpy()

max_idx = np.argmax(A[:len(effects), :] * np.where(W[1:, 1:] != 0, 1, 0))
print(max_idx, A.flatten()[max_idx])
print(np.linalg.matrix_rank(X), X.shape)

# Visualize the variances
px.imshow(np.abs(A)).show()


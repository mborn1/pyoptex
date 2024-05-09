import numpy as np
import pandas as pd
import os
import plotly.express as px
from pyoptex.doe.constraints import parse_script
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import x2fx, obs_var_from_Zs
from pyoptex.doe.cost_optimal.metric import Dopt
from pyoptex.doe.cost_optimal.wrapper import create_parameters, default_fn

# Get root folder
root = os.path.split(__file__)[0]

effects = {
    # Define effect type, model type
    'A': (1, 'tfi'),
    'B': (1, 'tfi'),
    'C': (1, 'quad'),
    'D': (1, 'quad'),
    'E': (1, 'quad'),
    'F': (1, 'quad'),
    'G': (1, 'quad'),
    'H': (1, 'tfi'),
    'I': (1, 'tfi'),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

# Cost function
nb_plots = 6
runs_per_plot_low = 7
runs_per_plot_high = 14
def cost_fn(Y):
    # Short-circuit
    if len(Y) == 0:
        return []

    # Determine the number of resets
    resets = np.zeros(len(Y))
    resets[1:] = np.any(np.diff(Y[:, :2], axis=0) != 0, axis=1).astype(np.int64)
    for i in range(0, len(resets)):
        # Reset when factor does not change for a period depending on the first factor
        if (Y[i-1, 0] == -1 and i >= runs_per_plot_low and np.all(resets[i-(runs_per_plot_low-1):i] == 0))\
                or (Y[i-1, 0] == 1 and i >= runs_per_plot_high and np.all(resets[i-(runs_per_plot_high-1):i] == 0)):
            resets[i] = 1

    # Determine number of runs per plot
    idx = np.concatenate([[0], np.flatnonzero(resets), [len(Y)]])
    plot_costs = [None] * (len(idx) - 1)
    for i in range(len(idx)-1):
        if Y[idx[i], 0] == -1:
            rp = runs_per_plot_low
        else:
            rp = runs_per_plot_high

        plot_costs[i] = (np.ones(idx[i+1] - idx[i]), rp, np.arange(idx[i], idx[i+1]))
    
    return [
        (resets, nb_plots - 1, np.arange(len(Y))),
        *plot_costs
    ]

# Compute D-optimality
def cov(Y, X, Zs, Vinv, costs, random=False):
    resets = costs[0][0].astype(np.int64)
    Z1 = np.cumsum(resets)
    Zs = [Z1]
    V = obs_var_from_Zs(Zs, len(Y), ratios=np.array([1.]))
    Vinv = np.expand_dims(np.linalg.inv(V), 0)
    return Y, X, Zs, Vinv
metric = Dopt(cov=cov)

# Coordinates
coords = [
    np.array([-1, 1]).reshape(-1, 1),
    np.array([-1, 1]).reshape(-1, 1),
    np.array([-1, -1/3, 1/3, 1]).reshape(-1, 1),
    None, None, None, None, None, None
]

# Define constraints
constraints = parse_script(f'((`B` > 0) & (`C` < -0.4)) | ((`B` < 0) & (`C` > 0.4))', effect_types).encode()

# Create parameters
fn = default_fn(1, cost_fn, metric, constraints=constraints)
params, _ = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, coords=coords)
params.fn.metric.init(params)

# Evaluate the ref model
Yref = pd.read_csv(f'{root}/../cost_optimal/data/ref_split_plot_size_dependent_70.csv').to_numpy()
Xref = params.Y2X(Yref)
metric_ref = params.fn.metric.call(Yref, Xref, None, None, params.fn.cost(Yref))
px.imshow(Yref, aspect='auto').show()

# Evaluate the cost model
r = f'{root}/../cost_optimal/results/example_split_plot_size_dependent'
for ex in os.listdir(r):
    Y = pd.read_csv(f'{r}/{ex}').to_numpy()
    X = params.Y2X(Y)
    metric = params.fn.metric.call(Y, X, None, None, params.fn.cost(Y))

    print(metric, metric_ref, metric/metric_ref, '--', len(Y), len(Yref), '--', [np.sum(c) for c, _, _ in params.fn.cost(Y)])
    if '0' in ex:
        px.imshow(Y, aspect='auto').show()

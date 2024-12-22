import numpy as np
import pandas as pd
import os
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.splitk_plot.metric import Dopt
from pyoptex.doe.splitk_plot.wrapper import create_parameters, default_fn
from pyoptex.doe.utils.design import x2fx

# Get root folder
root = os.path.split(__file__)[0]

effects = {
    # Define effect type, model type, stratum
    'A': (1, 'tfi', 1),
    'B': (1, 'tfi', 0),
    'H': (1, 'tfi', 0),
}
plot_sizes = np.array([4, 8])

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
effect_levels = {key: value[2] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})

# Cost function
nb_plots = 8
runs_per_plot = 4
def cost_fn(Y):
    # Short-circuit
    if len(Y) == 0:
        return []

    # Determine the number of resets
    resets = np.zeros(len(Y))
    resets[1:] = (np.diff(Y[:, 0]) != 0).astype(np.int64)
    for i in range(runs_per_plot, len(resets)):
        if np.all(resets[i-(runs_per_plot-1):i] == 0):
            resets[i] = 1

    # Determine number of runs per plot
    idx = np.concatenate([[0], np.flatnonzero(resets), [len(Y)]])
    plot_costs = [None] * (len(idx) - 1)
    for i in range(len(idx)-1):
        plot_costs[i] = (np.ones(idx[i+1] - idx[i]), runs_per_plot, np.arange(idx[i], idx[i+1]))
    
    return [
        (resets, nb_plots - 1, np.arange(len(Y))),
        *plot_costs
    ]

# Prepare the parameters
fn = default_fn(Dopt())
params, _ = create_parameters(fn, effect_types, effect_levels, plot_sizes, model=model)
params.fn.metric.preinit(params)

# Evaluate the ref model
Y = pd.read_csv(f'{root}/../splitk_plot/results/example_split_plot.csv')
X = params.Y2X(Y.to_numpy())
params.fn.metric.init(Y, X, params)
metric_ref = params.fn.metric.call(Y, X, params)

# Evaluate the cost model
Y = pd.read_csv(f'{root}/../cost_optimal/results/example_split_plot.csv')
X = params.Y2X(Y.to_numpy())
params.fn.metric.init(Y, X, params)
metric = params.fn.metric.call(Y, X, params)

print(metric, metric_ref)


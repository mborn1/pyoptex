import numpy as np
import pandas as pd
import os
from pyoptex.doe.constraints import parse_script
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import x2fx, obs_var_from_Zs
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.doe.cost_optimal.wrapper import create_parameters, default_fn

# Get root folder
root = os.path.split(__file__)[0]

# Define parameters
effects = {
    # Define effect type, model type
    'A': (1, 'quad'),
    'B': (1, 'quad'),
    # 'C': (1, 'tfi'),
}
# Second order sheffe model = full quadratic with 1 less term and constraint on sum

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

# Define new (augmented) sheffe model
model_aug = partial_rsm_names({key: 'tfi' for key in ['A', 'B', 'C']}).iloc[1:].to_numpy()

# Cost function
max_cost = np.array([2.5, 4, 10]) # In kg
def cost_fn(Y):
    # Extract experiment consumption costs
    c1 = Y[:, 0]
    c2 = Y[:, 1]

    # Return each subcost
    return [
        (c1, max_cost[0], np.arange(len(Y))),
        (c2, max_cost[1], np.arange(len(Y))),
        (1 - c1 - c2, max_cost[2], np.arange(len(Y)))
    ]

# Define coordinates
coords = [
    np.arange(0, 1.0001, 0.05)[:, np.newaxis],
    np.arange(0, 1.0001, 0.05)[:, np.newaxis],
]

# Covariance function
def cov(Y, X, Zs, Vinv, costs, random=False):
    # Define Y
    Y = np.concatenate((Y, np.expand_dims(1-Y[:,0]-Y[:,1], axis=1)), axis=1)

    # Recompute X given the augmented model
    X = x2fx(Y, model_aug)

    return Y, X, Zs, Vinv

# Define the metric
metric = Iopt(cov=cov)

# Define constraints
constraints = parse_script(
    f'(`A` + `B` > 1) | (`A` < 0.1) | (`B` < 0.2) | (`A` > 0.4) | (`B` > 0.5) | ((1 - `A` - `B`) < 0.1) | ((1 - `A` - `B`) > 0.7)', 
    effect_types
).encode()

# Prepare the parameters
fn = default_fn(1, cost_fn, metric, constraints=constraints)
params, _ = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, coords=coords)
params.fn.metric.init(params)

# Evaluate the ref model
Yref = pd.read_csv(f'{root}/../cost_optimal/data/ref_ingredient_stock.csv').to_numpy()
Xref = params.Y2X(Yref)
assert not np.any(params.fn.constraints(Yref))
metric_ref = params.fn.metric.call(Yref, Xref, [], np.expand_dims(np.eye(len(Yref)), 0), params.fn.cost(Yref))
print([np.sum(c) for c, _, _ in params.fn.cost(Yref)])

# Evaluate the cost model
Y = pd.read_csv(f'{root}/../cost_optimal/results/example_ingredient_stock.csv').to_numpy()
X = params.Y2X(Y)
assert not np.any(params.fn.constraints(Y))
metric = params.fn.metric.call(Y, X, [], np.expand_dims(np.eye(len(Y)), 0), params.fn.cost(Y))
print([np.sum(c) for c, _, _ in params.fn.cost(Y)])

print(metric, metric_ref)


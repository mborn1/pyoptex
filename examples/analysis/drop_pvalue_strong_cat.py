#!/usr/bin/env python3

import numpy as np
import pandas as pd

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names, term2strong, decode_term
from pyoptex.analysis import SimpleRegressor, PValueDropRegressor
from pyoptex.analysis.utils.plot import plot_res_diagnostics

# Seed randomization
set_seed(42)

# Define the factors
factors = [
    Factor('A', type='categorical', levels=['A_0', 'A_1', 'A_2'], coords=[[1, 0], [0, 1], [0, 0]]), 
    Factor('B'), Factor('C')
]
log_checkpoint("factor_names", [str(f.name) for f in factors])

# The number of random observations
N = 200

# Define the data
data = pd.DataFrame(np.random.rand(N, 0) * 2 - 1, columns=[])
for factor in factors:
    if factor.is_continuous:
        data[str(factor.name)] = np.random.rand(N) * 2 - 1
    else:
        data[str(factor.name)] = np.random.choice(factor.levels, N, replace=True)
data['Y'] = 2*np.where(data['A'] == 'A_0', 1, 0) \
            + 3*data['C'] \
            - 4*np.where(data['A'] == 'A_0', 1, 0)*data['B'] \
            + 5 \
            + np.random.normal(0, 1, N)
log_checkpoint("data_shape", list(data.shape))
log_checkpoint("data_Y_mean", float(data["Y"].mean()))

# Create the model
model = partial_rsm_names({str(f.name): 'quad' for f in factors})
Y2X = model2Y2X(model, factors)
log_checkpoint("model_shape", list(model.shape))
log_checkpoint("model_values", model.values.tolist())

# Define the dependencies
dependencies = order_dependencies(model, factors)
log_checkpoint("dependencies", dependencies.tolist())

# Create the regressor using weak heredity
regr = PValueDropRegressor(
    factors, Y2X,
    threshold=0.05, dependencies=dependencies, mode='weak'
)
regr.fit(data.drop(columns='Y'), data['Y'])
log_checkpoint("weak_terms", regr.terms_.tolist())
log_checkpoint("weak_formula", regr.model_formula(model=model))

# Convert the final model to strong and refit
terms_strong = term2strong(regr.terms_, dependencies)     # Convert to strong heredity model
terms_strong = decode_term(terms_strong, model, factors)  # Decode the categorical variable ('A_0' becomes 'A')
model = model.iloc[terms_strong]
Y2X = model2Y2X(model, factors)
log_checkpoint("strong_terms", terms_strong.tolist())

regr_simple = SimpleRegressor(factors, Y2X).fit(data.drop(columns='Y'), data['Y'])
log_checkpoint("model_formula", regr_simple.model_formula(model=model))

# Print the summary
print(regr_simple.summary())

# Print the formula in encoded form
print(regr_simple.model_formula(model=model))

# Predict
data['pred'] = regr_simple.predict(data.drop(columns='Y'))
log_checkpoint("pred_mean", float(data["pred"].mean()))
log_checkpoint("pred_std", float(data["pred"].std()))
log_checkpoint("predictions", data["pred"].values.tolist())

# Plot the residual diagnostics
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
).show()

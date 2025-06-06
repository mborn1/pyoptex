#!/usr/bin/env python3

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names, model2strong
from pyoptex.analysis import SimpleRegressor, PValueDropRegressor
from pyoptex.analysis.utils.plot import plot_res_diagnostics

# Seed randomization
set_seed(42)

# Define the factors
factors = [
    Factor('A'), Factor('B'), Factor('C')
]

# The number of random observations
N = 200

# Define the data
data = pd.DataFrame(np.random.rand(N, 3) * 2 - 1, columns=[str(f.name) for f in factors])
data['Y'] = 2*data['A'] + 3*data['C'] - 4*data['A']*data['B'] + 5\
                + np.random.normal(0, 1, N)

# Create the model
model = partial_rsm_names({str(f.name): 'quad' for f in factors})
Y2X = model2Y2X(model, factors)

# Define the dependencies
dependencies = order_dependencies(model, factors)

# Create the regressor using weak heredity
regr = PValueDropRegressor(
    factors, Y2X,
    threshold=0.05, dependencies=dependencies, mode='weak'
)
regr.fit(data.drop(columns='Y'), data['Y'])

# Convert the final model to strong and refit
terms_strong = model2strong(regr.terms_, dependencies)
model = model.iloc[terms_strong]
Y2X = model2Y2X(model, factors)

regr_simple = SimpleRegressor(factors, Y2X).fit(data.drop(columns='Y'), data['Y'])

# Print the summary
print(regr_simple.summary())

# Print the formula in encoded form
print(regr_simple.model_formula(model=model))

# Predict
data['pred'] = regr_simple.predict(data.drop(columns='Y'))

# Plot the residual diagnostics
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
).show()
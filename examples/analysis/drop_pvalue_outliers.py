#!/usr/bin/env python3

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names
from pyoptex.analysis import PValueDropRegressor, QuantileOutliersTransformer, SimpleRegressor
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
data.loc[np.arange(N//100) * 100, 'Y'] += 100

# Create the model
model = partial_rsm_names({str(f.name): 'quad' for f in factors})
Y2X = model2Y2X(model, factors)

# Define the dependencies
dependencies = order_dependencies(model, factors)

# Extract X and y
X = data.drop(columns='Y')
y = data['Y']

print('True model:', 'cst + A + B + A*C')

##############
# Create the regressor
regr = PValueDropRegressor(
    factors, Y2X,
    threshold=0.05, dependencies=dependencies, mode='weak'
)
regr.fit(X, y)

# Print the formula in encoded form
print('Without outlier removal:', regr.model_formula(model=model))

##############
# Detect and remove outliers
outlier_transformer = QuantileOutliersTransformer(
    factors, Y2X, threshold=5, stat='norm'
)
X, y = outlier_transformer.fit_transform(X, y)

# Create the regressor
regr = PValueDropRegressor(
    factors, Y2X,
    threshold=0.05, dependencies=dependencies, mode='weak'
)
regr.fit(X, y)

# Print the formula in encoded form
print('With outlier removal:',regr.model_formula(model=model))

##############
# Predict
data['pred'] = regr.predict(data.drop(columns='Y'))
data['outliers'] = outlier_transformer.outliers_

# Plot the residual diagnostics of everything
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
    color='outliers'
).show()

# Plot the residual diagnostics of the inliers
plot_res_diagnostics(
    data.loc[~outlier_transformer.outliers_], 
    y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
).show()

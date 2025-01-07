#!/usr/bin/env python3

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from pyoptex.analysis import SimpleRegressor
from pyoptex.analysis.utils.plot import plot_res_diagnostics

# Seed randomization
set_seed(42)

# Define the factors
factors = [
    Factor('A'), Factor('B'), Factor('C')
]

# The number of random observations
N = 200
nre = 5

# Define the data
data = pd.DataFrame(np.random.rand(N, 3) * 2 - 1, columns=[str(f.name) for f in factors])
data['RE'] = np.array([f'L{i}' for i in range(nre)])[np.repeat(np.arange(nre), N//nre)]
data['Y'] = 2*data['A'] + 3*data['C'] - 4*data['A']*data['B'] + 5\
                + np.random.normal(0, 1, N)\
                + np.repeat(np.random.normal(0, 1, nre), N//nre)

# Create the model
model = partial_rsm_names({str(f.name): 'quad' for f in factors})
Y2X = model2Y2X(model, factors)

# Define random effects
random_effects = ('RE',)

# Create the regressor
regr = SimpleRegressor(factors, Y2X, random_effects)
regr.fit(data.drop(columns='Y'), data['Y'])

# Print the summary
print(regr.summary())

# Print the formula in encoded form
print(regr.model_formula(model=model))

# Predict
data['pred'] = regr.predict(data.drop(columns='Y'))

# Create prediction plot
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
    color='RE'
).show()
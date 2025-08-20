
#!/usr/bin/env python3

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names
from pyoptex.analysis import SamsRegressor
from pyoptex.analysis.utils.plot import plot_res_diagnostics

# Seed randomization
set_seed(42)

# Define the factors
factors = [
    Factor('A'), Factor('B'), Factor('C'),
    Factor('D'), Factor('E'), Factor('F'),
]

# The number of random observations
N = 200

# Define the data
data = pd.DataFrame(np.random.rand(N, len(factors)) * 2 - 1, columns=[str(f.name) for f in factors])
data['Y'] = 2*data['A'] + 3*data['C'] - 4*data['A']*data['B'] + 5\
                + np.random.normal(0, 1, N)

# Define the model orders
model_order = {str(f.name): 'quad' for f in factors}

# Create the model
model = partial_rsm_names(model_order)
Y2X = model2Y2X(model, factors)

# Define the dependencies
dependencies = order_dependencies(model, factors)

# Create the regressor
regr = SamsRegressor(
    factors, Y2X,
    dependencies=dependencies, mode='weak',
    forced_model=np.array([0], np.int64),
    model_size=6, nb_models=5000, skipn=1000,
    entropy_model_order=model_order,
    ncluster='auto'
)
regr.fit(data.drop(columns='Y'), data['Y'])

# Plot the raster plot
regr.plot_selection().show()

# Print the summary
print(regr.summary())

# Print the formula in encoded form
print(regr.model_formula(model=model))

# Predict
data['pred'] = regr.predict(data.drop(columns='Y'))

# Plot the residual diagnostics
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
).show()


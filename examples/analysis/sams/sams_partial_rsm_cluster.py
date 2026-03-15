
#!/usr/bin/env python3

import numpy as np
import pandas as pd

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

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
log_checkpoint("factor_names", [str(f.name) for f in factors])

# The number of random observations
N = 200

# Define the data
data = pd.DataFrame(np.random.rand(N, len(factors)) * 2 - 1, columns=[str(f.name) for f in factors])
data['Y'] = 2*data['A'] + 3*data['C'] - 4*data['A']*data['B'] + 5\
                + np.random.normal(0, 1, N)
log_checkpoint("data_shape", list(data.shape))
log_checkpoint("data_Y_mean", float(data["Y"].mean()))

# Define the model orders
model_order = {str(f.name): 'quad' for f in factors}

# Create the model
model = partial_rsm_names(model_order)
Y2X = model2Y2X(model, factors)
log_checkpoint("model_shape", list(model.shape))
log_checkpoint("model_values", model.values.tolist())

# Define the dependencies
dependencies = order_dependencies(model, factors)
log_checkpoint("dependencies", dependencies.tolist())

# Create the regressor
regr = SamsRegressor(
    factors, Y2X,
    dependencies=dependencies, mode='weak',
    forced_model=np.array([0], np.int64),
    model_size=8, nb_models=5000, skipn=3000,
    entropy_model_order=model_order,
    ncluster='auto'
)
regr.fit(data.drop(columns='Y'), data['Y'])
log_checkpoint("nb_models", len(regr.models_))
log_checkpoint("model_formulas", [regr.model_formula(model=model, idx=i) for i in range(len(regr.models_))])
log_checkpoint("selected_models", [m.tolist() for m in regr.models_])

# Plot the raster plot
regr.plot_selection(model=model).show()

# Print the summary
print(regr.summary())

# Print the formula in encoded form
for i in range(len(regr.models_)):
    print(regr.model_formula(model=model, idx=i))
    log_checkpoint(f"model_coef_{i}", regr.model_coef_[i].tolist())
    log_checkpoint(f"models_{i}", regr.models_[i].tolist())

# Predict
data['pred'] = regr.predict(data.drop(columns='Y'))
log_checkpoint("pred_mean", float(data["pred"].mean()))
log_checkpoint("pred_std", float(data["pred"].std()))
log_checkpoint("predictions", data["pred"].values.tolist())

# Plot the residual diagnostics
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=[str(f.name) for f in factors],
).show()

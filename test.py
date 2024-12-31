import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X
from pyoptex.analysis.utils import order_dependencies, plot_res_diagnostics
from pyoptex.analysis.simple_model import SimpleRegressor
from pyoptex.analysis.p_value_drop_model import PValueDropRegressor

set_seed(42)

# Define the factors
factors = [
    Factor('A'), Factor('B'), Factor('C')
]

# TODO: add plot functions for every feature seperately

N = 200

# Define the data
data = pd.DataFrame(np.random.rand(N, 3) * 2 - 1, columns=['A', 'B', 'C'])
data['RE'] = np.array(['L1', 'L2', 'L3', 'L4', 'L5'])[np.repeat(np.arange(5), 200//5)]
data['Y'] = 2*data['A'] + 3*data['B'] - 4*data['A']*data['B'] + 5\
                + np.random.normal(0, 1, N)\
                + np.repeat(np.random.normal(0, 1, 5), N//5)

# Check if mixed model
is_mixed_lm = 'RE' in data.columns

# Define the model
model = pd.DataFrame([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [2, 0, 0],
    [3, 0, 0],
], columns=['A', 'B', 'C'])
dependencies = order_dependencies(model, factors)
Y2X = model2Y2X(model, factors)

# Define random effects
random_effects = ('RE',) if is_mixed_lm else ()

# Create the regressor
regr = PValueDropRegressor(
    factors, Y2X, random_effects, conditional=True,
    threshold=0.05, mode=None, dependencies=dependencies
)
regr.fit(data.drop(columns='Y'), data['Y'])
print(regr.summary())
print(regr.model_formula(model=model))

# Create prediction plot
data['pred'] = regr.predict(data.drop(columns='Y'))
plot_res_diagnostics(
    data, y_true='Y', y_pred='pred', 
    textcols=('A', 'B', 'C'),
    color='RE' if is_mixed_lm else None
).show()

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.utils.model import model2Y2X
from pyoptex.analysis.utils import order_dependencies
from pyoptex.analysis.simple_model import SimpleRegressor

set_seed(42)

# Define the factors
factors = [
    Factor('A'), Factor('B'), Factor('C')
]

# Define the data
data = pd.DataFrame(np.random.rand(200, 3) * 2 - 1, columns=['A', 'B', 'C'])
data['RE'] = np.array(['L1', 'L2', 'L3', 'L4', 'L5'])[np.repeat(np.arange(5), 200//5)]
data['Y'] = 2*data['A'] + 3*data['B'] - 4*data['A']*data['B'] + 5\
                + np.random.normal(0, 1, len(data))\
                + np.repeat(np.random.normal(0, 1, 5), 200//5)

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
dep = order_dependencies(model, factors)
Y2X = model2Y2X(model, factors)

# Define random effects
random_effects = ('RE',)

# Create the regressor
regr = SimpleRegressor(factors, Y2X, random_effects)
regr.fit(data.drop(columns='Y'), data['Y'])
print(regr.score(data.drop(columns=['Y']), data['Y']))

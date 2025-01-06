
import numba
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.design import x2fx
from pyoptex.utils.model import (
    partial_rsm_names, model2Y2X, model2encnames, order_dependencies,
    permitted_dep_add
)
from pyoptex.analysis import SamsRegressor

# Wolters & Bingham (2012)
# 5000 randomly generated models from PB12 and PB20 designs
# Model size, active variables, coefficients randomly generated
#       Hereditary (weak) and between 1 and s_max (=4 for PB12 and =6 for PB20) terms
#       Bounds for coefficient magnitudes

set_seed(42)

# Set the simulation parameters according to the paper supplement
smax = 4
nsims = 10
noise_var = 1
coeff_bounds = [(0.9, 3.6), (0.9, 1.9), (0.9, 1.4), (0.9, 1.1)]

# Define the data (PB12)
data = pd.DataFrame([
    [ 1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1],
    [ 1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1],
    [-1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1],
    [ 1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1],
    [ 1,  1, -1, -1, -1,  1, -1,  1,  1, -1,  1],
    [ 1, -1, -1, -1,  1, -1,  1,  1, -1,  1,  1],
    [-1, -1, -1,  1, -1,  1,  1, -1,  1,  1,  1],
    [-1, -1,  1, -1,  1,  1, -1,  1,  1,  1, -1],
    [-1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1],
    [ 1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1],
    [-1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
], columns=list('ABCDEFGHIJK'))

# Define the factors
factors = [Factor(name) for name in list('ABCDEFGHIJK')]

# Define a model and dependencies
model_order = {str(f.name): 'tfi' for f in factors}
model = partial_rsm_names(model_order)
dependencies = order_dependencies(model, factors)
Y2X = model2Y2X(model, factors)

# Define the names of the encoded matrix
effect_types = np.array([
    1 if f.is_continuous else len(f.levels) 
    for f in factors
])
names = np.array(model2encnames(model, effect_types))

#############################################################

encoding = ['Correct', 'Underfitted', 'Overfitted', 'Partial', 'Wrong']
enc_stats = ({k: 0 for k in encoding}, {k: 0 for k in encoding})

contains_true_model = np.zeros(nsims, dtype=np.bool_)
contains_true_model_order = np.zeros(nsims, dtype=np.bool_)
fit_types = np.zeros(nsims, dtype=np.int_)
fit_types_order = np.zeros(nsims, dtype=np.int_)
nc_arr = np.zeros(nsims, dtype=np.int_)
nc_order_arr = np.zeros(nsims, dtype=np.int_)
nw_arr = np.zeros(nsims, dtype=np.int_)
nw_order_arr = np.zeros(nsims, dtype=np.int_)
ntot = np.zeros(nsims, dtype=np.int_)
nmain = np.zeros(nsims, dtype=np.int_)

for i in tqdm(range(nsims)):
    # Sample a random number of terms
    s = np.random.randint(1, smax+1)
    ntot[i] = s

    # Sample q main effects (weights not specified in paper)
    q = np.random.randint(1, s+1)
    nmain[i] = q

    # Define the model (+1 to include the intercept)
    m = np.zeros(s+1, dtype=np.int_)

    # Sample main effects
    m[1:q+1] = np.random.choice(len(factors), q, replace=False) + 1

    # Check which effects can be sampled according to weak heredity
    tfi_offset = 1+len(factors)
    permitted = permitted_dep_add(m[:q+1], mode='weak', dep=dependencies, subset=np.arange(tfi_offset, len(model)))
    tfi = np.random.choice(np.flatnonzero(permitted), s-q, replace=False) + tfi_offset
    m[q+1:] = tfi

    # Sort the model
    m = np.sort(m)

    # Sample the coefficients
    coeff = np.random.rand(len(m)) * (coeff_bounds[s-1][1] - coeff_bounds[s-1][0]) + coeff_bounds[s-1][0]
    coeff = coeff * (-1) ** np.random.randint(0, 1+1, len(coeff))

    # Compute Y
    y = np.sum(x2fx(data.to_numpy(), model.to_numpy())[:, m] * coeff, axis=1) \
                + np.random.randn(len(data)) * np.sqrt(noise_var)

    # Compute SAMS base
    regr = SamsRegressor(
        factors, Y2X, 
        dependencies=dependencies, mode='weak', 
        forced_model=np.array([0], dtype=np.int_),
        model_size=8, nb_models=10000, skipn=2000,
        model_order=model_order, tqdm=False
    )
    regr.fit(data, y)

    # Check if the first five contains the true model
    contains_true_model[i] = any(
        len(regr.models_) > i
            and regr.models_[i].size == m.size 
            and np.all(np.sort(regr.models_[i]) == m) 
        for i in range(5)
    )
    contains_true_model_order[i] = any(
        len(regr.models_order_) > i
            and regr.models_order_[i].size == m.size 
            and np.all(np.sort(regr.models_order_[i]) == m) 
        for i in range(5)
    )

    # Check the best model and which type it is (approx model)
    nc = np.sum(np.isin(regr.models_[0], m, assume_unique=True))
    nc_arr[i] = nc
    nw = (regr.models_[0].size - nc)
    nw_arr[i] = nw
    if nc == s+1 and nw == 0:
        fit_types[i] = 0 # Correct
    elif 0 < nc < s+1 and nw == 0:
        fit_types[i] = 1 # Underfitted
    elif nc == s+1 and nw > 0:
        fit_types[i] = 2 # Overfitted
    elif 0 < nc < s+1 and nw > 0:
        fit_types[i] = 3 # Partial-truth
    else:
        fit_types[i] = 4 # Wrong
    enc_stats[0][encoding[fit_types[i]]] += 1

    # Check the best model and which type it is (order model)
    nc = np.sum(np.isin(regr.models_order_[0], m, assume_unique=True))
    nc_order_arr[i] = nc
    nw = (regr.models_order_[0].size - nc)
    nw_order_arr[i] = nw
    if nc == s+1 and nw == 0:
        fit_types_order[i] = 0 # Correct
    elif 0 < nc < s+1 and nw == 0:
        fit_types_order[i] = 1 # Underfitted
    elif nc == s+1 and nw > 0:
        fit_types_order[i] = 2 # Overfitted
    elif 0 < nc < s+1 and nw > 0:
        fit_types_order[i] = 3 # Partial-truth
    else:
        fit_types_order[i] = 4 # Wrong
    enc_stats[1][encoding[fit_types_order[i]]] += 1

    print()
    print('Current status')
    print('--------------')
    print(f'Approximate:', enc_stats[0])
    print(f'Paper      :', enc_stats[1])
    print()

# # Write the results
# with open('results.txt', 'w') as f:
#     np.savetxt(f, contains_true_model)
#     f.write('----\n')
#     np.savetxt(f, contains_true_model_order)
#     f.write('----\n')
#     np.savetxt(f, fit_types)
#     f.write('----\n')
#     np.savetxt(f, fit_types_order)
#     f.write('----\n')
#     np.savetxt(f, nc_arr)
#     f.write('----\n')
#     np.savetxt(f, nc_order_arr)
#     f.write('----\n')
#     np.savetxt(f, nw_arr)
#     f.write('----\n')
#     np.savetxt(f, nw_order_arr)
#     f.write('----\n')
#     np.savetxt(f, ntot)
#     f.write('----\n')
#     np.savetxt(f, nmain)

##############################################

# Read the results
with open('results_1.txt', 'r') as f:
    lines = f.read().split('----\n')
    nequal_entropies = np.array([round(float(x)) for x in lines[0].split('\n')[:-1]])
    contains_true_model = np.array([True if round(float(x)) == 1 else False for x in lines[1].split('\n')[:-1]])
    contains_true_model_order = np.array([True if round(float(x)) == 1 else False for x in lines[2].split('\n')[:-1]])
    fit_types = np.array([round(float(x)) for x in lines[3].split('\n')[:-1]])
    fit_types_order = np.array([round(float(x)) for x in lines[4].split('\n')[:-1]])
    nc_arr = np.array([round(float(x)) for x in lines[5].split('\n')[:-1]])
    nc_order_arr = np.array([round(float(x)) for x in lines[6].split('\n')[:-1]])
    nw_arr = np.array([round(float(x)) for x in lines[7].split('\n')[:-1]])
    nw_order_arr = np.array([round(float(x)) for x in lines[8].split('\n')[:-1]])
    ntot = np.array([round(float(x)) for x in lines[9].split('\n')[:-1]])
    nmain = np.array([round(float(x)) for x in lines[10].split('\n')[:-1]])

################
# # Plot the number of same order models

# # Count unique values and counts
# uniques, counts = np.unique(nequal_entropies, return_counts=True)

# # Initialize the bars
# nbars = 5+1
# bars = np.zeros(nbars)

# # Store the counts
# final_bar = np.sum(counts[uniques >= nbars-1])
# bars[uniques[uniques < nbars-1]] = counts[uniques < nbars-1]
# bars[nbars-1] = final_bar

# # Plot the bars
# px.bar(x=np.arange(nbars), y=bars).show()
################

################
# Print the number of true model in top 5
print('True model in top 5 (approximate):', np.sum(contains_true_model) / contains_true_model.size * 100, '%')
print('True model in top 5 (Wolter & Bingham):', np.sum(contains_true_model_order) / contains_true_model_order.size * 100, '%')
################

################
# Print and plot table of types for first model
uft, ft_counts = np.unique(fit_types, return_counts=True)
uftmo, ftmo_counts = np.unique(fit_types_order, return_counts=True)

print('Approximate method:', {encoding[uft[i]]: ft_counts[i] for i in range(len(uft))})
print('Wolter & Bingham method:', {encoding[uftmo[i]]: ftmo_counts[i] for i in range(len(uftmo))})

fig = go.Figure()
fig.add_trace(go.Bar(name='Approximate', x=encoding, y=[ft_counts[np.argmax(uft == i)] if i in uft else 0 for i in range(len(encoding))]))
fig.add_trace(go.Bar(name='Wolter & Bingham', x=encoding, y=[ftmo_counts[np.argmax(uftmo == i)] if i in uftmo else 0 for i in range(len(encoding))]))
fig.show()
################

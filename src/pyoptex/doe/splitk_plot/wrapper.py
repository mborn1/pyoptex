import numpy as np
import pandas as pd
from numba.typed import List
from tqdm import tqdm

from .utils import Parameters
from ..utils.design import create_default_coords, encode_design, x2fx, obs_var, decode_design
from ..utils.model import encode_model

# level_grps function: TODO

def _compute_cs(plot_sizes, ratios, thetas, thetas_inv):
    # Compute c-coefficients for all ratios
    c = np.zeros((ratios.shape[0], plot_sizes.size))
    for j, ratios in enumerate(ratios):
        c[j, 0] = 1
        for i in range(1, c.shape[1]):
            c[j, i] = -ratios[i] * np.sum(thetas[:i] * c[j, :i]) / np.sum(ratios[:i+1] * thetas[:i+1])
    c = c[:, 1:]

    return c

def create_parameters(fn, effect_types, effect_levels, plot_sizes, ratios=None, coords=None, model=None, Y2X=None):
    """
    effect_types : dict or np.array(1d)
        If dictionary, maps each column name to its type. Also extracts the column names. A 1 indicates
        a continuous factor, anything higher is a categorical factor with that many levels.
    """
    assert model is not None or Y2X is not None, 'Either a polynomial model or Y2X function must be provided'

    # Parse effect types
    if isinstance(effect_types, dict):
        # Detect effect types
        col_names = list(effect_types.keys())
        effect_types = np.array(list(effect_types.values()))
    else:
        # No column names known
        col_names = None

    # Parse effect levels
    if isinstance(effect_levels, dict):
        if col_names is not None:
            effect_levels = np.array([effect_levels[col] for col in col_names])
        else:
            col_names = list(effect_levels.keys())
            effect_levels = np.array(list(effect_levels.values()))

    # Ratios
    ratios = ratios if ratios is not None else np.ones((1, plot_sizes.size - 1))
    if len(ratios.shape) == 1:
        ratios = ratios.reshape(1, -1)
    assert ratios.shape[1] == plot_sizes.size - 1, f'Bad number of ratios for plotsizes (ratios shape: {ratios.shape}, plot sizes size: {plot_sizes.size})'

    # Set default coords
    coords = coords if coords is not None else [None]*effect_types.size
    coords = [create_default_coords(et) if coord is None else coord for coord, et in zip(coords, effect_types)]

    # Encode the coordinates
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))
    coords_enc = List([
        encode_design(coord, np.array([et]))
            if et > 1 and coord.shape[1] == 1 and np.all(np.sort(coord) == create_default_coords(et))
            else coord.astype(np.float64)
        for coord, et in zip(coords, effect_types)
    ])

    # Alphas and thetas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]
    thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    thetas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/thetas[1:])))

    # Compute cs
    cs = _compute_cs(plot_sizes, ratios, thetas, thetas_inv)

    # Compute Vinv
    Vinv = np.array([obs_var(plot_sizes, ratios=c) for c in cs])  

    # Set all groups
    if grps is None:
        grps = level_grps(np.zeros_like(plot_sizes), plot_sizes)
        grps = List([grps[lvl] for lvl, _ in factors])

    # Set the Y2X function
    if Y2X is None:
        # Detect model in correct order
        if isinstance(model, pd.DataFrame):
            if col_names is not None:
                model = model[col_names].to_numpy()
            else:
                col_names = model.columns
                model = model.to_numpy()

        # Encode model
        modelenc = encode_model(model, effect_types)

        # Create transformation function for polynomial models
        Y2X = lambda Y: x2fx(Y, modelenc)

    # Create the parameters
    params = Parameters(
        fn, effect_types, effect_levels, grps, plot_sizes, ratios, 
        coords, colstart, cs, alphas, thetas, thetas_inv, Vinv, Y2X
    )

    return params, col_names

def create_splitk_plot_design(
        fn, effect_types, effect_levels, plot_sizes, ratios=None, coords=None, model=None, Y2X=None,
        n_tries=10, max_it=10000
    ):
    assert n_tries > 0

    # Extract the parameters
    params, col_names = create_parameters(
        fn, effect_types, effect_levels, plot_sizes, ratios, coords, model, Y2X
    )

    # Main loop
    best_metric = -np.inf
    best_Y = np.zeros((params.alphas[0], params.effect_types.size))
    for i in tqdm(range(n_tries)):

        # Optimize the design
        Y, X = optimize(params, max_it)

        # Compute metric
        metric = params.fn.metric.call(Y, X)

        # Store the results
        if metric > best_metric:
            best_metric = metric
            best_Y = np.copy(Y)

    # Decode the final design
    Y = decode_design(best_Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=col_names)

    return Y, metric
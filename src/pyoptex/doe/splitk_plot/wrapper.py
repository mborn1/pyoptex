import numpy as np
import numba
import pandas as pd
from numba.typed import List
from tqdm import tqdm

from .optimize import optimize
from .init import initialize_feasible
from .utils import Parameters, FunctionSet, State, level_grps, obs_var, extend_design
from ..utils.design import create_default_coords, encode_design, x2fx, decode_design
from ..utils.model import encode_model
from ..constraints import no_constraints

def _compute_cs(plot_sizes, ratios, thetas, thetas_inv):
    # Compute c-coefficients for all ratios
    c = np.zeros((ratios.shape[0], plot_sizes.size))
    for j, ratio in enumerate(ratios):
        c[j, 0] = 1
        for i in range(1, c.shape[1]):
            c[j, i] = -ratio[i-1] * np.sum(thetas[:i] * c[j, :i]) / (thetas[0] + np.sum(ratio[:i] * thetas[1:i+1]))
    c = c[:, 1:]
    return c

def default_fn(metric, constraints=no_constraints, init=initialize_feasible):
    return FunctionSet(metric, constraints, init)

def create_parameters(fn, effect_types, effect_levels, plot_sizes, prior=None, ratios=None, coords=None, model=None, Y2X=None, cov=None, grps=None, compute_update=True):
    """
    effect_types : dict or np.array(1d)
        If dictionary, maps each column name to its type. Also extracts the column names. A 1 indicates
        a continuous factor, anything higher is a categorical factor with that many levels.
    Y2X : func
        Function to transform Y (with any covariates) to X
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

    # Set the Y2X function
    if Y2X is None:
        # Detect model in correct order
        if isinstance(model, pd.DataFrame):
            if col_names is not None:
                model = model[col_names].to_numpy()
            else:
                col_names = model.columns
                model = model.to_numpy()

        # Add covariates
        if cov is not None:
            # Expand covariates
            cov, model_cov, effect_types_cov = cov

            # Extend effect types
            cov_col_names = None
            if isinstance(effect_types_cov, dict):
                cov_col_names = list(effect_types_cov.keys())
                effect_types_cov = np.array(list(effect_types_cov.values()))
            
            # Parse covariates model
            if isinstance(model_cov, pd.DataFrame):
                # Sort the column names
                if col_names is not None:
                    if cov_col_names is not None:
                        cov_col_names = [col for col in model_cov.columns if col not in col_names]
                    model_cov = model_cov[[*col_names, *cov_col_names]]
                model_cov = model_cov.to_numpy()

            # Possibly encode cov
            cov = encode_design(cov, effect_types_cov)
            
            # Extend parameters
            et = np.concatenate((effect_types, effect_types_cov))
            model = np.concatenate((model, np.zeros((model.shape[0], model_cov.shape[1] - model.shape[1]))), axis=1)
            model = np.concatenate((model, model_cov), axis=0)
        else:
            # Default effect types
            et = effect_types

        # Encode model
        modelenc = encode_model(model, et)

        # Create transformation function for polynomial models
        Y2X = numba.njit(lambda Y: x2fx(Y, modelenc))
    else:
        # Make sure cov only specifies the ndarray
        if not isinstance(cov, np.ndarray):
            raise ValueEror(
                'When Y2X is specified, the model must already include covariates and '
                'cov is the array of elements to prepend to Y'
            )

    # Determine a prior
    if prior is not None:
        # Expand prior information
        prior, old_plot_sizes = prior

        # Convert prior to numpy
        if isinstance(prior, pd.DataFrame):
            if col_names is not None:
                prior = prior[col_names]
            prior = prior.to_numpy()

        # Make sure it's not encoded
        assert prior.shape[1] == effect_types.size, 'The prior must not be encoded'

        # Augment the design
        prior = extend_design(prior, old_plot_sizes, plot_sizes, effect_levels)
    else:
        # Nothing to start from
        old_plot_sizes = np.zeros_like(plot_sizes) 

    # Define which groups to optimize
    lgrps = level_grps(old_plot_sizes, plot_sizes)
    if grps is None:
        grps = List([lgrps[lvl] for lvl in effect_levels])
    else:
        grps = List([np.concatenate((grps[i].astype(np.int64), lgrps[effect_levels[i]]), dtype=np.int64) for i in range(len(effect_levels))])

    # Force types
    plot_sizes = plot_sizes.astype(np.int64)

    # Compile constraints
    fn = fn._replace(constraints=numba.njit(fn.constraints))

    # Create the parameters
    params = Parameters(
        fn, effect_types, effect_levels, grps, plot_sizes, ratios, 
        coords_enc, prior, colstart, cs, alphas, thetas, thetas_inv, Vinv, Y2X, cov,
        compute_update
    )
    
    return params, col_names

def create_splitk_plot_design(
        fn, effect_types, effect_levels, plot_sizes, prior=None, ratios=None, coords=None, model=None, Y2X=None, cov=None, 
        grps=None, compute_update=True, n_tries=10, max_it=10000, validate=False
    ):
    assert n_tries > 0

    # Extract the parameters
    params, col_names = create_parameters(
        fn, effect_types, effect_levels, plot_sizes, prior, ratios, coords, model, Y2X, cov, grps, compute_update
    )

    # Pre initialize metric
    params.fn.metric.preinit(params)

    # Main loop
    best_metric = -np.inf
    best_state = None
    for i in tqdm(range(n_tries)):

        # Optimize the design
        Y, state = optimize(params, max_it, validate=validate)

        # Store the results
        if state.metric > best_metric:
            best_metric = state.metric
            best_state = State(np.copy(state.Y), np.copy(state.X), state.metric)

    # Decode the final design
    Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=col_names)

    return Y, best_state

import numpy as np
import numba
import pandas as pd
from numba.typed import List
from tqdm import tqdm

from .optimize import optimize
from .init import initialize_feasible
from .utils import Parameters, FunctionSet, State, Factor, Plot, level_grps, obs_var, extend_design
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
    return FunctionSet(metric, constraints.encode(), constraints.func(), init)

def create_parameters(factors, fn, Y2X, prior=None, grps=None, use_formulas=True):
    """
    
    """
    # Assertions
    assert len(factors) > 0, 'At least one factor must be provided'
    for i, f in enumerate(factors):
        assert isinstance(f, Factor), f'Factor {i} is not of type Factor'
    if prior is not None:
        assert isinstance(prior, pd.DataFrame), f'The prior must be specified as a dataframe but is a {type(prior)}'
    assert min(f.plot.level for f in factors) == 0, f'The plots must start from level 0 (easy-to-change factors)'

    # Extract the plot sizes
    nb_plots = max(f.plot.level for f in factors) + 1
    plot_sizes = np.ones(nb_plots, dtype=np.int64) * -1
    ratios = [None] * nb_plots
    for f in factors:
        # Fix plot sizes
        if plot_sizes[f.plot.level] == -1:
            plot_sizes[f.plot.level] = f.plot.size
        else:
            assert plot_sizes[f.plot.level] == f.plot.size, f'Plot sizes at the same plot level must be equal, but are {plot_sizes[f.plot.level]} and {f.plot.size}'

        # Fix ratios
        r = np.sort(f.plot.ratio) if isinstance(f.plot.ratio, tuple) or isinstance(f.plot.ratio, list) \
                or isinstance(f.plot.ratio, np.ndarray) else [f.plot.ratio]
        if ratios[f.plot.level] is None:
            ratios[f.plot.level] = r
        else:
            assert all(i==j for i, j in zip(ratios[f.plot.level], r)), f'Plot ratios at the same plot level must be equal, but are {ratios[f.plot.level]} and {r}'

    # Align ratios
    nratios = max([len(r) for r in ratios])
    assert all(len(r) == 1 or len(r) == nratios for r in ratios), 'All ratios must be either a single number or and array of the same size'
    ratios = np.array([np.repeat(ratio, nratios) if len(ratio) == 1 else ratio for ratio in ratios]).T

    # Normalize ratios
    ratios = ratios[:, 1:] / ratios[:, 0]

    # Extract parameter arrays
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    effect_levels = np.array([f.plot.level for f in factors])
    coords = List([f.coords_ for f in factors])

    # Encode the coordinates
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))

    # Alphas and thetas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]
    thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    thetas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/thetas[1:])))

    # Compute cs
    cs = _compute_cs(plot_sizes, ratios, thetas, thetas_inv)

    # Compute Vinv
    Vinv = np.array([obs_var(plot_sizes, ratios=c) for c in cs])  

    # Determine a prior
    if prior is not None:
        # Expand prior
        prior, old_plots = prior
        assert all(isinstance(p, Plot) for p in old_plots), f'Old plots must be of type Plot'

        # Normalize factors
        for f in factors:
            prior[str(f.name)] = f.normalize(prior[str(f.name)])

        # Convert from pandas to numpy
        prior = prior[col_names].to_numpy()
        
        # Encode the design
        prior = encode_design(prior, effect_types, coords=coords)

        # Compute old plot sizes
        nb_old_plots = max(p.level for p in old_plots)
        old_plot_sizes = np.ones(nb_old_plots, dtype=np.int64) * -1
        for p in old_plots:
            if old_plot_sizes[p.level] == -1:
                old_plot_sizes[p.level] = p.size
            else:
                assert plot_sizes[p.level] == p.size, f'Prior plot sizes at the same prior plot level must be equal, but are {plot_sizes[p.level]} and {p.size}'

        # Assert the prior
        assert np.prod(old_plot_sizes) == len(prior), f'Prior plot sizes are misspecified, prior has {len(prior)} runs, but plot sizes require {np.prod(old_plot_sizes)} runs'
        assert nb_old_plots == nb_plots, f'The prior must specify the same number of levels as the factors: prior has {len(old_plot_sizes)} levels, but new design requires {len(plot_sizes)} levels'

        # TODO: validate prior (constraints)

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

    # Create the parameters
    params = Parameters(
        fn, effect_types, effect_levels, grps, plot_sizes, ratios, 
        coords, prior, colstart, cs, alphas, thetas, thetas_inv, Vinv, Y2X,
        use_formulas
    )
    
    return params

def create_splitk_plot_design(
        factors, fn, Y2X, prior=None, grps=None, use_formulas=True, 
        n_tries=10, max_it=10000, validate=False
    ):
    assert n_tries > 0, 'Must specify at least one random initialization (n_tries > 0)'

    # Extract the parameters
    params = create_parameters(factors, fn, Y2X, prior, grps, use_formulas)

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

    # Decode the design
    Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=[str(f.name) for f in factors])
    for f in factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])

    return Y, best_state

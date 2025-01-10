"""
Module for the interface to run the split^k-plot algorithm
"""

import numpy as np
import pandas as pd
from numba.typed import List
from tqdm import tqdm

from ...constraints import no_constraints, mixture_constraints
from ....utils.design import decode_design
from ..utils import Factor, FunctionSet, State
from .init import initialize_feasible
from .optimize import optimize
from .utils import (Parameters, Plot, extend_design, level_grps, obs_var, obs_var_Zs)


def _compute_cs(plot_sizes, ratios, thetas):
    """
    Computes the c-coefficients of the inverse of the 
    observation covariance matrix (Vinv).

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The plot sizes.
    ratios : np.array(2d)
        Multiple sets of a-priori variance ratios.
    thetas : np.array(1d)
        The array of thetas.
        thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    
    Returns
    -------
    cs : np.array(2d)
        The c-coefficients for each set of a-priori variance ratios.
    """
    # Compute c-coefficients for all ratios
    c = np.zeros((ratios.shape[0], plot_sizes.size))
    for j, ratio in enumerate(ratios):
        c[j, 0] = 1
        for i in range(1, c.shape[1]):
            c[j, i] = -ratio[i-1] * np.sum(thetas[:i] * c[j, :i])\
                         / (thetas[0] + np.sum(ratio[:i] * thetas[1:i+1]))
    c = c[:, 1:]
    return c

def default_fn(factors, metric, Y2X, constraints=no_constraints, init=initialize_feasible):
    """
    Create a functionset with the default operators. Each
    operator can be manually overriden by providing the parameter.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.doe.fixed_stucture.utils.Factor>`)
        The factors of the experiment.
    metric : :py:class:`SplitkPlotMetricMixin <pyoptex.doe.fixed_structure.splitk_plot.metric.SplitkPlotMetricMixin>`
        The metric object.
    Y2X : func
        The function converting from the design matrix to the
        model matrix.
    constraints : func
        The constraints function, 
        :py:func:`no_constraints <pyoptex.doe.constraints.no_constraints>` 
        by default.
    init : func
        The initialization function,
        :py:func:`initialize_feasible <pyoptex.doe.fixed_structure.splitk_plot.init.initialize_feasible>`
        by default.

    Returns
    -------
    fn : :py:class:`FunctionSet <pyoptex.doe.fixed_structure.utils.FunctionSet>`
        The function set.
    """

    # Check if factors contain mixtures
    if any(f.is_mixture for f in factors):
        # Create the mixture constraints
        mix_constr = mixture_constraints(
            [str(f.name) for f in factors if f.is_mixture], 
            factors
        )

        # Add the mixture constraints
        if constraints is None:
            constraints = mix_constr
        else:
            constraints = constraints | mix_constr

    # Default to no constraints
    if constraints is None:
        constraints = no_constraints

    return FunctionSet(metric, Y2X, constraints.encode(), constraints.func(), init)

def create_parameters(factors, fn, prior=None, grps=None, use_formulas=True):
    """
    Creates the parameters object by preprocessing the inputs. 
    This is a utility function to transform each variable 
    to its correct representation.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.doe.fixed_structure.utils.Factor>`)
        The list of factors.
    fn : :py:class:`FunctionSet <pyoptex.doe.fixed_structure.utils.FunctionSet>`
        A set of operators for the algorithm.
    prior : None or (pd.DataFrame, list(:py:class:`Plot <pyoptex.doe.fixed_structure.splitk_plot.utils.Plot>`)
        A possible prior design to use for augmentation. Must be 
        denormalized and decoded. The list of plots represents the configuration
        of the prior.
    grps : list(np.array(1d) or None)
        A list of groups to optimize for each factor. If None, all groups are optimized
        for that factor.
    use_formulas : bool
        Whether to use the internal update formulas or not.

    Returns
    -------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
        The simulation parameters.
    """
    # Assertions
    assert len(factors) > 0, 'At least one factor must be provided'
    for i, f in enumerate(factors):
        assert isinstance(f, Factor), f'Factor {i} is not of type Factor'
        assert isinstance(f.re, Plot), f'Factor {i} with name {f.name} does not have a Plot as random effect'
    if prior is not None:
        assert isinstance(prior[0], pd.DataFrame), f'The prior must be specified as a dataframe but is a {type(prior)}'
    assert min(f.re.level for f in factors) == 0, f'The plots must start from level 0 (easy-to-change factors)'

    # Extract the plot sizes
    nb_plots = max(f.re.level for f in factors) + 1
    plot_sizes = np.ones(nb_plots, dtype=np.int_) * -1
    ratios = [None] * nb_plots
    for f in factors:
        # Fix plot sizes
        if plot_sizes[f.re.level] == -1:
            plot_sizes[f.re.level] = f.re.size
        else:
            assert plot_sizes[f.re.level] == f.re.size, f'Plot sizes at the same plot level must be equal, but are {plot_sizes[f.re.level]} and {f.re.size}'

        # Fix ratios
        r = np.sort(f.re.ratio) \
                if isinstance(f.re.ratio, tuple) or isinstance(f.re.ratio, list) \
                or isinstance(f.re.ratio, np.ndarray) else [f.re.ratio]
        if ratios[f.re.level] is None:
            ratios[f.re.level] = r
        else:
            assert all(i==j for i, j in zip(ratios[f.re.level], r)), f'Plot ratios at the same plot level must be equal, but are {ratios[f.re.level]} and {r}'
    assert not any(r is None for r in ratios), 'Must specify every integer level in the interval [0, nb_plots)'

    # Compute number of runs
    nruns = np.prod(plot_sizes)

    # Align ratios
    nratios = max([len(r) for r in ratios])
    assert all(len(r) == 1 or len(r) == nratios for r in ratios), 'All ratios must be either a single number or and array of the same size'
    ratios = np.array([
        np.repeat(ratio, nratios) if len(ratio) == 1 else ratio 
        for ratio in ratios
    ]).T

    # Normalize ratios
    ratios = ratios[:, 1:] / ratios[:, 0:1]

    # Extract parameter arrays
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    effect_levels = np.array([f.re.level for f in factors])
    coords = List([f.coords_ for f in factors])

    # Encode the coordinates
    colstart = np.concatenate((
        [0], 
        np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))
    ))

    # Alphas and thetas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]
    thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    thetas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/thetas[1:])))

    # Compute cs
    cs = _compute_cs(plot_sizes, ratios, thetas)

    # Compute Zs
    Zs = obs_var_Zs(plot_sizes)

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
        
        # Don't encode the design
        # prior = encode_design(prior, effect_types, coords=coords)

        # Compute old plot sizes
        nb_old_plots = max(p.level for p in old_plots) + 1
        old_plot_sizes = np.ones(nb_old_plots, dtype=np.int_) * -1
        for p in old_plots:
            if old_plot_sizes[p.level] == -1:
                old_plot_sizes[p.level] = p.size
            else:
                assert plot_sizes[p.level] == p.size, f'Prior plot sizes at the same prior plot level must be equal, but are {plot_sizes[p.level]} and {p.size}'

        # Assert the prior
        assert np.prod(old_plot_sizes) == len(prior), f'Prior plot sizes are misspecified, prior has {len(prior)} runs, but plot sizes require {np.prod(old_plot_sizes)} runs'
        assert nb_old_plots == nb_plots, f'The prior must specify the same number of levels as the factors: prior has {len(old_plot_sizes)} levels, but new design requires {len(plot_sizes)} levels'

        # Validate prior
        assert not np.any(fn.constraintso(prior)), 'Prior contains constraint violating runs'
        alphas_old = np.cumprod(old_plot_sizes[::-1])[::-1]
        for i, f in enumerate(factors):
            if f.re.level != 0:
                p = prior[:, i].reshape(alphas_old[f.re.level], -1)
                assert np.all(np.all(p == np.expand_dims(p[:, 0], 1), axis=1)), f'Prior is not a split-plot design for factor {f.name}'

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
        grps = List([np.concatenate(
            (grps[i].astype(np.int_), lgrps[effect_levels[i]]), 
            dtype=np.int_
        ) for i in range(len(effect_levels))])
    
    # Create the parameters
    params = Parameters(
        fn, factors, nruns, effect_types, effect_levels, grps, ratios, 
        coords, prior, colstart, Zs, Vinv, plot_sizes, cs, alphas, thetas, thetas_inv,
        use_formulas
    )
    
    return params

def create_splitk_plot_design(params, n_tries=10, max_it=10000, validate=False):
    """
    Creates an optimal split^k-plot design using the parameters.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
        The simulation parameters.
    n_tries : int
        The number of random start repetitions. Must be larger than zero.
    max_it : int
        The maximum number of iterations per random initialization for the
        coordinate-exchange algorithm. Prevents infinite loop scenario.
    validate : bool
        Whether to validate each state.

    Returns
    -------
    Y : pd.DataFrame
        A pandas dataframe with the best found design. The
        design is decoded and denormalized.
    best_state : :py:class:`State <pyoptex.doe.fixed_structure.splitk_plot.utils.State>`
        The state corresponding to the returned design. 
        Contains the encoded design, model matrix, metric, etc.
    """
    assert n_tries > 0, 'Must specify at least one random initialization (n_tries > 0)'
    assert max_it > 0, 'Must specify at least one iteration of the coordinate-exchange per random initialization'

    # Pre initialize metric
    params.fn.metric.preinit(params)

    # Main loop
    best_metric = -np.inf
    best_state = None
    for _ in tqdm(range(n_tries)):

        # Optimize the design
        Y, state = optimize(params, max_it, validate=validate)

        # Store the results
        if state.metric > best_metric:
            best_metric = state.metric
            best_state = State(np.copy(state.Y), np.copy(state.X), state.metric)

    # Decode the design
    Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=[str(f.name) for f in params.factors])
    for f in params.factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])

    return Y, best_state

import numpy as np
import pandas as pd
import numba
from numba.typed import List
import numba
import warnings

from .simulation import simulate
from .init import init_feasible
from .sample import sample_random
from .temperature import LinearTemperature
from .accept import exponential_accept_rel
from .restart import RestartEveryNFailed
from .insert import insert_optimal
from .remove import remove_optimal_onebyone
from .utils import Parameters, FunctionSet, Factor
from ..utils.design import encode_design, decode_design, create_default_coords
from ..constraints import no_constraints

def default_fn(
    nsims, cost, metric, Y2X,
    init=init_feasible, sample=sample_random, temperature=None,
    accept=exponential_accept_rel, restart=None, insert=insert_optimal,
    remove=remove_optimal_onebyone, constraints=no_constraints
    ):
    """
    Create a functionset with the default operators as used in the paper. Each
    operator can be manually overriden by providing the parameter.
    This is a convenience function to avoid boilerplate code.

    For an idea on each operators interface (which inputs and outputs) see
    the examples in the code.

    Parameters
    ----------
    nsims : int
        The number of simulations for the algorithm
    cost : func
        The cost function
    metric : obj
        The metric object
    init : func
        The initialization function, :py:func:`init_feasible` by default.
    sample : func
        The sampling function, :py:func:`sample_random` by default.
    temperature : obj
        The temperature object, :py:class:`LinearTemperature` by default.
    accept : func
        The acceptance function, :py:func:`exponential_accept_rel` by default.
    restart : obj
        The restart object, :py:class:`RestartEveryNFailed` by default.
    insert : func
        The insertion function, :py:func:`insert_optimal` by default.
    remove : func
        The removal function, :py:func:`remove_optimal_onebyone` by default.
    constraints : func
        The constraints function, :py:func:`no_constraints` by default.

    Returns
    -------
    fn : :py:class:`FunctionSet`
        The function set.
    """
    # Set default objects
    if temperature is None:
        temperature = LinearTemperature(T0=1, nsims=nsims)
    if restart is None:
        restart = RestartEveryNFailed(nsims / 100)

    # Return the function set
    return FunctionSet(Y2X, init, sample, cost, metric, temperature, accept, restart, insert, remove, constraints.encode())

def create_parameters(factors, fn, prior=None, use_formulas=True):
    """
    Creates the parameters object by preprocessing some elements. This is a simple utility function
    to transform each variable to its correct representation.

    Y2X can allow the user to provide a custom design to model matrix function. This can be useful to incorporate
    non-polynomial columns.

    Parameters
    ----------
    factors : list(Factor)
        The list of factors.
    fn : :py:class:`FunctionSet`
        A set of operators for the algorithm. Must be specified up front.
    model : pd.DataFrame or np.array(2d)
        If pandas dataframe, extracts the columns names. If on top effect_types is a dictionary,
        it makes sure the columns are correctly ordered. A model is defined by the default regression
        notation (e.g. [0, ..., 0] for the intercept, [1, 0, ..., 0] for the first main effect, etc.).
        The parameter is ignored if a Y2X function is provided.
    prior : None or np.array(2d)
        A possible prior design to use for augmentation.
    Y2X : func(Y)
        Converts a design matrix to a model matrix. Defaults to :py:func:`x2fx <cost_optimal_designs.utils.x2fx>` with
        the provided polynomial model. This parameter can be used to create non-polynomial models. 

    Returns
    -------
    params : :py:func:`Parameters`
        The parameters object required for :py:func:`simulate`
    """
    # Initial input validation
    assert len(factors) > 0, 'At least one factor must be provided'
    for i, f in enumerate(factors):
        assert isinstance(f, Factor), f'Factor {i} is not of type Factor'
    if prior is not None:
        assert isinstance(prior, pd.DataFrame), f'The prior must be specified as a dataframe but is a {type(prior)}'

    # Extract the factor parameters
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    grouped_cols = np.array([bool(f.grouped) for f in factors])
    ratios = [f.ratio if isinstance(f.ratio, tuple) or isinstance(f.ratio, list) or isinstance(f.ratio, np.ndarray) else [f.ratio] for f in factors]
    coords = List([f.coords_ for f in factors])

    # Align ratios
    nratios = max([len(r) for r in ratios])
    assert all(len(r) == 1 or len(r) == nratios for r in ratios), 'All ratios must be either a single number or and array of the same size'
    ratios = np.array([np.repeat(ratio, nratios) if len(ratio) == 1 else ratio for ratio in ratios]).T

    # Define the starting columns
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))
        
    # Create the prior
    if prior is not None:
        # Normalize factors
        for f in factors:
            prior[str(f.name)] = f.normalize(prior[str(f.name)])

        # Convert from pandas to numpy
        prior = prior[col_names].to_numpy()
        
        # Encode the design
        prior = encode_design(prior, effect_types, coords=coords)

        # Validate prior
        assert not np.any(fn.constraints(prior)), 'Prior contains constraint violating runs'

    else:
        prior = np.empty((0, colstart[-1]))
    
    # Create the parameters
    params = Parameters(fn, colstart, coords, ratios, effect_types, grouped_cols, prior, {}, use_formulas)

    # Validate the cost of the prior
    if params.prior is not None:
        costs = params.fn.cost(params.prior, params)
        cost_Y = np.array([np.sum(c) for c, _, _ in costs])
        max_cost = np.array([m for _, m, _ in costs])
        assert np.all(cost_Y <= max_cost), 'Prior exceeds maximum cost'

    return params

def create_cost_optimal_design(factors, fn, prior=None, nreps=1, use_formulas=True, **kwargs):
    """
    Simulation wrapper dealing with some preprocessing for the algorithm. It creates the parameters and
    permits the ability to provided `nreps` random starts for the algorithm. Kwargs can contain any of
    the parameters specified in :py:func:`simulate` (apart from the parameters).
    The best design of all nreps is selected.

    Y2X can allow the user to provide a custom design to model matrix function. This can be useful to incorporate
    non-polynomial columns.
    
    .. warning::
        Make sure the order of the columns is as indicated in effect_types 
        (and it accounts for preceding categorical variables)!

    Parameters
    ----------
    factors : list(Factor)
        The factors for the design.
    fn : :py:class:`FunctionSet`
        A set of operators for the algorithm. Must be specified up front.
    model : pd.DataFrame or np.array(2d)
        If pandas dataframe, extracts the columns names. If on top effect_types is a dictionary,
        it makes sure the columns are correctly ordered. A model is defined by the default regression
        notation (e.g. [0, ..., 0] for the intercept, [1, 0, ..., 0] for the first main effect, etc.).
        The parameter is ignored if a Y2X function is provided.
    prior : None or np.array(2d)
        A possible prior design to use for augmentation.
    Y2X : func(Y)
        Converts a design matrix to a model matrix. Defaults to :py:func:`x2fx <cost_optimal_designs.utils.x2fx>` with
        the provided polynomial model. This parameter can be used to create non-polynomial models.
    nreps : int
        The number of random start repetitions. Must be larger than zero.
    kwargs : 
        Any other named parameters directly passed to simulate.

    Returns
    -------
    Y : pd.DataFrame
        A pandas dataframe with the best found design. It is decoded and contains the column names
        if found.
    best_state : :py:class:`State`
        The state corresponding to the returned design. Contains the encoded design, model matrix, 
        costs, metric, etc.
    """
    assert nreps > 0, 'Must specify at least one repetition for the algorithm'

    # Extract the parameters
    params = create_parameters(factors, fn, prior, use_formulas)

    # Simulation
    best_state = simulate(params, **kwargs)
    try:
        for i in range(nreps-1):
            try:
                state = simulate(params, **kwargs)
                if state.metric > best_state.metric:
                    best_state = state
            except ValueError as e:
                print(e)
    except KeyboardInterrupt:
        print('Interrupted: returning current results')

    # Decode the design
    Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=[str(f.name) for f in factors])
    for f in factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])
    return Y, best_state


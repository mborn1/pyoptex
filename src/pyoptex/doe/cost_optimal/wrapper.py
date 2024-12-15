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
from ..utils.model import encode_model
from ..utils.design import x2fx, encode_design, decode_design, create_default_coords
from ..constraints import no_constraints

def default_fn(
    nsims, cost, metric, 
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
    return FunctionSet(init, sample, cost, metric, temperature, accept, restart, insert, remove, constraints.encode())

def create_parameters(factors, fn, model=None, prior=None, Y2X=None, use_formulas=True):
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
    assert model is not None or Y2X is not None, 'Either a polynomial model or Y2X function must be provided'
    assert len(factors) > 0, 'At least one factor must be provided'
    for i, f in enumerate(factors):
        assert isinstance(f, Factor), f'Factor {i} is not of type Factor'
        assert f.type.lower() in ['cont', 'continuous', 'cat', 'categorical'], f'Factor {i} with name {f.name} has an unknown type {f.type}, must be "continuous" or "categorical"'
        if f.type.lower() in ['cont', 'continuous']:
            assert isinstance(f.min, float) or isinstance(f.min, int), f'Continuous factor {i} with name {f.name} requires an integer or a float as minimum, but received {f.min} with type {type(f.min)}'
            assert isinstance(f.max, float) or isinstance(f.max, int), f'Continuous factor {i} with name {f.name} requires an integer or a float as maximum, but received {f.max} with type {type(f.max)}'
            assert f.min < f.max, f'Continuous factor {i} with name {f.name} requires a strictly lower minimum than maximum, but has a minimum of {f.min} and a maximum of {f.max}'
            assert f.coords is None, f'Cannot specify coordinates for continuous factors, please specify the levels'
        else:
            assert len(f.levels) >= 2, f'Categorical factor {i} with name {f.name} has {len(f.levels)} levels, at least two required. Have you specified the "levels" parameters?'
            if f.coords is not None:
                coords = np.array(f.coords)
                assert len(coords.shape) == 2, f'Categorical factor {i} with name {f.name} requires a 2d array as coordinates, but has {len(coords.shape)} dimensions'
                assert coords.shape[0] == len(f.levels), f'Categorical factor {i} with name {f.name} requires one encoding for every level, but has {len(f.levels)} levels and {coords.shape[0]} encodings'
                assert coords.shape[1] == len(f.levels) - 1, f'Categorical factor {i} with name {f.name} and N levels requires N-1 dummy columns, but has {len(f.levels)} levels and {coords.shape[1]} dummy columns'
                assert np.linalg.matrix_rank(coords) == coords.shape[1], f'Categorical factor {i} with name {f.name} does not have a valid (full rank) encoding'
    if model is not None:
        assert isinstance(model, pd.DataFrame), f'The model must specified as a dataframe but is a {type(model)}'
    if prior is not None:
        assert isinstance(prior, pd.DataFrame), f'The prior must be specified as a dataframe but is a {type(prior)}'

    # Provide warnings
    if model is not None and Y2X is not None:
        warnings.warn('Both a model and Y2X function are specified, using Y2X')

    # Extract the factor parameters
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.type.lower() in ['cont', 'continuous'] else len(f.levels) for f in factors])
    grouped_cols = np.array([bool(f.grouped) for f in factors])
    ratios = [f.ratio if isinstance(f.ratio, tuple) or isinstance(f.ratio, list) or isinstance(f.ratio, np.ndarray)
                        else [f.ratio] for f in factors]
    
    # Extract coordinates
    def extract_coord(factor):
        if factor.coords is None:
            # Define the coordinates
            coord = create_default_coords(1)

            # Encode the coordinates for categorical factors
            if factor.type.lower() not in ['cont', 'continuous']:
                coord = encode_design(coord, np.array([len(factor.levels)]))

        else:
            # Extract the coordinates
            coord = np.array(factor.coords).astype(np.float64)

            # Normalize the continuous coordinates
            if factor.type.lower() in ['cont', 'continuous']:
                coord = (coord.reshape(-1, 1) - factor.min) / (factor.max - factor.min) * 2 - 1

        return coord
    coords = List([extract_coord(f) for f in factors])

    # Align ratios
    nratios = max([len(r) for r in ratios])
    assert all(len(r) == 1 or len(r) == nratios for r in ratios), 'All ratios must be either a single number or and array of the same size'
    ratios = np.array([np.repeat(ratio, nratios) if len(ratio) == 1 else ratio for ratio in ratios])

    # Define the starting columns
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))
    
    # Set the Y2X function
    if Y2X is None:
        # Detect model in correct order
        model = model[col_names].to_numpy()

        # Encode model
        modelenc = encode_model(model, effect_types)

        # Create transformation function for polynomial models
        Y2X = lambda Y: x2fx(Y, modelenc)
        
    # Create the prior
    if prior is not None:
        # Normalize factors
        for f in factors:
            if f.type.lower() in ['cont', 'continuous']:
                prior[str(f.name)] = ((prior[str(f.name)] - f.min) / (f.max - f.min)) * 2 - 1
            else:
                prior[str(f.name)] = prior[str(f.name)].map({lname: i for i, lname in enumerate(f.levels)})

        # Convert from pandas to numpy
        prior = prior[col_names].to_numpy()
        
        # Encode the design
        prior = encode_design(prior, effect_types)
    else:
        prior = np.empty((0, colstart[-1]))

    # Compile constraints
    fn = fn._replace(constraints=numba.njit(fn.constraints))
    
    # Create the parameters
    params = Parameters(fn, colstart, coords, ratios, effect_types, grouped_cols, prior, Y2X, {}, use_formulas)

    return params

def create_cost_optimal_design(factors, fn, model=None, prior=None, Y2X=None, nreps=1, use_formulas=True, **kwargs):
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
    params = create_parameters(factors, fn, model, prior, Y2X, use_formulas)

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
        if f.type.lower() in ['cont', 'continuous']:
            Y[str(f.name)] = (Y[str(f.name)] + 1) / 2 * (f.max - f.min) + f.min
        else:
            Y[str(f.name)] = Y[str(f.name)].astype(int).map({i: lname for i, lname in enumerate(f.levels)})
    return Y, best_state


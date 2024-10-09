import numpy as np
import pandas as pd
import numba
from numba.typed import List
import numba

from .simulation import simulate
from .init import init_feasible
from .sample import sample_random
from .temperature import LinearTemperature
from .accept import exponential_accept_rel
from .restart import RestartEveryNFailed
from .insert import insert_optimal
from .remove import remove_optimal_onebyone
from .utils import Parameters, FunctionSet
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

def create_parameters(effect_types, fn, model=None, coords=None, 
                        ratios=None, grouped_cols=None, prior=None, Y2X=None):
    """
    Creates the parameters object by preprocessing some elements. This is a simple utility function
    to transform each variable to its correct representation.

    Y2X can allow the user to provide a custom design to model matrix function. This can be useful to incorporate
    non-polynomial columns.

    .. warning::
        Make sure the order of the columns is as indicated in effect_types 
        (and it accounts for preceding categorical variables)!

    Parameters
    ----------
    effect_types : dict or np.array(1d)
        If dictionary, maps each column name to its type. Also extracts the column names. A 1 indicates
        a continuous factor, anything higher is a categorical factor with that many levels.
    fn : :py:class:`FunctionSet`
        A set of operators for the algorithm. Must be specified up front.
    model : pd.DataFrame or np.array(2d)
        If pandas dataframe, extracts the columns names. If on top effect_types is a dictionary,
        it makes sure the columns are correctly ordered. A model is defined by the default regression
        notation (e.g. [0, ..., 0] for the intercept, [1, 0, ..., 0] for the first main effect, etc.).
        The parameter is ignored if a Y2X function is provided.
    coords : None or list(np.array(2d) or None)
        If None or one element is None, the :py:func:`create_default_coords` are used.
    ratios : None or np.array(1d) or np.array(2d)
        The array of ratios vs. epsilon for each factors random effect variance. Set to all ones by default.
        Multi-dimensional ratios permit the computation of multiple Vinv simultaneously.
    grouped_cols : None or np.array(1d)
        A boolean array indicating which columns must receive a corresponding random effect. By default,
        all variables do.
    prior : None or np.array(2d)
        A possible prior design to use for augmentation.
    Y2X : func(Y)
        Converts a design matrix to a model matrix. Defaults to :py:func:`x2fx <cost_optimal_designs.utils.x2fx>` with
        the provided polynomial model. This parameter can be used to create non-polynomial models. 

    Returns
    -------
    params : :py:func:`Parameters`
        The parameters object required for :py:func:`simulate`
    col_names : list(str) or None
        A list of column names initially presented. None if no information was found.
    """
    assert model is not None or Y2X is not None, 'Either a polynomial model or Y2X function must be provided'

    if isinstance(effect_types, dict):
        # Detect effect types
        col_names = list(effect_types.keys())
        effect_types = np.array(list(effect_types.values()))
    else:
        # No column names known
        col_names = None

    # Set default grouped columns and ratios (force to 2d)
    grouped_cols = grouped_cols if grouped_cols is not None else np.ones(effect_types.size, dtype=np.bool_)
    ratios = ratios if ratios is not None else np.ones((1, effect_types.size))
    if len(ratios.shape) == 1:
        ratios = ratios.reshape(1, -1)

    # Set default coords
    coords = coords if coords is not None else [None]*effect_types.size
    coords = [create_default_coords(et) if coord is None else coord for coord, et in zip(coords, effect_types)]

    # Map ratios to grouped columns if necessary
    if ratios.shape[1] != grouped_cols.size:
        assert ratios.shape[1] == np.sum(grouped_cols), 'Must specify a ratio for each grouped column'
        r = np.ones((ratios.shape[0], effect_types.size))
        r[:, grouped_cols] = ratios
        ratios = r

    # Encode the coordinates
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))
    coords_enc = List([
        encode_design(coord, np.array([et]))
            if et > 1 and coord.shape[1] == 1 and np.all(np.sort(coord) == create_default_coords(et))
            else coord.astype(np.float64)
        for coord, et in zip(coords, effect_types)
    ])

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
        
    # Create the prior
    if prior is not None:
        # Convert from pandas to numpy
        if isinstance(prior, pd.DataFrame):
            if col_names is not None:
                prior = prior[col_names]
            prior = prior.to_numpy()
        
        # Possibly encode the design
        if prior.shape[1] == effect_types.size:
            prior = encode_design(prior, effect_types)
    else:
        prior = np.empty((0, colstart[-1]))

    # Compile constraints
    fn = fn._replace(constraints=numba.njit(fn.constraints))
    
    # Create the parameters
    params = Parameters(fn, colstart, coords_enc, ratios, effect_types, grouped_cols, prior, Y2X, {})

    return params, col_names

def create_cost_optimal_design(effect_types, fn, model=None, coords=None, ratios=None, grouped_cols=None, prior=None, 
                     Y2X=None, nreps=1, **kwargs):
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
    effect_types : dict or np.array(1d)
        If dictionary, maps each column name to its type. Also extracts the column names. A 1 indicates
        a continuous factor, anything higher is a categorical factor with that many levels.
    fn : :py:class:`FunctionSet`
        A set of operators for the algorithm. Must be specified up front.
    model : pd.DataFrame or np.array(2d)
        If pandas dataframe, extracts the columns names. If on top effect_types is a dictionary,
        it makes sure the columns are correctly ordered. A model is defined by the default regression
        notation (e.g. [0, ..., 0] for the intercept, [1, 0, ..., 0] for the first main effect, etc.).
        The parameter is ignored if a Y2X function is provided.
    coords : None or list(np.array(2d) or None)
        If None or one element is None, the :py:func:`_default_coords` are used.
    ratios : None or np.array(1d)
        The array of ratios vs. epsilon for each factors random effect variance. Set to all ones by default.
    grouped_cols : None or np.array(1d)
        A boolean array indicating which columns must receive a corresponding random effect. By default,
        all variables do.
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
    assert nreps > 0

    # Extract the parameters
    params, col_names = create_parameters(
        effect_types, fn, model, coords, ratios, grouped_cols, prior, Y2X
    )

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
        pass

    # Decode the design
    Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=col_names)
    return Y, best_state



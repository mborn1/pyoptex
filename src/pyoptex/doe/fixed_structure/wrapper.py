"""
Module for the interface to run the generic coordinate-exchange algorithm
"""

import numpy as np
import pandas as pd
from numba.typed import List
from tqdm import tqdm

from ..constraints import no_constraints
from ..utils.design import decode_design, obs_var_from_Zs
from .utils import (Factor, RandomEffect, FunctionSet, State, Parameters)
from .init import initialize_feasible
from .optimize import optimize


def default_fn(metric, Y2X, constraints=no_constraints, init=initialize_feasible):
    """
    Create a functionset with the default operators. Each
    operator can be manually overriden by providing the parameter.

    Parameters
    ----------
    metric : :py:class:`Metric <pyoptex.doe.fixed_structure.metric.Metric>`
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
        :py:func:`initialize_feasible <pyoptex.doe.fixed_structure.init.initialize_feasible>`
        by default.

    Returns
    -------
    fn : :py:class:`FunctionSet <pyoptex.doe.fixed_structure.utils.FunctionSet>`
        The function set.
    """
    return FunctionSet(metric, Y2X, constraints.encode(), constraints.func(), init)

def create_parameters(factors, fn, nruns, prior=None, grps=None):
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
    prior : None
        Not implemented yet.
    grps : None
        Not implemented yet.

    Returns
    -------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The simulation parameters.
    """
    # Assertions
    assert len(factors) > 0, 'At least one factor must be provided'
    for i, f in enumerate(factors):
        assert isinstance(f, Factor), f'Factor {i} is not of type Factor'
        assert f.re is None or isinstance(f.re, RandomEffect), f'Factor {i} with name {f.name} does not have a RandomEffect as random effect'    
        if f.re is not None:
            assert len(f.re.Z) == nruns, f'Factor {i} with name {f.name} does not have enough runs as random effect'
    assert prior is None, f'Priors have not yet been implemented'
    assert grps is None, f'Grouped optimization has not yet been implemented'

    # Extract the random effects
    re = []
    for f in factors:
        if f.re is not None and f.re not in re:
            re.append(f.re)

    # Extract the plot sizes
    ratios = []
    for r in re:
        # Extract ratios
        r = np.sort(r.ratio) \
                if isinstance(r.ratio, (tuple, list, np.ndarray))\
                else [r.ratio]
        
        # Append the ratios
        ratios.append(r)

    # Align ratios
    if len(ratios) > 0:
        nratios = max([len(r) for r in ratios])
        assert all(len(r) == 1 or len(r) == nratios for r in ratios), 'All ratios must be either a single number or and array of the same size'
        ratios = np.array([
            np.repeat(ratio, nratios) if len(ratio) == 1 else ratio 
            for ratio in ratios
        ]).T

    # Extract parameter arrays
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    effect_levels = np.array([re.index(f.re) + 1 if f.re is not None else 0 for f in factors])
    coords = List([f.coords_ for f in factors])

    # Encode the coordinates
    colstart = np.concatenate((
        [0], 
        np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))
    ))

    # Compute Zs and Vinv
    if len(re) > 0:
        Zs = np.array([np.array(r.Z) for r in re], dtype=np.int64)
        Vinv = np.linalg.inv(np.array([obs_var_from_Zs(Zs, N=nruns, ratios=r) for r in ratios]))
    else:
        Zs = np.empty((0, 0), dtype=np.int64)
        Vinv = np.expand_dims(np.eye(nruns), 0)
        
    # Define which groups to optimize
    lgrps = [np.arange(nruns)] + [np.arange(np.max(Z)+1) for Z in Zs]
    grps = List([lgrps[lvl] for lvl in effect_levels])

    # Create the parameters
    params = Parameters(
        fn, factors, nruns, effect_types, effect_levels, grps, ratios, 
        coords, prior, colstart, Zs, Vinv
    )
    
    return params

def create_fixed_structure_design(params, n_tries=10, max_it=10000, validate=False):
    """
    Creates an optimal design for the specified factors, using the parameters.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`)
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
    best_state : :py:class:`State <pyoptex.doe.fixed_structure.utils.State>`
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
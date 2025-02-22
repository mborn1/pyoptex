"""
Module for the interface to run the CODEX algorithm
"""

import numpy as np
import pandas as pd
from numba.typed import List

from ...constraints import no_constraints, mixture_constraints
from ....utils.design import decode_design, encode_design
from ..init import init_feasible
from ..utils import Factor, Parameters
from .utils import FunctionSet
from .accept import exponential_accept_rel
from .insert import insert_optimal
from .remove import remove_optimal_onebyone
from .restart import RestartEveryNFailed
from .sample import sample_random
from .simulation import simulate
from .temperature import LinearTemperature
from .optimization import CEOptimizer, CEStructOptimizer


def default_fn(
    nsims, factors, cost, metric, Y2X,
    init=init_feasible, sample=sample_random, temperature=None,
    accept=exponential_accept_rel, restart=None, insert=insert_optimal,
    remove=remove_optimal_onebyone, constraints=None,
    optimizers=[CEOptimizer(1), CEStructOptimizer(1)],
    final_optimizers=[CEOptimizer(1), CEStructOptimizer(1)]
    ):
    """
    Create a functionset with the default operators. Each
    operator can be manually overriden by providing the parameter.

    If any mixture components are present, the mixture constraint
    is automatically added.

    Parameters
    ----------
    nsims : int
        The number of simulations for the algorithm.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors of the experiment.
    cost : func(Y, params)
        The cost function.
    metric : :py:class:`Metric <pyoptex.doe.cost_optimal.metric.Metric>`
        The metric object.
    Y2X : func
        The function converting from the design matrix to the
        model matrix.
    init : func
        The initialization function, 
        :py:func:`init_feasible <pyoptex.doe.cost_optimal.init.init_feasible>` 
        by default.
    sample : func
        The sampling function, 
        :py:func:`sample_random <pyoptex.doe.cost_optimal.sample.sample_random>` 
        by default.
    temperature : obj
        The temperature object, 
        :py:class:`LinearTemperature <pyoptex.doe.cost_optimal.temperature.LinearTemperature>` 
        by default.
    accept : func
        The acceptance function, 
        :py:func:`exponential_accept_rel <pyoptex.doe.cost_optimal.accept.exponential_accept_rel>` 
        by default.
    restart : obj
        The restart object, 
        :py:class:`RestartEveryNFailed <pyoptex.doe.cost_optimal.restart.RestartEveryNFailed>` 
        by default.
    insert : func
        The insertion function, 
        :py:func:`insert_optimal <pyoptex.doe.cost_optimal.insert.insert_optimal>` 
        by default.
    remove : func
        The removal function, 
        :py:func:`remove_optimal_onebyone <pyoptex.doe.cost_optimal.remove.remove_optimal_onebyone>` 
        by default.
    constraints : func
        The constraints function, 
        :py:func:`no_constraints <pyoptex.doe.constraints.no_constraints>` 
        by default.
    optimizers : list(:py:class:`Optimizer <pyoptex.doe.cost_optimal.optimization.Optimizer>`)
        A list of optimizers. If None, it defaults to 
        :py:class:`CEOptimizer <pyoptex.doe.cost_optimal.optimization.CEOptimizer>` 
        and :py:class:`CEStructOptimizer <pyoptex.doe.cost_optimal.optimization.CEStructOptimizer>`.
        To provide no optimizers, pass an empty list. 
    final_optimizers : list(:py:class:`Optimizer <pyoptex.doe.cost_optimal.optimization.Optimizer>`)
        Similar to optimizers, but run at the very end of the algorithm to perform the
        final optimizations. These optimizers are run until no improvements are found.

    Returns
    -------
    fn : :py:class:`FunctionSet <pyoptex.doe.cost_optimal.codex.utils.FunctionSet>`
        The function set.
    """
    # Set default objects
    if temperature is None:
        temperature = LinearTemperature(T0=1, nsims=nsims)
    if restart is None:
        restart = RestartEveryNFailed(nsims / 100)

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
        
    # Return the function set
    return FunctionSet(
        Y2X, init, cost, metric, constraints.encode(), 
        sample, temperature,
        accept, restart, insert, remove, 
        optimizers, final_optimizers
    )

def create_parameters(factors, fn, prior=None, use_formulas=True):
    """
    Creates the parameters object by preprocessing the inputs. 
    This is a utility function to transform each variable 
    to its correct representation.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The list of factors.
    fn : :py:class:`FunctionSet <pyoptex.doe.cost_optimal.codex.utils.FunctionSet>`
        A set of operators for the algorithm.
    prior : None or pd.DataFrame
        A possible prior design to use for augmentation. Must be 
        denormalized and decoded.
    use_formulas : bool
        Whether to use the internal update formulas or not.

    Returns
    -------
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
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
    ratios = [f.ratio if isinstance(f.ratio, tuple) or isinstance(f.ratio, list)
                             or isinstance(f.ratio, np.ndarray) else [f.ratio] 
              for f in factors]
    coords = List([f.coords_ for f in factors])

    # Align ratios
    nratios = max([len(r) for r in ratios])
    assert all(len(r) == 1 or len(r) == nratios for r in ratios), 'All ratios must be either a single number or and array of the same size'
    ratios = np.array([
        np.repeat(ratio, nratios) if len(ratio) == 1 else ratio 
        for ratio in ratios
    ]).T

    # Define the starting columns
    colstart = np.concatenate((
        [0], 
        np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))
    ))
        
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
    params = Parameters(
        fn, factors, colstart, coords, ratios, effect_types, 
        grouped_cols, prior, {}, use_formulas
    )

    # Validate the cost of the prior
    if len(params.prior) > 0:
        costs = params.fn.cost(params.prior, params)
        cost_Y = np.array([np.sum(c) for c, _, _ in costs])
        max_cost = np.array([m for _, m, _ in costs])
        assert np.all(cost_Y <= max_cost), 'Prior exceeds maximum cost'

    return params

def create_cost_optimal_codex_design(params, nreps=10, nsims=7500, validate=True):
    """
    Creates an optimal design for the specified factors, using the CODEX algorithm.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    nreps : int
        The number of random start repetitions. Must be larger than zero.
    nsims : int
        The number of simulations (annealing steps) to run the algorithm for.
    validate : bool
        Whether to validate each state.

    Returns
    -------
    Y : pd.DataFrame
        A pandas dataframe with the best found design. The
        design is decoded and denormalized.
    best_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state corresponding to the returned design. 
        Contains the encoded design, model matrix, 
        costs, metric, etc.
    """
    assert nreps > 0, 'Must specify at least one repetition for the algorithm'

    # Simulation
    best_state = simulate(params, nsims=nsims, validate=validate)
    try:
        for i in range(nreps-1):
            try:
                state = simulate(params, nsims=nsims, validate=validate)
                if state.metric > best_state.metric:
                    best_state = state
            except ValueError as e:
                print(e)
    except KeyboardInterrupt:
        print('Interrupted: returning current results')

    # Decode the design
    Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=[str(f.name) for f in params.factors])
    for f in params.factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])
    return Y, best_state


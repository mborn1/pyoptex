import numpy as np
from tqdm import tqdm
from numba.typed import List

from .utils import obs_var_Zs, State
from .validation import validate_state
from .optimization import CEMetric, CEStructMetric
from ..utils.design import obs_var_from_Zs
from ..._profile import profile

def _validate_prior(params):
    """
    Validates the prior design passed to the algorithm. It requires the correct shape,
    that all runs are within the constraints and that the cost does not exceed the maximum cost.

    Parameters
    ----------
    params : :py:class:`Parameters`
        The parameters provided to the simulate function
    """
    assert params.prior.shape[1] == params.colstart[-1], 'Prior does not have the correct shape'
    assert not np.any(params.fn.constraints(params.prior)), 'Prior contains constraint violating runs'
    assert np.all(np.sum(params.fn.cost(params.prior), axis=-1) < params.max_cost), 'Prior exceeds maximum cost'

@profile
def simulate(params, optimizers=None, final=None, nsims=100, validate=False):
    """
    Performs the simulated annealing algorithm (SA). This is the main loop calling all of the
    operators.

    Parameters
    ----------
    params : :py:class:`Parameters`
        The constant parameters defining the optimization problem.
    optimizers : list(obj)
        A list of optimization procedures. Every object should have a .call()
        function optimizing the design, and an .init() function to initialize the
        optimizer. If None, it defaults to :py:class:`CEMetric` and :py:class:`CEStructMetric`.
        To provide no optimizers, pass an empty list. 
    final : list(obj)
        Similar to optimizersv, but run at the very end of the algorithm to perform the
        final optimizations.
    nsims : int
        The number of simulation iterations, accepted or rejected.
    validate : bool
        Whether to validate intermediate steps. Mostly used for debugging purposes.

    Returns
    -------
    best_state : :py:class:`State`
        The best found state containing the encoded design matrix, model matrix,
        costs, metric, etC.
    """
    # Validate the prior
    _validate_prior(params)

    # Default optimizers
    if optimizers is None:
        optimizers = [CEMetric(), CEStructMetric()]
    
    # Initialize functions
    params.fn.temp.reset()
    params.fn.restart.reset()
    params.fn.metric.init(params)
    for opt in optimizers:
        opt.reset()

    # Initialize final optimizations
    if final is None:
        final = optimizers
    else:
        for fin in final:
            fin.reset()

    # Initialize the design (state)
    Y = params.fn.init(params)
    X = params.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, params.grouped_cols)
    Vinv = np.array([np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) for ratios in params.ratios])
    costs = params.fn.cost(Y)
    cost_Y = np.sum(costs, axis=1)
    metric = params.fn.metric.call(Y, X, Zs, Vinv, costs)
    state = State(Y, X, Zs, Vinv, metric, cost_Y, costs)
    validate and validate_state(state, params)

    # Initialize best solution
    best_state = State(np.copy(Y), np.copy(X), tuple(np.copy(Zi) if Zi is not None else None for Zi in Zs), 
                       np.copy(Vinv), metric, np.copy(cost_Y), np.copy(costs))
    validate and validate_state(best_state, params)

    #######################################################################

    for _ in tqdm(range(nsims)):
        # Create a new random run
        accepted_sample = False
        while not accepted_sample:
            new_run = params.fn.sample(state, params).reshape((1, -1))
            accepted_sample = not params.fn.constraints(new_run)[0]

        # Insert in ideal position
        new_state = params.fn.insert(new_run, state, params)
        validate and validate_state(new_state, params)

        # Remove runs to be within cost constraints
        new_state = params.fn.remove(new_state, params)
        validate and validate_state(new_state, params)

        # Optimization
        for opt in optimizers:
            new_state = opt.call(new_state, params)
            validate and validate_state(new_state, params)

        # Accept
        accept = params.fn.accept(state.metric, new_state.metric, params.fn.temp.T) > np.random.rand()
        if accept:
            # Update the state and temperature
            state = new_state
            params.fn.temp.accepted()
            params.fn.restart.accepted()

            # Fix numerical issues with inverse update formulas
            state = State(
                state.Y, state.X, state.Zs, 
                np.array([np.linalg.inv(obs_var_from_Zs(state.Zs, len(state.Y), ratios)) for ratios in params.ratios]),
                state.metric, state.cost_Y, state.costs
            )

            if state.metric > best_state.metric:
                # Set the best state
                best_state = state
        else:
            params.fn.temp.rejected()
            params.fn.restart.rejected()

        # Restart policy
        state = params.fn.restart.call(state, best_state)
        validate and validate_state(state, params)

    # Final optimization
    optimized = True
    while optimized:
        best_metric = best_state.metric
        for opt in final:
            best_state = opt.call(best_state, params)
            validate and validate_state(best_state, params)
        optimized = best_state.metric > best_metric

    # Final validation
    try:
        validate_state(best_state, params)
    except AssertionError as e:
        print('Possible issue detected in final state')
        print(e)

    # Return the best state
    return best_state
        


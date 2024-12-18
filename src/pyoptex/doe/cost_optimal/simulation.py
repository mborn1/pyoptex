import numpy as np
from tqdm import tqdm
from numba.typed import List

from .utils import obs_var_Zs, State
from .validation import validate_state
from .optimization import CEMetric, CEStructMetric
from ..utils.design import obs_var_from_Zs
from ..._profile import profile

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
    # Initialize stats
    params.stats['it'] = 0
    params.stats['rejections'] = 0
    params.stats['insert_loc'] = -1 * np.ones(nsims, dtype=np.int64)
    params.stats['removed_insert'] = np.zeros(nsims, dtype=np.bool_)
    params.stats['metrics'] = np.zeros(nsims, dtype=np.float64)

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
    costs = params.fn.cost(Y, params)
    cost_Y = np.array([np.sum(c) for c, _, _ in costs])
    max_cost = np.array([m for _, m, _ in costs])
    metric = params.fn.metric.call(Y, X, Zs, Vinv, costs)
    state = State(Y, X, Zs, Vinv, metric, cost_Y, costs, max_cost)
    validate and validate_state(state, params)

    # Initialize best solution
    best_state = State(np.copy(Y), np.copy(X), tuple(np.copy(Zi) if Zi is not None else None for Zi in Zs), 
                       np.copy(Vinv), metric if not np.any(cost_Y > max_cost) else -np.inf,
                       np.copy(cost_Y), [(np.copy(c), m, np.copy(idx)) for c, m, idx in costs], np.copy(max_cost))
    validate and validate_state(best_state, params)

    #######################################################################

    for i in tqdm(range(nsims)):
        # Set iteration
        params.stats['it'] = i

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

        # Collect stats
        params.stats['metrics'][i] = new_state.metric

        # Accept
        cost_transition = (np.all(new_state.cost_Y <= new_state.max_cost) and np.any(state.cost_Y > state.max_cost))
        accept = params.fn.accept(state.metric, new_state.metric, params.fn.temp.T) > np.random.rand() or cost_transition \
                        or np.isinf(state.metric) or np.isnan(state.metric)
        if accept:
            # Update the state and temperature
            state = new_state
            params.fn.temp.accepted()
            params.fn.restart.accepted()

            # Fix numerical issues with inverse update formulas
            Vinv = np.array([np.linalg.inv(obs_var_from_Zs(state.Zs, len(state.Y), ratios)) for ratios in params.ratios])
            metric = params.fn.metric.call(state.Y, state.X, state.Zs, Vinv, state.costs)
            state = State(
                state.Y, state.X, state.Zs, Vinv, metric, state.cost_Y, state.costs, state.max_cost
            )

            # Set the best state
            cost_transition = (np.all(state.cost_Y <= state.max_cost) and np.any(best_state.cost_Y > best_state.max_cost))
            if state.metric > best_state.metric or cost_transition:
                best_state = state
        else:
            params.fn.temp.rejected()
            params.fn.restart.rejected()
            params.stats['rejections'] += 1

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
        


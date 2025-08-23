"""
Module containing all the cost functions of the cost optimal designs
"""

from functools import wraps, partial

import numba
import numpy as np
import pandas as pd

from ...utils.design import decode_design


def _fn_no_params(f, Y, params):
    return f(Y)

def _fn_decoded(f, Y, params):
    # Decode the design to a dataframe
    Y = decode_design(Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=[str(f.name) for f in params.factors])
    return f(Y, params)

def _fn_denormalized(f, Y, params):
    # Decode the design to a dataframe
    Y = decode_design(Y, params.effect_types, coords=params.coords)
    Y = pd.DataFrame(Y, columns=[str(f.name) for f in params.factors])
    for f in params.factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])
    return f(Y, params)

def __cost_fn(f, denormalize=True, decoded=True, contains_params=False):
    """
    Cost function decorator code.

    Parameters
    ----------
    f : func(Y, params) or func(Y)
        The cost function to be wrapped.
    denormalize : bool
        Whether to denormalize (and decode) the data before passing it to `f`.
    decoded : bool
        Whether to only decode, but not denormalize the data before passing it to `f`.
    contains_params : bool
        Whether the cost function requires the CODEX 
        :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`.
        This prevents numba compilation.

    Returns
    -------
    fn : func(Y, params)
        The decorated function
    """

    # Check if parameters are required (prevents direct use of numba.njit compilation)
    if not contains_params:
        f = wraps(f)(partial(_fn_no_params, f))

    # Check for denormalization in the cost function
    if denormalize:
        # Wrap the function
        f = wraps(f)(partial(_fn_denormalized, f))
        NOTE = 'This cost function works on denormalized inputs'

    elif decoded:
        # Wrap the function
        f = wraps(f)(partial(_fn_decoded, f))
        NOTE = 'This cost function works on decoded categorical inputs'

    else:
        NOTE = 'This cost function works on normalized (encoded) inputs'

    # Extend the documentation with a note on normalization
    if f.__doc__ is not None:
        params_pos = f.__doc__.find('    Parameters\n    ---------')
        f.__doc__ = f.__doc__[:params_pos] + f'\n    .. note::\n        {NOTE}\n\n' + f.__doc__[params_pos:]

    return f

def cost_fn(*args, **kwargs):
    """
    Parameters
    ----------
    f : func(Y, params) or func(Y)
        The cost function to be wrapped.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        A list of factors for the design.
    denormalize : bool
        Whether to denormalize (and decode) the data before passing it to `f`.
    decoded : bool
        Whether to only decode, but not denormalize the data before passing it to `f`.
    contains_params : bool
        Whether the cost function requires the
        :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`.
        This prevents numba compilation.

    Returns
    -------
    fn : func(Y, params)
        The decorated function
    """
    if len(args) > 0 and callable(args[0]):
        return __cost_fn(args[0], *args[1:], **kwargs)
    else:
        def wrapper(f):
            return __cost_fn(f, *args, **kwargs)
        return wrapper

############################################################

def combine_costs(costs):
    """
    Combine multiple cost functions together.

    Parameters
    ----------
    costs : iterable(func)
        An iterable of cost functions to concatenate
    
    Returns
    -------
    cost_fn : func(Y, params)
        The combined cost function for the simulation algorithm.
    """
    def _cost(Y, params):
        return [c for cf in costs for c in cf(Y, params)]

    # pylint: disable=line-too-long
    _cost.__doc__ = 'This is a combined cost function of:\n* ' + '\n* '.join(cf.__name__ for cf in costs)

    return _cost

def discount_cost(costs, factors, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = max(c1, c2, ..., base). 
    This means the total transition cost is determined by the most-hard-to-change factor.
    
    Parameters
    ----------
    costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    base_cost : float
        The base cost when no factors are changed, i.e., when a run
        is repeated.
    
    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Expand the costs to categorically encoded
    costs = np.array([
        c for f in factors 
            for c in ([costs[str(f.name)]] 
            if f.is_continuous 
            else [costs[str(f.name)]]*(len(f.levels)-1))
    ])

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        """Internal cost function according to 
        :py:function:`discount_cost <pyoptex.doe.cost_optimal.cost.discount_cost>`"""
        # Initialize costs
        cc = np.zeros(len(Y))
        cc[0] = base_cost

        # Loop for each cost
        for i in range(1, len(Y)):
            # Extract runs
            old_run = Y[i-1]
            new_run = Y[i]

            # Detect change in runs
            c = 0
            for j in range(old_run.size):
                if old_run[j] != new_run[j] and costs[j] > c:
                    c = costs[j]

            # Set base cost
            c = max(c, base_cost)

            # Set the cost
            cc[i] = c

        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)

def parallel_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    multiple workers can work on the transition between
    two consecutive runs in parallel. The total transition
    cost is determined by the most-hard-to-change factor.
    
    Parameters
    ----------
    transition_costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.
    
    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    return discount_cost(
        {k: v + execution_cost for k, v in transition_costs.items()}, 
        factors, max_cost, execution_cost
    )

def additive_cost(costs, factors, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = c1 + c2 + ... + base. 
    This means that every factor is independently, and sequentially changed.
    
    Parameters
    ----------
    costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    base_cost : float
        The base cost when no factors are changed, i.e., when a run
        is repeated.
    
    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Compute the column starts
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))
    costs = np.array([costs[str(f.name)] for f in factors])

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize the costs
        cc = np.zeros(len(Y))
        cc[0] = base_cost

        for i in range(1, len(Y)):
            # Base cost of a run
            tc = base_cost

            # Define the old / new run for transition
            old_run = Y[i-1]
            new_run = Y[i]

            # Additive costs
            for j in range(colstart.size-1):
                if np.any(old_run[colstart[j]:colstart[j+1]] != new_run[colstart[j]:colstart[j+1]]):
                    tc += costs[j]

            cc[i] = tc

        # Return the costs
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)

def single_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    only a single worker can work on the transition between
    two consecutive runs. The total transition
    cost is determined by the sum of all transition costs.
    
    Parameters
    ----------
    transition_costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.
    
    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    return additive_cost(
        {k: v + execution_cost for k, v in transition_costs.items()}, 
        factors, max_cost, execution_cost
    )

def scaled_parallel_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    multiple workers can work on the transition between
    two consecutive runs in parallel. The total transition
    cost is determined by the most-hard-to-change factor.
    The transition cost is determined by scaling the
    transition cost between start and stop with a base cost.
    See the parameters for more information.
    
    Parameters
    ----------
    transition_costs : dict(str, tuple(float, float, float, float) or float)
        A dictionary mapping the factor name to the transition cost.
        The cost is a tuple with as first element the base cost of any
        positive transition (-1 to +1), as second element the base cost
        of any negative transition (+1 to -1), as third element the 
        additional cost to positively scale between min (-1) and max (+1),
        and as third element the additional cost to negatively scale between 
        max (+1) and min (-1).
        Categorical factors should have only a float indicating the base cost
        of any transition.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.
    
    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Validate the categorical factors
    assert all(
        isinstance(transition_costs[str(f.name)], (int, float))
        for f in factors if f.is_categorical
    ), f'Categorical factors must only have a single float or integer representing the base transition cost of any transition'
    # Validate continuous factors
    assert all(
        len(transition_costs[str(f.name)]) == 4
        for f in factors if f.is_continuous
    ), f'Continuous variables must specify (base positive, base negative, scaling positive, scaling negative)'

    # Restructure transition costs
    transition_costs = {
        str(f.name): (
            transition_costs[str(f.name)] 
            if f.is_continuous
            else (transition_costs[str(f.name)], transition_costs[str(f.name)], 0, 0)
        ) for f in factors
    }

    # Compute the column starts
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))
    is_continuous = np.array([f.is_continuous for f in factors])

    # Expand the costs to arrays
    base_costs = np.array([
        [transition_costs[str(f.name)][0], transition_costs[str(f.name)][1]] 
        for f in factors
    ])
    scale_costs = np.array([
        [transition_costs[str(f.name)][2] / 2, transition_costs[str(f.name)][3] / 2]
        for f in factors
    ])

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize the costs
        cc = np.zeros(len(Y))
        cc[0] = execution_cost

        for i in range(1, len(Y)):
            # Define the old / new run for transition
            old_run = Y[i-1]
            new_run = Y[i]

            # Additive costs
            cc_ = np.zeros(colstart.size-1)
            for j in range(colstart.size-1):
                # Check for a transition
                if np.any(old_run[colstart[j]:colstart[j+1]] != new_run[colstart[j]:colstart[j+1]]):
                    # Check if continuous or categorical factor
                    if is_continuous[j]:
                        diff = new_run[colstart[j]] - old_run[colstart[j]]
                        if diff > 0:
                            # Positive transition
                            cc_[j] = base_costs[j][0] + scale_costs[j][0] * diff
                        else:
                            # Negative transition
                            cc_[j] = base_costs[j][1] - scale_costs[j][1] * diff
                    else:
                        # Categorical base cost
                        cc_[j] = base_costs[j][0]
                    
            # Take the maximum as most-hard-to-change
            cc[i] = np.max(cc_) + execution_cost

        # Return the costs
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)

def scaled_single_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    only a single worker can work on the transition between
    two consecutive runs. The total transition
    cost is determined by the sum of all transition costs.
    The transition cost is determined by scaling the
    transition cost between start and stop with a base cost.
    See the parameters for more information.
    
    Parameters
    ----------
    transition_costs : dict(str, tuple(float, float, float, float) or float)
        A dictionary mapping the factor name to the transition cost.
        The cost is a tuple with as first element the base cost of any
        positive transition (-1 to +1), as second element the base cost
        of any negative transition (+1 to -1), as third element the 
        additional cost to positively scale between min (-1) and max (+1),
        and as third element the additional cost to negatively scale between 
        max (+1) and min (-1).
        Categorical factors should have only a float indicating the base cost
        of any transition.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.
    
    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Validate the categorical factors
    assert all(
        isinstance(transition_costs[str(f.name)], (int, float))
        for f in factors if f.is_categorical
    ), f'Categorical factors must only have a single float or integer representing the base transition cost of any transition'
    # Validate continuous factors
    assert all(
        len(transition_costs[str(f.name)]) == 4
        for f in factors if f.is_continuous
    ), f'Continuous variables must specify (base positive, base negative, scaling positive, scaling negative)'

    # Restructure transition costs
    transition_costs = {
        str(f.name): (
            transition_costs[str(f.name)] 
            if f.is_continuous
            else (transition_costs[str(f.name)], transition_costs[str(f.name)], 0, 0)
        ) for f in factors
    }

    # Compute the column starts
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))
    is_continuous = np.array([f.is_continuous for f in factors])

    # Expand the costs to arrays
    base_costs = np.array([
        [transition_costs[str(f.name)][0], transition_costs[str(f.name)][1]] 
        for f in factors
    ])
    scale_costs = np.array([
        [transition_costs[str(f.name)][2] / 2, transition_costs[str(f.name)][3] / 2]
        for f in factors
    ])

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize the costs
        cc = np.zeros(len(Y))
        cc[0] = execution_cost

        for i in range(1, len(Y)):
            # Define the old / new run for transition
            old_run = Y[i-1]
            new_run = Y[i]

            # Additive costs
            cc_ = np.zeros(colstart.size-1)
            for j in range(colstart.size-1):
                # Check for a transition
                if np.any(old_run[colstart[j]:colstart[j+1]] != new_run[colstart[j]:colstart[j+1]]):
                    # Check if continuous or categorical factor
                    if is_continuous[j]:
                        diff = new_run[colstart[j]] - old_run[colstart[j]]
                        if diff > 0:
                            # Positive transition
                            cc_[j] = base_costs[j][0] + scale_costs[j][0] * diff
                        else:
                            # Negative transition
                            cc_[j] = base_costs[j][1] - scale_costs[j][1] * diff
                    else:
                        # Categorical base cost
                        cc_[j] = base_costs[j][0]
                    
            # Take the maximum as most-hard-to-change
            cc[i] = np.sum(cc_) + execution_cost

        # Return the costs
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)

def fixed_runs_cost(max_runs):
    """
    Cost function to deal with a fixed maximum number of experiments.
    The maximum cost is supposed to be the number of runs, and this cost function
    simply returns 1 for each run.

    Parameters
    ----------
    max_runs : int
        The maximum number of runs.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    def _cost_fn(Y):
        return [(np.ones(len(Y)), max_runs, np.arange(len(Y)))]

    return cost_fn(_cost_fn, denormalize=False, decoded=False, contains_params=False)

def max_changes_cost(factor, factors, max_changes):
    """
    Cost function to deal with a fixed maximum number of changes in a specific factor.
    The maximum cost is supposed to be the number of changes, and this cost function
    simply returns 1 for each change.
    
    .. note::
        It does not account for the initial setup and final result

    Parameters
    ----------
    factor : str or int
        The name or index of the factor
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_changes : int
        The maximum number of changes in the specified factor.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Expand factor for categorical variables
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))

    # Determine the columns of the factor
    if isinstance(factor, str):
        factor = [str(f.name) for f in factors].index(factor)
    factor = slice(colstart[factor], colstart[factor+1])

    # Create cost function
    def _cost_fn(Y):
        changes = np.zeros(len(Y))
        changes[1:] = np.any(np.diff(Y[:, factor], axis=0), axis=1).astype(int)
        return [(changes, max_changes, np.arange(len(Y)))]

    return cost_fn(_cost_fn, denormalize=False, decoded=False, contains_params=False)

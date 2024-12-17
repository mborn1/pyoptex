import numba
import numpy as np
from functools import wraps

def __cost_fn(f, factors=None, denormalize=True, decoded=True, contains_params=False):

    # Check if parameters are required (prevents direct use of numba.njit compilation)
    if contains_params:
        fn1 = f
    else:
        @wraps(f)
        def fn1(Y, params):
            return f(Y)

    # Check for denormalization in the cost function
    if denormalize:
        # Extract factor parameters
        col_names = [str(f.name) for f in factors]

        # Compute denormalization parameters
        norm_mean = np.array([f.mean for f in factors])
        norm_scale = np.array([f.scale for f in factors])

        # Extract categorical factors
        cat_factors = {str(f.name): {float(i): lname for i, lname in enumerate(f.levels)} 
                        for f in factors if f.is_categorical}
        
        # Wrap the function
        @wraps(fn1)
        def fn(Y, params):
            # Decode the design to a dataframe
            Y = decode_design(Y, params.effect_types, coords=params.coords)
            Y = (Y - norm_mean) / norm_scale
            Y = pd.DataFrame(Y, columns=col_names)
            for f, l in cat_factors.items():
                Y[f] = Y[f].map(l)
            
            return fn1(Y, params)
        NOTE = 'This cost function works on denormalized inputs'
    elif decoded:
        # Wrap the function
        @wraps(fn1)
        def fn(Y, params):
            # Decode the design to a dataframe
            Y = decode_design(Y, params.effect_types, coords=params.coords)
            return fn1(Y, params)
        NOTE = 'This cost function works on decoded categorical inputs'  
    else:
        # Wrap the function
        @wraps(fn1)
        def fn(Y, params):
            return fn1(Y, params)
        NOTE = 'This cost function works on normalized (encoded) inputs'

    # Extend the documentation with a note on normalization
    if fn.__doc__ is not None:
        params_pos = fn.__doc__.find('    Parameters\n    ---------')
        fn.__doc__ = fn.__doc__[:params_pos] + f'\n    .. note::\n        {NOTE}\n\n' + fn.__doc__[params_pos:]

    return fn

def cost_fn(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        return __cost_fn(args[0], *args[1:], **kwargs)
    else:
        def wrapper(f):
            return __cost_fn(f, *args, **kwargs)
        return wrapper

############################################################

def combine_costs(cost_fn):
    """
    Combine multiple cost functions together.

    Parameters
    ----------
    cost_fn : iterable(func)
        An iterable of cost functions to concatenate
    
    Returns
    -------
    cost_fn : func
        The combined cost function for the simulation algorithm.
    """
    def _cost(Y, params):
        return [c for cf in cost_fn for c in cf(Y, params)]

    _cost.__doc__ = 'This is a combined cost function of:\n* ' + '\n* '.join(cf.__name__ in cost_fn)

    return _cost

def discount_cost(costs, factors, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = max(c1, c2, ..., base). 
    This means that if a harder to vary factor changes, the easier factors comes for free.
    
    Parameters
    ----------
    costs : np.array(1d)
        The cost of each factor in effect_types.
    effect_types : dict or np.array(1d)
        The type of each effect. If a dictionary, the values are taken as the array.
    max_cost : float
        The maximum cost of this function.
    base_cost : float
        The base cost when nothing is altered.
    
    Returns
    -------
    cost_fn : func
        The cost function for the simulation algorithm.
    """
    # Expand the costs to categorically encoded
    costs = np.array([c for f in factors for c in ([costs[str(f.name)]] if f.is_continuous else [costs[str(f.name)]]*(len(f.levels)-1))])

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize costs
        cc = np.zeros(len(Y))

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
            if c < base_cost:
                c = base_cost

            # Set the cost
            cc[i] = c
        
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)

def transition_discount_cost(costs, factors, max_cost, base_cost=1):
    return discount_cost({k: v + base_cost for k, v in costs.items()}, factors, max_cost, base_cost)

def additive_effect_trans_cost(costs, factors, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = c1 + c2 + ... + base. 
    This means that every factor is independently, and sequentially changed.

    The function can deal with categorical factors, correctly expanding the costs array.
    
    Parameters
    ----------
    costs : np.array(1d)
        The cost of each factor in effect_types.
    effect_types : dict or np.array(1d)
        The type of each effect. If a dictionary, the values are taken as the array.
    max_cost : float
        The max cost of this function.
    base_cost : float
        The base cost when nothing is altered.
    
    Returns
    -------
    cost_fn : func
        The cost function for the simulation algorithm.
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

        for i in range(len(Y)):
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

def fixed_runs_cost(max_cost):
    """
    Cost function to deal with a fixed maximum number of experiments.
    The maximum cost is supposed to be the number of runs, and this cost function
    simply returns 1 for each run.

    Parameters
    ----------
    max_cost : float
        The maximum number of runs.

    Returns
    -------
    cost_fn : func
        The cost function for the simulation algorithm.
    """
    def _cost_fn(Y):
        return [(np.ones(len(Y)), max_cost, np.arange(len(Y)))]

    return cost_fn(_cost_fn, denormalize=False, decoded=False, contains_params=False)

def max_changes_cost(factor, factors, max_cost):
    """
    Cost function to deal with a fixed maximum number of changes in a specific factor.
    The maximum cost is supposed to be the number of changes, and this cost function
    simply returns 1 for each change.
    
    .. note::
        It does not account for the initial setup and final result

    Parameters
    ----------
    factor : str or int
        The index of the factor (in effect_types)
    effect_types : dict or np.array(1d)
        The type of each effect. If a dictionary, the values are taken as the array.
    max_cost : float
        The maximum number of changes in the specified factor.

    Returns
    -------
    cost_fn : func
        The cost function for the simulation algorithm.
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
        return [(changes, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost_fn, denormalize=False, decoded=False, contains_params=False)

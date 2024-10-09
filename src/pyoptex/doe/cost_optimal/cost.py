import numba
import numpy as np

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
    def _cost(Y):
        return [c for cf in cost_fn for c in cf(Y)]

    return _cost

def discount_effect_trans_cost(costs, effect_types, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = max(c1, c2, ..., base). 
    This means that if a harder to vary factor changes, the easier factors comes for free.

    The function can deal with categorical factors, correctly expanding the costs array.

    .. note::
        The function requires an ordering in the costs for each effect. It only looks at
        the first factor which changes and defines the cost based on that.
    
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
    # Convert effect types
    if isinstance(effect_types, dict):
        effect_types = np.array(list(effect_types.values()))

    # Expand the costs to categorically encoded
    costs = np.array([c for cost, et in zip(costs, effect_types) for c in ([cost] if et == 1 else [cost]*(et-1))])

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

    return _cost

def additive_effect_trans_cost(costs, effect_types, max_cost, base_cost=1):
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
    # Convert effect types
    if isinstance(effect_types, dict):
        effect_types = np.array(list(effect_types.values()))

    # Compute the column starts
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))
    costs = np.array(costs)
    
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

    return _cost

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
    def cost_fn(Y):
        return [(np.ones(len(Y)), max_cost, np.arange(len(Y)))]

    return cost_fn

def max_changes_cost(factor, effect_types, max_cost):
    """
    Cost function to deal with a fixed maximum number of changes in a specific factor.
    The maximum cost is supposed to be the number of changes, and this cost function
    simply returns 1 for each change.
    
    .. note::
        It does not account for the initial setup and final result

    Parameters
    ----------
    factor : int
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
    # Convert effect types
    if isinstance(effect_types, dict):
        effect_types = np.array(list(effect_types.values()))

    # Expand factor for categorical variables
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))))
    factor = slice(colstart[factor], colstart[factor+1])

    # Create cost function
    def cost_fn(Y):
        changes = np.zeros(len(Y))
        changes[1:] = np.any(np.diff(Y[:, factor], axis=0), axis=1).astype(int)
        return [(changes, max_cost, np.arange(len(Y)))]

    return cost_fn

import numpy as np
from tqdm import tqdm

from ..utils.design import x2fx
from ..utils.init import full_factorial, init_single_unconstrained

def greedy_cost_minimization(Y, params):
    """
    Greedily minimizes the cost of the design Y.

    Parameters
    ----------
    Y : np.array(2d)
        The design to cost minimize.
    params : :py:class:`Parameters <cost_optimal_designs.simulation.Parameters>`
        The simulation parameters.

    Returns
    -------
    Y : np.array(2d)
        The greedily cost minimized design.
    """
    # Initialization (force prior)
    nprior = len(params.prior)
    Yn = np.zeros_like(Y)
    Yn[:nprior] = params.prior
    chosen = np.zeros(len(Y), dtype=np.bool_)
    chosen[:nprior] = True

    # Iteratively use greedy cost minimization
    for i in range(nprior, len(Y)):
        # Find parameters that are not chosen
        non_chosen = np.where(~chosen)[0]

        # # Initialize all costs
        costs = [None] * non_chosen.size
        
        # Compute all costs
        for k in range(non_chosen.size):
            Yn[i] = Y[non_chosen[k]]
            costs[k] = params.fn.cost(Yn[:i+1])

        # Compute the total cost of each operation
        costs_Y = np.array([np.sum([np.sum(c) / m * c.size / (i+1) for c, m, _ in cost]) for cost in costs])
        min_cost_idx = non_chosen[np.argmin(costs_Y)]

        # Chose the index
        Yn[i] = Y[min_cost_idx]
        chosen[min_cost_idx] = True

    return Yn

################################################

def init(params, n=1, complete=False):
    """
    Initialize a design with `n` randomly sampled runs. They must
    be within the constraints.

    Parameters
    ----------
    params : :py:class:`Parameters <cost_optimal_designs.simulation.Parameters>`
        The simulation parameters.
    n : int
        The number of runs
    complete : bool
        False means use the coordinates and prior specified in params, otherwise, no
        coords or prior are used. Can be used to perform a complete sample of the design space.
    
    Returns
    -------
    run : np.array(2d)
        The resulting design.
    """
    # Initialize
    run = np.zeros((n, params.colstart[-1]), dtype=np.float64)
    invalid = np.ones(n, dtype=np.bool_)

    # Adjust for completeness
    if not complete:
        nprior = len(params.prior)
        run[:nprior] = params.prior
        invalid[:nprior] = False
        coords = params.coords
    else:
        coords = None

    # Loop until all are valid
    while np.any(invalid):
        run[invalid] = init_single_unconstrained(params.colstart, coords, run[invalid], params.effect_types)
        invalid[invalid] = params.fn.constraints(run[invalid])

    return run

def init_feasible(params, max_tries=3, max_size=None, force_cost_feasible=True):
    """
    Generate a random initial and feasible design. From a random
    permutation of a full factorial design, the runs are dropped one-by-one
    as long as they still provide a feasible design. Finally, the design
    is greedily reordered for minimal cost.

    Parameters
    ----------
    params : :py:class:`Parameters <cost_optimal_designs.simulation.Parameters>`
        The simulation parameters.
    max_tries : int
        The maximum number of random tries. If all random tries fail, a 
        final non-randomized design is created. If this also fails, a ValueError is thrown.
    max_size : int
        The maximum number of runs before iteratively removing them.
    force_cost_feasible : bool
        Force a final cost feasibility check.

    Returns
    -------
    Y : np.array(2d)
        The initial design.
    """
    # Initialize the tries for randomization
    tries = -1
    reverse = False

    # Check if prior is estimeable
    Xprior = params.Y2X(params.prior)
    if Xprior.shape[0] != 0 and np.linalg.matrix_rank(Xprior) == Xprior.shape[1]:
        return params.prior 
    nprior = len(Xprior)

    feasible = False
    while not feasible:
        # Add one try
        tries += 1

        # Create a full factorial design
        Y = full_factorial(params.colstart, params.coords)

        # Permute to randomize
        if tries < max_tries:
            Y = np.random.permutation(Y)

        # Drop impossible combinations
        Y = Y[~params.fn.constraints(Y)]

        # Define a maximum size (for feasibility)
        if max_size is not None:
            Y = Y[:max_size]

        # Compute X
        X = params.Y2X(Y)

        # Augmentation
        Y = np.concatenate((params.prior, Y), axis=0)
        X = np.concatenate((Xprior, X), axis=0)

        # Drop runs
        keep = np.ones(len(Y), dtype=np.bool_)
        keep[:nprior] = True
        r = range(nprior, len(Y)) if not reverse else range(len(Y)-1, nprior-1, -1)
        for i in tqdm(r):
            keep[i] = False
            Xk = X[keep]
            if np.linalg.matrix_rank(Xk) != X.shape[1]:
                keep[i] = True
        Y = Y[keep]

        # Reorder for cost optimization (greedy)
        if tries < max_tries:
            Y = greedy_cost_minimization(Y, params)

        # Fill it up
        X = params.Y2X(Y)
        costs = params.fn.cost(Y)
        cost_Y = np.array([np.sum(c) for c, _, _ in costs])
        max_cost = np.array([m for _, m, _ in costs])
        feasible = (np.linalg.matrix_rank(X) == X.shape[1]) and (np.all(cost_Y <= max_cost) or not force_cost_feasible)

        # Raise an error if no feasible design can be found
        if tries >= max_tries and not feasible:
            if reverse:
                raise ValueError('Unable to find a feasible design within the cost constraints')
            else:
                reverse = True 

    return Y


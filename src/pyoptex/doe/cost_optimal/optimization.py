import numpy as np

from .formulas import ce_update_vinv, detect_block_end_from_start
from .simulation import State
from ..utils.design import force_Zi_asc
from ...utils.numba import numba_any_axis1, numba_diff_axis0
from ..._profile import profile


def adapt_group(groups, factor, row_start, row_end):
    """
    Detects which groups should be adapted by having changed a specific
    factor column from row_start to row_end.

    Parameters
    ----------
    groups : np.array(1d)
        The grouping matrix for this factor.
    factor : np.array(2d)
        The factor column with its coordinate changed already.
    row_start : int
        The starting row (included) where the coordinate was changed.
    row_end : int
        The final row (excluded) where the coordinate was changed.
    
    Returns
    -------
    b : list(tuple(row_start, row_end, group_from, group_to))
        A list of operations to apply changing the groups of the factors.
    """
    # Short-circuit
    if groups is None:
        return []

    # Check if it is a complete group / merge
    merge_above, same_above = False, False
    merge_below, same_below = False, False
    if row_start > 0:
        same_above = groups[row_start-1] == groups[row_start]
        merge_above = np.all(factor[row_start-1] == factor[row_start])
    if row_end < len(groups):
        same_below = groups[row_end] == groups[row_end-1]
        merge_below = np.all(factor[row_end] == factor[row_start])
        
    # Initialize groups
    b = []
    max_grp = groups[-1]
    if merge_above:
        # Set the group
        b.append((row_start, row_end, groups[row_start], groups[row_start-1]))

        # Validate if below must also be merged (and group dropped)
        if merge_below:
            block_end = detect_block_end_from_start(groups, row_end)
            b.append((row_end, block_end, groups[row_end], groups[row_start-1]))
    else:
        if merge_below:
            # Set the group
            b.append((row_start, row_end, groups[row_start], groups[row_end]))
        else:
            # Check if split from above / below (new group)
            if same_above or same_below:
                b.append((row_start, row_end, groups[row_start], max_grp+1))

                # Check for double split
                if same_above and same_below:
                    block_end = detect_block_end_from_start(groups, row_end)
                    b.append((row_end, block_end, groups[row_end], max_grp+2))

    return np.array(b)

######################################################

@profile
def ce_metric(state, params):
    """
    Optimizes the design by applying a single coordinate exchange pass
    over the design.

    Parameters
    ----------
    state : :py:class:`State <cost_optimal_designs.simulation.State>`
        The state from which to sample.
    params : :py:class:`Parameters <cost_optimal_designs.simulation.Parameters>`
        The simulation parameters.

    Returns
    -------
    new_state : :py:class:`State <cost_optimal_designs.simulation.State>`
        The new state after optimization.
    """
    nprior = len(params.prior)

    # Loop over all coordinates
    for row in range(state.Y.shape[0] - 1, nprior-1, -1):
        for col in range(params.colstart.size - 1):
            # Store original values
            Ycoord = np.copy(state.Y[row, params.colstart[col]:params.colstart[col+1]])
            Xrow = np.copy(state.X[row])

            # Store original coordinate
            co = Ycoord

            # Loop over possible coordinates
            for coord in params.coords[col]:
                # Skip original coordinate
                if np.any(co != coord):
                    # Initialize accept
                    accept = False

                    # Set coordinate
                    state.Y[row, params.colstart[col]:params.colstart[col+1]] = coord

                    # Check the constraints
                    if not params.fn.constraints(state.Y[row:row+1])[0]:
                        state.X[row] = params.Y2X(state.Y[row])

                        # Compute costs
                        new_costs = params.fn.cost(state.Y)
                        new_cost = np.array([np.sum(c) for c, _, _ in new_costs])
                        max_cost = np.array([m for _, m, _ in new_costs])

                        # Check constraints
                        if np.all(new_cost <= max_cost):
                            # Update Zsn, Vinv
                            b = adapt_group(state.Zs[col], state.Y[:, params.colstart[col]:params.colstart[col+1]], row, row+1)
                            if len(b) == 0:
                                Zsn = state.Zs
                                Vinvn = state.Vinv
                            else:
                                Zin, Vinvn = ce_update_vinv(
                                    np.copy(state.Vinv), 
                                    np.copy(state.Zs[col]), b, params.ratios[:, col]
                                )
                                Zsn = tuple([Zi if i != col else force_Zi_asc(Zin) for i, Zi in enumerate(state.Zs)])

                            # Check metric
                            new_metric = params.fn.metric.call(state.Y, state.X, Zsn, Vinvn, new_costs)

                            # Compute accept
                            accept = new_metric > state.metric
                    
                    # Accept the update
                    if accept:
                        # Store metric and design
                        Ycoord = coord
                        Xrow = np.copy(state.X[row])

                        # Update the state
                        state = State(state.Y, state.X, Zsn, Vinvn, new_metric, new_cost, new_costs, max_cost)

                    else:
                        # Reset values
                        state.Y[row, params.colstart[col]:params.colstart[col+1]] = Ycoord
                        state.X[row] = Xrow

    return state

class CEMetric:
    """
    Applies the coordinate exchange function on the design every
    n iterations.

    Attributes
    ----------
    i : int
        The current iteration
    n : int
        Every nth iteration, the optimization is applied.
    """
    def __init__(self, n=1):
        self.i = 0
        self.n = n

    def reset(self):
        self.i = 0

    def call(self, state, params):
        self.i += 1
        if self.i % self.n == 0:
            state = ce_metric(state, params)
        return state

#########################################################

@profile
def ce_struct_metric(state, params):
    """
    Optimizes the design by applying a single structured coordinate exchange pass
    over the design. By structured is meant that not every individual coordinate
    is adjusted, rather, entire groups of coordinates are adjusted per factor.

    Parameters
    ----------
    state : :py:class:`State <cost_optimal_designs.simulation.State>`
        The state from which to sample.
    params : :py:class:`Parameters <cost_optimal_designs.simulation.Parameters>`
        The simulation parameters.

    Returns
    -------
    new_state : :py:class:`State <cost_optimal_designs.simulation.State>`
        The new state after optimization.
    """
    nprior = len(params.prior)

    # Loop over all coordinates
    for col in range(params.colstart.size - 1):
        # Detect blocks
        blocks = np.concatenate((
            np.array([0], np.int64), 
            np.where(numba_any_axis1(numba_diff_axis0(state.Y[:, params.colstart[col]:params.colstart[col+1]]) != 0))[0] + 1,
            np.array([len(state.Y)], np.int64)
        ))

        # Extract blocks with no overlap in prior
        blocks = blocks[blocks >= nprior]
        if blocks[0] != nprior:
            blocks = np.concatenate((np.array([nprior], np.int64), blocks))

        for b in range(blocks.size - 1):
            # Rows from that block
            rows = np.arange(blocks[b], blocks[b+1])

            # Store original values
            Ycoord = np.copy(state.Y[rows[0], params.colstart[col]:params.colstart[col+1]])
            Xrows = np.copy(state.X[rows])

            # Store original coordinate
            co = Ycoord

            # Loop over possible coordinates
            for coord in params.coords[col]:

                # Short-circuit original coordinate
                if np.any(co != coord):

                    # Initialize accept
                    accept = False

                    # Set coordinate
                    state.Y[rows, params.colstart[col]:params.colstart[col+1]] = coord

                    # Check constraints
                    if not np.any(params.fn.constraints(state.Y[rows])):
                        state.X[rows] = params.Y2X(state.Y[rows])

                        # Compute costs
                        new_costs = params.fn.cost(state.Y)
                        new_cost = np.array([np.sum(c) for c, _, _ in new_costs])
                        max_cost = np.array([m for _, m, _ in new_costs])
                        
                        if np.all(new_cost <= max_cost):
                            # Update Zsn, Vinv
                            b = adapt_group(state.Zs[col], state.Y[:, params.colstart[col]:params.colstart[col+1]], rows[0], rows[-1]+1)
                            if len(b) == 0:
                                Zsn = state.Zs
                                Vinvn = state.Vinv
                            else:
                                Zin, Vinvn = ce_update_vinv(
                                    np.copy(state.Vinv), 
                                    np.copy(state.Zs[col]), b, params.ratios[:, col]
                                )
                                Zsn = tuple([Zi if i != col else force_Zi_asc(Zin) for i, Zi in enumerate(state.Zs)])

                            # Compute new metric
                            new_metric = params.fn.metric.call(state.Y, state.X, Zsn, Vinvn, new_costs)

                            # Compute accept
                            accept = new_metric > state.metric

                    # Accept the update
                    if accept:
                        # Store metric and design
                        Ycoord = coord
                        Xrows = np.copy(state.X[rows])

                        # Update the state
                        state = State(state.Y, state.X, Zsn, Vinvn, new_metric, new_cost, new_costs, max_cost)
                    else:
                        # Reset values
                        state.Y[rows, params.colstart[col]:params.colstart[col+1]] = Ycoord
                        state.X[rows] = Xrows

    return state

class CEStructMetric:
    """
    Applies the structured coordinate exchange function on the design every
    n iterations.

    Attributes
    ----------
    i : int
        The current iteration
    n : int
        Every nth iteration, the optimization is applied.
    """
    def __init__(self, n=1):
        self.i = 0
        self.n = n

    def reset(self):
        self.i = 0

    def call(self, state, params):
        self.i += 1
        if self.i % self.n == 0:
            state = ce_struct_metric(state, params)
        return state

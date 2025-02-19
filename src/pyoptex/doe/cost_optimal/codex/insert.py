"""
Module for all insert functions of the CODEX algorithm
"""

import numpy as np

from ...._profile import profile
from ....utils.numba import numba_insert_axis0
from ....utils.design import force_Zi_asc, obs_var_from_Zs
from .formulas import (NO_UPDATE, detect_block_end_from_start,
                       insert_update_vinv)
from .simulation import State
from ..utils import obs_var_Zs


def groups_insert(Yn, Zs, pos, colstart):
    """
    When run at position `pos` is inserted, compute
    how the groups for the different factors change.

    Parameters
    ----------
    Yn : np.array(2d)
        The design matrix after inserting the new run at position `pos`
    Zs : list(np.array(1d) or None)
        The grouping matrices of the old design (before insertion)
    pos : int
        The position of the inserted run
    colstart : np.array(1d)
        The starting column of each factor
    
    Returns
    -------
    a : np.array(1d)
        For each factor, the group that run belongs to
    b : list(tuple(row_start, row_end, group_from, group_to))
        A list of additional group changes per factor. For example,
        in case a group was broken up due to the insertion.
    """

    # Initialization
    a = np.zeros(len(Zs), dtype=np.int_)
    b = [() for _ in range(a.size)]

    # Loop over all factors
    for i in range(colstart.size - 1):
        # Check if factor is grouped
        if Zs[i] is not None:
            # Loop initialization
            Zi = Zs[i]
            max_grp = Zi[-1]
            cols = slice(colstart[i], colstart[i+1])

            # Detect change types
            if pos > 0 and np.all(Yn[pos-1, cols] == Yn[pos, cols]):
                # Merge above
                a[i] = Zi[pos-1]
            elif pos < len(Zi) and np.all(Yn[pos+1, cols] == Yn[pos, cols]):
                # Merge below
                a[i] = Zi[pos]
            else:
                # Split group
                a[i] = max_grp + 1

                # Double split
                if 0 < pos < len(Zi) and np.all(Yn[pos-1, cols] == Yn[pos+1, cols]):
                    block_end = detect_block_end_from_start(Zi, pos)
                    b[i] = (pos+1, block_end+1, Zi[pos], max_grp + 2)
        else:
            # Set non-update
            a[i] = NO_UPDATE

    return a, b

def _insert_position(new_run, pos, state, params, new_X=None):
    """
    Inserts a new run at the specified position and returns a new state.

    Parameters
    ----------
    new_run : np.array(1, 1d)
        The new run to be added to the design.
    pos : int
        The position at which to insert the new run.
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to start.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    new_X : np.array(1, 1d)
        The model matrix part of that run = x2fx(new_run)
    
    Returns
    -------
    new_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The new state after inserting the run at that position.
    """
    # Compute new X
    if new_X is None:
        new_X = params.fn.Y2X(new_run)

    # Create the new design
    Y = numba_insert_axis0(state.Y, pos, new_run[0])
    X = numba_insert_axis0(state.X, pos, new_X[0])

    # Update Zs and Vinv
    if any(Zi is not None for Zi in state.Zs):
        if params.use_formulas:
            a, b = groups_insert(Y, state.Zs, pos, params.colstart)
            Zs, Vinv = insert_update_vinv(state.Vinv, state.Zs, pos, a, b, params.ratios)
            Zs = tuple(force_Zi_asc(Zi) if Zi is not None else None for Zi in Zs)
        else:
            Zs = obs_var_Zs(Y, params.colstart, params.grouped_cols)
            Vinv = np.array([
                np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) 
                for ratios in params.ratios
            ])
    else:
        # Shortcut as there are no hard-to-vary factors
        Zs = state.Zs
        Vinv = np.broadcast_to(
            np.eye(len(Y)), 
            (state.Vinv.shape[0], len(Y), len(Y))
        )

    # Update costs
    costs = params.fn.cost(Y, params)
    cost_Y = np.sum(costs, axis=1)

    # Compute the new metric
    metric = params.fn.metric.call(Y, X, Zs, Vinv, costs)

    # Collect stats
    params.stats['insert_loc'][params.stats['it']] = pos

    return State(Y, X, Zs, Vinv, metric, cost_Y, costs)

###################################################

def insert_last(new_run, state, params):
    """
    Inserts a new run at the last position and returns a new state.

    Parameters
    ----------
    new_run : np.array(1, 1d)
        The new run to be added to the design.
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to start.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    
    Returns
    -------
    new_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The new state after inserting the run at the last position.
    """
    # Insert in last position
    return _insert_position(new_run, len(state.Y), state, params)

@profile
def insert_optimal(new_run, state, params):
    """
    Inserts a new run in the optimal position by simultaneously
    maximizing the metric and minimizing the cost increase.

    Parameters
    ----------
    new_run : np.array(1, 1d)
        The new run to be added to the design.
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to start.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    
    Returns
    -------
    new_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The new state after inserting the run.
    """
    # Compute new X
    new_X = params.fn.Y2X(new_run)
    nprior = len(params.prior)

    ############################################################

    # Find ideal insert position
    best_metric = 0
    exceeds_budget = True
    best_state = state

    # Loop over all possible positions
    for k in range(state.Y.shape[0], nprior-1, -1):
        # Insert run
        Yn = numba_insert_axis0(state.Y, k, new_run[0])
        Xn = numba_insert_axis0(state.X, k, new_X[0])

        # Compute new observation variance
        if any(Zi is not None for Zi in state.Zs):
            if params.use_formulas:
                a, b = groups_insert(Yn, state.Zs, k, params.colstart)
                Zsn, Vinvn = insert_update_vinv(
                    state.Vinv, state.Zs, k, a, b, params.ratios
                )
                Zsn = tuple(
                    force_Zi_asc(Zi) if Zi is not None else None 
                    for Zi in Zsn
                )
            else:
                Zsn = obs_var_Zs(Yn, params.colstart, params.grouped_cols)
                Vinvn = np.array([
                    np.linalg.inv(obs_var_from_Zs(Zsn, len(Yn), ratios)) 
                    for ratios in params.ratios
                ])
        else:
            # Shortcut as there are no hard-to-vary factors
            Zsn = state.Zs
            Vinvn = np.broadcast_to(
                np.eye(len(Yn)), 
                (state.Vinv.shape[0], len(Yn), len(Yn))
            )

        # Compute cost increase
        costsn = params.fn.cost(Yn, params)
        cost_Yn = np.array([np.sum(c) for c, _, _ in costsn])
        max_cost = np.array([m for _, m, _ in costsn])

        # Compute metric
        metricn = params.fn.metric.call(Yn, Xn, Zsn, Vinvn, costsn)

        # Create the new state
        staten = State(Yn, Xn, Zsn, Vinvn, metricn, cost_Yn, costsn, max_cost)

        # Target
        # pylint: disable=line-too-long
        mt = np.sum(staten.cost_Y / staten.max_cost * np.array([c.size for c, _, _ in staten.costs])) / len(staten.Y) \
                - np.sum(state.cost_Y / state.max_cost * np.array([c.size for c, _, _ in state.costs])) / len(state.Y)
        metric_temp = (staten.metric - state.metric) / (mt / len(state.costs))

        # Exceeds budget
        exceeds_budget_temp = np.any(cost_Yn > max_cost)

        # Maximize
        if (metric_temp > best_metric and exceeds_budget == exceeds_budget_temp) \
                or (exceeds_budget and not exceeds_budget_temp):
            best_metric = metric_temp
            best_state = staten
            exceeds_budget = exceeds_budget_temp
            params.stats['insert_loc'][params.stats['it']] = k

    ############################################################

    # Insert in position
    return best_state

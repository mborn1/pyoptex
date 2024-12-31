"""
Module for all metrics of the CODEX algorithm
"""

import numpy as np

from ...._profile import profile
from ....utils.design import force_Zi_asc, obs_var_from_Zs
from .formulas import detect_block_end_from_start, remove_update_vinv
from .simulation import State
from ..utils import obs_var_Zs


def groups_remove(Yn, Zs, pos, colstart):
    """
    Computes the change in groups when removing run at `pos`.

    Parameters
    ----------
    Yn : np.array(2d)
        The design matrix.
    Zs : list(np.array(1d) or None)
        The grouping matrices.
    pos : int
        The index of the run to remove.
    colstart : np.array(1d)
        The start column of each factor.

    Returns
    -------
    b : list(tuple(start, end, group_from, group_to))
        Each element represents a factor random effect and specifies
        whether runs from this factor should be moved to another group.
    """
    # Initialization
    b = [() for _ in range(len(Zs))]

    # Loop over all factors
    for i in range(colstart.size - 1):
        # Check if it is a grouped column
        if Zs[i] is not None:
            # Loop initialization
            Zi = Zs[i]
            cols = slice(colstart[i], colstart[i+1])
            
            # Detect double split
            if pos > 0 and pos < len(Zi) - 1 \
                        and np.all(Yn[pos-1, cols] == Yn[pos, cols]) \
                        and Zi[pos-1] != Zi[pos+1]:
                block_end = detect_block_end_from_start(Zi, pos+1)
                b[i] = (pos, block_end-1, Zi[pos+1], Zi[pos-1])

    return b

###################################################

@profile
def remove_optimal_onebyone(state, params, prevent_insert=False):
    """
    Removes runs from the design until within the cost constraints. Runs
    are selected and removed one-by-one for minimal metric loss and 
    maximal cost reduction.

    Parameters
    ----------
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to sample.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    prevent_insert : bool
        Whether to prevent the removal of the recently inserted run.

    Returns
    -------
    new_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The new state after inserting the run.
    """
    nprior = len(params.prior)

    # Temporary variables
    keep = np.ones(len(state.Y), dtype=np.bool_)

    # Stats
    insert_loc = params.stats['insert_loc'][params.stats['it']]

    # Find which to drop
    while np.any(state.cost_Y > state.max_cost):

        # Loop initialization
        best_metric = np.inf
        best_state = state
        best_k = -1

        # Compute bottleneck indices
        idx = np.unique(np.concatenate([idx for _, _, idx in state.costs]))
        idx = idx[idx >= nprior]

        # Loop over all available runs
        for k in idx:
            # Set keep to false
            keep[k] = False

            # Define new design
            Yn = state.Y[keep[:len(state.Y)]]
            Xn = state.X[keep[:len(state.Y)]]

            # Compute Zsn and Vinvn
            if any(Zi is not None for Zi in state.Zs):
                if params.use_formulas:
                    b = groups_remove(Yn, state.Zs, k, params.colstart)
                    Zsn, Vinvn = remove_update_vinv(
                        state.Vinv, state.Zs, k, b, params.ratios
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

            # Compute cost reduction
            costsn = params.fn.cost(Yn, params)
            cost_Yn = np.array([np.sum(c) for c, _, _ in costsn])
            max_cost = np.array([m for _, m, _ in costsn])

            # Compute new metric
            metricn = params.fn.metric.call(Yn, Xn, Zsn, Vinvn, costsn)

            # Create new state
            staten = State(Yn, Xn, Zsn, Vinvn, metricn, cost_Yn, costsn, max_cost)
                
            # Compute metric loss per cost
            # pylint: disable=line-too-long
            mt = np.sum(state.cost_Y / state.max_cost * np.array([c.size for c, _, _ in state.costs])) / len(state.Y) \
                - np.sum(staten.cost_Y / staten.max_cost * np.array([c.size for c, _, _ in staten.costs])) / len(staten.Y)
            metric_temp = (state.metric - staten.metric) / (mt / len(state.costs))

            # Minimize
            if (metric_temp < best_metric or np.isinf(best_metric)) \
                    and (k != insert_loc or not prevent_insert or insert_loc < 0):
                best_metric = metric_temp
                best_state = staten
                best_k = k
            
            # Set keep to true
            keep[k] = True

        # Drop the run
        state = best_state
        if best_k == insert_loc:
            params.stats['removed_insert'][params.stats['it']] = True
        elif best_k < insert_loc:
            insert_loc -= 1

    return state

def remove_optimal_onebyone_prevent(state, params):
    """
    Similar to 
    :py:func:`remove_optimal_onebyone <pyoptex.doe.cost_optimal.codex.remove.remove_optimal_onebyone>`, 
    but with prevent_insert = True.

    Parameters
    ----------
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to sample.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    new_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The new state after inserting the run.
    """
    return remove_optimal_onebyone(state, params, prevent_insert=True)

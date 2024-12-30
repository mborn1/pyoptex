"""
Module containing all the covariate functions of the cost optimal designs
"""

import numpy as np

# Update function when adding new blocking factors
def _update_woodbury(Vinv, new_Zs, new_ratios):
    """
    Update formula designed to add random effects 
    which do not depend on any factor reset (e.g. blocking factors). 
    
    Parameters
    ----------
    Vinv : np.array(3d)
        The inverses of the observation covariance matrices.
    new_Zs : list(np.array(1d))
        The additional random effect groupings.
    new_ratios : np.array(2d) or np.array(1d)
        The ratios for each of the Vinv and Zs. Shape[0] must be
        the number of Vinv, and shape[1] must be the number of new Zs.
        If 1d array is provided, the blocks are simply broadcasted.

    Returns
    -------
    Vinv : np.array(3d)
        The updated inverses of the observation covariance matrices.
    """
    # Broadcast ratios
    if len(new_ratios.shape) == 1:
        new_ratios = np.broadcast_to(new_ratios, (Vinv.shape[0], *new_ratios.shape))
    elif new_ratios.shape[0] == 1:
        new_ratios = np.broadcast_to(new_ratios, (Vinv.shape[0], *new_ratios.shape[1:]))

    # Make sure there are enough ratios
    assert Vinv.shape[0] == new_ratios.shape[0], 'Every Vinv requires a ratio'
    assert len(new_Zs) == new_ratios.shape[1], 'Every grouping requires a ratio'

    # Convert to expanded format
    new_Zs = [np.eye(Zi[-1]+1, dtype=np.bool_)[Zi] for Zi in new_Zs]

    # Apply each Zs sequentially
    for i in range(len(new_Zs)):
        # Extract parameter
        Zi = new_Zs[i]
        ratios = new_ratios[:, i]

        # Compute VR
        VR = np.stack([np.sum(Vinv[:, :, Zi[:, j]], axis=-1) for j in range(Zi.shape[1])], axis=-1)
        SVR = np.stack([np.sum(VR[:, Zi[:, j]], axis=1) for j in range(Zi.shape[1])], axis=1)
        idx = np.diag_indices_from(SVR[0])
        SVR[:, idx[0], idx[0]] += 1/ratios[:, np.newaxis]

        # Compute woodbury
        Vinv -= VR @ np.linalg.solve(SVR, np.swapaxes(VR, -2, -1))

    return Vinv

# pylint: disable=unused-argument,too-many-arguments
def no_cov(Y, X, Zs, Vinv, costs, random=False):
    """
    Function to indicate no covariate is added.

    Parameters
    ----------
    Y : np.array(2d)
        The design matrix
    X : np.array(2d)
        The model matrix
    Zs : list(np.array(1d))
        The grouping matrices
    Vinv : np.array(3d)
        The inverses of the multiple covariance matrices for each
        set of a-priori variance ratios.
    costs : list(np.array(1d), float, np.array(1d))
        The list of different costs.
    random : bool
        Whether to add covariates at random or predetermined. The random
        aspect is used for sampling random points in the design space.

    Returns
    -------
    Y : np.array(2d)
        The updated design matrix with covariates.
    X : np.array(2d)
        The updated model matrix with covariates.
    Zs : list(np.array(1d))
        The updated grouping matrices with added random covariate effects.
    Vinv = np.array(3d)
        The updated inverses of the covariance matrices with the added
        random covariate effects.
    """
    return Y, X, Zs, Vinv

def cov_time_trend(time=1, cost_index=0):
    """
    Covariance function to account for time trends.
    The cost at `cost_index` is assumed to represent some form
    of time. Every `time`, the level of the time trend is increased
    by one unit.

    For example, if time is 2 and we have runs with
    cumulative time [0, 1, 2, 3, 4, 5], the added time column
    will be [-1, -1, 0, 0, 1, 1].

    Parameters
    ----------
    time : float
        Every `time` in cumulative cost, the time
        column progresses one equidistant step.
    cost_index : int
        The index in the multi-cost objective to look at.
    
    Returns
    -------
    cov : func(Y, X, Zs, Vinv, costs)
        The covariance function.
    """
    # Define the covariance function
    def _cov(Y, X, Zs, Vinv, costs, random=False):
        # Define time array
        if random:
            t = np.random.rand(Y.shape[0]) * 2 - 1
        else:
            cum_cost = np.cumsum(costs[cost_index][0])
            t = np.floor_divide(cum_cost, time)
            t = t / t[-1] * 2 - 1

        # Concatenate time array
        Y = np.concatenate((Y, t[:, np.newaxis]), axis=1)
        X = np.concatenate((X, t[:, np.newaxis]), axis=1)

        return Y, X, Zs, Vinv

    return _cov

def cov_double_time_trend(time_outer=1, time_inner=1, cost_index=0):
    """
    Covariance function to account for double time trends. The inner
    time column is reset every time the outer time column resets.
    The cost at `cost_index` is assumed to represent some form
    of time. Every `time_outer`, the level of the outer time trend is 
    increased by one unit. Every `time_inner`, the level of the inner
    time trend is increased by one unit, being reset whenever the outer
    time trend changes.

    For example, if outer time is 2 and inner time is 1 we have runs with
    cumulative cost [0, 1, 2, 3, 4, 5], the added outer time column
    will be [-1, -1, 0, 0, 1, 1], and inner time column 
    [-1, 1, -1, 1, -1, 1].

    Parameters
    ----------
    time_outer : float
        Every `time_outer` in cumulative cost, the first time
        column progresses one equidistant step.
    time_inner : float
        Every `time_inner` in cumulative cost, the second time
        column progresses one equidistant step. It is reset
        whenever `time_outer` is adjusted.
    cost_index : int
        The index in the multi-cost objective to look at.
    
    Returns
    -------
    cov : func(Y, X, Zs, Vinv, costs)
        The covariance function.
    """
    # Define the covariance function
    def _cov(Y, X, Zs, Vinv, costs, random=False):
        if random:
            # Define random outer and inner array
            t_outer = np.random.rand(Y.shape[0]) * 2 - 1
            t_inner = np.random.rand(Y.shape[0]) * 2 - 1
        else:
            # Define outer time array
            cum_cost = np.cumsum(costs[cost_index][0])
            t_outer = np.floor_divide(cum_cost, time_outer)
            if cum_cost[-1] > time_outer:
                t_outer = t_outer / t_outer[-1] * 2 - 1

            # Define inner time array
            t_inner = np.mod(cum_cost, time_outer)
            t_inner = np.floor_divide(t_inner, time_inner)
            t_inner = t_inner / (np.floor_divide(time_outer, time_inner)-1) * 2 - 1

        # Concatenate time array
        Y = np.concatenate((Y, t_outer[:, np.newaxis], t_inner[:, np.newaxis]), axis=1)
        X = np.concatenate((X, t_outer[:, np.newaxis], t_inner[:, np.newaxis]), axis=1)

        return Y, X, Zs, Vinv

    return _cov

def cov_block(cost=1, ratios=1., cost_index=0):
    """
    Covariance function to add a blocking factor to the
    system every `cost` in the cumulative cost. This
    is mostly used when cost is time related.

    For example, if cost = 2 and the cumulative cost is
    [0, 1, 2, 3, 4, 5], the added blocking groups are
    [0, 0, 1, 1, 2, 2].

    Parameters
    ----------
    cost : float
        Every `cost` in cumulative cost, a new block
        is started.
    ratio : float or np.array(1d)
        The ratios for each of the Vinv (in a Bayesian approach).
        In case the ratio is a float, it is broadcasted accordingly.
    cost_index : int
        The index in the multi-cost objective to look at.

    Returns
    -------
    cov : func(Y, X, Zs, Vinv, costs)
        The covariance function.
    """
    # Convert number to array
    if not isinstance(ratios, np.ndarray):
        ratios = np.array([ratios], dtype=np.float64)
    
    # Expand array dimensions
    ratios = ratios[:, np.newaxis]
    
    # Define the covariance function
    def _cov(Y, X, Zs, Vinv, costs, random=False):
        # Define blocking update
        if random:
            pass
        else:
            # Define blocks and ratios
            cum_cost = np.cumsum(costs[cost_index][0])
            blocks = np.floor_divide(cum_cost, cost).astype(np.int_)
            
            # Update Zs, Vinv
            Zs = list(Zs)
            Zs.append(blocks)
            Vinv = _update_woodbury(np.copy(Vinv), [blocks], ratios)

        return Y, X, Zs, Vinv

    return _cov

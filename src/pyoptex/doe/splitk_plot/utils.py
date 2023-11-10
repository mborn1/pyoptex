import numpy as np
import numba
from numba.typed import List
from collections import namedtuple
from ..constraints import no_constraints

FunctionSet = namedtuple('FunctionSet', 'metric constraints')
Parameters = namedtuple('Parameters', 'fn effect_types effect_levels grps plot_sizes ratios coords colstart c alphas thetas thetas_inv Vinv Y2X')
Update = namedtuple('Update', 'level grp runs cols new_coord old_coord old_metric U D')
State = namedtuple('State', 'Y X metric')

def obs_var_Zs(plot_sizes):
    """
    Compute the Zs from the plot_sizes for the splitk plot designs.

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The sizes of each plot. e.g. [3, 4] is a split-plot design
        with 4 plots and 3 runs per plot.
    
    Returns
    -------
    Zs : tuple(np.array(1d))
        A tuple of grouping matrices.
    """
    # Initialize alphas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]

    # Compute (regular) groupings
    Zs = tuple([np.repeat(np.arange(alpha), int(alphas[0] / alpha)) for alpha in alphas[1:]])
    return Zs

@numba.njit
def obs_var(plot_sizes, ratios=None):
    """
    Compute the observation covariance matrix from the plot sizes and
    ratios.

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The sizes of each plot. e.g. [3, 4] is a split-plot design
        with 4 plots and 3 runs per plot.
    ratios : np.array(1d) or None
        The ratios for each of the random effects. This is of size plot_sizes.size - 1.
    
    Returns
    -------
    V : np.array(2d)
        The observation covaraince matrix.
    """
    # Initialize alphas and thetas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]
    thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    if ratios is None:
        ratios = np.ones_like(plot_sizes[1:], dtype=np.float64)

    # Compute variance-covariance of observations
    V = np.eye(alphas[0])
    for i in range(ratios.size):
        Zi = np.kron(np.eye(alphas[i+1]), np.ones((thetas[i+1], 1)))
        V += ratios[i] * Zi @ Zi.T

    return V

################################################

# TODO
def level_grps(s0, s1):
    """
    Determine which groups should be updated per level
    considering the old plot sizes and the new (after augmentation).

    Parameters
    ----------
    s0 : np.array
        The initial plot sizes
    s1 : np.array
        The new plot sizes

    Returns
    -------
    grps : list(np.array)
        A list of numpy arrays indicating which groups should
        be updated per level. E.g. grps[0] indicates all level-zero
        groups that should be updated.
    """
    # Initialize groups
    grps = List()
    grps.append(np.arange(s0[-1], s1[-1]))

    for k, i in enumerate(range(s1.size - 2, -1, -1)):
        # Indices from current level
        g0 = np.arange(s0[i], s1[i])
        for j in range(i+1, s1.size):
            g0 = (np.expand_dims(np.arange(s0[j]) * np.prod(s1[i:j]), 1) + g0).flatten()

        # All indices from added runs in higher levels
        g1 = (np.expand_dims(grps[k] * s1[i], 1) + np.arange(s1[i])).flatten()

        # Concatenate both and save
        g = np.concatenate((g0, g1))
        grps.append(g)
    
    # Return reverse of groups
    return grps[::-1]

def extend_design(Y, plot_sizes, new_plot_sizes, factors):
    """
    Extend an existing design Y with initial plot sizes (`plot_sizes`) to
    a new design. This function only extends the existing design by adding new
    runs in the correct positions and forcing the correct level factors.

    It does not perform any optimization or initialization of the new runs

    Parameters
    ----------
    Y : np.array
        The initial design. If all initial plot sizes are zero, a new design is
        created with all zeros.
    plot_sizes : np.array
        The initial plot sizes of the design
    new_plot_sizes : np.array
        The new plot sizes after augmentation
    factors : np.array 
        The main terms of the design with their level

    Returns
    -------
    Yext : np.array 
        The extended (feasible) design.
    """
    # Return full matrix if all zeros
    if np.all(plot_sizes) == 0:
        return np.zeros((np.prod(new_plot_sizes), factors.shape[0]), dtype=np.float64)

    # Difference in plot sizes
    plot_sizes_diff = new_plot_sizes - plot_sizes
    thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))

    # Add new runs in the correct places
    new_runs = list()
    for i in range(new_plot_sizes.size):
        g = np.repeat(np.arange(thetas[i+1], thetas[-1] + thetas[i+1], thetas[i+1]), np.prod(new_plot_sizes[:i]) * plot_sizes_diff[i])
        new_runs.extend(g)
    Y = np.insert(Y, new_runs, 0, axis=0)

    # Compute new alphas and thetas
    nthetas = np.cumprod(np.concatenate((np.array([1]), new_plot_sizes)))
    nalphas = np.cumprod(new_plot_sizes[::-1])[::-1]

    # Fix the levels 
    for col in range(factors.shape[0]):
        level = factors[col, 0]
        if level != 0:
            size = nthetas[level]
            for grp in range(nalphas[level]):
                Y[grp*size:(grp+1)*size, col] = Y[grp*size, col]
    
    return Y

def terms_per_level(factors, model):
    """
    Compute the amount of coefficients to be estimated per split-level.
    """
    # Initialize
    max_split_level = np.max(factors[:, 0])
    split_levels = np.zeros(max_split_level+1, np.int64)

    # Compute amount of terms with only factors higher or equal to current split-level
    for i in range(max_split_level + 1):
        split_factors = factors[:, 0] >= i
        nterms_in_level = np.all(model[:, ~split_factors] == 0, axis=1) & np.any(model[:, split_factors] != 0, axis=1)
        split_levels[i] = np.sum(nterms_in_level)

    # Adjust to account for terms already counted in higher levels
    split_levels[:-1] -= split_levels[1:]

    return split_levels

def min_split_levels(split_levels):
    """
    Compute the minimum amount of split levels from an
    array containing the amount of terms (or coefficients to be estimated)
    per split-level

    Parameters
    ----------
    split_levels : np.array
        The result of :py:func:`terms_per_level`

    Returns
    -------
    req : np.array
        The absolute minimum amount of runs per shift-level
    """
    req = np.zeros_like(split_levels)

    req[-1] = (split_levels[-1] + 1) + 1
    for i in range(2, split_levels.shape[0] + 1):
        req[-i] = np.ceil((split_levels[-i] + 1) / np.prod(req[-i+1:]) + 1)

    return req

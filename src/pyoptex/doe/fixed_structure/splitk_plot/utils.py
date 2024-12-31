"""
Module for all utility functions of the split^k-plot algorithm
"""

from collections import namedtuple

import numba
import numpy as np
import pandas as pd

from ..utils import Parameters as Parameterso, RandomEffect as RandomEffect

Parameters = namedtuple('Parameters', ' '.join(Parameterso._fields) + ' plot_sizes c alphas thetas thetas_inv compute_update')
Update = namedtuple('Update', 'level grp runs cols new_coord old_coord Xi_old old_metric')

__Plot__ = namedtuple('__Plot__', 'level size ratio', defaults=(0, 1, 1))
class Plot(__Plot__):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        self = super(Plot, cls).__new__(cls, *args, **kwargs)
        assert self.level >= 0, f'Plot levels must be larger than or equal to zero, but is {self.level}'
        assert self.size > 0, f'Plot sizes must be larger than zero, but is {self.size}'
        if isinstance(self.ratio, tuple) or isinstance(self.ratio, list) or isinstance(self.ratio, np.ndarray):
            assert all(r >= 0 for r in self.ratio), f'Variance ratios must be larger than or equal to zero, but is {self.ratio}'
        else:
            assert self.ratio >= 0, f'Variance ratios must be larger than or equal to zero, but is {self.ratio}'
        return self

################################################

def obs_var_Zs(plot_sizes):
    """
    Create the grouping matrices (1D array) for each plot according
    to the provided plot sizes.

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The sizes of each plot. e.g. [3, 4] is a split-plot design
        with 4 plots and 3 runs per plot.

    Returns
    -------
    Zs : tuple(np.array(1d) or None)
        A tuple of grouping matrices for each plot.
    """
    # Initialize alphas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]

    # Compute (regular) groupings
    Zs = tuple([np.repeat(np.arange(alpha), int(alphas[0] / alpha)) for alpha in alphas[1:]])
    return Zs

@numba.njit
def obs_var(plot_sizes, ratios=None):
    """
    Directly computes the observation matrix from the design. Is similar to
    :py:func:`obs_var_Zs <pyoptex.doe.fixed_structure.splitk_plot.utils.obs_var_Zs>` 
    followed by :py:func:`obs_var_from_Zs <pyoptex.utils.design.obs_var_from_Zs>`.

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The sizes of each plot. e.g. [3, 4] is a split-plot design
        with 4 plots and 3 runs per plot.
    ratios : np.array(1d)
        The variance ratios of the different groups compared to the variance of 
        the random errors.

    Returns
    -------
    V : np.array(2d)
        The observation covariance matrix.
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

def level_grps(s0, s1):
    """
    Determines which groups should be updated per level
    considering the old plot sizes and the new (after augmentation).

    Parameters
    ----------
    s0 : np.array(1d)
        The initial plot sizes
    s1 : np.array(1d)
        The new plot sizes

    Returns
    -------
    grps : list(np.array(1d))
        A list of numpy arrays indicating which groups should
        be updated per level. E.g. grps[0] indicates all level-zero
        groups that should be updated.
    """
    # Initialize groups
    grps = []
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

def extend_design(Y, plot_sizes, new_plot_sizes, effect_levels):
    """
    Extend an existing design Y with initial plot sizes (`plot_sizes`) to
    a new design with `new_plot_sizes`. This function only extends the 
    existing design by adding new runs in the correct positions and forcing 
    the correct factor levels where necessary. It does not perform any 
    optimization or initialization of the new runs.

    Parameters
    ----------
    Y : np.array(2d)
        The initial design. If all initial plot sizes are zero, 
        a new design is created with all zeros.
    plot_sizes : np.array(1d)
        The initial plot sizes of the design.
    new_plot_sizes : np.array(1d)
        The new plot sizes after augmentation.
    effect_levels : np.array(1d)
        The plot level of each factor.

    Returns
    -------
    Yext : np.array(2d)
        The extended design.
    """
    # Return full matrix if all zeros
    if np.all(plot_sizes) == 0:
        return np.zeros((np.prod(new_plot_sizes), effect_levels.size), dtype=np.float64)

    # Difference in plot sizes
    plot_sizes_diff = new_plot_sizes - plot_sizes
    thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))

    # Add new runs in the correct places
    new_runs = list()
    for i in range(new_plot_sizes.size):
        g = np.repeat(
            np.arange(thetas[i+1], thetas[-1] + thetas[i+1], thetas[i+1]), 
            np.prod(new_plot_sizes[:i]) * plot_sizes_diff[i]
        )
        new_runs.extend(g)
    Y = np.insert(Y, new_runs, 0, axis=0)

    # Compute new alphas and thetas
    nthetas = np.cumprod(np.concatenate((np.array([1]), new_plot_sizes)))
    nalphas = np.cumprod(new_plot_sizes[::-1])[::-1]

    # Fix the levels 
    for col in range(effect_levels.size):
        level = effect_levels[col]
        if level != 0:
            size = nthetas[level]
            for grp in range(nalphas[level]):
                Y[grp*size:(grp+1)*size, col] = Y[grp*size, col]
    
    return Y

def terms_per_plot_level(factors, model):
    """
    Computes the amount of coefficients to be estimated per plot level.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.doe.fixed_structure.utils.Factor>`)
        The factors of the design.
    model : pd.DataFrame
        The model dataframe

    Returns
    -------
    nb_terms_per_level : np.array(1d)
        The number of terms for each plot level. E.g., element
        zero is the numbe of terms for the easy-to-change degrees
        of freedom.
    """
    # Reorder model
    assert isinstance(model, pd.DataFrame), 'The model must be a dataframe'
    col_names = [str(f.name) for f in factors]
    model = model[col_names].to_numpy()

    # Initialize
    plot_levels = np.array([f.re.level for f in factors])
    max_split_level = np.max(plot_levels)
    split_levels = np.zeros(max_split_level+1, np.int_)

    # Compute amount of terms with only factors higher or equal to current split-level
    for i in range(max_split_level + 1):
        split_factors = plot_levels >= i
        nterms_in_level = np.all(model[:, ~split_factors] == 0, axis=1) & np.any(model[:, split_factors] != 0, axis=1)
        split_levels[i] = np.sum(nterms_in_level)

    # Adjust to account for terms already counted in higher levels
    split_levels[:-1] -= split_levels[1:]

    # Add the intercept
    split_levels[-1] += 1

    return split_levels

def min_plot_levels(tppl):
    """
    Computes the required number of degrees of freedom
    at each plot level in order to estimate all fixed
    effects and variances of the random effects.

    Parameters
    ----------
    tppl : np.array(1d)
        The number of terms per plot level. Is the result
        of calling 
        :py:func:`terms_per_plot_level <pyoptex.doe.fixed_structure.splitk_plot.utils.terms_per_plot_level>`.

    Returns
    -------
    req : np.array(1d)
        The absolute minimum sized split^k-plot design to fit
        all fixed effects and estimate all variances of the random
        effects.
    """
    req = np.zeros_like(tppl)

    req[-1] = (tppl[-1] + 1)
    for i in range(2, tppl.shape[0] + 1):
        req[-i] = np.ceil((tppl[-i] + 1) / np.prod(req[-i+1:]) + 1)

    return req

def validate_plot_sizes(factors, model):
    """
    Validates that this configuration of split-plot sizes
    can estimate all fixed effects and variances of the random
    effects.

    Parameters
    ----------
    factors : list(:py:class:`Factor pyoptex.doe.fixed_structure.utils.Factor`)
        The factors of the design.
    model : pd.DataFrame
        The model dataframe
    """
    # Compute plot sizes
    nb_plots = max(f.re.level for f in factors) + 1
    plot_sizes = np.zeros(nb_plots, dtype=np.int_)
    for f in factors:
        plot_sizes[f.re.level] = f.re.size

    # Compute the terms per level
    tppl = terms_per_plot_level(factors, model)

    # Compute and check the requirements per level
    req = np.zeros_like(tppl)
    req[-1] = (tppl[-1] + 1)
    for i in range(2, tppl.shape[0] + 1):
        req[-i] = np.ceil((tppl[-i] + 1) / np.prod(plot_sizes[-i+1:]) + 1)

    # Validate they are above minimum
    min_levels = min_plot_levels(terms_per_plot_level(factors, model))
    assert np.all(plot_sizes >= req), f'The minimum sized split^k-plot design has sizes {min_levels}'

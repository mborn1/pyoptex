import pandas as pd
import numpy as np
import statsmodels.api as sm
from functools import cached_property
from statsmodels.regression.mixed_linear_model import VCSpec
from sklearn.metrics import r2_score

from ..doe.utils.model import encode_model
from ..doe.utils.design import obs_var_from_Zs

def identity(Y):
    """
    The identity function.

    Parameters
    ----------
    Y : np.array
        The input

    Returns
    -------
    Y : np.array
        The input returned
    """
    return Y

def order_dependencies(model, factors):
    """
    Create a dependency matrix from a model where
    interactions and higher order effects depend
    on their components and lower order effects.

    For example:
    * :math:`x_0`: depends only on the intercept.
    * :math:`x_0^2`: depends on :math:`x_0`, which in turn depends on the intercept.
    * :math:`x_0 x_1`: depends on both :math:`x_0` and :math:`x_1`, which both depend on the intercept.
    * :math:`x_0^2 x_1` : depends on both :math:`x_0^2` and :math:`x_1`, which depend on :math:`x_0` and the intercept.

    Parameters
    ----------
    model : pd.DataFrame
        The model
    factors : list(:py:class:`Cost_optimal factor <pyoptex.doe.cost_optimal.utils.Factor>` or :py:class:`Splitk_plot factor <pyoptex.doe.splitk_plot.utils.Factor>`)
        The list of factors in the design.

    Returns
    -------
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model. Term i depends on term j
        if dep(i, j) = true.
    """
    # Validation
    assert isinstance(model, pd.DataFrame), 'Model must be a dataframe'
    assert np.all(model >= 0), 'All powers must be larger than zero'

    col_names = [str(f.name) for f in factors]
    assert all(col in col_names for col in model.columns), 'Not all model parameters are factors'
    assert all(col in model.columns for col in col_names), 'Not all factors are in the model'

    # Extract factor parameters
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])

    # Detect model in correct order
    model = model[col_names].to_numpy()

    # Encode model
    modelenc = encode_model(model, effect_types)

    # Compute the possible dependencies
    eye = np.expand_dims(np.eye(modelenc.shape[1]), 1)
    model = np.expand_dims(modelenc, 0)
    all_dep = model - eye # all_dep[:, i] are all possible dependencies for term i

    # Valid dependencies
    all_dep_valid = np.where(np.all(all_dep >= 0, axis=2))
    from_terms = all_dep_valid[1]

    # Extract the dependent terms
    to_terms = np.argmax(np.all(
        np.expand_dims(modelenc, 0) == np.expand_dims(all_dep[all_dep_valid], 1), 
        axis=2
    ), axis=1)

    # Compute dependencies
    dep = np.zeros((modelenc.shape[0], modelenc.shape[0]), dtype=np.bool_)
    dep[from_terms, to_terms] = True

    return dep

def r2adj(fit):
    if fit.k_fe < len(fit.params):
        # Extract Xnumber of observations
        nobs = len(fit.model.exog)

        # Compute P
        P = np.eye(nobs) - np.ones((nobs, nobs)) / nobs

        # Fit intercept model
        fit0 = sm.MixedLM(
            fit.model.endog, np.ones((nobs, 1)), fit.model.groups,
            fit.model.exog_re, fit.model.exog_vc, fit.model.use_sqrt
        ).fit()

        # Extract the groups
        vc_mats = fit.model.exog_vc.mats
        Zs = np.stack([np.argmax(vc_mats[i][0], axis=1) for i in range(len(vc_mats))]).T

        # Compute intercept semi-variance
        V0 = obs_var_from_Zs(Zs, nobs, fit0.params[fit.k_fe:]) * fit0.scale
        rss0 = np.sum(V0 * P.T) # = np.trace(V0 @ P)

        # Compute model semi-variance
        V1 = obs_var_from_Zs(Zs, nobs, fit.params[fit.k_fe:]) * fit.scale
        rss = np.sum(V1 * P.T)

        # Compute adjusted R2
        r2a = 1 - rss / rss0
    else:

        # Attribute already exists for OLS
        r2a = fit.rsquared_adj

    return r2a

def fit_ols(X, y):
    fit = sm.OLS(y, X).fit()
    fit.k_fe = len(fit.params)
    fit.vcomp = np.array([], dtype=np.float64)
    fit.converged = True
    return fit

def fit_mixedlm(X, y, groups):
    # Retrieve dummy encoding for each group
    dummies = [pd.get_dummies(group) for group in groups]

    # Create the mixed lm spec
    exog_vc = VCSpec(
        [f'g{i}' for i in range(len(groups))],
        [[[f'g{i}[{col}]' for col in dummy.columns]] for i, dummy in enumerate(dummies)],
        [[dummy.to_numpy()] for dummy in dummies]
    )

    # Fit the model
    fit = sm.MixedLM(y, X, np.ones(len(X)), exog_vc=exog_vc).fit()

    # Add additional values
    fit.rsquared = cached_property(
        lambda self: r2_score(self.model.endog, self.predict(self.model.exog))
    )
    fit.rsquared_adj = cached_property(
        lambda self: r2adj(self)
    )

    return fit


from functools import cached_property

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.regression.mixed_linear_model import VCSpec

from ...utils.design import obs_var_from_Zs

def r2adj(fit):
    """
    Computes the adjust r-square from a model fit.
    When fit is from an OLS, the r2adj is already precomputed
    as rsquared_adj. When fit is from a mixed LM, the r2adj
    is based on the average estimated semi-variance as denoted
    in `Piepho (2019) <https://pubmed.ncbi.nlm.nih.gov/30957911/>`_.

    .. math::

        R_{adj}^2 &= 1 - \\frac{trace(V1 P)}{trace(V0 P)} \\\\
        P &= I_N - \\frac{1}{n} 1_N 1_N^T 

    N is the number of runs.
    :math:`V_0` and :math:`V_1` are the observation covariance matrices
    of the model fitted with only the intercept and the complete model.
    :math:`I_N` is the identity matrix of size (N, N). :math:`1_N` is 
    a vector of ones of size N.

    Parameters
    ----------
    fit : :py:class:`statsmodels.regression.linear_model.RegressionResults` or :py:class:`statsmodels.regression.mixed_linear_model.MixedLMResults`
        The results after fitting a statsmodels OLS or mixed LM.

    Returns
    -------
    r2adj : float
        The adjust r-squared of the results.
    """
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
    """
    Wrapper to fit a statsmodels OLS model. It includes
    the following additional attributes required for a 
    more synchronized output between OLS and Mixed LM.

    * **k_fe**: The number of parameters.
    * **vcomp**: An empty array.
    * **converged**: True. 

    Parameters
    ----------
    X : np.array(2d)
        The normalized, encoded model matrix of the data.
    y : np.array(1d)
        The output variable
    
    Returns
    -------
    fit : :py:class:`statsmodels.regression.linear_model.RegressionResults`
        The statsmodels regression results with some additional
        attributes.
    """
    fit = sm.OLS(y, X).fit()
    fit.k_fe = len(fit.params)
    fit.vcomp = np.array([], dtype=np.float64)
    fit.converged = True
    return fit

def fit_mixedlm(X, y, groups):
    """
    Wrapper to fit a statsmodels Mixed LM model. It includes
    the following additional attributes required for a 
    more synchronized output between OLS and Mixed LM.

    * **rsquared** : The simple r2_score of the fit.
    * **rsquared_adj** : The adjust r2 score according to
      :py:func:`r2_adj <pyoptex.analysis.utils.r2_adj>`.

    Parameters
    ----------
    X : np.array(2d)
        The normalized, encoded model matrix of the data.
    y : np.array(1d)
        The output variable
    groups : np.array(2d)
        The 2d array of the groups for the random effects
        of size (g, N), with g the number of random effects
        and N the number of runs. Each row is an integer
        array with values from 0 until the total number of groups
        for that random effect. For example [0, 0, 1, 1] indicates
        that the first two runs are correlated, and the last two.
    
    Returns
    -------
    fit : :py:class:`statsmodels.regression.mixed_linear_model.MixedLMResults`
        The statsmodels Mixed LM results with some additional
        attributes.
    """
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

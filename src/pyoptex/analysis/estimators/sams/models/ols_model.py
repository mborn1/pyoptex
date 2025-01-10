"""
Module for the SAMS OLS modelling.
"""

import numpy as np
import statsmodels.api as sm

from .model import Model, ModelResults


class OlsModel(Model):
    """
    A default OLS model for use with the SAMS algorithm which
    extends the 
    :py:class:`Model <pyoptex.analysis.estimators.sams.models.model.Model>`
    interface.

    Attributes
    ----------
    X : np.array(2d)
        The encoded, normalized model matrix of the data
    y : np.array(1d)
        The output variable.
    forced : np.array(1d)
        Any terms that must be included in the model.
    mode : None or 'weak' or 'strong'
        The heredity model during sampling.
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    ss_intercept : float
        The sum of squared residuals for a model with only the intercept.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the OLS model

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The output variable.
        forced : np.array(1d)
            Any terms that must be included in the model.
        mode : None or 'weak' or 'strong'
            The heredity model during sampling.
        dep : np.array(2d)
            The dependency matrix of size (N, N) with N the number
            of terms in the encoded model (output from Y2X). Term i depends on term j
            if dep(i, j) = true.
        """
        super().__init__(*args, **kwargs)
        self.ss_intercept = np.var(self.y) * len(self.y)

    def _fit(self, X, y):
        """
        Internal fit function based on X and y data.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix with specific
            selected terms.
        y : np.array(1d)
            The output variable.
        
        Returns
        -------
        params : np.array(1d)
            The coefficients of the linear regression
        r2adj : float
            The adjusted coefficient of determination.
        mse_resid : float
            The sum of squared residuals divided by the degrees
            of freedom (= X.shape[0] - X.shape[1]).
        """
        # Fit OLS (performance in numpy with fallback for nearly singular designs)
        params, se, n, _ = np.linalg.lstsq(X, y, rcond=None)

        # Check for rank deficiency
        if n < X.shape[1]:
            # Fit with statsmodels. Is slower, but more accurate.
            ols = sm.OLS(y, X).fit()
            params = ols.params
            mse_resid = ols.mse_resid

        else:
            # Compute results
            mse_resid = (se / (X.shape[0] - X.shape[1]))
        
        # Compute the adjusted R2
        r2adj = 1 - mse_resid / (self.ss_intercept)

        return params, r2adj, mse_resid

    def fit(self, model):
        """
        Fits an OLS model

        Parameters
        ----------
        model : np.array(1d)
            The current model terms.

        Returns
        -------
        fit : :py:class:`ModelResults <pyoptex.analysis.estimators.sams.models.ModelResults>`
            An object of type model results containing the optimization
            metric and the estimated coefficients.
        """
        # Create the exog matrix
        X = self.X[:, model]

        # Drop rows with nan values
        complete = ~np.any(np.isnan(X), axis=1)

        params, r2adj, _ = self._fit(X[complete], self.y[complete])
        return ModelResults(r2adj, params)

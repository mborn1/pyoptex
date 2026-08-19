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
        n, k = X.shape
        # Fit OLS (performance in numpy with fallback for nearly singular designs)
        params, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)

        # Check for rank deficiency
        if rank < k:
            # Fit with statsmodels. Is slower, but more accurate.
            ols = sm.OLS(y, X).fit()
            params = ols.params
            RSS = ols.ssr
            df_resid = int(ols.df_resid)
        else:
            RSS = residuals[0]
            df_resid = n - k

        mse_resid = RSS / df_resid

        # Compute adjusted R2
        r2 = 1.0 - RSS / self.ss_intercept
        r2adj = 1.0 - (1.0 - r2) * (n - 1) / df_resid

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

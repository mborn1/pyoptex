"""
Module for the SAMS MixedLM modelling.
"""
import numpy as np
from scipy import linalg

from .ols_model import OlsModel


def Vsqrt_inv(V):
    """
    Compute the :math:`\\sqrt{V^{-1}}` by means of an eigenvalue
    expansion.

    Parameters
    ----------
    V : np.array(2d)
        The matrix on which the operator applies.
    
    Returns
    -------
    Vsqrt_inv : np.array(2d)
        The result of the operator.
    """
    D, E = linalg.eigh(V)
    D[D < 0] = 0
    return E @ np.diag(1 / np.sqrt(D)) @ E.T


class MixedLMModel(OlsModel):
    """
    A Mixed linear model for use with the SAMS algorithm which
    extends the 
    :py:class:`OLSModel <pyoptex.analysis.estimators.sams.models.ols_model.OLSModel>`.

    The data (X) and output variable (y) are adjusted to an OLS model
    based on the estimation of the observation covariance matrix specified
    by the groups and ratios.

    If we assume that y is normally distributed with constant covariance
    matrix :math:`V`

    .. math::

        y \\sim \\mathcal{N}(X \\beta, \\sigma_\\epsilon^2 V)

    we can note that

    .. math::

        V^{-1/2} y \\sim \\mathcal{N}(V^{-1/2} X \\beta, \\sigma_\\epsilon^2 I_N)


    .. note::
        This assumes for computational purposes that the observation covariance
        matrix is accurately specified using the provided ratios. Once a model has
        been obtained, use REML to estimate the true variance ratios.
        Potentially rerun the SAMS algorithm with the updated variance ratios to
        validate that the same model is selected.

    Attributes
    ----------
    X : np.array(2d)
        The encoded, normalized model matrix of the data. This data is
        corrected for the mixedlm approximation.
    y : np.array(1d)
        The output variable. This data is
        corrected for the mixedlm approximation.
    forced : np.array(1d)
        Any terms that must be included in the model.
    mode : None or 'weak' or 'strong'
        The heredity model during sampling.
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    ss_intercept : float
        The sum of squared residuals for a model with only the intercept. Recomputed
        to account for the mixedlm approximation.
    """

    def __init__(self, *args, V=None, **kwargs):
        """
        Creates the MixedLM model for SAMS.

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
        V : np.array(2d)
            The observation variance-covariance matrix.
        """
        super().__init__(*args, **kwargs)

        # Check if there is an observation covariance matrix
        if V is not None:
            # Compute observation variance-covariance
            sqrtV_inverted = Vsqrt_inv(V)

            # Adjust data and output variable
            self.y = np.squeeze(sqrtV_inverted @ np.expand_dims(self.y, 1))
            self.X = sqrtV_inverted @ self.X

            # Compute the intercept variance
            intercept = sqrtV_inverted @ np.ones((len(self.X), 1))
            self.ss_intercept = np.var(self.y) * len(self.y)

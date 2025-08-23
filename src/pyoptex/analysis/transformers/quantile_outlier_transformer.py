import scipy.stats as spstats
import numpy as np

from ..mixins.fit_mixin import OutlierTransformerMixin
from ...utils.model import identityY2X

class QuantileOutliersTransformer(OutlierTransformerMixin):
    """
    Transformer using the Quantile-quantile plot method to detect outliers based on a threshold
    from the ideal value. Drops the terms one-by-one based on the largest deviation as
    long as the value is above the threshold.

    The :py:func:`fit_transform <pyoptex.analysis.transformers.quantile_outlier_transformer.QuantileOutliersTransformer.fit_transform>` 
    function fits the data and removes the detected outliers. During regular transform, nothing happens as this
    should only remove training outliers.

    .. note::
        It is extended by :py:class:`OutlierTransformerMixin <pyoptex.analysis.mixins.fit_mixin.OutlierTransformerMixin>`.

    Attributes
    ----------
    threshold : float
        The threshold for dropping terms on the deviation from the quantile line.
    stat : str
        The distribution to use for the quantile-quantile plot.
    errors_ : np.ndarray(1d)
        The errors (pred - y) for a simple model fit.
    outliers_ : np.ndarray(1d)
        A boolean array marking which rows are considered
        outliers in the training dataset.
    """
    def __init__(self, factors=(), Y2X=identityY2X, random_effects=(), 
                 threshold=1, stat='norm'):
        """
        Creates the outlier transformer

        Parameters
        ----------
        factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
            A list of factors to be used during fitting. It contains
            the categorical encoding, continuous normalization, etc.
        Y2X : func(Y)
            The function to transform a design matrix Y to a model matrix X.
        random_effects : list(str)
            The names of any random effect columns. Every random effect
            is interpreted as a string column and encoded using 
            effect encoding.
        threshold : float
            The threshold for dropping terms on the deviation from the quantile line.
        stat : str
            The distribution to use for the quantile-quantile plot.
        """
        super().__init__(factors, Y2X, random_effects)
        self.threshold = threshold
        self.stat = stat

    def _fit(self, X, y):
        """
        Internal fit function for the quantile outlier transformer.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
        # Define the terms
        self.terms_ = np.arange(X.shape[1])

        # Fit the data
        fit_ = self.fit_fn_(X, y, self.terms_)

        # Store the final results
        self.coef_ = fit_.params[:fit_.k_fe]

        # Fit and compute errors
        pred = self._predict(X)
        self.errors_ = (pred - np.array(y)) * self.y_std_

        # Detect the outliers
        self.outliers_, self.distances_ = self.quantile_outliers(
            self.errors_, self.threshold, stat=getattr(spstats, self.stat)
        )

        # Return self
        return self
    
    def _apply_transform(self, X, y):
        """
        Internal transform function for the quantile outlier transformer.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series or np.array(1d)
            The output variable

        Returns
        -------
        X : pd.DataFrame
            The transformed data
        y : pd.Series or np.array(1d)
            The transformed output variable
        """
        # Remove the outliers
        X = X.loc[~self.outliers_]
        y = y[~self.outliers_]
        return X, y
    
    def _predict(self, X):
        """
        Internal predict function based on the
        encoded, normalized model matrix of the data.
        It applies the simple linear regression formula.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.

        Returns
        -------
        pred : np.array(1d)
            The predictions.
        """
        # Predict based on linear regression
        return np.sum(X[:, self.terms_] * np.expand_dims(self.coef_, 0), axis=1)

    def quantile_outliers(self, errors, threshold, stat=spstats.norm):
        """
        Quantile-quantile outlier detector based on the desired distribution
        and a threshold value. This is usually performed on the regression errors.

        Parameters
        ----------
        errors : np.ndarray(1d)
            The data which should follow the specified distribution.
        threshold : float
            The threshold to mark outliers.
        stat : `scipy.stats.rv_continuous`
            A scipy distribution object.

        Returns
        -------
        outliers : np.ndarray(1d)
            A boolean array marking which elements are considered outliers.
        """
        # Find indices
        idx = np.argsort(errors)

        # Compute initial difference from quantile line
        se0 = errors[idx]
        se0 = (se0 - np.mean(se0)) / np.std(se0)
        ppf = np.linspace(0, 1, len(se0) + 2)[1:-1]
        te0 = stat.ppf(ppf)
        d0 = np.abs(se0 - te0)

        # Start the distances 
        distances = np.copy(d0)

        # Outlier mask
        outliers = np.zeros(len(d0), dtype=np.bool_)

        # Drop them one-by-one
        while np.any(d0 > threshold):
            # Mark as an outlier
            outliers[np.argmax(d0)] = True

            # Break when only one sample is left
            if np.sum(~outliers) == 1:
                break

            # Recompute distances
            se1 = se0[~outliers]
            se1 = (se1 - np.mean(se1)) / np.std(se1)
            ppf = np.linspace(0, 1, se1.size + 2)[1:-1]
            te1 = stat.ppf(ppf)
            d1 = np.abs(se1 - te1)
            distances[~outliers] = d1

            # Set the distances on the correct positions
            d0 = np.zeros(outliers.size)
            d0[~outliers] = d1

        # Return outliers according to the original array
        return outliers[np.argsort(idx)], distances[np.argsort(idx)]

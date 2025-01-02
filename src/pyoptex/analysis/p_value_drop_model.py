import numpy as np
from sklearn.base import BaseEstimator

from .mixins.fit_mixin import RegressionMixin
from .mixins.conditional_mixin import ConditionalRegressionMixin
from .utils import identity


class PValueDropRegressor(ConditionalRegressionMixin, RegressionMixin, BaseEstimator):
    """
    A regressor implementing the
    :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>` and
    :py:class:`ConditionalRegressionMixin <pyoptex.analysis.mixins.conditional_mixin.ConditionalRegressionMixin>`
    interfaces.

    Permits to fit a simple model provided in Y2X with optionally random effects. Model selection
    from the model matrix is done by dropping terms one-by-one based on the highest p-value,
    higher than the threshold.

    If desired, the model can force weak and strong heredity by setting the mode to
    'weak' or 'strong' respectively, and providing a dependency matrix.

    .. note::
        It also includes all parameters and attributes from 
        :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>` and
        :py:class:`ConditionalRegressionMixin <pyoptex.analysis.mixins.conditional_mixin.ConditionalRegressionMixin>`

    Attributes
    ----------
    threshold : float
        The p-value threshold, any term above is dropped (while adhering to the selected mode).
    dependencies : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    mode : None, 'weak' or 'strong'
        The heredity mode to adhere to.
    """

    def __init__(self, factors=(), Y2X=identity, random_effects=(), 
                 conditional=False, 
                 threshold=0.05, dependencies=None, mode=None):
        """
        P-value based model selection regressor.
        
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
        conditional : bool
            Whether to create a conditional model or not.
        threshold : float
            The p-value threshold, any term above is dropped (while adhering to the selected mode).
        dependencies : np.array(2d)
            The dependency matrix of size (N, N) with N the number
            of terms in the encoded model (output from Y2X). Term i depends on term j
            if dep(i, j) = true.
        mode : None, 'weak' or 'strong'
            The heredity mode to adhere to.
        """
        super().__init__(
            factors=factors, Y2X=Y2X, random_effects=random_effects,
            conditional=conditional
        )
        self.threshold = threshold
        self.dependencies = dependencies
        self.mode = mode

    def _can_drop(self, terms, idx, mode, dependencies):
        """
        Determines if the term specified by at `idx` of `terms` can be dropped,
        given the other existing terms in the model, the mode, and
        the dependency matrix.

        Parameters
        ----------
        terms : np.array(1d)
            The terms in the current model. 
        idx : int
            The index of the term in the current model to be dropped.
        mode : None or 'weak' or 'strong'
            The heredity mode.
        dependencies : np.array(2d)
            The dependency matrix of size (N, N) with N the number
            of terms in the encoded model (output from Y2X). Term i depends on term j
            if dep(i, j) = true.

        Returns
        -------
        can_drop : bool
            Whether this term can be dropped given the dependencies
            and heredity mode.
        """
        # Retrieve the terms depending on this term
        dep_terms = dependencies[:, terms[idx]]

        # Check for the mode
        if mode == 'strong':
            # No dependent terms, otherwise violation of strong heredity
            drop = ~np.any(dep_terms)

        elif mode == 'weak':
            # No single dependent terms left, otherwise violation of weak heredity
            single_deps = np.sum(dependencies[dep_terms][:, terms], axis=1) == 1
            drop = ~np.any(single_deps)
        
        else:
            # No restrictions
            drop = True

        return drop

    def _drop_one_by_one(self, X, y, threshold, mode, dependencies):
        """
        Drops the terms of the model one-by-one, starting from all
        terms in the model matrix X.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        threshold : float
            The p-value threshold, any term above is dropped (while adhering to the selected mode).
        mode : None or 'weak' or 'strong'
            The heredity mode.
        dependencies : np.array(2d)
            The dependency matrix of size (N, N) with N the number
            of terms in the encoded model (output from Y2X). Term i depends on term j
            if dep(i, j) = true.

        Returns
        -------
        terms : np.array(1d)
            An array of indices corresponding to the columns
            which are kept after the model selection.
        """
        # Define the terms to keep
        keep = np.arange(X.shape[1])

        # Fit the model repeatedly and drop terms
        removed = True
        while removed:
            # Fit the data
            fit = self.fit_fn_(X, y, keep)
            pvalues = fit.pvalues[:fit.k_fe]
            sorted_p_idx = np.argsort(pvalues)[::-1]

            # Find the first droppable index
            i = 0
            while i < keep.size \
                    and pvalues[sorted_p_idx[i]] > threshold \
                    and not self._can_drop(keep, sorted_p_idx[i], mode, dependencies):
                i += 1

            # Check for a valid index
            if i < keep.size and pvalues[sorted_p_idx[i]] > threshold:
                keep = np.delete(keep, sorted_p_idx[i])
            else:
                removed = False

        return keep
    
    def _validate_fit(self, X, y):
        """
        Additional validation of the threshold, mode, and
        dependency matrix.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
        # Super validation
        super()._validate_fit(X, y)

        # Validate dependencies and mode
        assert 0 <= self.threshold <= 1, 'Threshold must be in the range [0, 1]'
        assert self.mode in (None, 'weak', 'strong'), 'The drop-mode must be None, weak or strong'
        if self.mode in ('weak', 'strong'):
            assert self.dependencies is not None, 'Must specify dependency matrix if using weak or strong heredity'
            assert len(self.dependencies.shape) == 2, 'Dependencies must be a 2D array'
            assert self.dependencies.shape[0] == self.dependencies.shape[1], 'Dependency matrix must be square'
            assert self.dependencies.shape[0] == X.shape[1], 'Must specify a dependency for each term'

    def _fit(self, X, y):
        """
        Internal fit function to apply the drop-one-by-one algorithm
        based on p-values of the terms.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
        # Drop terms one-by-one based on p-value
        self.terms_ = self._drop_one_by_one(X, y, self.threshold, self.mode, self.dependencies)

        # Fit the resulting model
        self.fit_ = self.fit_fn_(X, y, self.terms_)

        # Store the final results
        self.coef_ = self.fit_.params[:self.fit_.k_fe]
        self.scale_ = self.fit_.scale
        self.vcomp_ = self.fit_.vcomp

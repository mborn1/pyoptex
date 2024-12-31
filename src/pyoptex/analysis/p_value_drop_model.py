import numpy as np
from sklearn.base import BaseEstimator

from .mixins.fit_mixin import RegressionMixin
from .mixins.conditional_mixin import ConditionalRegressionMixin
from .utils import identity


class PValueDropRegressor(ConditionalRegressionMixin, RegressionMixin, BaseEstimator):

    def __init__(self, factors=(), Y2X=identity, random_effects=(), 
                 conditional=False, 
                 threshold=0.05, dependencies=None, mode=None):
        super().__init__(
            factors=factors, Y2X=Y2X, random_effects=random_effects,
            conditional=conditional
        )
        self.threshold = threshold
        self.dependencies = dependencies
        self.mode = mode

    def _can_drop(self, terms, idx, mode, dependencies):
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
        # Drop terms one-by-one based on p-value
        self.terms_ = self._drop_one_by_one(X, y, self.threshold, self.mode, self.dependencies)

        # Fit the resulting model
        self.fit_ = self.fit_fn_(X, y, self.terms_)

        # Store the final results
        self.coef_ = self.fit_.params[:self.fit_.k_fe]
        self.scale_ = self.fit_.scale
        self.vcomp_ = self.fit_.vcomp

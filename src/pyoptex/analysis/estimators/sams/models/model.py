"""
Module for the SAMS base model.
"""

import numpy as np
from collections import namedtuple

from .....utils.model import permitted_dep_add

ModelResults = namedtuple('ModelResults', 'metric params')

class Model:
    """
    Base class of a model specifying sampling and mutation functionalities
    in the SAMS algorithm.

    This class can be extended by overwriting the
    :py:func:`fit <pyoptex.analysis.estimators.sams.model.fit>` function.

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
    """
    def __init__(self, X, y, forced=None, mode='weak', dep=None):
        """
        Initialize of the base model in the SAMS algorithm.

        Parameters
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
        """
        # Validate the inputs
        assert len(X) == len(y), 'Must have the same number of runs for the data as the output variable'
        assert forced is None or np.all(forced < X.shape[1]), 'The forced terms are not in the model matrix, index is too high'
        assert mode in (None, 'weak', 'strong'), 'The mode must be None, weak or strong heredity'
        if mode is not None:
            assert dep is not None, 'Must specify a dependency matrix if the mode is weak or strong heredity'
            assert len(dep.shape) == 2, 'Dependencies must be a 2D array'
            assert dep.shape[0] == dep.shape[1], 'Dependency matrix must be square'
            assert dep.shape[0] == X.shape[1], 'Must specify a dependency for each term'

        # Create default forced
        if forced is None:
            forced = np.array([], dtype=np.int_)

        # Store
        self.X = X 
        self.y = y 
        self.forced = forced
        self.mode = mode
        self.dep = dep

    def _sort(self, model):
        """
        Sort the model to force uniqueness in lexical order.

        .. note::
            This operation happens in-place.

        Parameters
        ----------
        model : np.array(1d)
            The current model

        Returns
        -------
        model : np.array(1d)
            The unique model, by sorting everything but the
            forced terms.
        """
        # Sort in-place (non-forced part)
        model[self.forced.size:].sort()

        # Return the model
        return model

    def _sample(self, model):
        """
        Sample a new term in the model according to the
        heredity mode.

        Parameters
        ----------
        model : np.array(1d)
            The current model.

        Returns
        -------
        term : int
            The index of the sampled term.
        """
        # Find the locations of the options
        valid = permitted_dep_add(model, self.mode, self.dep)
        valid[model] = False
        options = np.flatnonzero(valid)

        # Return a random sample
        return np.random.choice(options)

    def _remove(self, model, idx):
        """
        Remove the specified term, alongside all terms violating the
        heredity constraints.

        Parameters
        ----------
        model : np.array(1d)
            The model from which to remove the term.
        idx : int
            The index in the model to remove.

        Returns
        -------
        idx : np.array(1d)
            A boolean array of size model.size specifying which terms
            are invalid.      
        """
        # Initialize indices to remove
        o = np.zeros(len(model), dtype=np.bool_)
        o[idx] = True

        # Look for indices to remove
        while True:
            # Find indices of the remaining model terms
            i = np.flatnonzero(~o)

            # Check which indices are invalid
            model_terms = model[i]
            invalid = i[~permitted_dep_add(model_terms, self.mode, self.dep, model_terms)]

            # If no indices are updated, break
            if invalid.size == 0:
                break

            # Set those indices to True
            o[invalid] = True

        return o

    def _validate(self, models):
        """
        Validates whether the models are valid according to the
        heredity mode.

        Parameters
        ----------
        models : list(np.array(1d))
            The models to be validated.
        """
        for model in models:
            assert np.all(permitted_dep_add(model, self.mode, self.dep, model))

    def init(self, model):
        """
        Create a random model by sequential sampling.
        The first terms are filled with the forced terms.

        .. note::
            This operation happens in-place.

        Parameters
        ----------
        model : np.array(1d)

        Returns
        -------
        model : np.array(1d)
            A random model represented by the indices
            in the model matrix X.
        """
        # Sample sequentially
        model[:self.forced.size] = self.forced
        for i in range(self.forced.size, len(model)):
            model[i] = self._sample(model[:i])

        # Force uniqueness of model
        return self._sort(model)

    def mutate(self, model):
        """
        Mutate the model by removing atleast one term
        and replacing it.

        .. note::
            This operation happens in-place.

        Parameters
        ----------
        model : np.array(1d)
            The model to be mutated.

        Returns
        -------
        model : np.array(1d)
            The mutated model
        """
        # Remove (atleast) one random term from the model
        orphan_idx = np.random.randint(self.forced.size, len(model))
        to_remove = self._remove(model, orphan_idx)

        # Fix all removed terms by sampling sequentially
        for i in np.flatnonzero(to_remove):
            model[i] = self._sample(model[~to_remove])
            to_remove[i] = False

        # Force uniqueness
        return self._sort(model)

    def fit(self, model):
        """
        Fit a regression model on the current terms specified by `model`.

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
        raise NotImplementedError('This function must be implemented')

"""
Module for the SAMS branch-and-bound code.
"""

import numpy as np
from scipy import sparse

from .bnb import BnB
from .....utils.model import permitted_dep_add

def sparsify(models, nterms):
    """
    Makes a certain collection of models sparse using a boolean mask
    for the model terms.

    Parameters
    ----------
    models : np.ndarray(2d)
        The set of models, each row is one model.
    nterms : int
        The total number of possible terms (to define the sparse matrix
        dimensions).

    Returns
    -------
    sparse_matrix : `scipy.sparse.csc_matrix`
        The sparse csc matrix.
    """
    return sparse.csc_matrix((
                np.ones(models.size, dtype=np.bool_), 
                (np.repeat(np.arange(len(models)), models.shape[1]), models.flatten())
            ), shape=(len(models), nterms), dtype=np.bool_)

class SamsBnB(BnB):
    """
    Runs the BnB algorithm for SAMS automated model selection.

    Attributes
    ----------
    model_size : int
        The size of the overfitted models.
    models : np.array(2d)
        The returned results from the SAMS simulation.
        A numpy array with a special datatype where each element contains
        two arrays of size `model_size` ('model', np.int\_), ('coeff', np.float64),
        and one scalar ('metric', np.float64).
    nterms : int
        The total number of fixed effects in the encoded, normalized
        model matrix (=X.shape[1] after encoding and normalization).
        No element in models should be larger than or equal to this value.
    mode : None or 'weak' or 'strong'
        The heredity mode during sampling.
    dependencies : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    forced_model : np.array(1d)
        The terms which were forced to be in the simulation models
        as an integer array. Often the intercept.
    kill : np.array(1d)
        A boolean array of which terms should not be investigated as
        they cannot be in the top performing models. Updated during the
        algorithm.
    spm : :py:class:`scipy.sparse.csc_array`
        A sparse boolean matrix of the `models`. Has dimensions
        (models.shape[0], nterms).

    """

    def __init__(self, model_size, models, nterms, 
                 mode=None, dependencies=None, forced_model=None):
        """
        Initializes the branch-and-bound object. 

        Parameters
        ----------
        model_size : int
            The size of the overfitted models.
        models : np.array(2d)
            The returned results from the SAMS simulation.
            A numpy array with a special datatype where each element contains
            two arrays of size `model_size` ('model', np.int\_), ('coeff', np.float64),
            and one scalar ('metric', np.float64).
        nterms : int
            The total number of fixed effects in the encoded, normalized
            model matrix (=X.shape[1] after encoding and normalization).
            No element in models should be larger than or equal to this value.
        mode : None or 'weak' or 'strong'
            The heredity mode during sampling.
        dependencies : np.array(2d)
            The dependency matrix of size (N, N) with N the number
            of terms in the encoded model (output from Y2X). Term i depends on term j
            if dep(i, j) = true.
        forced_model : None or np.array(1d)
            The terms which were forced to be in the simulation models
            as an integer array. Often the intercept.
        """

        # Input validation
        assert mode in (None, 'weak', 'strong'), 'The drop-mode must be None, weak or strong'
        if mode in ('weak', 'strong'):
            assert dependencies is not None, 'Must specify dependency matrix if using weak or strong heredity'
            assert len(dependencies.shape) == 2, 'Dependencies must be a 2D array'
            assert dependencies.shape[0] == dependencies.shape[1], 'Dependency matrix must be square'
            assert dependencies.shape[0] == nterms, 'Must specify a dependency for each term'

        # Default values
        if forced_model is None:
            forced_model = np.zeros((0,), dtype=np.int_)

        # Store the variables
        self.model_size = model_size
        self.models = models
        self.nterms = nterms
        self.mode = mode
        self.dependencies = dependencies
        self.forced_model = forced_model
        self.kill = None

        # State
        self.spm = sparsify(models, self.nterms)
         
    def initialize(self, nfit):
        """
        Initializes the results using a greedy search.

        Parameters
        ----------
        nfit : int
            The number of top results to find.

        Returns
        -------
        top_results : list(node)
            A list of initial optimal node guesses.
        top_scores : np.ndarray(1d)
            The corresponding scores.
        """
        # Initialize the best size 'nfit' models
        top_models = np.zeros((nfit, self.model_size), dtype=np.int_)
        top_frequencies = np.zeros(nfit, dtype=np.int_)

        # Initialization procedure
        models = self.spm

        for j in range(nfit):
            # Create a view to the start_model
            models_with_submodel = models

            for i in range(top_models.shape[1]):
                # Reset counts and extract current model
                submodel = top_models[j, :i]

                # Extract permitted terms: TODO: possible only compute freq for permitted terms
                permitted = permitted_dep_add(submodel, self.mode, self.dependencies)
                permitted[submodel] = False

                # Compute frequency of terms
                freq = models_with_submodel.sum(axis=0).A1

                # Remove from frequencies
                freq[~permitted] = 0
                    
                # Get the most frequent term
                term = np.argmax(freq)
                top_models[j, i] = term

                # Only keep the samples with this term
                keep = models_with_submodel[:, term].toarray().flatten().astype(np.bool_)
                models_with_submodel = models_with_submodel[keep]    

            # Drop all results with this model sequence
            models_with_sequence = (models[:, top_models[j]].sum(axis=1).A1 == top_models.shape[1])
            models = models[~models_with_sequence]

            # Compute the frequency of this model
            top_frequencies[j] = np.sum(self.spm[:, top_models[j]].sum(axis=1).A1 == top_models.shape[1])
            
            # Stop if no models are left
            if models.shape[0] == 0:
                break

        # Sort model columns for lexicographical order
        top_models.sort(axis=1)

        # Create nodes
        top_models = [(model, self.model_size) for model in top_models]
        
        return top_models, top_frequencies
    
    def init_queue(self, top_results, top_scores):
        """
        Initializes the branches queue, starting from the forced model and
        yielding all possible one-term extensions.

        Parameters
        ----------
        top_results : list(node)
            A list of top result nodes from initialization. Sorted
            by top_scores (lowest first).
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.

        Returns
        -------
        branches : iterable or generator
            The potential branches from all the provided nodes.
        """
        # Initial options
        valid = permitted_dep_add(self.forced_model, self.mode, self.dependencies)
        valid[self.forced_model] = False
        valid[self.kill] = False
        options = np.where(valid)[0]

        for i in options:
            # Create a node
            node = np.zeros(self.model_size, dtype=np.int_)
            node[:self.forced_model.size] = self.forced_model
            node[self.forced_model.size] = i
            node[self.forced_model.size + 1:] = -1
            yield (node, self.forced_model.size+1)
    
    def upperbound(self, node):
        """
        Compute the upperbound on the amount of times this submodel
        occurs in the set (=frequency of this submodel).

        Parameters
        ----------
        node : obj
            The object for which to compute the upperbound.

        Returns
        -------
        score : int or float
            The upperbound score.
        """
        # Create the sized model
        node, size = node
        sized_model = node[:size]

        # Compute frequency of submodel
        freq = np.sum(self.spm[:, sized_model].sum(axis=1).A1 == size)
        
        return freq

    def leaf(self, node):
        """
        Checks whether the model is of full size.

        Parameters
        ----------
        node : obj
            The node to check.

        Returns
        -------
        is_leaf : bool 
            Whether the node is a leaf.
        """
        # Check if a full model
        node, size = node
        return node.size == size

    def branches(self, node):
        """
        Generates branches by adding possible where permitted.

        Parameters
        ----------
        node : obj
            The root node

        Returns
        -------
        branches : iterable or generator
            The new branches appearing from this node
        """
        # Create the sized model
        node, size = node
        sized_model = node[:size]

        # Detect which models are still permitted
        permitted = np.ones(self.nterms, dtype=np.bool_)
        permitted[self.forced_model] = False
        permitted[self.kill] = False
        permitted[:sized_model[-1]+1] = False # Only allow new terms to the right, otherwise the same model is checked multiple times
        permitted[permitted] = permitted_dep_add(
            sized_model, self.mode, self.dependencies, subset=permitted
        )

        # Add the permitted combinations to the queue
        for i in np.flatnonzero(permitted):
            n = node.copy()
            n[size] = i
            yield (n, size+1)
        
    def node_in_results(self, node, results):
        """
        Check whether the model is already in the results.

        Parameters
        ----------
        node : obj
            The node to check.
        results : list(node)
            The list of current optimal nodes.
        
        Returns
        -------
        in_results : bool 
            Whether the node is in the results or not.
        """
        return any(np.all(results[i][0] == node[0]) for i in range(len(results)))
    
    def preloop(self, top_results, top_scores):
        """
        Kills any terms which do not occur frequently enough.

        Parameters
        ----------
        top_results : list(node)
            A list of initial optimal node guesses. Sorted
            by top_scores (lowest first).
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.

        Returns
        -------
        top_results : list(node)
            A list of initial optimal node guesses. Sorted
            by top_scores (lowest first).
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.
        """
        # Set kill sequence, any term which does not occur frequently enough
        self.kill = self.spm.sum(axis=0).A1 < top_scores[0]
        return top_results, top_scores
    
    def postnew(self, old, new, top):
        """
        Kills any terms which do not occur frequently enough.

        Parameters
        ----------
        old : (node, score)
            The old node and score that got removed.
        new : (node, score)
            The new node and score that got added.
        top : (list(node), np.ndarray(1d))
            The new set of optimal nodes and scores, sorted
            by lowest score first.
        """
        # Update kill sequence
        self.kill = self.spm.sum(axis=0).A1 < top[1][0]

"""
Module for the base branch-and-bound code.
"""

import queue
import numpy as np


class BnB:
    """
    Base branch-and-bround class to retain ntop models (instead of only the best).
    It is a maximization algorithm.
    """
    def __init__(self):
        pass
    
    def initialize(self, nfit):
        """
        Initialize a guess for the optimal nodes.

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
        raise NotImplementedError('Class BnB should be extended')
        
    def init_queue(self, top_results, top_scores):
        """
        Initializes the branches queue from the top results and scores.

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
        raise NotImplementedError('Class BnB should be extended')
        
    def upperbound(self, node):
        """
        Function to compute the upperbound for the maximization
        branch-and-bound algorithm.

        Parameters
        ----------
        node : obj
            The object for which to compute the upperbound.

        Returns
        -------
        score : int or float
            The upperbound score.
        """
        raise NotImplementedError('Class BnB should be extended')

    def leaf(self, node):
        """
        Checks whether a node is a leaf.

        Parameters
        ----------
        node : obj
            The node to check.

        Returns
        -------
        is_leaf : bool 
            Whether the node is a leaf.
        """
        raise NotImplementedError('Class BnB should be extended')

    def branches(self, node):
        """
        Function to generates the branches from a node

        Parameters
        ----------
        node : obj
            The root node

        Returns
        -------
        branches : iterable or generator
            The new branches appearing from this node
        """
        raise NotImplementedError('Class BnB should be extended')
        
    def node_in_results(self, node, results):
        """
        Check whether the node is already in the results.

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
        raise NotImplementedError('Class BnB should be extended')
        
    def preloop(self, top_results, top_scores):
        """
        Callback to run before starting the branch-and-bound algorithm
        (but after intialization).

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
        return top_results, top_scores
        
    def postloop(self, top_results, top_scores):
        """
        Callback to run after the branch-and-bound algorithm has run.

        Parameters
        ----------
        top_results : list(node)
            A list of optimal nodes. Sorted
            by top_scores (lowest first).
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.

        Returns
        -------
        top_results : list(node)
            A list of optimal nodes. Sorted
            by top_scores (lowest first).
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.
        """
        return top_results, top_scores
        
    def prenew(self, old, new, top):
        """
        Function defining what to do after
        finding a new optimal node and before adding it
        to the top.

        Parameters
        ----------
        old : (node, score)
            The old node and score that got removed.
        new : (node, score)
            The new node and score that got added.
        top : (list(node), np.ndarray(1d))
            The old set of optimal nodes and scores, sorted
            by lowest score first.
        """
        pass

    def postnew(self, old, new, top):
        """
        Function defining what to do after
        finding a new optimal node and adding it
        to the top.

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
        pass
    
    def loop(self, top_results, top_scores):
        """
        Loops through the branch-and-bound algorithm keeping
        topn results.

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
            A list of optimal nodes. Sorted
            by top_scores (lowest first).
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.
        """
        # Callback before everything starts
        top_results, top_scores = self.preloop(top_results, top_scores)
        
        # Create the queue
        q = queue.SimpleQueue()
        
        # Initialize queue
        for node in self.init_queue(top_results, top_scores):
            q.put(node)
        
        # Loop until the queue is empty
        while not q.empty():
            
            # Get the next node
            node = q.get()
            
            # Compute its upperbound
            upperbound = self.upperbound(node)
            
            # If the upperbound is higher than the lowest score
            if upperbound > top_scores[0]:

                # If the node is a leaf
                if self.leaf(node):
                    
                    # if not yet in the top results
                    if not self.node_in_results(node, top_results):

                        # Create the updates
                        old = top_results[0], top_scores[0]
                        new = node, upperbound
                        
                        # Callback before adding the node
                        self.prenew(old, new, (top_results, top_scores))
                        
                        # Store the model
                        top_results[0] = new[0]
                        top_scores[0] = new[1]

                        # Resort the sequence
                        idx = np.argsort(top_scores)
                        top_results = [top_results[i] for i in idx]
                        top_scores = top_scores[idx]
                        
                        # Callback after adding the node
                        self.postnew(old, new, (top_results, top_scores))

                else:
                    # Add the subnodes to the queue
                    for n in self.branches(node):
                        q.put(n)
                    
        # Callback after the loop completed
        top_results, top_scores = self.postloop(top_results, top_scores)
        
        return top_results, top_scores
    
    def top(self, nfit):
        """
        Returns the top `nfit` results using the branch-and-bound algorithm.

        Parameters
        ----------
        nfit : int
            The number of top models to return.

        Returns
        -------
        top_results : list(node)
            A list of nfit top nodes, sorted by top_scores lowest first.
        top_scores : np.ndarray(1d)
            The corresponding scores, sorted lowest first.
        """
        # Generate initial results
        top_results, top_scores = self.initialize(nfit)
        
        # Sort rows according to frequencies
        idx = np.argsort(top_scores)
        top_scores = top_scores[idx]
        top_results = [top_results[i] for i in idx]
                
        # Perform branch and bound algorithm
        top_results, top_scores = self.loop(top_results, top_scores)
        return top_results, top_scores

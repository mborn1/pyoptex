"""
Module for all restart functions of the CODEX algorithm
"""

class RestartEveryNFailed:
    """
    Restarts every `max_it` rejected samples. This is to counter
    bad search regions and reset to the best previsouly found
    design (state).

    Attributes
    ----------
    i : int
        The current number of consecutively rejected iterations.
    max_it : int
        The maximum number of consecutively rejected iterations.
    """
    def __init__(self, max_it):
        """
        Initializes the restart function

        Parameters
        ----------
        max_it : int
            The maximum number of consecutively rejected iterations.
        """
        self.i = 0
        self.max_it = max_it

    def reset(self):
        """
        Resets the restart function
        """
        self.i = 0

    def accepted(self):
        """
        Log an accepted iteration.
        """
        self.i = 0

    def rejected(self):
        """
        Log a rejected iteration.
        """
        self.i += 1

    def call(self, state, best_state):
        """
        Calls the restart function. If the number of 
        consecutively rejected iterations is larger
        than `max_it`, the `best_state` is returned.
        Otherwise the `state` is returned.

        Parameters
        ----------
        state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
            The state accepted or rejected state.
        best_state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
            The best state until now.

        Returns
        -------
        state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
            Either the `state` or `best_state`.
        """
        if self.i > self.max_it:
            self.i = 0
            print('Restarted the optimization from optimum')
            return best_state
        else:
            return state

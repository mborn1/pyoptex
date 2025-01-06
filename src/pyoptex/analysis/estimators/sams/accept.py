"""
Module for the SAMS accept functions.
"""
import numpy as np


class Accept:
    """
    The base class to create an acceptance function for the SAMS
    procedure. Requires the functions
    :py:func:`reset <pyoptex.analysis.estimators.sams.accept.Accept.reset>`, 
    :py:func:`accepted <pyoptex.analysis.estimators.sams.accept.Accept.accepted>`, 
    :py:func:`rejected <pyoptex.analysis.estimators.sams.accept.Accept.rejected>`,
    and :py:func:`__call__ <<pyoptex.analysis.estimators.sams.accept.Accept.__call__>`.
    """

    def reset(self):
        """
        Should reset the acceptance function to its original temperature.
        """
        raise NotImplementedError('This function has not been implemented')

    def __call__(self, d):
        """
        Computes the acceptance probability based on the distance
        (m_new - m_old), with m the metric to be maximized.

        Parameters
        ----------
        d : float
            The distance m_new - m_old when maximizing the metric m.

        Returns
        -------
        prob : float
            The acceptance probability between 0 and 1.
        """
        raise NotImplementedError('This function has not been implemented')

    def accepted(self):
        """
        Called when the state got accepted. Reduces the temperature.
        """
        raise NotImplementedError('This function has not been implemented')

    def rejected(self):
        """
        Called when the state got rejected. Increases the temperature.
        """
        raise NotImplementedError('This function has not been implemented')

class ExponentialAccept:
    """
    Compute the temperature of the Simulated Annealing system
    according to an exponential.
    
    * Updates the temperature with T = T * rho if the model got accepted.
    * Updates the temperature with T = T / alpha if the model got rejected.
      alpha = rho ** (1/kappa).

    Attributes
    ----------
    T0 : float
        The initial temperature of the system.
    rho : float 
        The reduction factor when the new state is accepted.
        Should be less than 1.
    alpha : float
        The increase factor when the new state is rejected.
        Computed as :math:`\\rho^{1/\\kappa}`.
    """

    def __init__(self, T0=1, rho=0.95, kappa=4):
        """
        Create the acceptance function.

        Parameters
        ----------
        T0 : float
            The initial temperature
        rho : float
            The reduction factor when a state is accepted.
        kappa : float
            The exponential factor when computing alpha, the increase
            factor when a state is rejected. When accepting a state
            and rejecting kappa states, the temperature will remain
            constant.
        """
        assert 0 < rho <= 1, 'The reduction factor rho should be between 0 and 1'
        self.rho = rho
        self.alpha = rho ** (1 / kappa)
        self._T = T0
        self.reset()

    def reset(self):
        """
        Resets the temperature to its original T0 value.
        """
        self.T = self._T

    def __call__(self, d):
        """
        Computes the exponential probability

        .. math::
            
            prob = e^{-d / T}

        Parameters
        ----------
        d : float
            The distance m_new - m_old when maximizing the metric m.

        Returns
        -------
        prob : float
            The acceptance probability.
        """
        return np.exp(-d / self.T)

    def accepted(self):
        """
        Called when the state got accepted. Reduces the temperature by
        multiplying with rho.
        """
        self.T *= self.rho

    def rejected(self):
        """
        Called when the state got rejected. Increases the temperature
        by dividing by alpha.
        """
        self.T /= self.alpha

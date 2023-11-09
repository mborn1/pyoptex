
class LinearTemperature:
    """
    Linear temperature operator. Computes temperature as
    T = (1-i/nsims) * T0 for each iteration.

    Attributes
    ----------
    T0 : float
        The starting temperature.
    nsims : int
        The total number of simulations.
    i : int
        The current iteration
    T : float
        The current temperature
    """
    def __init__(self, T0, nsims):
        self.T0 = T0
        self.nsims = nsims
        self.i = 0
        self.T = T0

    def reset(self):
        self.i = 0
        self.T = self.T0

    def accepted(self):
        self.T = (1-(self.i+1)/self.nsims) * self.T0
        self.i += 1

    def rejected(self):
        self.accepted()

class ExponentialTemperature:
    """
    Exponential temperature which decreases with an accepted
    state, and increases with a rejected state. The rate of 
    increase/decrease is represented by kappa. E.g. kappa = 4
    requires 4 rejections to reach the same temperature as one
    acceptation.

    Attributes
    ----------
    T0 : float
        The starting temperature.
    rho : float
        The rate at which the temperature decreases.
    kappa : int or float
        The number of rejections per accept for a stable temperature.
    alpha : float
        Derived value from kappa and rho
    T : float
        The current temperature
    """
    
    def __init__(self, T0, rho=0.95, kappa=4):
        self.rho = rho
        self.kappa = kappa
        self.alpha = rho ** (1 / kappa)
        self.T0 = T0
        self.T = self.T0

    def reset(self):
        self.T = self.T0

    def accepted(self):
        self.T *= self.rho

    def rejected(self):
        self.T /= self.alpha

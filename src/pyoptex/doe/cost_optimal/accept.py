import numpy as np

def exponential_accept_rel(m0, m1, T):
    """
    Computes the accept probability as an exponential function of the
    ratio between the new and old metric, and the temperature.

    * ositive metrics: (m1/m0)**(1/T)
    * Negative metrics: (m0/m1)**(1/T)

    .. note::
        This requires the metric to have one distinct sign.

    Parameters
    ----------
    m0 : float
        The old metric
    m1 : float
        The new metric
    T : float
        The current temperature

    Returns
    -------
    alpha : float
        The accept probability.
    """
    if m0 == 0:
        return 1
    else:
        d = m1/m0 if m0 > 0 else m0/m1
        return d ** (1/T)
"""
Module for all accept functions of the CODEX algorithm
"""

def exponential_accept_rel(m0, m1, T):
    """
    Computes the accept probability as an exponential function of the
    ratio between the new and old metric, and the temperature.

    * Positive metrics: (m1/m0)**(1/T)
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
    # If old metric is zero, always accept
    if m0 == 0:
        return 1

    # Compute accept probability
    d = m1/m0 if m0 > 0 else m0/m1
    return d ** (1/T)

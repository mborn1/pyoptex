"""
Module for all sample / generated functions of the CODEX algorithm
"""

import numpy as np


def sample_last(state, params):
    """
    Samples a new run by taking the last run 
    and randomly changing one coordinate.

    Parameters
    ----------
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to sample.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    run : np.array(1d)
        The newly sampled run.
    """
    # Determine which factor changes
    change = np.random.randint(params.colstart.size - 1)
    coords = params.coords[change]

    # Create a new run out of it
    new_run = np.copy(state.Y[-1])
    new_run[params.colstart[change]:params.colstart[change+1]] = \
        coords[np.random.randint(len(coords))]
    return new_run

def sample_random(state, params):
    """
    Samples a new run by taking a random run from the design
    and randomly changing one coordinate.

    Parameters
    ----------
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to sample.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    run : np.array(1d)
        The newly sampled run.
    """
    # Determine which factor changes
    change = np.random.randint(params.colstart.size - 1)
    coords = params.coords[change]

    # Create a new random base run
    idx = np.random.randint(len(state.Y))
    new_run = np.copy(state.Y[idx])

    # Change the coordinate
    new_run[params.colstart[change]:params.colstart[change+1]] = \
        coords[np.random.randint(len(coords))]
    return new_run

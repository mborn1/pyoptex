"""
Module containing the simulation function for SAMS.
"""

import numpy as np
from tqdm import tqdm as tqdm_

from .accept import ExponentialAccept


def simulate_sams(model, model_size, accept_fn=None, nb_models=10, minprob=0.01, tqdm=True):
    """
    Sample models using SAMS algorithm.

    Parameters
    ----------
    model : :py:class:`Model <pyoptex.analysis.estimators.sams.models.model.Model>`
        The model to fit such as an OLS or a mixed model.
    model_size : int
        The total size of each overfitted model.
    accept_fn : func(d)
        The acceptance function. Defaults to the
        :py:class:`exponential accept <pyoptex.analysis.estimators.sams.accept.ExponentialAccept>`.
    nb_models : int
        The number of models to sample.
    minprob : float
        The minimum probability before accepting.
    tqdm : bool
        Whether to use tqdm to track the progress.

    Returns
    -------
    results : np.array(1d)
        A numpy array with a special datatype where each element contains
        two arrays of size `model_size` ('model', np.int\_), ('coeff', np.float64),
        and one scalar ('metric', np.float64). Results contains `nb_models` elements.
    """
    # Initialize variables
    model_it = 0
    if accept_fn is None:
        accept_fn = ExponentialAccept()
    accept_fn.reset()

    # Initialize model storage
    rdtype = np.dtype([
        ('model', np.int_, model_size), 
        ('coeff', np.float64, model_size),
        ('metric', np.float64)
    ])   
    results = np.zeros(nb_models, dtype=rdtype)
    models = np.zeros((nb_models, model_size), dtype=np.int_)

    # Create initial model
    m = np.zeros(model_size, dtype=np.int_)
    m = model.init(m)

    # Compute initial metric
    fit = model.fit(m)
    metric0 = fit.metric
    
    # Start the main simulation loop
    with tqdm_(total=nb_models, disable=(not tqdm)) as pbar:

        while model_it < nb_models:
            # Mutate to a proposed model
            pm = np.copy(m)
            pm = model.mutate(pm)

            # Compute metric
            fit = model.fit(pm)
            metric1 = fit.metric

            # Compute distance
            d = metric0 - metric1
            if np.random.rand() < max(accept_fn(d), minprob):
                # Accept the proposed model
                metric0 = metric1
                m = pm

                # Store if unique: TODO faster by first validating on metric? Maybe a sorted array?
                if not np.any(np.all(models[:model_it] == model, axis=1)):
                    
                    # Store the model
                    models[model_it] = m
                    results[model_it] = m, fit.params, fit.metric

                    # Increase progress
                    model_it += 1
                    pbar.update(1)

                # Decrease temperature
                accept_fn.accepted()

            else:
                # Increase temperature
                accept_fn.rejected()

    return results
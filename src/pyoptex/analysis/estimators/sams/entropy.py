"""
Module for the SAMS entropy calculations.
"""

import numpy as np

from ....utils.model import sample_model_dep
from ....utils.numba import numba_int2bool

from .models.model import Model
def sample_mcmc(dep, size, forced=None, mode=None, N=1, skip=10):
    # Create the SAMS modeller
    m = Model(np.zeros((0, len(dep))), np.zeros((0,)), mode=mode, forced=forced, dep=dep)

    # Initialize a random model
    model = np.zeros((size,), dtype=np.int_)
    m.init(model)

    # Intialize the samples
    samples = np.zeros((N, size), dtype=np.int_)

    # Warmup phase
    for i in range(1000):
        m.mutate(model)

    # Main sampling loop
    for i in range(N*skip):
        # Mutate the model
        m.mutate(model)

        # Every skip, store the result
        if i % skip == 0:
            samples[int(i/skip)] = model

    return samples

def sample_random(dep, size, forced=None, mode=None, N=1):
    assert mode == 'weak', 'Mode must be weak'

    #########################
    # Initialize number of dependencies
    nb_dep = np.ma.masked_where(~dep, np.zeros_like(dep, dtype=np.int_)).harden_mask()

    # At the true positions in these columns, set a 1
    affected = ~np.any(dep, axis=1)
    nb_dep[:, affected] = 1
    affected = np.any(dep[:, affected], axis=1)

    while np.any(affected):
        # Alter the affected positions
        nb_dep[:, affected] = np.min(nb_dep[affected], axis=1).compressed() + 1
        affected = np.any(dep[:, affected], axis=1)

    #########################

    # Initialize the models
    models = np.zeros((N, size), dtype=np.int_)
    models[:, :forced.size] = forced

    # Fix the forced model
    if forced is not None and forced.size > 0:
        # Convert submodel to binary array
        affected = model[:forced.size]
        submodelb = np.zeros(len(dep), dtype=np.int_)
        submodelb[affected] = 1
        
        # Update the model
        nb_dep[:, affected] -= 1
        affected = np.any(dep[:, affected], axis=1)
        while np.any(affected):
            # Alter the affected positions
            nb_dep[:, affected] = np.min(nb_dep[affected], axis=1) - submodelb[affected] + 1
            affected = np.any(dep[:, affected], axis=1)
    
    # Sample all models
    for model in models:
        # Initialize i
        i = forced.size
        j = forced.size
        nb_dep_ = nb_dep.copy()

        # Loop until a full model
        while i < size:

            # Compute the minimal path for each term
            min_path = np.min(nb_dep_, axis=1).filled(0)

            # Sample the first
            choices = np.ones(len(dep), dtype=np.bool_)
            choices[min_path >= size - i] = False # Remove those with too many dependencies
            choices[model[:i]] = False # Remove already in the model
            choices = np.flatnonzero(choices)
            model[i] = np.random.choice(choices)

            # TODO: purely random sampling is a problem for true sampling

            # Check if already hereditary
            if min_path[model[i]] > 0:
                # Update with dependencies
                choices = np.copy(dep[model[i]])
                choices[min_path >= size - i - 1] = False
                choices[model[:i+1]] = False
                choices = np.flatnonzero(choices)

                # Check if there are any choices
                while choices.size != 0:
                    # Sample a new term
                    i += 1
                    model[i] = np.random.choice(choices)

                    # Check for heredity
                    if min_path[model[i]] <= 0:
                        break

                    # Determine new choices
                    choices = np.copy(dep[model[i]])
                    choices[min_path >= size - i - 1] = False
                    choices[model[:i+1]] = False
                    choices = np.flatnonzero(choices)

            # Increase the model size        
            i += 1

            # Convert submodel to binary array
            affected = model[j:i]
            submodelb = np.zeros(len(dep), dtype=np.int_)
            submodelb[affected] = 1
            
            # Update the model
            nb_dep_[:, affected] -= 1
            affected = np.any(dep[:, affected], axis=1)
            while np.any(affected):
                # Alter the affected positions
                nb_dep_[:, affected] = np.min(nb_dep_[affected], axis=1) - submodelb[affected] + 1
                affected = np.any(dep[:, affected], axis=1)

            # Set j to i for next iteration
            j = i

    return models

def entropies_approx(submodels, freqs, model_size, dep, mode, 
                     forced=None, N=10000, eps=1e-6):
    """
    Compute the approximate entropy by sampling N random models
    and observing the frequency of each submodel.

    The entropy is computed as
     
    .. math:

        f_{o} * log_2(f_{o} / f_{t}) + (1 - f_{o}) * log_2((1 - f_{o}) / (1 - f_{t}))

    where :math:`f_{o}` is the observed frequency of the submodel in the SAMS
    procedure and :math:`f_{t}` is the theoretical frequency when sampling at random.
    A higher entropy indicates more "surprise" and therefore more likely to be
    the correct model.

    Parameters
    ----------
    submodels : list(np.array(1d))
        The list of top submodels for each size.
    freqs : np.array(1d)
        The frequencies of these submodels in the raster plot.
    model_size : int
        The size of the overfitted models. 
        The overfitted model includes the forced model,
        and its size must thus be larger than the forced model.
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    mode : None or 'weak' or 'strong'
         The heredity mode during sampling.
    forced : None or np.array(1d)
        Any terms that must be included in the model.
    N : int
        The number of random samples to draw to compute the
        theoretical frequency of a submodel.
    eps : float
        A numerical stability parameter in computing the entropy.

    Returns
    -------
    entropy : np.array(1d)
        An array of floats of the same length as the submodels.
    """
    # Generate random samples
    # samples = sample_model_dep(dep, model_size, N, forced, mode)
    # samples = sample_mcmc(dep, model_size, forced, mode, N, skip=10)
    samples = sample_random(dep, model_size, forced, mode, N)

    # Convert samples to a boolean array
    samples = numba_int2bool(samples, len(dep))

    # Initialize entropies
    entropies = np.empty(len(submodels), dtype=np.float64)

    for i in range(len(submodels)):
        # Extract model parameters
        submodel = submodels[i]

        # Theoretical frequency
        theoretical_freq = np.sum(np.all(samples[:, submodel], axis=1)) / samples.shape[0]

        # Observed frequency
        obs_freq = freqs[i]

        # Compute entropy
        entropies[i] = obs_freq * np.log2(obs_freq / theoretical_freq) \
                        + (1 - obs_freq + eps) * np.log2((1 - obs_freq + eps) / (1 - theoretical_freq))
    
    return entropies

####################################################################

from scipy.special import comb

def count_models(max_model, model_size, model=None):
    """
    Counts the number of models of a given size in the max model
    assuming weak heredity.

    .. warning::
        This assumes weak heredity!

    Parameters
    ----------
    max_model : (n_main, n_tfi, n_quad)
        The number of main, tfi and quadratic effects in the main model.
        Each time the total amount.
    model_size : int
        The size of the overfitted models.
    model : (me_pp, me_pm, me_mm, mtfi, mquad)
        The submodel parameters.
        - me_pp: The number of effects that can create quadratic and TFI
        - me_pm: The number of effects that can only create TFI
        - me_mm: The number of effects that cannot create quadratic or TFI
        - mtfi: The number of TFI
        - mquad: The number of quadratic effects
    """
    # Extract encoder and model values
    if model is None:
        model = (0, 0, 0, 0, 0)
    me_pp, me_pm, me_mm, mtfi, mquad = model
    me = me_pp + me_pm + me_mm
    terms = me + mtfi + mquad

    # Extract number of main terms in each section (main, tfi and quadratic effects)
    n_main, n_tfi, n_quad = max_model

    # Extract parameters
    wpp = n_quad
    wpm = n_tfi - wpp
    wmm = n_main - wpm - wpp

    # Count models
    count = 0
    for ypp in range(0, model_size + 1 - terms):
        p1 = comb(wpp - me_pp, ypp)
        for ypm in range(1 if me == 0 and ypp == 0 else 0, model_size + 1 - terms - ypp):
            p2 = comb(wpm - me_pm, ypm)
            for ymm in range(0, model_size + 1 - terms - ypp - ypm):
                p3 = comb(wmm - me_mm, ymm)
                y1 = ypp + ypm + ymm
                for y2 in range(0, me_pp + ypp - mquad + 1):
                    p4 = comb(me_pp + ypp - mquad, y2)
                    P = (ypp + ypm) * (wpp + wpm - me_pp - me_pm - 1) - comb(ypp + ypm, 2)
                    Q = (me_pp + me_pm) * (wpp + wpm - 1) - comb(me_pp + me_pm, 2) - mtfi
                    p5 = comb(P + Q, model_size - terms - y1 - y2)
                    count += p1 * p2 * p3 * p4 * p5
              
    return count

def entropies(submodels, freqs, model_size, max_model, eps=1e-6):
    """
    Compute the entropies of the submodels given the total set of models

    Assertions: 
    
    * encoding: Models are encoded by intercept, quad, TFI, main effects (in that order)
      (see partial_rsm)
    * simulation: The simulation is performed in weak heredity conditions

    Parameters
    ----------
    submodels : list(np.array(1d))
        The submodels to compute the entropy for
    models : np.array(2d)
        All the models of the simulation
    freqs : np.array(1d)
        An array of (observed) frequencies from each submodel
    max_model : (nquad, ntwo, nlin)
        A tuple with the number of quadratic, TFI and linear effects
    eps : float
        The numerical stability parameter in computing the logarithms
    """
    # Extract global parameters
    nquad, ntwo, nlin = max_model
    nint = nquad + ntwo
    nmain = nlin + nint
    nterms = nmain + int(nint * (nint - 1) / 2) + nquad + 1

    # Redefine max model for counting
    max_model = (nmain + 1, nint, nquad)

    # Count total number of models
    ct = count_models(max_model, model_size)

    # Initialize entropies
    entropies = np.empty(len(submodels), dtype=np.float64)

    for i in range(len(submodels)):
        # Extract model parameters
        submodel = submodels[i]
        
        # Extract amount of terms in submodel
        me_pp = np.sum((submodel <= nquad) & (submodel > 0))
        me_pm = np.sum((submodel <= nint) & (submodel > nquad))
        me_mm = np.sum((submodel <= nmain) & (submodel > nint))
        mquad = np.sum(submodel >= nterms - nquad)
        mtfi = submodel.size - me_pp - me_pm - me_mm - mquad
        model = (me_pp, me_pm, me_mm, mtfi, mquad)

        # Theoretical frequency
        theoretical_freq = count_models(max_model, model_size, model) / ct

        # Observed frequency
        obs_freq = freqs[i]

        # Compute entropy
        entropies[i] = obs_freq * np.log2(obs_freq / theoretical_freq) \
                        + (1 - obs_freq + eps) * np.log2((1 - obs_freq + eps) / (1 - theoretical_freq))

    return entropies






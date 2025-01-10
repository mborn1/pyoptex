"""
Module for the SAMS entropy calculations.
"""

import numpy as np
from scipy.special import comb

from ....utils.model import sample_model_dep_onebyone
from ....utils.numba import numba_int2bool

def entropies_approx(submodels, freqs, model_size, dep, mode, 
                     forced=None, N=10000, sampler=sample_model_dep_onebyone, eps=1e-6):
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
    sampler : func(dep, model_size, N, forced, mode)
        The sampler to use when generating random hereditary models.
    eps : float
        A numerical stability parameter in computing the entropy.

    Returns
    -------
    entropy : np.array(1d)
        An array of floats of the same length as the submodels.
    """
    # Generate random samples
    samples = sampler(dep, model_size, N, forced, mode)

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

    Returns
    -------
    nb_models : int
        The number of hereditary models with the specified submodel.
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
    Compute the entropies of the submodels given the total set of models.
    Please read the warning in the documentation on customizing SAMS.

    .. warning::
        Asserts weak heredity and a partial response surface model 
        in a particular order.

    Parameters
    ----------
    submodels : list(np.array(1d))
        The submodels to compute the entropy for
    freqs : np.array(1d)
        An array of (observed) frequencies from each submodel
    model_size : int
        The size of the overfitted models.
        The overfitted model includes the forced model.
    max_model : (nquad, ntwo, nlin)
        A tuple with the number of quadratic, TFI and linear effects
    eps : float
        The numerical stability parameter in computing the logarithms

    Returns
    -------
    entropy : np.array(1d)
        An array of floats with the entropy for each submodel.
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






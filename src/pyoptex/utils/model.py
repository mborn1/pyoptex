"""
Module containing all the generic model functions
"""

from collections import Counter

import numpy as np
import pandas as pd

from .design import x2fx
from .numba import numba_choice_bool_axis0


def partial_rsm(nquad, ntfi, nlin):
    """
    Creates a partial response surface model from a number of quadratic,
    two-factor interactions (tfi) and linear terms.
    First come the quadratic terms which can have linear, tfi and quadratic effects.
    Then come the tfi which can only have linear and tfi. Finally, there
    are the linear only effects.

    Parameters
    ----------
    nquad : int
        The number of main effects capable of having quadratic effects.
    ntfi : int
        The number of main effects capable of two-factor interactions.
    nlin : int
        The number of main effects only capable of linear effects.
    
    Returns
    -------
    model : np.array(2d)
        The model array where each term is a row and the value
        specifies the power. E.g. [1, 0, 2] represents x0 * x2^2.
    """
    # Compute terms
    nint = nquad + ntfi
    nmain = nlin + nint
    nterms = nmain + int(nint * (nint - 1) / 2) + nquad + 1

    # Initialize (pre-allocation)
    max_model = np.zeros((nterms, nmain), dtype=np.int_)
    stop = 1

    # Main effects
    max_model[stop + np.arange(nmain), np.arange(nmain)] = 1
    stop += nmain

    # Interaction effects
    for i in range(nint - 1):
        max_model[stop + np.arange(nint - 1 - i), i] = 1
        max_model[stop + np.arange(nint - 1 - i), i + 1 + np.arange(nint - 1 - i)] = 1
        stop += (nint - 1 - i)

    # Quadratic effects
    max_model[stop + np.arange(nquad), np.arange(nquad)] = 2

    return max_model

def partial_rsm_names(effects):
    """
    Creates a partial response surface model 
    :py:func:`partial_rsm <pyoptex.utils.model.partial_rsm>` 
    from the provided effects. The effects is a dictionary mapping 
    the column name to one of ('lin', 'tfi', 'quad').

    Parameters
    ----------
    effects : dict
        A dictionary mapping the column name to one of ('lin', 'tfi', 'quad')

    Returns
    -------
    model : pd.DataFrame
        A dataframe with the regression model, in the same order as effects.
    """
    # Sort the effects
    sorted_effects = sorted(effects.items(), key=lambda x: {'lin': 3, 'tfi': 2, 'quad': 1}[x[1]])

    # Count the number
    c = Counter(map(lambda x: x[1], sorted_effects))

    # Create the model
    model = partial_rsm(c['quad'], c['tfi'], c['lin'])

    return pd.DataFrame(model, columns=[e[0] for e in sorted_effects])[list(effects.keys())]

################################################

def encode_model(model, effect_types):
    """
    Encodes the model according to the effect types.
    Each continuous variable is encoded as a single column,
    each categorical variable is encoded by creating n-1 columns 
    (with n the number of categorical levels).

    Parameters
    ----------
    model : np.array(2d)
        The initial model, before encoding
    effect_types : np.array(1d)
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).

    Returns
    -------
    model : np.array(2d)
        The newly encoded model.
    """
    # Number of columns required for encoding
    cols = np.where(effect_types > 1, effect_types - 1, effect_types)

    ####################################

    # Insert extra columns for the encoding
    extra_columns = cols - 1
    a = np.zeros(np.sum(extra_columns), dtype=np.int_)
    start = 0
    for i in range(extra_columns.size):
        a[start:start+extra_columns[i]] = np.full(extra_columns[i], i+1)
        start += extra_columns[i]
    model = np.insert(model, a, 0, axis=1)

    ####################################

    # Loop over all terms and insert identity matrix (with rows)
    # if the term is present
    current_col = 0
    # Loop over all factors
    for i in range(cols.size):
        # If required more than one column
        if cols[i] > 1:
            j = 0
            # Loop over all rows
            while j < model.shape[0]:
                if model[j, current_col] == 1:
                    # Replace ones by identity matrices
                    ncols = cols[i]
                    model = np.insert(model, [j] * (ncols - 1), model[j], axis=0)
                    model[j:j+ncols, current_col:current_col+ncols] = np.eye(ncols)
                    j += ncols
                else:
                    j += 1
            current_col += cols[i]
        else:
            current_col += 1

    return model

def model2Y2X(model, factors):
    """
    Creates a Y2X function from a model.

    Parameters
    ----------
    model : pd.DataFrame
        The model
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.

    Returns
    -------
    Y2X : func(Y)
        The function transforming the design matrix (Y) to
        the model matrix (X).
    """
    # Validation
    assert isinstance(model, pd.DataFrame), 'Model must be a dataframe'
    
    col_names = [str(f.name) for f in factors]
    assert all(col in col_names for col in model.columns), 'Not all model parameters are factors'
    assert all(col in model.columns for col in col_names), 'Not all factors are in the model'

    # Extract factor parameters
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])

    # Detect model in correct order
    model = model[col_names].to_numpy()

    # Encode model
    modelenc = encode_model(model, effect_types)

    # Create transformation function for polynomial models
    Y2X = lambda Y: x2fx(Y, modelenc)

    return Y2X

def mixture_scheffe_model(mixture_effects, process_effects=dict(), cross_order=None, mcomp='_mixture_comp_'):
    """
    Creates a Scheffe model with potential process effects and
    potential cross-terms between the mixture effects and process effects.

    A mixture model with N components is fully defined by N-1 components.
    Therefore, the `mixture_effects` parameter should include all but one
    mixture components as the first element, and the degree as the second element.
    For example, a mixture with three components is specified by two
    factors A and B. The degree specifies wether to only include all main effects,
    or also interactions between the components.

    Examples:

    * mixture = [('A', 'B'), 'lin'] will yield (as defined in `Scheffé (1958) <https://www-jstor-org.kuleuven.e-bronnen.be/stable/2983895?sid=primo&seq=4>`_) 
      
      .. math::
        
        \sum_{k=1}^3 \\beta_k x_k
    * mixture = [('A', 'B'), 'tfi] will yield (as defined in `Scheffé (1958) <https://www-jstor-org.kuleuven.e-bronnen.be/stable/2983895?sid=primo&seq=4>`_)
      
      .. math::
        \sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l
    * process = {'D': 'quad', 'E': 'quad'} will yield
      
      .. math::
        \\alpha_0 + \sum_{k=1}^2 \\alpha_k z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    * mixture = [('A', 'B'), 'lin'], process = {'D': 'quad', 'E': 'quad'} will yield
      
      .. math::
      
        \sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \\alpha_k z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    * mixture = [('A', 'B'), 'tfi'], process = {'D': 'quad', 'E': 'quad'} will yield
      
      .. math::
      
        \sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l + \sum_{k=1}^2 \\alpha_k z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    * mixture = [('A', 'B'), 'tfi'], process = {'D': 'quad', 'E': 'quad'}, cross_order='lin' will yield (as defined by `Kowalski et al. (2002) <https://www.jstor.org/stable/1270686>`_)
      
      .. math::

        &\sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l + \\\\
        &\sum_{k=1}^2 [ \sum_{i=1}^3 \\gamma_{k,i} x_i ] z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    
    * mixture = [('A', 'B'), 'tfi'], process = {'D': 'quad', 'E': 'quad'}, cross_order='tfi' will yield
      
      .. math::

        &\sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l + \\\\
        &\sum_{k=1}^2 [ \sum_{i=1}^3 \\gamma_{k,i} x_i ] z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 [\sum_{i=1}^3 \\gamma_{k,l,i} x_i] z_k z_l + \sum_{i=1}^2 [\sum_{k=1}^2 \sum_{l=k+1}^3 \\gamma_{k,l,i} x_k x_l] z_i + \sum_{k=1}^2 z_k^2
      
    .. warning::
        This function is only to see the model used by
        :py:func:`mixtureY2X <pyoptex.utils.model.mixtureY2X>`.
        Do not use this with :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`.

    Parameters
    ----------
    mixture_effects : tuple(list(str), str)
        The mixture effects is a tuple with as first element the names of the mixture
        components, and as second element the model order. All but one mixture
        component should be specified, e.g., a mixture with three components A, B, and
        C should only specify A and B in the `factors` list and as a first element here.
        The model order is either 'lin' or 'tfi'.
    process_effects : dict(str, str)
        Maps the process variable names to their order. The order can be
        'lin', 'tfi', 'quad.
    cross_order : str or None
        The cross order which is either None, 'lin', or 'tfi'
    mcomp : str
        The name of the last mixture component.

    Returns
    -------
    model : pd.DataFrame
        A dataframe with the Scheffe model with process variables.
    """
    # Split in effects and order
    me, mo = mixture_effects
    me = [*me, mcomp]

    # Validate all mixture effects are continuous
    assert mo in ('lin', 'tfi'), f'The order of a mixture experiment cannot be higher than two-factor interactions'
    assert cross_order in (None, 'lin', 'tfi'), 'Can only consider no cross, linear crossing, or two-factor interaction (tfi) crossing'

    # Create the scheffe model and process model
    scheffe_model = partial_rsm_names({e: mo for e in me}).iloc[1:]
    process_model = partial_rsm_names(process_effects).iloc[1:]

    # Extract components
    mixture_comps = scheffe_model.columns
    process_comps = process_model.columns

    # Cross both dataframes
    scheffe_model[process_comps] = 0
    process_model[mixture_comps] = 0
    model = pd.concat((scheffe_model, process_model), ignore_index=True)

    # Cross the model
    if cross_order == 'tfi':
        # Extract tfi process effects
        tfi_process_effects = (
            (model[process_comps].sum(axis=1) == 2) # Sum is 2
            & (model[process_comps].astype(np.bool_).sum(axis=1) == 2) # Count is 2
            & (model[mixture_comps].sum(axis=1) == 0) # No other components
        )

        # Extract linear process effects
        lin_process_effects = (
            (model[process_comps].sum(axis=1) == 1)
            & (model[mixture_comps].sum(axis=1) == 0)
        )

        # Cross the linear process and two-factor mixture
        tfi_mixt_lin_process = pd.merge(
            scheffe_model.iloc[len(mixture_comps):][list(mixture_comps)], 
            model.loc[lin_process_effects, list(process_comps)], 
            how='cross'
        )

        # Combine them with linear mixture effects (apart from M_COMP)
        lin_mixt_tfi_process = pd.merge(
            model.loc[tfi_process_effects, list(process_comps) + [mcomp]],
            pd.DataFrame(
                np.eye(len(mixture_comps) - 1, dtype=np.int_), 
                columns=mixture_comps.drop(mcomp)
            ), 
            how='cross'
        )[model.columns]

        # Change tfi process effect to interaction with M_COMP
        model.loc[tfi_process_effects, mcomp] = 1

        # Combine all terms
        model = pd.concat(
            (model, tfi_mixt_lin_process, lin_mixt_tfi_process), 
            ignore_index=True
        )

    if cross_order in ('lin', 'tfi'):
        # Extract linear process effects
        lin_process_effects = (
            (model[process_comps].sum(axis=1) == 1)
            & (model[mixture_comps].sum(axis=1) == 0)
        )

        # Combine them with linear mixture effects (apart from M_COMP)
        additional_terms = pd.merge(
            model.loc[lin_process_effects, list(process_comps) + [mcomp]],
            pd.DataFrame(
                np.eye(len(mixture_comps) - 1, dtype=np.int_), 
                columns=mixture_comps.drop(mcomp)
            ), 
            how='cross'
        )[model.columns]

        # Change linear process effect to interaction with M_COMP
        model.loc[lin_process_effects, mcomp] = 1

        # Combine all terms
        model = pd.concat((model, additional_terms), ignore_index=True)

    return model

def mixtureY2X(factors, mixture_effects, process_effects=dict(), cross_order=None):
    """
    Creates a Scheffe model Y2X with potential process effects and
    potential cross-terms between the mixture effects and process effects.

    A mixture model with N components is fully defined by N-1 components.
    Therefore, the `mixture_effects` parameter should include all but one
    mixture components as the first element, and the degree as the second element.
    For example, a mixture with three components is specified by two
    factors A and B. The degree specifies wether to only include all main effects,
    or also interactions between the components.

    Examples:

    * mixture = [('A', 'B'), 'lin'] will yield (as defined in `Scheffé (1958) <https://www-jstor-org.kuleuven.e-bronnen.be/stable/2983895?sid=primo&seq=4>`_) 
      
      .. math::
        
        \sum_{k=1}^3 \\beta_k x_k
    * mixture = [('A', 'B'), 'tfi] will yield (as defined in `Scheffé (1958) <https://www-jstor-org.kuleuven.e-bronnen.be/stable/2983895?sid=primo&seq=4>`_)
      
      .. math::
        \sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l
    * process = {'D': 'quad', 'E': 'quad'} will yield
      
      .. math::
        \\alpha_0 + \sum_{k=1}^2 \\alpha_k z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    * mixture = [('A', 'B'), 'lin'], process = {'D': 'quad', 'E': 'quad'} will yield
      
      .. math::
      
        \sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \\alpha_k z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    * mixture = [('A', 'B'), 'tfi'], process = {'D': 'quad', 'E': 'quad'} will yield
      
      .. math::
      
        \sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l + \sum_{k=1}^2 \\alpha_k z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    * mixture = [('A', 'B'), 'tfi'], process = {'D': 'quad', 'E': 'quad'}, cross_order='lin' will yield (as defined by `Kowalski et al. (2002) <https://www.jstor.org/stable/1270686>`_)
      
      .. math::

        &\sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l + \\\\
        &\sum_{k=1}^2 [ \sum_{i=1}^3 \\gamma_{k,i} x_i ] z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 \\alpha_{k,l} z_k z_l + \sum_{k=1}^2 z_k^2
    
    * mixture = [('A', 'B'), 'tfi'], process = {'D': 'quad', 'E': 'quad'}, cross_order='tfi' will yield
      
      .. math::

        &\sum_{k=1}^3 \\beta_k x_k + \sum_{k=1}^2 \sum_{l=k+1}^3 \\beta_{k,l} x_k x_l + \\\\
        &\sum_{k=1}^2 [ \sum_{i=1}^3 \\gamma_{k,i} x_i ] z_k + \sum_{k=1}^1 \sum_{l=k+1}^2 [\sum_{i=1}^3 \\gamma_{k,l,i} x_i] z_k z_l + \sum_{i=1}^2 [\sum_{k=1}^2 \sum_{l=k+1}^3 \\gamma_{k,l,i} x_k x_l] z_i + \sum_{k=1}^2 z_k^2

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The factors of the experiment.
    mixture_effects : tuple(list(str), str)
        The mixture effects is a tuple with as first element the names of the mixture
        components, and as second element the model order. All but one mixture
        component should be specified, e.g., a mixture with three components A, B, and
        C should only specify A and B in the `factors` list and as a first element here.
        The model order is either 'lin' or 'tfi'.
    process_effects : dict(str, str)
        Maps the process variable names to their order. The order can be
        'lin', 'tfi', 'quad.
    cross_order : str or None
        The cross order which is either None, 'lin', or 'tfi'

    Returns
    -------
    Y2X : func(Y)
        The function transforming the design matrix (Y) to
        the model matrix (X).
    """
    # Validation
    assert all(f.is_mixture for f in factors if str(f.name) in mixture_effects[0]), f'Mixture factors must be of type mixture'

    # Create the mixture model
    me, _ = mixture_effects
    mcomp = '_mixture_comp_'
    model = mixture_scheffe_model(mixture_effects, process_effects, cross_order, mcomp=mcomp)

    # Validate all factors are in model and vice-versa
    col_names = [str(f.name) for f in factors]
    assert all(col in col_names for col in model if col != mcomp), 'Not all model parameters are factors'
    assert all(col in model.columns for col in col_names), 'Not all factors are in the model'

    ################################################

    # Extract factor parameters
    effect_types = np.concatenate((
        np.array([1 if f.is_continuous else len(f.levels) for f in factors]),
        np.array([1]) # Add continuous final mixture component
    ))

    # Retrieve start of columns
    colstart = np.concatenate((
        [0], 
        np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))
    ))

    # Retrieve indices of mixture components (encoded)
    me_idx_enc = np.array([colstart[col_names.index(e)] for e in me])

    # Detect model in correct order
    model = model[col_names + [mcomp]].to_numpy()

    # Encode model
    modelenc = encode_model(model, effect_types)

    # Define Y2X
    def Y2X(Y):
        Y = np.concatenate((
            Y,
            np.expand_dims(1 - np.sum(Y[:, me_idx_enc], axis=1), 1)
        ), axis=1)
        return x2fx(Y, modelenc)

    return Y2X

def identityY2X(Y):
    """
    The identity function.

    Parameters
    ----------
    Y : np.array
        The input

    Returns
    -------
    Y : np.array
        The input returned
    """
    return Y

################################################

def encode_names(col_names, effect_types):
    """
    Encodes the column names according to the categorical
    expansion of the factors.

    For example, if there is one categorical factor with
    three levels 'A' and one continuous factor, the encoded
    names are ['A_0', 'A_1', 'B'].

    Parameters
    ----------
    col_names : list(str)
        The base column names
    effect_types : np.array(1d)
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).

    Returns
    -------
    enc_names : list(str)
        The list of encoded column names.
    """
    lbls = [
        lbl for i in range(len(col_names)) 
            for lbl in (
                [col_names[i]] if effect_types[i] <= 2 
                else [f'{col_names[i]}_{j}' for j in range(effect_types[i] - 1)]
            )
    ]
    return lbls

def model2names(model, col_names=None):
    """
    Converts the model to parameter names. Each row of the
    model represents one term. 

    For example, the row [1, 2] with column names ['A', 'B']
    is converted to 'A * B^2'.

    Parameters
    ----------
    model : np.array(2d) or pd.DataFrame
        The model
    col_names : None or list(str)
        The name of each column of the model. If not provided
        and a dataframe is provided as the model, the names are
        taken from the model dataframe. If the model is a numpy
        array, the columns are named as ['1', '2', ...]

    Returns
    -------
    param_names : list(str)
        The names of the parameters in the model.
    """
    # Convert model to columns
    if isinstance(model, pd.DataFrame):
        col_names = list(model.columns)
        model = model.to_numpy()

    # Set base column names
    if col_names is None:
        col_names = list(np.arange(model.shape[1]).astype(str))
    col_names = np.asarray(col_names)

    def __comb(x):
        # Select the model term
        term = model[x]

        # Create higher order representations
        higher_order_effects = (term != 1) & (term != 0)
        high = np.char.add(np.char.add(col_names[higher_order_effects], '^'), term[higher_order_effects].astype(str))

        # Concatenate with main effects and join
        term_repr = np.concatenate((col_names[term == 1], high))
        term_repr = f' * '.join(term_repr)

        # Constant term
        if term_repr == '':
            term_repr = 'cst'

        return term_repr 
        
    return list(np.vectorize(__comb)(np.arange(model.shape[0])))

def model2encnames(model, effect_types, col_names=None):
    """
    Retrieves the names of the encoded parameters. Similar to
    :py:func:`model2names <pyoptex.utils.model.model2names>`, but also
    categorically encodes the necessary factors.

    Parameters
    ----------
    model : np.array(2d) or pd.DataFrame
        The model
    effect_types : np.array(1d)
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).
    col_names : None or list(str)
        The name of each column of the model. If not provided
        and a dataframe is provided as the model, the names are
        taken from the model dataframe. If the model is a numpy
        array, the columns are named as ['1', '2', ...]

    Returns
    -------
    enc_param_names : list(str)
        The names of the parameters in the model.
    """
    # Convert model to columns
    if isinstance(model, pd.DataFrame):
        col_names = list(model.columns)
        model = model.to_numpy()

    # Convert to encoded names
    model_enc = encode_model(model, effect_types)
    col_names_enc = encode_names(col_names, effect_types)
    col_names_model = model2names(model_enc, col_names_enc)

    return col_names_model

################################################

def order_dependencies(model, factors):
    """
    Create a dependency matrix from a model where
    interactions and higher order effects depend
    on their components and lower order effects.

    For example:
    * :math:`x_0`: depends only on the intercept.
    * :math:`x_0^2`: depends on :math:`x_0`, which in turn depends on the intercept.
    * :math:`x_0 x_1`: depends on both :math:`x_0` and :math:`x_1`, which both depend on the intercept.
    * :math:`x_0^2 x_1` : depends on both :math:`x_0^2` and :math:`x_1`, which depend on :math:`x_0` and the intercept.

    Parameters
    ----------
    model : pd.DataFrame
        The model
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.

    Returns
    -------
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model. Term i depends on term j
        if dep(i, j) = true.
    """
    # Validation
    assert isinstance(model, pd.DataFrame), 'Model must be a dataframe'
    assert np.all(model >= 0), 'All powers must be larger than zero'

    col_names = [str(f.name) for f in factors]
    assert all(col in col_names for col in model.columns), 'Not all model parameters are factors'
    assert all(col in model.columns for col in col_names), 'Not all factors are in the model'

    # Extract factor parameters
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])

    # Detect model in correct order
    model = model[col_names].to_numpy()

    # Encode model
    modelenc = encode_model(model, effect_types)

    # Compute the possible dependencies
    eye = np.expand_dims(np.eye(modelenc.shape[1]), 1)
    model = np.expand_dims(modelenc, 0)
    all_dep = model - eye # all_dep[:, i] are all possible dependencies for term i

    # Valid dependencies
    all_dep_valid = np.where(np.all(all_dep >= 0, axis=2))
    from_terms = all_dep_valid[1]

    # Extract the dependent terms
    to_terms = np.argmax(np.all(
        np.expand_dims(modelenc, 0) == np.expand_dims(all_dep[all_dep_valid], 1), 
        axis=2
    ), axis=1)

    # Compute dependencies
    dep = np.zeros((modelenc.shape[0], modelenc.shape[0]), dtype=np.bool_)
    dep[from_terms, to_terms] = True

    return dep

def model2strong(model, dep):
    """
    Convert an existing model to its strong heredity
    variant according to the provided dependency matrix.
    A model is a strong heredity model if the for
    every term i, all its dependencies are also included
    in the model.

    Parameters
    ----------
    model : np.array(1d)
        The array with indices of the terms included in the
        initial model
    dep : np.array(2d)
        A matrix of size (N, N) with N the total number of terms
        in the encoded model. Term i depends on term j
        if dep(i, j) = true.

    Returns
    -------
    strong : np.array(1d)
        The strong heredity model based on the initial model.
    """
    # Create a mask
    strong = np.zeros(dep.shape[0], dtype=np.bool_)
    strong[model] = True
    nterms_old = 0
    nterms = np.sum(strong)

    # Loop until no new terms are added
    while nterms_old < nterms:
        # Add dependencies
        strong[np.any(dep[strong], axis=0)] = True

        # Update number of terms
        nterms_old = nterms
        nterms = np.sum(strong)

    return np.flatnonzero(strong)

def permitted_dep_add(model, mode=None, dep=None, subset=None):
    """
    Computes which terms are permitted to be added to this model
    such that adding any of the returned terms does not violate
    the heredity constraints.

    .. note::
        Does not check whether the term already exists in the model.

    Parameters
    ----------
    model : np.array(1d)
        The current model.
    mode : None, 'weak' or 'strong'
        The heredity mode to adhere to.
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    subset : np.array(1d)
        The subset of terms to validate. 
        If None, all terms are validated.

    Returns
    -------
    valid : np.ndarray(1d)
        A boolean array indicating which terms are valid to be added.
        Has the same length as the subset.
    """
    # Subset the relations
    dep = dep if subset is None else dep[subset]

    if mode == 'weak':
        # Take all terms without dependencies
        valid = ~np.any(dep, axis=1)

        # If a model is present, also add the valid dependencies
        if len(model) > 0:
            valid = valid | np.any(dep[:, model], axis=1)

    elif mode == 'strong':
        # Take all terms without dependencies
        valid = ~np.any(dep, axis=1)

        # If a model is present, also add the valid dependencies
        if len(model) > 0:
            # All dependencies must be in the model
            valid = valid | (np.sum(dep, axis=1) == np.sum(dep[:, model], axis=1))

    else:
        # All terms are valid
        valid = np.ones(len(dep), dtype=np.bool_)

    return valid

def permitted_dep_drop(model, mode=None, dep=None, subset=None):
    """
    Determines if the term specified by at `idx` of `model` can be dropped,
    given the other existing terms in the model, the mode, and
    the dependency matrix.

    Parameters
    ----------
    model : np.array(1d)
        The terms in the current model. 
    mode : None or 'weak' or 'strong'
        The heredity mode.
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    subset : np.array(1d)
        The subset of terms to validate, represented as indices in the
        `model` parameter. 
        If None, all terms in the model are validated.

    Returns
    -------
    can_drop : np.array(bool)
        Whether this term can be dropped given the dependencies
        and heredity mode.
    """
    # Short-circuit
    if len(model) == 0:
        return np.zeros((0,), dtype=np.bool_)

    # Extract the subset
    if subset is None:
        subset = np.arange(len(model))
    subset = model[subset]

    # Check for the mode
    if mode == 'strong':
        # No dependent terms, otherwise violation of strong heredity
        drop = ~np.any(dep[:, subset][model], axis=0)

    elif mode == 'weak':
        # No single dependent terms left, otherwise violation of weak heredity
        single_deps = np.sum(dep[:, model][model], axis=1) == 1
        drop = ~np.any(dep[:, subset][model[single_deps]], axis=0)
    
    else:
        # No restrictions
        drop = np.ones(len(subset), dtype=np.bool_)

    return drop

def sample_model_dep_onebyone(dep, size, n_samples=1, forced=None, mode=None):
    """
    Sample a model given the dependency matrix of a
    fixed size. The terms are sampled one-by-one.

    Parameters
    ----------
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    size : int
        The size of the model to sample.
    n_samples : int
        The number of samples to draw.
    forced : np.array(1d)
        A model which must be included at all times.
    mode : None or 'weak' or 'strong'
        The heredity mode during sampling.

    Returns
    -------
    model : np.array(2d)
        The sampled model which is an array of integers of size (n_samples, size).
    """
    # The output
    out = np.zeros((n_samples, size), dtype=np.int_)

    # No dep
    no_dep = np.flatnonzero(~np.any(dep, axis=1))

    # Check for a forced model
    if forced is not None:
        # Set forced model as the beginning of each sample
        out[:, :forced.size] = forced
        start = forced.size
    else:
        # Sample the initial value
        if mode is None:
            # Any value is possible
            out[:, 0] = np.random.choice(np.arange(len(dep)), out.shape[0])
        else:
            # Only no dependency terms are possible
            out[:, 0] = np.random.choice(no_dep, out.shape[0])
        start = 1

    if mode is None:
        # Loop to generate a sample
        for i in range(start, out.shape[1]):
            # Determine which ones are valid
            valids = np.ones((out.shape[0], len(dep)), dtype=np.bool_)
            valids[np.repeat(np.arange(out.shape[0]), i), out[:, :i].flatten()] = False
            
            # Random sampling
            out[:, i] = numba_choice_bool_axis0(valids)

    elif mode == 'weak':
        # Loop to generate a sample
        for i in range(start, out.shape[1]):
            # Compute which terms are valid
            valids = np.any(dep[:, out[:, :i]], axis=-1).T
            valids[:, no_dep] = True
            valids[np.repeat(np.arange(out.shape[0]), i), out[:, :i].flatten()] = False

            # Random sampling
            out[:, i] = numba_choice_bool_axis0(valids)

    elif mode == 'strong':
        # Compute total number of dependencies
        nb_dep = np.sum(dep, axis=1)

        # Loop to generate a sample
        for i in range(start, out.shape[1]):
            # Compute which terms are valid
            valids = (np.sum(dep[:, out[:, :i]], axis=-1).T == nb_dep)
            valids[:, no_dep] = True
            valids[np.repeat(np.arange(out.shape[0]), i), out[:, :i].flatten()] = False

            # Random sampling
            out[:, i] = numba_choice_bool_axis0(valids)

    else:
        raise ValueError('Mode not recognized, must be either None, "weak" or "strong"')

    return out

def sample_model_dep_random(dep, size, n_samples=1, forced=None, mode=None):
    """
    Sample a model given the dependency matrix of a
    fixed size. The terms are as follows:

    * First you uniformly sample any term.
    * Then you look at the necessary dependencies and add these one-by-one.
    * If multiple dependencies exist, you sample from them uniformly. 
    * Go back to step one and continue until you sampled `size` terms.

    .. note::
        The mode must be weak heredity as of now.

    Parameters
    ----------
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model (output from Y2X). Term i depends on term j
        if dep(i, j) = true.
    size : int
        The size of the model to sample.
    n_samples : int
        The number of samples to draw.
    forced : np.array(1d)
        A model which must be included at all times.
    mode : None or 'weak' or 'strong'
        The heredity mode during sampling.

    Returns
    -------
    model : np.array(2d)
        The sampled model which is an array of integers of size (n_samples, size).
    """
    # Validate the mode
    assert mode != 'strong', 'Mode must be None or weak'

    # Return a one-by-one sampler if the mode is None
    if mode is None:
        return sample_model_dep_onebyone(dep, size, n_samples, forced, mode)

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
    models = np.zeros((n_samples, size), dtype=np.int_)
    models[:, :forced.size] = forced

    # Fix the forced model
    if forced is not None and forced.size > 0:
        # Convert submodel to binary array
        affected = forced
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

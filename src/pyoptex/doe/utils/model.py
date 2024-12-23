"""
Module containing all the generic model functions
"""

from collections import Counter

import numpy as np
import pandas as pd

from .design import x2fx


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
    max_model = np.zeros((nterms, nmain), dtype=np.int64)
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
    :py:func:`pyoptex.doe.utils.model.partial_rsm` 
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
    a = np.zeros(np.sum(extra_columns), dtype=np.int64)
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
    factors : list(:py:class:`Cost_optimal factor <pyoptex.doe.cost_optimal.utils.Factor>` or :py:class:`Splitk_plot factor <pyoptex.doe.splitk_plot.utils.Factor>`)
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
    
    .. warning::
        This function is only to see the model used by
        :py:func:`mixtureY2X <pyoptex.doe.utils.model.mixtureY2X>`.
        Do not use this with :py:func:`model2Y2X <pyoptex.doe.utils.model.model2Y2X>`.
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
    Creates a mixture model Y2X function. 

    TODO: note Scheffe model. Note which components to specify
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
    :py:func:`pyoptex.doe.utils.model.model2names`, but also
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

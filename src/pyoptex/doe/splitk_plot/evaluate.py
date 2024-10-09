import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
import pandas as pd
import numpy as np
from .metric import Aopt, Dopt, Iopt
from .wrapper import default_fn, create_parameters
from ..utils.model import encode_names, model2names, encode_model
from ..utils.design import x2fx, encode_design
from ..constraints import no_constraints

def evaluate_metrics(Y, effect_types, plot_sizes, model=None, ratios=None,
                        constraints=no_constraints, Y2X=None, cov=None):
    # Create the design parameters
    effect_levels = np.ones(len(effect_types), dtype=np.int64)
    fn = default_fn(None, constraints=constraints)
    params, col_names = create_parameters(fn, effect_types, effect_levels, plot_sizes, ratios=ratios, model=model, Y2X=Y2X, cov=cov)

    # Transform Y to numpy
    if isinstance(Y, pd.DataFrame):
        if col_names is not None:
            Y = Y[col_names]
        Y = Y.to_numpy()

    # Encode the design    
    Y = encode_design(Y, params.effect_types)
    if params.cov is not None:
        Y = np.concatenate((Y, params.cov), axis=1)

    # Define the metric inputs
    X = params.Y2X(Y)

    # Initialize the metrics
    n = 10000
    iopt = Iopt(n=n)
    iopt.preinit(params)
    iopt.init(Y, X, params)
    dopt = Dopt()
    dopt.preinit(params)
    dopt.init(Y, X, params)
    aopt = Aopt()
    aopt.preinit(params)
    aopt.init(Y, X, params)

    # Compute the metrics
    m_iopt = iopt.call(Y, X, params)
    m_dopt = dopt.call(Y, X, params)
    m_aopt = aopt.call(Y, X, params)

    # Return the metrics
    return (m_iopt, m_dopt, m_aopt)

def fraction_of_design_space(Y, effect_types, plot_sizes, model=None, ratios=None,
                                constraints=no_constraints, Y2X=None, cov=None):
    assert ratios is None or len(ratios.shape) == 1 or ratios.shape[0] == 1, 'Can only specify one set of variance ratios'

    # Create the design parameters
    effect_levels = np.ones(len(effect_types), dtype=np.int64)
    fn = default_fn(None, constraints=constraints)
    params, col_names = create_parameters(fn, effect_types, effect_levels, plot_sizes, ratios=ratios, model=model, Y2X=Y2X, cov=cov)

    # Transform Y to numpy
    if isinstance(Y, pd.DataFrame):
        if col_names is not None:
            Y = Y[col_names]
        Y = Y.to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types)
    if params.cov is not None:
        Y = np.concatenate((Y, params.cov), axis=1)

    # Define the metric inputs
    X = params.Y2X(Y)
    
    # Initialize Iopt
    n = 10000
    iopt = Iopt(n=n)
    iopt.preinit(params)
    iopt.init(Y, X, params)

    # Compute information matrix
    M = X.T @ params.Vinv[0] @ X

    # Compute prediction variances
    pred_var = np.sum(iopt.samples.T * np.linalg.solve(M, iopt.samples.T), axis=0)

    return np.sort(pred_var)

def plot_fraction_of_design_space(Y, effect_types, plot_sizes, model=None, ratios=None,
                                    constraints=no_constraints, Y2X=None, cov=None):
    # Compute prediction variances
    pred_var = fraction_of_design_space(Y, effect_types, plot_sizes, model, ratios, constraints, Y2X, cov)

    # Create the figure
    fig = go.Figure()
    color = DEFAULT_PLOTLY_COLORS[0]
    fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(pred_var)), y=pred_var, marker_color=color))
    fig.add_hline(y=np.mean(pred_var), annotation_text=f'{np.mean(pred_var):.3f}', annotation_font_color=DEFAULT_PLOTLY_COLORS[0], 
                  line_dash='dash', line_width=1, line_color=DEFAULT_PLOTLY_COLORS[0], annotation_position='bottom right')

    return fig

def estimation_variance_matrix(Y, effect_types, plot_sizes, model=None, ratios=None,
                                  constraints=no_constraints, Y2X=None, cov=None):
    assert ratios is None or len(ratios.shape) == 1 or ratios.shape[0] == 1, 'Can only specify one set of variance ratios'
    
    # Create the design parameters
    effect_levels = np.ones(len(effect_types), dtype=np.int64)
    fn = default_fn(None, constraints=constraints)
    params, col_names = create_parameters(fn, effect_types, effect_levels, plot_sizes, ratios=ratios, model=model, Y2X=Y2X, cov=cov)

    # Transform Y to numpy
    if isinstance(Y, pd.DataFrame):
        if col_names is not None:
            Y = Y[col_names]
        Y = Y.to_numpy()

    # Encode the columns names
    col_names_model = None
    if model is not None:
        # Set default column names
        if col_names is None:
            col_names = list(np.arange(model.shape[1]).astype(str))

        # Extract covariate parameter names
        if params.cov is not None:
            # Extract covariate names
            if isinstance(cov[2], dict):
                cov_col_names = list(cov[2].keys())
            else:
                cov_col_names = [f'cov_{i}' for i in range(len(cov[2]))]

            # Extend model with covariates
            if isinstance(cov[1], pd.DataFrame):
                model_cov = model_cov[[*col_names, *cov_col_names]]
                model_cov = model_cov.to_numpy()
            
            # Append covariate names to column names
            col_names = col_names.extend(cov_col_names)

            # Extend the model with covariates
            model = np.concatenate((model, np.zeros((model.shape[0], model_cov.shape[1] - model.shape[1]))), axis=1)
            model = np.concatenate((model, model_cov), axis=0)

        # Encode the model
        model_enc = encode_model(model, params.effect_types)

        # Create the column names
        col_names_enc = encode_names(col_names, params.effect_types)
        col_names_model = model2names(model_enc, col_names_enc)

    # Encode the design
    Y = encode_design(Y, params.effect_types)
    if params.cov is not None:
        Y = np.concatenate((Y, params.cov), axis=1)

    # Define the metric inputs
    X = params.Y2X(Y)

    # Compute information matrix
    M = X.T @ params.Vinv[0] @ X

    # Compute inverse of information matrix
    Minv = np.linalg.inv(M)

    # Attach the column names
    if col_names_model is not None:
        Minv = pd.DataFrame(Minv, columns=col_names_model, index=col_names_model)

    # Create figure
    return Minv

def plot_estimation_variance_matrix(Y, effect_types, plot_sizes, model=None, ratios=None,
                                        constraints=no_constraints, Y2X=None, cov=None):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, effect_types, plot_sizes, model, ratios, constraints, Y2X, cov)

    # Return the plot
    return px.imshow(Minv)

def estimation_variance(Y, effect_types, plot_sizes, model=None, ratios=None,
                            constraints=no_constraints, Y2X=None, cov=None):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, effect_types, plot_sizes, model, ratios, constraints, Y2X, cov)

    return np.diag(Minv) 

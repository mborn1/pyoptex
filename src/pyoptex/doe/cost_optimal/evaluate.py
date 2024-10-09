import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
import pandas as pd
import numpy as np
from .metric import Aopt, Dopt, Iopt
from .wrapper import default_fn, create_parameters
from .utils import obs_var_Zs
from ..utils.design import x2fx, encode_design, obs_var_from_Zs
from ..utils.model import encode_model, encode_names, model2names
from ..constraints import no_constraints

def evaluate_metrics(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                        constraints=no_constraints, Y2X=None, cov=None):
    # Create the design parameters
    fn = default_fn(1, cost_fn, None, constraints=constraints)
    params, col_names = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, ratios=ratios, Y2X=Y2X)

    # Transform Y to numpy
    if isinstance(Y, pd.DataFrame):
        if col_names is not None:
            Y = Y[col_names]
        Y = Y.to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types)

    # Define the metric inputs
    X = params.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) for ratios in params.ratios])
    costs = None
    if params.fn.cost is not None:
        costs = params.fn.cost(Y)

    # Initialize the metrics
    iopt = Iopt(n=10000, cov=cov)
    iopt.init(params)
    dopt = Dopt(cov=cov)
    dopt.init(params)
    aopt = Aopt(cov=cov)
    aopt.init(params)

    # Compute the metrics
    m_iopt = -iopt.call(Y, X, Zs, Vinv, costs)
    m_dopt = dopt.call(Y, X, Zs, Vinv, costs)
    m_aopt = -aopt.call(Y, X, Zs, Vinv, costs)

    # Return the metrics
    return (m_iopt, m_dopt, m_aopt)

def fraction_of_design_space(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                constraints=no_constraints, Y2X=None, cov=None):
    assert ratios is None or len(ratios.shape) == 1 or ratios.shape[0] == 1, 'Can only specify one set of variance ratios'

    # Create the design parameters
    fn = default_fn(1, cost_fn, None, constraints=constraints)
    params, col_names = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, ratios=ratios, Y2X=Y2X)

    # Transform Y to numpy
    if isinstance(Y, pd.DataFrame):
        if col_names is not None:
            Y = Y[col_names]
        Y = Y.to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types)

    # Define the metric inputs
    X = params.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.linalg.inv(obs_var_from_Zs(Zs, len(Y), params.ratios[0]))
    costs = None
    if params.fn.cost is not None:
        costs = params.fn.cost(Y)
    
    # Initialize Iopt
    iopt = Iopt(n=10000, cov=cov)
    iopt.init(params)

    # Compute information matrix
    if cov is not None:
        _, X, _, Vinv = cov(Y, X, Zs, Vinv, costs)
    M = X.T @ Vinv @ X

    # Compute prediction variances
    pred_var = np.sum(iopt.samples.T * np.linalg.solve(M, iopt.samples.T), axis=0)

    return np.sort(pred_var)

def plot_fraction_of_design_space(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                    constraints=no_constraints, Y2X=None, cov=None):
    # Compute prediction variances
    pred_var = fraction_of_design_space(Y, effect_types, cost_fn, model, grouped_cols, ratios, constraints, Y2X, cov)

    # Create the figure
    fig = go.Figure()
    color = DEFAULT_PLOTLY_COLORS[0]
    fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(pred_var)), y=pred_var, marker_color=color))
    fig.add_hline(y=np.mean(pred_var), annotation_text=f'{np.mean(pred_var):.3f}', annotation_font_color=DEFAULT_PLOTLY_COLORS[0], 
                  line_dash='dash', line_width=1, line_color=DEFAULT_PLOTLY_COLORS[0], annotation_position='bottom right')

    return fig

def estimation_variance_matrix(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                  constraints=no_constraints, Y2X=None, cov=None):
    assert ratios is None or len(ratios.shape) == 1 or ratios.shape[0] == 1, 'Can only specify one set of variance ratios'
    
    # Create the design parameters
    fn = default_fn(1, cost_fn, None, constraints=constraints)
    params, col_names = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, ratios=ratios, Y2X=Y2X)

    # Transform Y to numpy
    if isinstance(Y, pd.DataFrame):
        if col_names is not None:
            Y = Y[col_names]
        Y = Y.to_numpy()

    # Encode the columns names
    col_names_model = None
    if model is not None and cov is None:
        # Convert model to numpy
        if isinstance(model, pd.DataFrame):
            if col_names is not None:
                model = model[col_names]
            model = model.to_numpy()

        # Set default column names
        if col_names is None:
            col_names = list(np.arange(model.shape[1]).astype(str))
        
        # Encode the model
        model_enc = encode_model(model, params.effect_types)

        # Create the column names
        col_names_enc = encode_names(col_names, params.effect_types)
        col_names_model = model2names(model_enc, col_names_enc)

    # Encode the design
    Y = encode_design(Y, params.effect_types)

    # Define the metric inputs
    X = params.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.linalg.inv(obs_var_from_Zs(Zs, len(Y), params.ratios[0]))
    costs = None
    if params.fn.cost is not None:
        costs = params.fn.cost(Y)

    # Compute information matrix
    if cov is not None:
        _, X, _, Vinv = cov(Y, X, Zs, Vinv, costs)
    M = X.T @ Vinv @ X

    # Compute inverse of information matrix
    Minv = np.linalg.inv(M)

    # Add column names to information matrix
    if col_names_model is not None:
        Minv = pd.DataFrame(Minv, columns=col_names_model, index=col_names_model)

    # Create figure
    return Minv

def plot_estimation_variance_matrix(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                        constraints=no_constraints, Y2X=None, cov=None):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, effect_types, cost_fn, model, grouped_cols, ratios, constraints, Y2X, cov)

    # TODO: add columns names to imshow

    # Return the plot
    return px.imshow(Minv)

def estimation_variance(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                            constraints=no_constraints, Y2X=None, cov=None):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, effect_types, cost_fn, model, grouped_cols, ratios, constraints, Y2X, cov)

    return np.diag(Minv) 

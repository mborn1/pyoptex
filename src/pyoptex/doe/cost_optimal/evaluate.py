import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from .metric import Aopt, Dopt, Iopt
from .wrapper import default_fn, create_parameters
from .utils import obs_var_Zs
from ..utils.design import x2fx, encode_design

def evaluate_metrics(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                        constraints=no_constraints, Y2X=None, cov=None):
    # Create the design parameters
    fn = default_fn(1, cost_fn, None, constraints=constraints)
    params, _ = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, ratios=ratios, Y2X=Y2X)

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
    iopt = Iopt(n=1000000, cov=cov)
    iopt.init(params)
    dopt = Dopt(cov=cov)
    dopt.init(params)
    aopt = Aopt(cov=cov)
    aopt.init(params)

    # Compute the metrics
    m_iopt = iopt.call(Y, X, Zs, Vinv, costs)
    m_dopt = dopt.call(Y, X, Zs, Vinv, costs)
    m_aopt = aopt.call(Y, X, Zs, Vinv, costs)

    # Return the metrics
    return (m_iopt, m_dopt, m_aopt)

def fraction_of_design_space(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                constraints=no_constraints, Y2X=None, cov=None):
    assert len(ratios.shape) == 1 or ratios.shape[0] == 1, 'Can only specify one set of variance ratios'

    # Create the design parameters
    fn = default_fn(1, cost_fn, None, constraints=constraints)
    params, _ = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, ratios=ratios, Y2X=Y2X)

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
    iopt = Iopt(n=1000000, cov=cov)
    iopt.init(params)

    # Compute information matrix
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
    fig.add_hline(y=np.mean(pred_var), annotation_text=f'{np.mean(pred_var):.3f}', annotation_font_color=DEFAULT_PLOTLY_COLORS[i], 
                  line_dash='dash', line_width=1, line_color=DEFAULT_PLOTLY_COLORS[i], annotation_position='bottom right')

    return fig

def estimation_variance_matrix(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                  constraints=no_constraints, Y2X=None, cov=None):
    assert len(ratios.shape) == 1 or ratios.shape[0] == 1, 'Can only specify one set of variance ratios'
    
    # Create the design parameters
    fn = default_fn(1, cost_fn, None, constraints=constraints)
    params, _ = create_parameters(effect_types, fn, model=model, grouped_cols=grouped_cols, ratios=ratios, Y2X=Y2X)

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
    _, X, _, Vinv = cov(Y, X, Zs, Vinv, costs)
    M = X.T @ Vinv @ X

    # Compute inverse of information matrix
    Minv = np.linalg.inv(M)

    # Create figure
    return Minv

def plot_estimation_variance_matrix(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                                        constraints=no_constraints, Y2X=None, cov=None):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, effect_types, cost_fn, model, grouped_cols, ratios, constraints, Y2X, cov)

    # Return the plot
    return px.imshow(Minv)

def estimation_variance(Y, effect_types, cost_fn=None, model=None, grouped_cols=None, ratios=None,
                            constraints=no_constraints, Y2X=None, cov=None):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, effect_types, cost_fn, model, grouped_cols, ratios, constraints, Y2X, cov)

    return np.diag(Minv) 

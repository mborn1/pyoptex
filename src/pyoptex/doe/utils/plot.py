import plotly.express as px

def design_heatmap(Y):
    # Top-down plotting
    return px.imshow(Y, aspect='auto')

import plotly.express as px
import numpy as np
import pandas as pd
# from .evaluate import correlation_map

def design_heatmap(Y):
    # TODO: fix categorical factors

    # Top-down plotting
    return px.imshow(Y, aspect='auto')

def plot_correlation_map(Y, effect_types, model=None, Y2X=None, method='pearson', drop_nans=True):
    # Compute correlation map
    corr = correlation_map(Y, effect_types, model, Y2X, method)
    
    # Iteratively drop entire rows and columns of nans
    if drop_nans:
        bad = np.all(np.isnan(corr), axis=1)
        while np.any(bad):
            if isinstance(corr, pd.DataFrame):
                bad = bad.to_numpy()
                corr = corr.iloc[~bad, ~bad]
            else:
                corr = corr[~bad][:, ~bad]
            bad = np.all(np.isnan(corr), axis=1)

    return px.imshow(corr)

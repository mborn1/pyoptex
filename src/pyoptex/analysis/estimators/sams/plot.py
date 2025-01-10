"""
Module for SAMS plotting functions.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_raster(results, terms, skipn=0, metric_name='metric',
                forced=None, raster_terms=None, kmeans=None,
                fig=None):
    """
    Plot a raster of the results. This plot contains one row per model, sorted
    by metric to identify good performing submodels (=densly colored columns).

    Parameters
    ----------
    results : np.ndarray(1d)
        A complex numpy object with 'metric', 'model' and 'coeff' as
        subtypes. This is the result returned by
        :py:func:`simulate_sams <pyoptex.analysis.estimators.sams.simulation.simulate_sams>`.
    terms : list(str)
        A list the names of all the terms in the model.
    skipn : int
        The number of terms to indicate as skipped.
    metric_name : str
        The name of the metric, e.g., r2(adj)
    forced : None or np.array(1d)
        The terms to ignore in the raster plot. Sets the coefficients
        of these terms to zero.
    raster_terms : list(int) or np.array(1d)
        The terms to indicate in the raster.
    kmeans : :py:class:`sklearn.cluster.KMeans`
        The number of clusters in the raster plot. The kmeans should be trained by
        ordering the results according to 'metric' and skipping the first
        `skipn`. The object can optionally have an additional `skips` attribute
        to indicate a skip for each cluster. The skips should be
        an array of integers, with element i indicating the number of models
        to skip for group i.
    fig : None or tuple(:py:class:`plotly.graph_objects.Figure`, (int, int), (int, int))
        Either None or a tuple containing the subplots figure, the (row, col) to draw
        the raster on, and the (row, col) to draw the metric on. Most often, these
        are horizontally next to each other.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        A Plotly figure object of the raster.
    """
    # Order the results ascending
    idx = np.argsort(results['metric'])
    results = results[idx]
    
    # Check if we require kmeans plotting
    if kmeans is not None:
        # Initialize indices with first skipn
        idx = np.zeros(len(results), dtype=np.int_)
        idx[:skipn] = np.arange(skipn)

        # Initialize the array of skips and thresholds
        skips = [skipn]
        cluster_thresholds = [skipn]

        # Add default skips
        if not hasattr(kmeans, 'skips'):
            kmeans.skips = np.zeros(kmeans.n_clusters, dtype=np.int_)

        # Loop over all clusters
        for i in range(kmeans.n_clusters):
            # Extract indices of this cluster (offset by skipn)
            idx_ = np.flatnonzero(kmeans.labels_ == i) + skips[0]

            # Add the new skip and threshold
            if kmeans.skips[i] > 0:
                skips.append(cluster_thresholds[i] + kmeans.skips[i])
            cluster_thresholds.append(cluster_thresholds[i] + idx_.size)
            
            # Add the models to the index sorted by metric
            idx[cluster_thresholds[i]: cluster_thresholds[i+1]] = \
                    idx_[np.argsort(results['metric'][idx_])]

        # Order the results again
        results = results[idx]

        # Drop zero from skips if necessary
        if skips[0] == 0:
            skips = skips[1:]

        # Drop initial from cluster_thresholds
        cluster_thresholds = cluster_thresholds[1:]
    else:
        # Set no kmeans thresholds and only a single skip
        cluster_thresholds = []
        skips = [skipn] if skipn > 0 else []

    # Create the raster
    raster = np.zeros((len(results), len(terms)))
    if forced is not None:
        # Force uniqueness for speedup
        forced = np.unique(forced)

        # Set the coefficients
        for i, res in enumerate(results):
            model = res['model']
            non_forced_terms = np.isin(model, forced, assume_unique=True, invert=True)
            raster[i, model[non_forced_terms]] = res['coeff'][non_forced_terms]
    else:
        # Set the coefficients
        for i, res in enumerate(results):
            raster[i, res['model']] = res['coeff']

    # Normalize the coefficients based on absolute value
    raster /= np.expand_dims(np.max(np.abs(raster), 1), 1)

    ############################################

    # Create the figure
    if fig is None:
        # Create a new figure
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2])
        rc1 = {'row': 1, 'col': 1}
        rc2 = {'row': 1, 'col': 2}
        fig.update_layout(height=750)
    else:
        # Unpack the figure
        fig, rc1, rc2 = fig
        rc1 = {'row': rc1[0], 'col': rc1[1]}
        rc2 = {'row': rc2[0], 'col': rc2[1]}

    # Add the heatmap
    fig.add_trace(
        go.Heatmap(z=raster,
                   x=terms,
                   colorscale=[(0, 'red')] + [(0.5-1e-5, 'lightgray'), (0.5, 'white'), (0.5+1e-5, 'lightgray')] + [(1, 'blue')],
                   zmid=0,
                   customdata=np.expand_dims(np.broadcast_to(
                        np.arange(raster.shape[1]), 
                        raster.shape
                   ), 2),
                   hovertemplate='Term: %{x}'+
                                 '<br>Run: %{y}'+
                                 '<br>Coefficient: %{z}'+
                                 '<br>Term ID: %{customdata[0]}'
                ),
        **rc1
    )

    # Add the term annotations
    if raster_terms is not None:
        # Start y-value for the raster term
        y = 0.1

        # Add all raster terms as annotations
        for t in raster_terms:
            fig.add_annotation(
                x=t, 
                y=y * len(raster),
                text=f'<b>{terms[t]}</b>',
                font_size=18,
                showarrow=True,
                arrowhead=1,
                ax=(t - raster.shape[1]/2) / (raster.shape[1]/2) * -20,
                **rc1
            )
            y = (y + 0.2) % 0.8

    # Add the metric line plot
    fig.add_trace(
        go.Scatter(
            y=np.arange(raster.shape[0]), 
            x=results['metric'], 
            name=metric_name, 
            marker_color='black', 
            showlegend=False
        ), **rc2
    )
    fig.update_xaxes(title_text=metric_name, **rc2)

    # Add final metric annotation
    fig.add_annotation(
        x=results['metric'][-1], 
        y=results.size - 1,
        text=f'{results["metric"][-1]:.3f}',
        showarrow=True,
        arrowhead=1,
        ax=-50, ay=20,
        **rc2
    )

    # Add the thresholds
    for t in cluster_thresholds:
        fig.add_hline(
            y=t, line_dash='dot', line_color='black', 
            line_width=0.5, **rc1
        )
        fig.add_hline(
            y=t, line_dash='dot', line_color='black', 
            line_width=0.5, **rc2
        )

    # Add skipn horizontal lines
    for s in skips:
        fig.add_hline(
            s, line_dash='dash', **rc1
        )
        fig.add_hline(
            s, annotation_text='Skipped', annotation_position='bottom left', 
            line_dash='dash', **rc2
        )

    # The figure size and yrange
    fig.update_yaxes(range=[0, raster.shape[0]-1], **rc1)
    fig.update_yaxes(range=[0, raster.shape[0]-1], **rc2)

    # Return the figure
    return fig
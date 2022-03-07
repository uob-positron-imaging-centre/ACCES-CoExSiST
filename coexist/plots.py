#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plots.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.06.2021


import  colorsys

import  numpy               as      np
from    scipy.interpolate   import  NearestNDInterpolator

import  matplotlib.colors   as      mc
import  plotly.express      as      px
import  plotly.graph_objs   as      go
from    plotly.subplots     import  make_subplots

import  coexist




class LightAdjuster:
    '''Darken / lighten a given colour. Instantiate the class with the colour
    wanted, then call the object with a float / list of floats - <1.0 darkens,
    1.0 does not change the colour, >1.0 lightens.
    '''

    def __init__(self, color):
        try:
            c = mc.cnames[color]
        except KeyError:
            c = color

        if c.startswith("rgb"):
            self.c = colorsys.rgb_to_hls(*(
                co / 255 for co in px.colors.unlabel_rgb(color)
            ))
        else:
            self.c = colorsys.rgb_to_hls(*mc.to_rgb(c))


    def adjust(self, amount):
        # Shortname
        c = self.c

        color_tuple = colorsys.hls_to_rgb(
            c[0], max(0, min(1, amount * c[1])), c[2]
        )

        return px.colors.label_rgb([co * 255 for co in color_tuple])


    def __call__(self, amounts):
        if not hasattr(amounts, "__iter__"):
            return self.adjust(amounts)

        return [self.adjust(am) for am in amounts]




def format_fig(fig, size=20, font="Computer Modern", template="plotly_white"):
    '''Format a Plotly figure to a consistent theme for the Nature
    Computational Science journal.'''

    # LaTeX font
    fig.update_layout(
        font_family = font,
        font_size = size,
        title_font_family = font,
        title_font_size = size,
    )

    for an in fig.layout.annotations:
        an["font"]["size"] = size

    fig.update_xaxes(title_font_family = font, title_font_size = size)
    fig.update_yaxes(title_font_family = font, title_font_size = size)
    fig.update_layout(template = template)


def access(
    access_data,
    select = lambda results: results[:, -1] < np.inf,
    epochs = ...,
    colors = px.colors.qualitative.Set1,
    overall = False,
    means = True,
    confidence = True,
):
    '''Create a Plotly figure showing the solutions tried, uncertainties and
    error values found in a `coexist.Access` run.

    Parameters
    ----------
    access_data : coexist.AccessData or str
        An `AccessData` object containing all information about an ACCES run;
        you can initialise it with ``coexist.AccessData.read("folder_path")``.
        Alternatively, supply the ``folder_path`` directly.

    select : function, default lambda results: results[:, -1] < np.inf
        A filtering function used to plot only selected solutions tried, based
        on an input 2D table `results`, with columns formatted as [param1,
        param2, ..., param1_std, param2_std, ..., overall_std, error_value].
        E.g. to only plot solutions with an error value smaller than 100:
        ``select = lambda results: results[:, -1] < 100``.

    epochs : int or iterable or Ellipsis, default Ellipsis
        The index or indices of the epochs to plot. An `int` signifies a single
        epoch, an iterable (list-like) signifies multiple epochs, while an
        Ellipsis (`...`) signifies all epochs.

    colors : list[str], default plotly.express.colors.qualitative.Set1
        A list of colors used for each parameter plotted.

    overall : bool, default False
        If `True`, also plot the overall standard deviation progression; note
        that sometimes all parameters converge but the overall std-dev remains
        high.

    means : bool, default True
        If `True`, also plot the centre of the region explored by CMA-ES.

    confidence : bool, default True
        If `True`, also plot the standard deviation of each parameter as
        confidence intervals.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure containing subplots with the solutions tried. Call the
        `.show()` method to display it.

    Examples
    --------
    If `coexist.Access(filepath, random_seed = 12345)` was run, the
    directory "access_seed12345" would have been created. Plot its results:

    >>> import coexist
    >>> data = coexist.AccessData.read("access_seed12345")
    >>> fig = coexist.plots.access(data)
    >>> fig.show()

    Or more tersely:

    >>> import coexist
    >>> coexist.plots.access("access_seed12345").show()

    Only plot solution combinations that yielded an error value < 100:

    >>> coexist.plots.access(
    >>>      data,
    >>>      select = lambda results: results[:, -1] < 100,
    >>> ).show()
    '''

    # Type-checking inputs
    if not isinstance(access_data, coexist.AccessData):
        access_data = coexist.AccessData.read(access_data)

    # Check if sample_indices is an iterable collection (list-like)
    # otherwise just "iterate" over the single number or Ellipsis
    if not hasattr(epochs, "__iter__"):
        epochs = [epochs]

    # Extract data needed from `access_data`
    parameters = access_data.parameters

    # The data columns: [param1, param2, ..., error]
    results = access_data.results.to_numpy()

    epochs_unscaled = access_data.epochs.to_numpy()
    epochs_scaled = access_data.epochs_scaled.to_numpy()

    ns = access_data.population

    # The number of parameters
    num_parameters = len(parameters)
    num_errors = results.shape[1] - num_parameters
    names = parameters.index

    # Create a subplots grid
    ncols = int(np.ceil(np.sqrt(num_parameters + 1 + num_errors)))
    nrows = int(np.ceil((num_parameters + 1 + num_errors) / ncols))

    fig = make_subplots(
        rows = nrows,
        cols = ncols,
        shared_xaxes = True,
    )

    # Plot the parameter values checked per epoch
    num_epochs = access_data.num_epochs
    epochs_params = np.repeat(np.arange(num_epochs), ns)

    # Filter results plotted based on the error value and selected epochs
    selection = select(results)
    if epochs[0] is not Ellipsis:
        missing = set(range(num_epochs)) - set(epochs)
        for e in missing:
            selection[e * ns:e * ns + ns] = False

    # Compute relative error between 0 and 1
    error = results[selection, -1]
    relative_error = (error - error.min()) / (error.max() - error.min())

    # Plot solutions tried for each parameter
    for i in range(num_parameters):
        row = i // ncols + 1
        col = i % ncols + 1

        # Ensure the color_index does not go beyond the number of colours
        color_index = i
        while color_index >= len(colors):
            color_index -= len(colors)
        color = colors[color_index]
        adjuster = LightAdjuster(color)

        fig.add_trace(
            go.Scatter(
                name = names[i],
                x = epochs_params[selection],
                y = results[selection, i],
                mode = "markers",
                marker = dict(
                    size = 8,
                    opacity = 0.4,
                    color = adjuster(1 - relative_error),
                ),
                showlegend = False,
            ),
            row = row,
            col = col,
        )

        # Plot the unscaled means and std-dev as confidence intervals
        mu = epochs_unscaled[selection[::ns], i]

        if means:
            fig.add_trace(
                go.Scatter(
                    x = epochs_params[selection][::ns],
                    y = mu,
                    mode = "lines",
                    line = dict(
                        width = 2,
                        color = color,
                    ),
                    showlegend = False,
                ),
                row = row,
                col = col,
            )

        # Add transparency to confidence interval colour
        if confidence:
            color_alpha = color.replace("rgb", "rgba").split(")")[0] + ",0.2)"
            std_x = epochs_params[selection][::ns]
            std_lo = mu - epochs_unscaled[selection[::ns], i + num_parameters]
            std_hi = mu + epochs_unscaled[selection[::ns], i + num_parameters]

            fig.add_trace(
                go.Scatter(
                    x = std_x,
                    y = std_lo,
                    mode = "lines",
                    line_width = 0,
                    hoverinfo = "skip",
                    showlegend = False,
                ),
                row = row,
                col = col,
            )

            fig.add_trace(
                go.Scatter(
                    x = std_x,
                    y = std_hi,
                    mode = "lines",
                    line_width = 0,
                    fill = 'tonexty',
                    fillcolor = color_alpha,
                    hoverinfo = "skip",
                    showlegend = False,
                ),
                row = row,
                col = col,
            )

        # Plot the scaled standard deviations after parameter values
        fig.add_trace(
            go.Scatter(
                name = names[i],
                x = epochs_params[selection][::ns],
                y = epochs_scaled[selection[::ns], i + num_parameters],
                mode = "lines",
                line = dict(
                    color = color,
                ),
            ),
            row = num_parameters // ncols + 1,
            col = num_parameters % ncols + 1,
        )

    # Plot the overall standard deviation
    if overall:
        fig.add_trace(
            go.Scatter(
                name = "Overall standard deviation",
                x = epochs_params[selection],
                y = results[selection, -2],
                mode = "lines",
                line = dict(
                    color = "black",
                )
            ),
            row = num_parameters // ncols + 1,
            col = num_parameters % ncols + 1,
        )

    # Plot the error values
    for i in range(num_errors):
        row = (num_parameters + 1 + i) // ncols + 1
        col = (num_parameters + 1 + i) % ncols + 1

        fig.add_trace(
            go.Scatter(
                x = epochs_params[selection],
                y = results[selection, num_parameters + i],
                mode = "markers",
                marker = dict(
                    size = 8,
                    opacity = 0.4,
                    color = results[selection, -1],
                    colorscale = "cividis",
                ),
                showlegend = False,
            ),
            row = row,
            col = col,
        )

    # Set graph ranges and axis labels
    for i in range(num_parameters):
        xaxis = "xaxis" if i == 0 else f"xaxis{i + 1}"
        yaxis = "yaxis" if i == 0 else f"yaxis{i + 1}"

        fig.layout[xaxis].update(title = "Epoch")
        fig.layout[yaxis].update(title = names[i])

    # Set axis labels for the standard devation and error value subplots
    fig.layout[f"xaxis{num_parameters + 1}"].update(title = "Epoch")
    fig.layout[f"xaxis{num_parameters + 2}"].update(title = "Epoch")

    fig.layout[f"yaxis{num_parameters + 1}"].update(
        title = "Standard Deviation"
    )

    # If the default error names are given, capitalise them
    for i in range(num_errors):
        title = access_data.results.columns[num_parameters + i]
        if title.startswith("error"):
            title = "E" + title[1:]

            if i == num_errors - 1:
                title = "Combined " + title

        fig.layout[f"yaxis{num_parameters + 2 + i}"].update(title = title)

    format_fig(fig)
    fig.update_layout(title = dict(
        text = "ACCES Convergence Plot",
        font_size = 25,
    ))

    return fig


def access2d(
    access_data,
    resolution = (500, 500),
    width = 0.2,
    select = lambda results: results[:, -1] < np.inf,
    epochs = ...,
    colorscale = "Blues_r",
    scaled = False,
    seeds = True,
):
    '''Create a Plotly figure showing 2D Voronoi diagram of the error values
    found in 2D slices of the parameters explored in a `coexist.Access` run.

    Parameters
    ----------
    access_data : coexist.AccessData or str
        An `AccessData` object containing all information about an ACCES run;
        you can initialise it with ``coexist.AccessData.read("folder_path")``.
        Alternatively, supply the ``folder_path`` directly.

    resolution : 2-tuple, default (1000, 1000)
        The number of pixels in the heatmap / Voronoi diagram shown in the
        x- and y-dimensions.

    width : float, default 0.1
        The width of the slices as a ratio of the parameter range.

    select : function, default lambda results: results[:, -1] < np.inf
        A filtering function used to plot only selected solutions tried, based
        on an input 2D table `results`, with columns formatted as [param1,
        param2, ..., param1_std, param2_std, ..., overall_std, error_value].
        E.g. to only plot solutions with an error value smaller than 100:
        `select = lambda results: results[:, -1] < 100`.

    epochs : int or iterable or Ellipsis, default Ellipsis
        The index or indices of the epochs to plot. An `int` signifies a single
        epoch, an iterable (list-like) signifies multiple epochs, while an
        Ellipsis (`...`) signifies all epochs.

    colorscale : str, default "Blues_r"
        The colorscale used to colour-code the error value. For a list of
        possible colorscales, see `plotly.com/python/builtin-colorscales`.

    seeds : bool, default True
        If True, also plot the points representing parameter combinations
        tried.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure containing subplots with the solutions tried. Call the
        `.show()` method to display it.

    Examples
    --------
    If `coexist.Access(filepath, random_seed = 12345)` was run, the
    directory "access_seed12345" would have been created. Plot its results:

    >>> import coexist
    >>>
    >>> data = coexist.AccessData.read("access_seed12345")
    >>> fig = coexist.plots.access2d(data)
    >>> fig.show()

    Or more tersely:

    >>> import coexist
    >>> coexist.plots.access2d("access_seed12345").show()

    Only plot the results from epochs 5, 6, 7:

    >>> coexist.plots.access2d(data, epochs = [4, 5, 6]).show()

    Only plot a slice through a 3D parameter space for `fp1` and `fp3`, with
    0.4 < `fp2` < 0.6.

    >>> coexist.plot_access2d(
    >>>     data,
    >>>     columns = [0, 2]
    >>>     select = lambda res: (res[:, 1] > 0.4) & (res[:, 1] < 0.6),
    >>> ).show()
    '''

    # Type-checking inputs
    if not isinstance(access_data, coexist.AccessData):
        access_data = coexist.AccessData.read(access_data)

    # Check if sample_indices is an iterable collection (list-like)
    # otherwise just "iterate" over the single number or Ellipsis
    if not hasattr(epochs, "__iter__"):
        epochs = [epochs]

    # Extract data needed from `access_data`
    if scaled:
        results = access_data.results_scaled.to_numpy()
        epochs_raw = access_data.epochs_scaled.to_numpy()
        parameters = access_data.parameters_scaled
    else:
        results = access_data.results.to_numpy()
        epochs_raw = access_data.epochs.to_numpy()
        parameters = access_data.parameters

    num_params = len(parameters)
    names = parameters.index

    ns = access_data.population
    num_epochs = access_data.num_epochs

    # Create a subplots grid
    ncols = num_params - 1
    nrows = num_params - 1

    fig = make_subplots(
        rows = nrows,
        cols = ncols,
        shared_xaxes = True,
        shared_yaxes = True,
        horizontal_spacing = 0.1 / 2 / ncols,
        vertical_spacing = 0.2 / 2 / nrows,
    )

    # Filter results plotted based on the error value and selected epochs
    selection = select(results)
    if epochs[0] is not Ellipsis:
        # Handle negative epoch indices
        epochs = (e if e >= 0 else e + num_epochs for e in epochs)

        # Set the booleans in `selection` to False for epochs not requested
        missing = set(range(num_epochs)) - set(epochs)
        for e in missing:
            selection[e * ns:e * ns + ns] = False

    results = results[selection]
    epochs_raw = epochs_raw[selection[::ns]]

    # Create a 2D map of pixels coloured by the closest measured point's error
    error = results[:, -1]
    error_bounds = [error.min(), error.max()]
    error_scaled = (
        (error - error_bounds[0]) /
        (error_bounds[1] - error_bounds[0])
    )

    # Plot a lower triangular matrix without the diagonal, so for 3 parameters
    # => lower 2x2 triangle
    for i in range(1, num_params):
        for j in range(i):
            row = i
            col = j + 1

            # Create a 2D error map with each pixel mappend to the closest
            # sample's error
            x = np.linspace(parameters["min"][j], parameters["max"][j],
                            resolution[0])
            y = np.linspace(parameters["min"][i], parameters["max"][i],
                            resolution[1])

            # Select parameter space slice of given `width`
            cond = np.full(len(results), True)
            others = set(range(num_params)) - {i, j}
            for o in others:
                param_values = results[:, o]
                param_range = parameters[["min", "max"]].iloc[o]
                param_range = param_range[1] - param_range[0]
                mean = epochs_raw[-1, o]
                cond = cond & (
                    (param_values > mean - 0.5 * width * param_range) &
                    (param_values < mean + 0.5 * width * param_range)
                )

            xx, yy = np.meshgrid(x, y)
            error_map = NearestNDInterpolator(
                results[cond][:, [j, i]],
                error[cond],
            )(xx, yy)

            # Plot the 2D error map
            fig.add_trace(
                go.Heatmap(
                    x = x,
                    y = y,
                    z = error_map,
                    zmin = error_bounds[0],
                    zmax = error_bounds[1],
                    colorscale = colorscale,
                    colorbar_title = "Error",
                    showscale = row == col == 1,
                    showlegend = False,
                ),
                row = row,
                col = col,
            )

            # Plot points tried. Use inverted colorscale for good contrast
            # (i.e. color = 1 / errors)
            if seeds:
                fig.add_trace(
                    go.Scatter(
                        x = results[cond, j],
                        y = results[cond, i],
                        mode = "markers",
                        marker = dict(
                            size = 1 + 10 * error_scaled,
                            color = 1 / (error - error_bounds[0] + 1),
                            colorscale = colorscale,
                            colorbar_title = None,
                        ),
                        showlegend = False,
                    ),
                    row = row,
                    col = col,
                )

            # Set axis labels
            isub = (row - 1) * ncols + col
            xaxis = "xaxis" if isub == 1 else f"xaxis{isub}"
            yaxis = "yaxis" if isub == 1 else f"yaxis{isub}"

            # bounds = parameters[["min", "max"]]
            # fig.layout[xaxis].update(range = bounds.iloc[j].to_numpy())
            # fig.layout[yaxis].update(range = bounds.iloc[i].to_numpy())

            if col == 1:
                fig.layout[yaxis].update(title = names[i])

            if row == nrows:
                fig.layout[xaxis].update(title = names[j])


    format_fig(fig)
    fig.update_xaxes(showgrid = False, zeroline = False)
    fig.update_yaxes(showgrid = False, zeroline = False)

    prefix = "Scaled " if scaled else ""
    fig.update_layout(title = dict(
        text = (
            f"{prefix}ACCES Voronoi Plot - Parameter Slices Width = "
            f"{width * 100:3.1f}% Data Range"
        ),
        font_size = 25,
    ))

    return fig


def surrogate2d(
    access_surrogate,
    resolution = (500, 500),
    width = 0.2,
    select = lambda results: results[:, -1] < np.inf,
    epochs = ...,
    colorscale = "Blues_r",
    scaled = False,
    seeds = True,
):
    '''Create a Plotly figure showing 2D Voronoi diagram of the error values
    found in 2D slices of the parameters explored in a `coexist.Access` run.

    Parameters
    ----------
    access_data : coexist.AccessData or str
        An `AccessData` object containing all information about an ACCES run;
        you can initialise it with ``coexist.AccessData.read("folder_path")``.
        Alternatively, supply the ``folder_path`` directly.

    resolution : 2-tuple, default (1000, 1000)
        The number of pixels in the heatmap / Voronoi diagram shown in the
        x- and y-dimensions.

    width : float, default 0.1
        The width of the slices as a ratio of the parameter range.

    select : function, default lambda results: results[:, -1] < np.inf
        A filtering function used to plot only selected solutions tried, based
        on an input 2D table `results`, with columns formatted as [param1,
        param2, ..., param1_std, param2_std, ..., overall_std, error_value].
        E.g. to only plot solutions with an error value smaller than 100:
        `select = lambda results: results[:, -1] < 100`.

    epochs : int or iterable or Ellipsis, default Ellipsis
        The index or indices of the epochs to plot. An `int` signifies a single
        epoch, an iterable (list-like) signifies multiple epochs, while an
        Ellipsis (`...`) signifies all epochs.

    colorscale : str, default "Blues_r"
        The colorscale used to colour-code the error value. For a list of
        possible colorscales, see `plotly.com/python/builtin-colorscales`.

    seeds : bool, default True
        If True, also plot the points representing parameter combinations
        tried.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure containing subplots with the solutions tried. Call the
        `.show()` method to display it.

    Examples
    --------
    If `coexist.Access(filepath, random_seed = 12345)` was run, the
    directory "access_seed12345" would have been created. Plot its results:

    >>> import coexist
    >>>
    >>> data = coexist.AccessData.read("access_seed12345")
    >>> fig = coexist.plots.access2d(data)
    >>> fig.show()

    Or more tersely:

    >>> import coexist
    >>> coexist.plots.access2d("access_seed12345").show()

    Only plot the results from epochs 5, 6, 7:

    >>> coexist.plots.access2d(data, epochs = [4, 5, 6]).show()

    Only plot a slice through a 3D parameter space for `fp1` and `fp3`, with
    0.4 < `fp2` < 0.6.

    >>> coexist.plot_access2d(
    >>>     data,
    >>>     columns = [0, 2]
    >>>     select = lambda res: (res[:, 1] > 0.4) & (res[:, 1] < 0.6),
    >>> ).show()
    '''

    # Type-checking inputs
    if not isinstance(access_surrogate, coexist.AccessSurrogate):
        access_data = coexist.AccessData.read(access_data)

    # Check if sample_indices is an iterable collection (list-like)
    # otherwise just "iterate" over the single number or Ellipsis
    if not hasattr(epochs, "__iter__"):
        epochs = [epochs]

    # Extract data needed from `access_data`
    parameters = access_data.parameters
    num_params = len(parameters)
    names = parameters.index

    # The data columns: [param1, param2, ..., error]
    if scaled:
        results = access_data.results_scaled.to_numpy()
        epochs_raw = access_data.epochs_scaled.to_numpy()

        # Scale parameter values
        scaling = (
            access_data.results.to_numpy() /
            access_data.results_scaled.to_numpy()
        )
        scaling = np.mean(scaling[:, :-1], axis = 0)
        parameters = parameters.copy()
        parameters["min"] /= scaling
        parameters["max"] /= scaling
    else:
        results = access_data.results.to_numpy()
        epochs_raw = access_data.epochs.to_numpy()

    ns = access_data.population
    num_epochs = access_data.num_epochs

    # Create a subplots grid
    ncols = num_params - 1
    nrows = num_params - 1

    fig = make_subplots(
        rows = nrows,
        cols = ncols,
        shared_xaxes = True,
        shared_yaxes = True,
        horizontal_spacing = 0.1 / 2 / ncols,
        vertical_spacing = 0.2 / 2 / nrows,
    )

    # Filter results plotted based on the error value and selected epochs
    selection = select(results)
    if epochs[0] is not Ellipsis:
        # Handle negative epoch indices
        epochs = (e if e >= 0 else e + num_epochs for e in epochs)

        # Set the booleans in `selection` to False for epochs not requested
        missing = set(range(num_epochs)) - set(epochs)
        for e in missing:
            selection[e * ns:e * ns + ns] = False

    results = results[selection]
    epochs_raw = epochs_raw[selection[::ns]]

    # Create a 2D map of pixels coloured by the closest measured point's error
    error = results[:, -1]
    error_bounds = [error.min(), error.max()]
    error_scaled = (
        (error - error_bounds[0]) /
        (error_bounds[1] - error_bounds[0])
    )

    # Plot a lower triangular matrix without the diagonal, so for 3 parameters
    # => lower 2x2 triangle
    for i in range(1, num_params):
        for j in range(i):
            row = i
            col = j + 1

            # Create a 2D error map with each pixel mappend to the closest
            # sample's error
            x = np.linspace(parameters["min"][j], parameters["max"][j],
                            resolution[0])
            y = np.linspace(parameters["min"][i], parameters["max"][i],
                            resolution[1])

            # Select parameter space slice of given `width`
            cond = np.full(len(results), True)
            others = set(range(num_params)) - {i, j}
            for o in others:
                param_values = results[:, o]
                param_range = parameters[["min", "max"]].iloc[o]
                param_range = param_range[1] - param_range[0]
                mean = epochs_raw[-1, o]
                cond = cond & (
                    (param_values > mean - 0.5 * width * param_range) &
                    (param_values < mean + 0.5 * width * param_range)
                )

            xx, yy = np.meshgrid(x, y)
            error_map = NearestNDInterpolator(
                results[cond][:, [j, i]],
                error[cond],
            )(xx, yy)

            # Plot the 2D error map
            fig.add_trace(
                go.Heatmap(
                    x = x,
                    y = y,
                    z = error_map,
                    zmin = error_bounds[0],
                    zmax = error_bounds[1],
                    colorscale = colorscale,
                    colorbar_title = "Error",
                    showscale = row == col == 1,
                    showlegend = False,
                ),
                row = row,
                col = col,
            )

            # Plot points tried. Use inverted colorscale for good contrast
            # (i.e. color = 1 / errors)
            if seeds:
                fig.add_trace(
                    go.Scatter(
                        x = results[cond, j],
                        y = results[cond, i],
                        mode = "markers",
                        marker = dict(
                            size = 1 + 10 * error_scaled,
                            color = 1 / (error - error_bounds[0] + 1),
                            colorscale = colorscale,
                            colorbar_title = None,
                        ),
                        showlegend = False,
                    ),
                    row = row,
                    col = col,
                )

            # Set axis labels
            isub = (row - 1) * ncols + col
            xaxis = "xaxis" if isub == 1 else f"xaxis{isub}"
            yaxis = "yaxis" if isub == 1 else f"yaxis{isub}"

            # bounds = parameters[["min", "max"]]
            # fig.layout[xaxis].update(range = bounds.iloc[j].to_numpy())
            # fig.layout[yaxis].update(range = bounds.iloc[i].to_numpy())

            if col == 1:
                fig.layout[yaxis].update(title = names[i])

            if row == nrows:
                fig.layout[xaxis].update(title = names[j])


    format_fig(fig)
    # fig.update_xaxes(showgrid = False, zeroline = False)
    # fig.update_yaxes(showgrid = False, zeroline = False)

    prefix = "Scaled " if scaled else ""
    fig.update_layout(title = dict(
        text = (
            f"{prefix}ACCES Voronoi Plot - Parameter Slices Width = "
            f"{width * 100:3.1f}% Data Range"
        ),
        font_size = 25,
    ))

    return fig



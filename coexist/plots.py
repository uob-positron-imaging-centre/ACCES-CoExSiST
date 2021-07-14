#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plots.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.06.2021


import  os
import  re
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




def format_fig(fig):
    '''Format a Plotly figure to a consistent theme for the Nature
    Computational Science journal.'''

    # LaTeX font
    font = "Computer Modern"
    size = 20
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
    fig.update_layout(template = "plotly_white",)


def find_history(history_path):
    '''Find the `opt_history_<num_solutions>.csv` file for the following cases:

        1. `history_path` points to the actual file
        2. `history_path` points to an `access_info_<hash>` folder
        3. `history_path` is a directory that contains `opt_history_<ns>`
        4. `history_path` is a directory that contains `access_info_<hash>`

    Returns the full path to the history file.
    '''

    ns_finder = re.compile(r"opt_history_[0-9]+\.csv")

    # If we received e.g. "simulation/access_info_123456/opt_history_100.csv"
    if os.path.split(history_path)[1].startswith("opt_history"):
        pass

    # If we received e.g. "simulation/access_info_123456"
    elif os.path.split(history_path)[1].startswith("access_info"):
        for f in os.listdir(history_path):
            if ns_finder.search(f):
                history_path = os.path.join(history_path, f)

    # If we received e.g. "simulation/"
    else:
        files = os.listdir(history_path)

        # First check if there is an `opt_history_<ns>` file
        for f in files:
            if ns_finder.search(f):
                history_path = os.path.join(history_path, f)
                break

        # Otherwise check if there is an `access_info_<hash>` folder
        else:
            access_finder = re.compile(r"access_info_[0-9]+")
            access_infos = [f for f in files if access_finder.search(f)]
            if len(access_infos) > 1:
                raise ValueError((
                    f"The path provided in `{history_path=}` contains "
                    "multiple `access_info_<hash>` folders:\n"
                    f"{access_infos}\nPlease provide a path to a specific "
                    "`access_info_<hash>`."
                ))

            if len(access_infos) == 0:
                raise FileNotFoundError((
                    f"The path provided in `{history_path=}` does not point "
                    "to a `opt_history_<ns>.csv` file, or an ACCESS folder."
                ))

            history_path = access_infos[0]
            for f in os.listdir(history_path):
                if ns_finder.search(f):
                    history_path = os.path.join(history_path, f)

    return history_path


def plot_access(
    history_path,
    parameters = None,
    select = lambda results: results[:, -1] < np.inf,
    epochs = ...,
    colors = px.colors.qualitative.Set1,
    overall = True,
):
    '''Create a Plotly figure showing the solutions tried, uncertainties and
    error values found in a `coexist.AccessScript` run.

    Parameters
    ----------
    history_path: str
        A path to an ACCES history file
        (e.g. "sim/access_info_123456/opt_history_100.csv"), an ACCES run
        folder (e.g. "sim/access_info_123456"), or a directory where a single
        ACCES run exists (e.g. "sim/")

    parameters: pandas.DataFrame, optional
        The `parameters` dataframe used to run ACCESS, exactly the same as in
        the simulation script used. If provided, extra information will be
        plotted.

    select: function, default lambda results: results[:, -1] < np.inf
        A filtering function used to plot only selected solutions tried, based
        on an input 2D table `results`, with columns formatted as [param1,
        param2, ..., param1_std, param2_std, ..., overall_std, error_value].
        E.g. to only plot solutions with an error value smaller than 100:
        `select = lambda results: results[:, -1] < 100`.

    epochs: int or iterable or Ellipsis, default Ellipsis
        The index or indices of the epochs to plot. An `int` signifies a single
        epoch, an iterable (list-like) signifies multiple epochs, while an
        Ellipsis (`...`) signifies all epochs.

    colors: list[str], default plotly.express.colors.qualitative.Set1
        A list of colors used for each parameter plotted.

    overall: bool, default True
        If `True`, also plot the overall standard deviation.

    Returns
    -------
    fig: plotly.graph_objs.Figure
        A Plotly figure containing subplots with the solutions tried. Call the
        `.show()` method to display it.

    Example
    -------
    If `coexist.AccessScript(filepath, random_seed = 12345)` was run, the
    directory "access_info_227336" would have been created. Plot its results:

    >>> import coexist
    >>> fig = coexist.plot_access("access_info_227336")
    >>> fig.show()

    Only plot solution combinations that yielded an error value < 100:

    >>> coexist.plot_access(
    >>>     "access_info_227336",
    >>>      select = lambda results: results[:, -1] < 100,
    >>> ).show()

    If the ACCESS parameters used (created in the simulation script) are given,
    the parameters' names will be plotted and the standard deviations will be
    scaled between 0 and 1:

    >>> parameters = coexist.create_parameters(
    >>>     variables = ["fp1", "fp2"],
    >>>     minimums = [0, 0],
    >>>     maximums = [1, 2],
    >>> )
    >>>
    >>> coexist.plot_access("access_info_227336", parameters).show()

    Notes
    -----
    At least the "opt_history_<num_solutions>.csv" file is needed. If the
    "opt_history_<num_solutions>_scaled.csv" file is found too, then the
    scaled standard deviations will be plotted (i.e. between 0 and 1).
    '''

    # Type-checking inputs
    if parameters is not None:
        coexist.AccessScript.validate_parameters(parameters)

    # Check if sample_indices is an iterable collection (list-like)
    # otherwise just "iterate" over the single number or Ellipsis
    if not hasattr(epochs, "__iter__"):
        epochs = [epochs]

    # Find the path to the `opt_history_<num_solutions>.csv` file
    history_path = find_history(history_path)

    # The data columns: [param1, param2, ..., param1_stddev, param2_stddev,
    # ..., overall_std_dev, error]
    results = np.loadtxt(history_path)

    # Load scaled results for the scaled standard deviation
    history_path_scaled = history_path.split(".csv")[0] + "_scaled.csv"
    if not os.path.isfile(history_path_scaled):
        results_scaled = results
    else:
        results_scaled = np.loadtxt(history_path_scaled)

    # Number of solutions (ns) per epoch
    ns_extractor = re.compile(r"opt_history_|\.csv")
    ns = int(ns_extractor.split(os.path.split(history_path)[1])[1])

    # The number of parameters
    num_parameters = (results.shape[1] - 2) // 2

    # If `parameters` were provided, plot extra information
    if parameters is not None:
        names = parameters.index

        # If no scaled results were provided, we can still scale by the
        # parameters' sigma
        if not os.path.isfile(history_path_scaled):
            results[:, num_parameters:-2] /= \
                parameters["sigma"].to_numpy()

    else:
        names = [f"Parameter {i + 1}" for i in range(num_parameters)]

    # Create a subplots grid
    ncols = int(np.ceil(np.sqrt(num_parameters + 2)))
    nrows = int(np.ceil((num_parameters + 2) / ncols))

    fig = make_subplots(
        rows = nrows,
        cols = ncols,
        shared_xaxes = True,
    )

    # Plot the parameter values checked per epoch
    num_epochs = results.shape[0] // ns
    epochs_params = np.repeat(np.arange(num_epochs), ns)

    # Filter results plotted based on the error value and selected epochs
    selection = select(results)
    if epochs[0] is not Ellipsis:
        missing = set(range(num_epochs)) - set(epochs)
        for e in missing:
            selection[e * ns:e * ns + ns] = False

    # Compute relative error between 0 and 2
    error = results[selection, -1]
    relative_error = (error - error.min()) / (error.max() - error.min())

    # Plot solutions tried for each parameter
    for i in range(num_parameters):
        row = i // ncols + 1
        col = i % ncols + 1

        color = colors[i]
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

        # Plot the standard deviations and results on the bottom row
        fig.add_trace(
            go.Scatter(
                name = names[i],
                x = epochs_params[selection],
                y = results_scaled[selection, i + num_parameters],
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
    fig.add_trace(
        go.Scatter(
            name = "Error values",
            x = epochs_params[selection],
            y = results[selection, -1],
            mode = "markers",
            marker = dict(
                size = 8,
                opacity = 0.4,
                color = "black",
            )
        ),
        row = (num_parameters + 1) // ncols + 1,
        col = (num_parameters + 1) % ncols + 1,
    )

    # Set graph ranges and axis labels
    for i in range(num_parameters):
        row = i // ncols + 1
        col = i % ncols + 1

        xaxis = "xaxis" if i == 0 else f"xaxis{i + 1}"
        yaxis = "yaxis" if i == 0 else f"yaxis{i + 1}"

        fig.layout[xaxis].update(title = "Epoch")
        fig.layout[yaxis].update(title = names[i])

        if parameters is not None:
            param = parameters.iloc[i]
            fig.layout[yaxis].update(range = [param["min"], param["max"]])

    # Set axis labels for the standard devation and error value subplots
    fig.layout[f"xaxis{num_parameters + 1}"].update(title = "Epoch")
    fig.layout[f"xaxis{num_parameters + 2}"].update(title = "Epoch")

    fig.layout[f"yaxis{num_parameters + 1}"].update(
        title = "Standard Deviation"
    )
    fig.layout[f"yaxis{num_parameters + 2}"].update(title = "Error Value")

    format_fig(fig)

    return fig


def plot_access2d(
    history_path,
    parameters,
    resolution = (1000, 1000),
    columns = [0, 1],
    select = lambda results: results[:, -1] < np.inf,
    epochs = ...,
    colorscale = "Blues_r",
):
    '''Create a Plotly figure showing a 2D Voronoi diagram of the error values
    found in a `coexist.AccessScript` run.

    This can be used to visualise a 2D optimisation problem, or a slice through
    higher-dimensional error functions.

    Parameters
    ----------
    history_path: str
        A path to an ACCES history file
        (e.g. "sim/access_info_123456/opt_history_100.csv"), an ACCES run
        folder (e.g. "sim/access_info_123456"), or a directory where a single
        ACCES run exists (e.g. "sim/")

    parameters: pandas.DataFrame
        The `parameters` dataframe used to run ACCESS, exactly the same as in
        the simulation script used.

    resolution: 2-tuple, default (1000, 1000)
        The number of pixels in the heatmap / Voronoi diagram shown in the
        x- and y-dimensions.

    columns: 2-list, default [0, 1]
        The columns in the ACCES history file to plot, corresponding to the
        free parameter indices; e.g. [0, 1] represents the first two
        parameters.

    select: function, default lambda results: results[:, -1] < np.inf
        A filtering function used to plot only selected solutions tried, based
        on an input 2D table `results`, with columns formatted as [param1,
        param2, ..., param1_std, param2_std, ..., overall_std, error_value].
        E.g. to only plot solutions with an error value smaller than 100:
        `select = lambda results: results[:, -1] < 100`.

    epochs: int or iterable or Ellipsis, default Ellipsis
        The index or indices of the epochs to plot. An `int` signifies a single
        epoch, an iterable (list-like) signifies multiple epochs, while an
        Ellipsis (`...`) signifies all epochs.

    colorscale: str, "Blues_r"
        The colorscale used to colour-code the error value. For a list of
        possible colorscales, see `plotly.com/python/builtin-colorscales`.

    Returns
    -------
    fig: plotly.graph_objs.Figure
        A Plotly figure containing subplots with the solutions tried. Call the
        `.show()` method to display it.

    Example
    -------
    If `coexist.AccessScript(filepath, random_seed = 12345)` was run, the
    directory "access_info_227336" would have been created. Plot its results:

    >>> import coexist
    >>>
    >>> parameters = coexist.create_parameters(
    >>>     variables = ["fp1", "fp2", "fp3"],
    >>>     minimums = [0, 0, 0],
    >>>     maximums = [1, 2, 1],
    >>> )
    >>>
    >>> fig = coexist.plot_access2d("access_info_227336", parameters)
    >>> fig.show()

    Only plot the results from epochs 5, 6, 7:

    >>> coexist.plot_access2d(
    >>>     "access_info_227336",
    >>>     parameters,
    >>>     epochs = [4, 5, 6],
    >>> ).show()

    Only plot a slice through a 3D parameter space for `fp1` and `fp3`, with
    0.4 < `fp2` < 0.6.

    >>> coexist.plot_access2d(
    >>>     "access_info_227336",
    >>>     parameters,
    >>>     columns = [0, 2]
    >>>     select = lambda res: (res[:, 1] > 0.4) & (res[:, 1] < 0.6),
    >>> ).show()

    '''

    # Type-checking inputs
    coexist.AccessScript.validate_parameters(parameters)
    if len(columns) != 2:
        raise ValueError((
            f"The input `{columns=}` must contain exactly two columns to plot "
            "corresponding to the parameter indices (e.g. [0, 1] to plot the "
            "first two parameters)."
        ))

    # Check if sample_indices is an iterable collection (list-like)
    # otherwise just "iterate" over the single number or Ellipsis
    if not hasattr(epochs, "__iter__"):
        epochs = [epochs]

    # Find the path to the `opt_history_<num_solutions>.csv` file
    history_path = find_history(history_path)

    # The data columns: [param1, param2, ..., param1_stddev, param2_stddev,
    # ..., overall_std_dev, error]
    results = np.loadtxt(history_path)

    # Number of solutions (ns) per epoch
    ns_extractor = re.compile(r"opt_history_|\.csv")
    ns = int(ns_extractor.split(os.path.split(history_path)[1])[1])

    # Plot the parameter values checked per epoch
    num_epochs = results.shape[0] // ns

    # Filter results plotted based on the error value and selected epochs
    selection = select(results)
    if epochs[0] is not Ellipsis:
        # Handle negative epoch indices
        epochs = (e if e >= 0 else e + num_epochs for e in epochs)

        # Set the booleans in `selection` to False for epochs not requested
        missing = set(range(num_epochs)) - set(epochs)
        for e in missing:
            selection[e * ns:e * ns + ns] = False

    # Create a 2D map of pixels coloured by the closest measured point's error
    x = np.linspace(parameters["min"][0], parameters["max"][0], resolution[0])
    y = np.linspace(parameters["min"][1], parameters["max"][1], resolution[1])

    xx, yy = np.meshgrid(x, y)
    error_map = NearestNDInterpolator(
        results[selection][:, columns],
        results[selection][:, -1],
    )(xx, yy)

    # Plot the 2D error map
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = x,
            y = y,
            z = error_map,
            colorscale = colorscale,
        )
    )
    fig.update_xaxes(title = parameters.index[0])
    fig.update_yaxes(title = parameters.index[1])

    format_fig(fig)
    return fig

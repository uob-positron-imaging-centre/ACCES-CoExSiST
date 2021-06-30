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
            self.c = colorsys.rgb_to_hls(*[
                co / 255 for co in px.colors.unlabel_rgb(color)
            ])
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
        font_family=font,
        font_size=size,
        title_font_family=font,
        title_font_size=size,
    )

    for an in fig.layout.annotations:
        an["font"]["size"] = size

    fig.update_xaxes(title_font_family=font, title_font_size=size)
    fig.update_yaxes(title_font_family=font, title_font_size=size)

    fig.update_layout(
        # legend = dict(yanchor="top", xanchor="right"),
        template = "plotly_white",
    )


def plot_access(
    history_path,
    parameters = None,
    select = lambda errors: errors < np.inf,
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

    select: function, default lambda errors: errors < np.inf
        A filtering function used to plot only the selected solutions tried,
        based on its error value. E.g. only plot solutions with an error value
        smaller than 100: `select = lambda errors: errors < 100`.

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
    >>>      select = lambda errors: errors < 100,
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

    # Find the history file and extract the number of solutions (ns) from the
    # file name
    ns_finder = re.compile(r"opt_history_[0-9]+\.csv")
    ns_extractor = re.compile(r"opt_history_|\.csv")

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
        access_finder = re.compile(r"access_info_[0-9]+")
        access_infos = [f for f in os.listdir() if access_finder.search(f)]
        if len(access_infos) > 1:
            raise ValueError((
                f"The path provided in `{history_path=}` contains multiple "
                f"`access_info_<hash>` folders:\n{access_infos}\n"
                "Please provide a path to a specific `access_info_<hash>`."
            ))
        if len(access_infos) == 0:
            raise FileNotFoundError((
                f"The path provided in `{history_path=}` does not point to a "
                "`opt_history_<ns>.csv` file, or an ACCESS folder."
            ))

        history_path = access_infos[0]
        for f in os.listdir(history_path):
            if ns_finder.search(f):
                history_path = os.path.join(history_path, f)

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
    ns = int(ns_extractor.split(os.path.split(history_path)[1])[1])

    # The number of parameters
    num_parameters = (results.shape[1] - 2) // 2

    # If `parameters` were provided, plot extra information
    names = parameters.index if parameters else np.full(num_parameters, None)
    if parameters is not None:
        names = parameters.index

        # If no scaled results were provided, we can still scale by the
        # parameters' sigma
        if not os.path.isfile(history_path_scaled):
            results[num_parameters:num_parameters - 2] /= \
                parameters["sigma"].to_numpy()

    else:
        names = [f"Parameter {i + 1}" for i in range(num_parameters)]

    # Colorscheme used
    colors = px.colors.qualitative.Set1

    # Create a subplots grid
    ncols = int(np.ceil(np.sqrt(num_parameters + 2)))
    nrows = int(np.ceil((num_parameters + 2) / ncols))

    subplot_titles = list(names) + [
        None for _ in range(ncols * nrows - num_parameters)
    ]
    subplot_titles[-2] = "Standard Deviations"
    subplot_titles[-1] = "Error Values"

    fig = make_subplots(
        rows = nrows,
        cols = ncols,
        shared_xaxes = True,
        subplot_titles = subplot_titles,
    )

    # Plot the parameter values checked per epoch
    num_epochs = results.shape[0] // ns
    epochs = np.arange(num_epochs)
    epochs_params = np.repeat(epochs, ns)

    # Filter results plotted based on the error value
    selection = select(results[:, -1])

    # Compute relative error between 0 and 2
    error = results[selection, -1]
    relative_error = (error - error.min()) / (error.max() - error.min())

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
                )
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
                )
            ),
            row = nrows,
            col = ncols - 1,
        )

    # Plot the overall standard deviation
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
        row = nrows,
        col = ncols - 1,
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
        row = nrows,
        col = ncols,
    )

    # Set graph ranges
    for i in range(num_parameters):
        row = i // ncols + 1
        col = i % ncols + 1

        xaxis = "xaxis" if i == 0 else f"xaxis{i + 1}"
        yaxis = "yaxis" if i == 0 else f"yaxis{i + 1}"

        fig.layout[xaxis].update(range = [0, num_epochs])

        if parameters is not None:
            param = parameters.iloc[i]
            fig.layout[yaxis].update(range = [param["min"], param["max"]])

    format_fig(fig)

    return fig

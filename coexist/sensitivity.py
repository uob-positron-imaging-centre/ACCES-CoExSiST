#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sensitivity.py
# License: GNU v3.0


'''Local parameter sensitivity around an existing optimisation result.

Fit a Gaussian process to a sampled objective and find parameter variations
keeping its prediction within a symmetric tolerance band around a reference.
'''


import  json
from    dataclasses                      import  dataclass
from    itertools                        import  combinations
from    pathlib                          import  Path

import  numpy                            as      np
import  pandas                           as      pd
import  matplotlib.pyplot                as      plt
from    matplotlib.colors                import  Normalize, LogNorm, SymLogNorm
from    matplotlib.lines                 import  Line2D
from    matplotlib.patches               import  Patch
from    scipy.optimize                   import  brentq, minimize

try:
    import  sklearn
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    raise ModuleNotFoundError(
        'Sensitivity analysis requires scikit-learn. Install it with '
        '`python -m pip install "coexist[sensitivity]"`.',
        name = "sklearn",
    ) from error

from    sklearn.compose                  import  TransformedTargetRegressor
from    sklearn.gaussian_process         import  GaussianProcessRegressor
from    sklearn.gaussian_process.kernels import  ConstantKernel, Matern
from    sklearn.gaussian_process.kernels import  WhiteKernel
from    sklearn.pipeline                 import  make_pipeline
from    sklearn.preprocessing            import  StandardScaler




@dataclass
class SensitivityResult:
    '''Tables, plots and fitted model from a parameter sensitivity analysis.

    All parameter values and objective responses are in their original units.
    The ranges describe the fitted response within the supplied tolerance;
    they are not experimental confidence intervals.

    Attributes
    ----------
    reference : pd.Series
        The parameter vector around which variations are evaluated.

    ranges : pd.DataFrame
        Accepted parameter intervals, with one row per parameter, mode and
        interval. "slice" holds the other parameters fixed; "profile" allows
        them to adjust. Multiple rows preserve disconnected intervals.
        ``contains_reference_value`` identifies intervals containing that
        parameter's reference coordinate; it does not establish a connected
        path to the reference in the full parameter space.

    importance : pd.DataFrame
        Root-mean-square (RMS) and maximum absolute objective changes over
        each parameter's reporting window, holding the others fixed.

    ranking : pd.DataFrame
        Dimensionless ``relative_sensitivity`` scores: each parameter's RMS
        change divided by the sum across parameters. Scores sum to one, or
        are all zero for a constant response. They depend on the reporting
        window and are not fractions of objective variance.

    importance_full : pd.DataFrame
        The same objective-change measures as ``importance``, evaluated over
        each parameter's complete min/max bounds using the same fitted model.

    ranking_full : pd.DataFrame
        Dimensionless ranking over the complete bounds, normalised separately
        from ``ranking``. The other parameters remain fixed at the reference.
        These curves can include regions with few observations.

    interactions : pd.DataFrame
        RMS and maximum absolute departures of paired responses from the sum
        of their individual effects, with other parameters fixed.

    single : pd.DataFrame
        Individual response curves, acceptance flags and compensating
        parameter values for each grid point.

    pairs : dict[tuple, pd.DataFrame]
        Parameter-name pairs mapped to their joint response grids. Each table
        includes fixed and adjusted responses and compensating parameters.

    sampling : pd.DataFrame
        Bounds, reporting windows and sampled spans for each parameter.

    observed_acceptable : pd.DataFrame
        Actual evaluations within the reporting box whose objective lies
        inside the tolerance band. Acceptance applies to each full vector.

    metadata : dict
        Reference objective, tolerance limits, fitting settings and counts.

    model : sklearn.compose.TransformedTargetRegressor
        The fitted Gaussian process and its transformations. ``model.predict``
        expects inputs scaled as ``(parameters - reference) / bound_widths``.

    Examples
    --------
    Analyse a saved ACCES run and inspect its parameter ranges and rankings:

    >>> import coexist
    >>> data = coexist.AccessData("access_seed42")
    >>> result = data.sensitivity()
    >>> result.ranges
    >>> result.ranking
    >>> result.ranking_full
    >>> result.save("sensitivity_output")
    '''

    reference: pd.Series
    ranges: pd.DataFrame
    importance: pd.DataFrame
    ranking: pd.DataFrame
    importance_full: pd.DataFrame
    ranking_full: pd.DataFrame
    interactions: pd.DataFrame
    single: pd.DataFrame
    pairs: dict
    sampling: pd.DataFrame
    observed_acceptable: pd.DataFrame
    metadata: dict
    model: object


    def save(self, directory, **plot_options):
        '''Save the analysis tables, settings and both sensitivity matrices.

        Tables are saved as CSV files and settings as "summary.json". The
        fixed and adjusted matrices are each saved as PNG and PDF figures.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory to create or write the analysis into. Existing files
            with the same names are overwritten.

        **plot_options : other keyword arguments
            Display options forwarded to ``plot``: ``labels``,
            ``objective_label`` and ``objective_scale``.

        Examples
        --------
        Save figures with an objective label including its units:

        >>> result.save("sensitivity_output", objective_label = "Area (mm²)")
        '''
        directory = Path(directory)
        directory.mkdir(parents = True, exist_ok = True)

        tables = {
            "ranges": self.ranges,
            "importance": self.importance,
            "ranking": self.ranking,
            "importance_full": self.importance_full,
            "ranking_full": self.ranking_full,
            "interactions": self.interactions,
            "single": self.single,
            "observed_acceptable": self.observed_acceptable,
            "sampling": self.sampling,
        }
        for name, table in tables.items():
            table.to_csv(directory / f"{name}.csv", index = False)

        for i, table in enumerate(self.pairs.values(), 1):
            table.to_csv(directory / f"pair_{i}.csv", index = False)

        summary = json.dumps(self.metadata, indent = 2, allow_nan = False)
        (directory / "summary.json").write_text(summary + "\n")
        for fixed, name in [(True, "fixed"), (False, "adjusted")]:
            fig = self.plot(others_fixed = fixed, **plot_options)
            fig.savefig(directory / f"sensitivity_{name}.png", dpi = 160)
            fig.savefig(directory / f"sensitivity_{name}.pdf")
            plt.close(fig)


    def plot(
        self,
        others_fixed: bool = True,
        *,
        labels = None,
        objective_label = "Objective",
        objective_scale = "linear",
    ):
        '''Plot a matrix of individual and paired parameter responses.

        The diagonal shows individual responses and the observed reference.
        Below it, contours show absolute objective changes, with acceptable
        regions coloured green. Above it, heatmaps show objective values on
        a common colour scale shared between the fixed and adjusted plots.

        Parameters
        ----------
        others_fixed : bool, default True
            Hold the remaining parameters at the reference. If False, allow
            them to adjust and show projections of acceptable observations
            as purple diamonds. These observations can vary all parameters;
            they do not establish acceptance of intermediate combinations.

        labels : dict, optional
            Parameter names mapped to display labels, optionally with units.

        objective_label : str, default "Objective"
            Objective label to show on the colour scale and response axes.

        objective_scale : str, default "linear"
            Colour scaling: "linear", "log" for positive objectives, or
            "symlog" for objectives including zero or negative values.
            Display options do not change the fitted model or the results.

        Returns
        -------
        matplotlib.figure.Figure
            The sensitivity matrix, including legends for highlighted points.

        Examples
        --------
        Compare fixed-parameter responses with responses allowing adjustment:

        >>> fixed = result.plot(others_fixed = True)
        >>> adjusted = result.plot(others_fixed = False)
        >>> adjusted.savefig("sensitivity_adjusted.pdf")
        '''
        names = list(self.reference.index)
        labels = {} if labels is None else labels
        count = len(names)
        column = "slice_objective" if others_fixed else "profile_objective"
        lower, upper = self.metadata["objective_limits"]
        reference = self.metadata["reference_value"]
        reference_id = self.metadata.get("reference_evaluation")
        mode = "Others fixed" if others_fixed else "Others adjusted"

        values = [np.array([lower, upper, reference])]
        for table in [self.single, *self.pairs.values()]:
            values.append(table[["slice_objective", "profile_objective"]]
                          .to_numpy().ravel())
        values = np.concatenate(values)
        low, high = np.nanmin(values), np.nanmax(values)
        if objective_scale == "log":
            if low <= 0:
                raise ValueError("log requires positive values; use symlog")
            norm = LogNorm(vmin = low, vmax = high)
        elif objective_scale == "symlog":
            scale = self.metadata["absolute_tolerance"] or abs(reference) or 1
            norm = SymLogNorm(scale, vmin = low, vmax = high, base = 10)
        elif objective_scale == "linear":
            norm = Normalize(vmin = low, vmax = high)
        else:
            raise ValueError("objective_scale must be linear, log or symlog")

        fig = plt.figure(
            figsize = (max(7, 3 * count + 0.5), max(4, 2.8 * count + 1)),
            constrained_layout = True,
        )
        layout = fig.add_gridspec(
            count + 1, count, height_ratios = [0.3] + [1] * count,
        )
        axes = np.array([
            [fig.add_subplot(layout[row + 1, col]) for col in range(count)]
            for row in range(count)
        ])
        observed = self.observed_acceptable
        alternative = observed.loc[
            (observed[names].to_numpy() != self.reference.to_numpy())
            .any(axis = 1)
        ]
        for i, name in enumerate(names):
            ax = axes[i, i]
            table = self.single[self.single.parameter == name]
            ax.plot(table.value, table[column], color = "#236b8e", lw = 1.8)
            ax.axhspan(lower, upper, color = "#45a66c", alpha = 0.22)
            for limit in np.unique([lower, upper]):
                ax.axhline(limit, color = "#28824c", ls = "--", lw = 0.8)
            ax.axvline(self.reference[name], color = "#555555", ls = ":")
            ax.scatter(
                self.reference[name], reference, marker = "D",
                color = "#202b33", s = 28, zorder = 5,
            )
            if not others_fixed:
                ax.scatter(
                    alternative[name], alternative.observed_objective,
                    marker = "D", color = "#b13e79", s = 28, zorder = 5,
                )
            if objective_scale == "symlog":
                ax.set_yscale("symlog", linthresh = scale, base = 10)
            else:
                ax.set_yscale(objective_scale)
            ax.set_ylabel(objective_label, fontsize = 9)
            ax.grid(axis = "y", alpha = 0.15)

        heatmap = None
        for (a, b), table in self.pairs.items():
            i, j = names.index(a), names.index(b)
            grid = table.pivot(
                index = "value_b", columns = "value_a", values = column,
            )
            xx, yy = grid.columns.to_numpy(), grid.index.to_numpy()
            z = grid.to_numpy()
            below, above = axes[j, i], axes[i, j]
            below.set_facecolor("#f4f6f7")
            if len(xx) > 1 and len(yy) > 1 and np.isfinite(z).any():
                heatmap = above.pcolormesh(
                    yy, xx, np.ma.masked_invalid(z.T),
                    shading = "gouraud", cmap = "viridis", norm = norm,
                    rasterized = True,
                )
                departure = abs(z - reference)
                if np.nanmin(departure) < np.nanmax(departure):
                    contours = below.contour(
                        xx, yy, departure, levels = 4,
                        colors = "#929da3", linewidths = 0.65,
                    )
                    below.clabel(contours, fontsize = 7, fmt = "%.2g")
                if lower < upper:
                    below.contourf(
                        xx, yy, z, levels = [lower, upper],
                        colors = ["#b7dfc3"],
                    )
                levels = [
                    v for v in np.unique([lower, upper])
                    if np.nanmin(z) < v < np.nanmax(z)
                ]
                if levels:
                    below.contour(
                        xx, yy, z, levels = levels,
                        colors = "#237747", linewidths = 1.5,
                    )
                    above.contour(
                        yy, xx, z.T, levels = levels,
                        colors = "white", linewidths = 1.2,
                    )
                if np.nanmin(z) > upper or np.nanmax(z) < lower:
                    below.text(
                        0.04, 0.96, "No GP region within tolerance",
                        transform = below.transAxes, va = "top", fontsize = 7,
                        color = "#58646b",
                        bbox = dict(facecolor = "white", edgecolor = "none",
                                    alpha = 0.85, pad = 2),
                    )
            for ax, x, y in [(below, a, b), (above, b, a)]:
                ax.plot(
                    self.reference[x], self.reference[y],
                    marker = "+", color = "#db513b", ms = 8, mew = 1.5,
                )
                ax.set_ylabel(str(labels.get(y, y)), fontsize = 9)
            if not others_fixed:
                below.scatter(
                    alternative[a], alternative[b], marker = "D",
                    facecolor = "#b13e79", edgecolor = "white",
                    linewidth = 0.7, s = 36, zorder = 5,
                )
                below.scatter(
                    self.reference[a], self.reference[b], marker = "D",
                    facecolor = "#202b33", edgecolor = "white",
                    linewidth = 0.7, s = 36, zorder = 5,
                )
                for label, x, y in zip(
                    observed.evaluation.astype(str), observed[a], observed[b],
                ):
                    below.annotate(
                        label, (x, y), xytext = (-5, -10), ha = "right",
                        textcoords = "offset points", fontsize = 7,
                        color = ("#202b33" if label == reference_id
                                 else "#8c2860"),
                    )

        for row in range(count):
            for col in range(count):
                ax = axes[row, col]
                ax.tick_params(labelsize = 8)
                ax.set_xlabel(str(labels.get(names[col], names[col])),
                              fontsize = 9)
                window = self.sampling.iloc[col]
                if window.window_lower < window.window_upper:
                    ax.set_xlim(window.window_lower, window.window_upper)
                if row != col:
                    window = self.sampling.iloc[row]
                    if window.window_lower < window.window_upper:
                        ax.set_ylim(window.window_lower, window.window_upper)
                if row == 0:
                    ax.set_title(str(labels.get(names[col], names[col])),
                                 fontsize = 11, pad = 10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
        if heatmap is not None:
            bar = fig.colorbar(heatmap, ax = axes.ravel().tolist(),
                               shrink = 0.78, fraction = 0.025, pad = 0.02)
            bar.set_label(f"{objective_label} ({objective_scale} scale)")
            bar.ax.tick_params(labelsize = 9)

        reference_label = "Observed reference"
        if reference_id is not None:
            reference_label += f" (ID {reference_id})"
        handles = [
            Line2D([], [], color = "#236b8e", lw = 1.8,
                   label = ("GP response" if others_fixed else
                            "GP response after compensation")),
            Patch(facecolor = "#b7dfc3", edgecolor = "#237747",
                  label = "GP within objective band"),
            Line2D([], [], marker = "D", color = "#202b33", ls = "",
                   label = reference_label),
        ]
        if count > 1:
            handles.extend([
                Line2D([], [], color = "#929da3", lw = 0.8,
                       label = "Absolute change from reference"),
                Line2D([], [], marker = "+", color = "#db513b", ls = "",
                       markersize = 8, markeredgewidth = 1.5,
                       label = "Reference coordinates"),
            ])
        if not others_fixed and not alternative.empty:
            label = "Other accepted observations"
            if len(alternative) == 1:
                evaluation = alternative.evaluation.iloc[0]
                label = f"Accepted observation (ID {evaluation})"
            handles.append(Line2D(
                [], [], marker = "D", color = "#b13e79", ls = "",
                label = label,
            ))
        legend = fig.add_subplot(layout[0, :])
        legend.set_axis_off()
        legend.legend(handles = handles, loc = "center",
                      ncol = 2 if count < 3 else 3,
                      frameon = False, fontsize = 9)
        fit = ("interpolating GP" if self.metadata.get("interpolate", False)
               else "smoothed GP")
        fig.suptitle(
            f"Local sensitivity | {fit} | {mode.lower()}\n"
            f"Observed reference {reference:.4g} | band [{lower:.4g}, "
            f"{upper:.4g}] | GP at reference "
            f"{self.metadata['predicted_reference_value']:.4g}",
            fontsize = 11,
        )
        return fig



@dataclass
class SurfaceData:
    '''Structure storing a fitted response and its reporting box.

    Parameter coordinates are relative to the reference and divided by the
    original parameter-bound widths.

    Attributes
    ----------
    model : sklearn.compose.TransformedTargetRegressor
        Fitted response model, returning predictions in objective units.

    box : np.ndarray
        Lower and upper bounds of the reporting box in scaled coordinates.

    reference_value : float
        Observed objective at the reference vector.

    tolerance : float
        Allowed absolute objective departure from the reference.

    target : float
        Reference objective after applying the fitted response transformation.

    starting_points : np.ndarray
        Scaled training vectors used to initialise compensation searches.

    anchors : np.ndarray
        Scaled acceptable observations inside the reporting box.

    profile_starts : int
        Maximum number of optimisation starts per compensation search.
    '''

    model: object
    box: np.ndarray
    reference_value: float
    tolerance: float
    target: float
    starting_points: np.ndarray
    anchors: np.ndarray
    profile_starts: int




def fit_gaussian_process(
    x, objective, reference_value, seed,
    interpolate, response_transform, kernel,
):
    '''Fit an interpolating or smoothing GP using sklearn estimators.
    '''
    if kernel is None:
        kernel = ConstantKernel() * Matern(np.ones(x.shape[1]), nu = 1.5)
    if not interpolate:
        kernel = kernel + WhiteKernel(noise_level = 0.01)
    regressor = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel = kernel,
            normalize_y = True,
            alpha = 1e-6 if interpolate else 1e-10,
            n_restarts_optimizer = 10,
            random_state = seed,
        ),
    )
    if response_transform == "log":
        if (objective <= 0).any() or reference_value <= 0:
            raise ValueError("log response_transform requires positive values")
        forward, inverse = np.log, np.exp
    elif response_transform == "asinh":
        scale = abs(reference_value) or float(np.std(objective)) or 1.0
        forward = lambda values: np.arcsinh(values / scale)
        inverse = lambda values: scale * np.sinh(values)
    else:
        raise ValueError("response_transform must be asinh or log")
    model = TransformedTargetRegressor(
        regressor = regressor,
        func = forward, inverse_func = inverse,
    )
    return model.fit(x, objective)




def within_objective_band(values, lower, upper):
    '''Test predicted values against the closed band, allowing for roundoff.

    Root finding and batched GP predictions need not agree in their last
    digits. Allow 1e-10 of the band magnitude when classifying predictions;
    this scales with objective units and leaves an all-zero band exact.
    '''
    margin = 1e-10 * max(abs(lower), abs(upper))
    return (values >= lower - margin) & (values <= upper + margin)




def profile_match(surface, fixed, previous = None):
    '''Find a compensating vector whose prediction lies in the tolerance band.

    Try the fixed-other slice, continuation point and observed starting points.
    Minimise squared distance to the reference in transformed response units
    with bounded optimisation. Stop after a solve yields a feasible witness.
    An unsuccessful search does not prove that no compensating vector exists.
    '''
    fixed = dict(fixed)
    for j, (low, high) in enumerate(surface.box):
        if low == high:
            fixed[j] = low

    free = [j for j in range(len(surface.box)) if j not in fixed]
    point = np.zeros(len(surface.box))
    for j, value in fixed.items():
        point[j] = value

    def expand(values):
        expanded = point.copy()
        expanded[free] = values
        return expanded

    scaler = surface.model.regressor_.named_steps["standardscaler"]
    gp = surface.model.regressor_.named_steps["gaussianprocessregressor"]

    def loss(values):
        scaled = (expand(values) - scaler.mean_) / scaler.scale_
        response = gp.predict(scaled[None, :])[0]
        return (response - surface.target) ** 2

    candidates = [point, *surface.starting_points]
    if previous is not None:
        candidates.append(previous)
    starts = np.clip(candidates, surface.box[:, 0], surface.box[:, 1])
    for j, value in fixed.items():
        starts[:, j] = value
    starts = np.unique(starts, axis = 0)

    values = surface.model.predict(starts)
    distances = abs(values - surface.reference_value)
    closest = int(np.argmin(distances))
    best_value, best_point = float(values[closest]), starts[closest]
    best_distance = float(distances[closest])
    lower = surface.reference_value - surface.tolerance
    upper = surface.reference_value + surface.tolerance
    if best_distance == 0:
        return best_value, best_point, "within_tolerance"
    if not free:
        accepted = within_objective_band(best_value, lower, upper)
        status = "within_tolerance" if accepted else "outside_tolerance"
        return best_value, best_point, status

    bounds = surface.box[free]
    converged = False
    for i in np.argsort(distances, kind = "stable")[:surface.profile_starts]:
        solution = minimize(
            loss, starts[i][free], method = "L-BFGS-B", bounds = bounds,
            options = dict(ftol = 1e-8, maxiter = 100),
        )
        converged |= solution.success
        candidate = expand(solution.x)
        value = float(surface.model.predict(candidate[None, :])[0])
        distance = abs(value - surface.reference_value)
        if distance < best_distance:
            best_value, best_point, best_distance = value, candidate, distance
        if within_objective_band(best_value, lower, upper):
            return best_value, best_point, "within_tolerance"

    status = "outside_tolerance" if converged else "solver_failed"
    return best_value, best_point, status




def _intervals(
    table, column, limits, reference, lower_reason, upper_reason,
    evaluate = None,
):
    '''Intersect the piecewise-linear response curve with a two-sided band.

    Keep disconnected intervals separate. An optional evaluator refines each
    crossing with Brent's method. Otherwise use linear interpolation, including
    crossings between nodes on opposite sides of the entire band.
    '''
    values = table.value.to_numpy()
    response = table[column].to_numpy()
    lower, upper = limits
    segments = []
    for i, value in enumerate(values):
        if np.isfinite(response[i]) and lower <= response[i] <= upper:
            segments.append((value, value))
        if i == len(values) - 1 or not np.isfinite(response[i:i + 2]).all():
            continue

        change = response[i + 1] - response[i]
        if change == 0:
            if lower <= response[i] <= upper:
                segments.append((value, values[i + 1]))
            continue

        crossings = sorted([(lower - response[i]) / change,
                            (upper - response[i]) / change])
        start, stop = max(0, crossings[0]), min(1, crossings[1])
        if start <= stop:
            width = values[i + 1] - value
            endpoints = []
            for fraction in [start, stop]:
                endpoint = value + fraction * width
                if evaluate is not None and 0 < fraction < 1:
                    target = min(
                        limits, key = lambda v:
                        abs(v - (response[i] + fraction * change)),
                    )
                    fraction = brentq(
                        lambda f: evaluate(value + f * width) - target, 0, 1,
                    )
                    endpoint = value + fraction * width
                endpoints.append(endpoint)
            segments.append(tuple(endpoints))

    merged = []
    for low, high in sorted(segments):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])

    rows = []
    for low, high in merged:
        left = lower_reason if low == values[0] else "threshold"
        right = upper_reason if high == values[-1] else "threshold"
        left_index = np.searchsorted(values, low, side = "left") - 1
        right_index = np.searchsorted(values, high, side = "right")
        if left_index >= 0 and not np.isfinite(response[left_index]):
            left = "solver_failed"
        if (right_index < len(values) and
            not np.isfinite(response[right_index])):
            right = "solver_failed"
        rows.append(dict(
            lower = float(low), upper = float(high), width = float(high - low),
            lower_change = float(low - reference),
            upper_change = float(high - reference),
            contains_reference_value = bool(low <= reference <= high),
            lower_limit = left, upper_limit = right,
        ))
    return rows




def evaluate_grid(surface, points, fixed_indices, center, span, names):
    '''Evaluate slices and find compensated witnesses on the same grid.
    '''
    slices = surface.model.predict(points)
    rows = []
    previous = None
    for coordinates, value in zip(points, slices):
        fixed = {j: coordinates[j] for j in fixed_indices}
        matched, witness, status = profile_match(surface, fixed, previous)
        previous = witness
        row = dict(
            slice_objective = float(value),
            profile_objective = (matched if status != "solver_failed"
                                 else np.nan),
            slice_accepted = within_objective_band(
                value, surface.reference_value - surface.tolerance,
                surface.reference_value + surface.tolerance,
            ),
            profile_accepted = status == "within_tolerance",
            profile_status = status,
        )
        row.update({
            f"compensating_{name}": float(v)
            for name, v in zip(names, center + span * witness)
        })
        rows.append(row)
    return pd.DataFrame(rows)




def grid_rms(values, *axes):
    '''RMS over a rectangular grid, with uniform weight per physical unit.

    Trapezoidal integration accounts for nonuniform grid spacing, so adding
    observed coordinates or tolerance crossings does not overweight them.
    '''
    squared = np.asarray(values) ** 2
    for axis in reversed(axes):
        if len(axis) == 1:
            squared = squared[..., 0]
        else:
            squared = np.sum(
                (squared[..., 1:] + squared[..., :-1]) * np.diff(axis) / 2,
                axis = -1,
            ) / (axis[-1] - axis[0])
    return float(np.sqrt(squared))




def rank_parameters(single, reference_prediction):
    '''Rank fixed-other slices by RMS change, in objective and relative units.
    '''
    rows = []
    for name, table in single.groupby("parameter", sort = False):
        change = table.slice_objective.to_numpy() - reference_prediction
        rows.append(dict(
            parameter = name,
            rms_objective_change = grid_rms(change, table.value.to_numpy()),
            max_abs_objective_change = float(np.max(abs(change))),
            tested_parameter_span = float(np.ptp(table.value)),
        ))
    importance = pd.DataFrame(rows).sort_values(
        "rms_objective_change", ascending = False, kind = "mergesort",
    ).reset_index(drop = True)
    importance.insert(0, "rank", importance.rms_objective_change.rank(
        method = "min", ascending = False,
    ).astype(int))
    ranking = importance[["rank", "parameter"]].copy()
    total = importance.rms_objective_change.sum()
    ranking["relative_sensitivity"] = (
        importance.rms_objective_change / total if total > 0 else 0.0
    )
    return importance, ranking




def analyse_parameters(
    surface, center, span, limits, names, grid_size, verbose,
):
    '''Compute individual curves and tolerance intervals in the local window.
    '''
    tables, intervals = [], []
    objective_limits = [
        surface.reference_value - surface.tolerance,
        surface.reference_value + surface.tolerance,
    ]
    for j, name in enumerate(names):
        if verbose:
            print(f"Parameter {name}", flush = True)
        low, high = surface.box[j]
        axis = np.unique(np.r_[
            np.linspace(low, high, grid_size), 0.0, surface.anchors[:, j],
        ])
        points = np.zeros((len(axis), len(names)))
        points[:, j] = axis
        table = evaluate_grid(surface, points, [j], center, span, names)
        table.insert(0, "value", center[j] + span[j] * axis)
        table.insert(0, "parameter", name)
        tables.append(table)

        reasons = [
            "original_bound" if np.isclose(
                center[j] + span[j] * value, bound,
                rtol = 0, atol = span[j] * 1e-10,
            ) else "parameter_window"
            for value, bound in zip([low, high], limits[j])
        ]
        def response(value):
            point = np.zeros(len(names))
            point[j] = (value - center[j]) / span[j]
            return surface.model.predict(point[None, :])[0]

        for mode in ["slice", "profile"]:
            accepted = _intervals(
                table, mode + "_objective", objective_limits,
                center[j], *reasons,
                evaluate = (response if mode == "slice" or len(names) == 1
                            else None),
            )
            intervals.extend(
                dict(parameter = name, mode = mode, **interval)
                for interval in accepted
            )

    ranges = pd.DataFrame(intervals, columns = [
        "parameter", "mode", "lower", "upper", "width", "lower_change",
        "upper_change", "contains_reference_value", "lower_limit",
        "upper_limit",
    ])
    return pd.concat(tables, ignore_index = True), ranges




def analyse_pairs(
    surface, center, span, names, reference_prediction,
    grid_size, ranges, verbose,
):
    '''Compute joint tolerance maps and fixed-other non-additive contrasts.
    '''
    pairs, interactions = {}, []
    for a, b in combinations(range(len(names)), 2):
        if verbose:
            print(f"Pair {names[a]} / {names[b]}", flush = True)
        axis_a = np.unique(np.r_[
            np.linspace(*surface.box[a], grid_size), 0.0,
            surface.anchors[:, a],
            (ranges.loc[ranges.parameter == names[a], ["lower", "upper"]]
             .to_numpy().ravel() - center[a]) / span[a],
        ])
        axis_b = np.unique(np.r_[
            np.linspace(*surface.box[b], grid_size), 0.0,
            surface.anchors[:, b],
            (ranges.loc[ranges.parameter == names[b], ["lower", "upper"]]
             .to_numpy().ravel() - center[b]) / span[b],
        ])
        coordinates = np.array([(v, w) for w in axis_b for v in axis_a])
        points = np.zeros((len(coordinates), len(names)))
        points[:, [a, b]] = coordinates
        table = evaluate_grid(surface, points, [a, b], center, span, names)
        table.insert(0, "value_b", center[b] + span[b] * coordinates[:, 1])
        table.insert(0, "value_a", center[a] + span[a] * coordinates[:, 0])
        table.insert(0, "parameter_b", names[b])
        table.insert(0, "parameter_a", names[a])

        # Anchored interaction: f(a, b) - f(a, b0) - f(a0, b) + f(a0, b0)
        points_a, points_b = points.copy(), points.copy()
        points_a[:, b] = 0.0
        points_b[:, a] = 0.0
        contrast = (
            table.slice_objective.to_numpy() -
            surface.model.predict(points_a) -
            surface.model.predict(points_b) + reference_prediction
        )
        table["interaction"] = contrast
        pairs[(names[a], names[b])] = table
        interactions.append(dict(
            parameter_a = names[a], parameter_b = names[b],
            interaction_rms = grid_rms(
                contrast.reshape(len(axis_b), len(axis_a)), axis_b, axis_a,
            ),
            max_abs_interaction = float(np.max(abs(contrast))),
        ))

    interactions = pd.DataFrame(interactions, columns = [
        "parameter_a", "parameter_b", "interaction_rms", "max_abs_interaction",
    ])
    return pairs, interactions.sort_values(
        "interaction_rms", ascending = False,
    ).reset_index(drop = True)




def analyse(
    parameters,
    objective,
    bounds = None,
    *,
    reference = None,
    reference_value = None,
    parameter_window = 0.2,
    objective_tolerance = 0.1,
    absolute_tolerance = None,
    interpolate = True,
    response_transform = "asinh",
    kernel = None,
    n_neighbors = None,
    profile_starts = 5,
    grid_size = 101,
    pair_grid_size = 31,
    random_state = 42,
    verbose = True,
):
    '''Analyse a scalar response around the best observed or supplied vector.

    Parameters
    ----------
    parameters : pd.DataFrame or np.ndarray
        Physical parameters of shape (n_samples, n_parameters), one row per
        evaluation. DataFrame column names
        are preserved; arrays use "Parameter 1", "Parameter 2", etc.

    objective : pd.Series, pd.DataFrame or np.ndarray
        Scalar response for each row; a DataFrame must have one column.
        Negative values and zeros are supported.
        To analyse several responses, call this function once per response.

    bounds : pd.DataFrame or np.ndarray, optional
        Min/max bounds of shape (n_parameters, 2), in the same order. A
        DataFrame is indexed by parameter with "min" and "max" columns.
        Defaults to the column minima/maxima of complete evaluations. Each
        minimum must be strictly below its maximum; omit constant parameters.

    reference : pd.Series or np.ndarray, optional
        Parameter vector at the optimum of interest, in physical units.
        Defaults to the complete observation with the smallest objective,
        choosing the first in input order for a tie. Supply another vector
        when studying a maximum or a different optimisation result.

    reference_value : float, optional
        Observed objective at that vector. Defines the tolerance band centre
        and magnitude; it is not replaced by the GP prediction. Defaults to
        the recorded objective at the reference. An unobserved reference or
        repeated reference with differing responses needs an explicit value.

    parameter_window : float in [0, 1], default 0.2
        Box half-width as a fraction of each bound width, clipped to those
        bounds. With inferred bounds this uses the observed parameter spans.
        Zero tests only the reference; one covers all bounds.

    objective_tolerance : float >= 0, default 0.1
        Symmetric allowance around the observed reference:
        ``abs(prediction - reference_value)`` must not exceed
        ``objective_tolerance * abs(reference_value)``.

    absolute_tolerance : float >= 0, optional
        Override the relative allowance with this half-width in objective
        units. Useful when the reference is zero or percentages are unsuitable.

    interpolate : bool, default True
        Interpolate the supplied evaluations with a small numerical nugget.
        False fits a white-noise kernel to estimate a smoothed response.
        Interpolation describes the recorded objective; it does not remove
        variability in stochastic evaluations or uncertainty between samples.

    response_transform : {"asinh", "log"}, default "asinh"
        "asinh" supports signed objectives. "log" requires positive objectives
        and keeps predictions positive, useful for calibration discrepancies.

    kernel : sklearn.gaussian_process.kernels.Kernel, optional
        Signal covariance. Defaults to a constant times an anisotropic
        Matern-3/2 kernel. Smoothing adds a fitted white-noise component.

    n_neighbors : int, optional
        Fit at most this many nearest complete evaluations (at least two),
        measured in bound-scaled coordinates. The default uses all evaluations.
        Training may extend outside the reporting box.

    profile_starts : int >= 1, default 5
        Number of optimisation starts. Project training vectors, the reference
        and the previous solution onto the fixed coordinates; start from those
        whose GP predictions are closest to the reference objective.

    grid_size, pair_grid_size : int >= 3, defaults 101 and 31
        Grid resolutions; axes also include the reference and coordinates of
        actual acceptable evaluations within the reporting box.
        Pair axes additionally include the single-parameter range endpoints.
        The full-bound ranking uses grid_size points plus the reference and
        coordinates of acceptable observations across the complete bounds.

    random_state : int, default 42
        Seed for GP hyperparameter optimisation. Identical ordered inputs,
        settings and software environment give repeatable numerical results.
        Profile searches use deterministic starting points. Exact agreement
        across library versions or numerical backends is not guaranteed.

    verbose : bool, default True
        Print fitting counts, the objective band and parameter/pair progress.

    Returns
    -------
    SensitivityResult
        Ranges and signed variations in parameter units, fixed-other RMS
        sensitivity and pair interactions in objective units, dimensionless
        rankings over the local window and full bounds, joint maps,
        compensating vectors and sampling counts.

    Notes
    -----
    By default the GP models asinh(objective / scale), where scale is the
    absolute reference value, or the sample standard deviation when it is
    zero. Constant zero responses use scale 1. This compresses large values
    while supporting both signs. Predictions are inverse-transformed GP means
    in objective units. Interpolation uses a 1e-6 normalised numerical nugget.
    Repeated parameter vectors with different responses require smoothing.

    The band is centred on the observed reference value. A smoothed GP can
    miss this band even at the reference vector, so accepted ranges can be
    empty. The observed and predicted reference values are reported separately.
    Actual acceptable observations use the same band as the GP predictions.
    Predicted acceptance flags allow roundoff of 1e-10 times the larger
    absolute band limit. Observations and range crossings use the exact band.

    Ranges describe the fitted response across the reporting box. Sample
    counts and the prediction at the reference are included with the results.
    ``ranking`` describes this window; ``ranking_full`` spans the complete
    bounds. Both use the same fitted response and hold other inputs fixed.
    The full-bound ranking requires extra GP predictions but no refit or
    additional profile optimisation. It can cover poorly sampled regions,
    especially when fitting only n_neighbors evaluations.

    Profiles search for any vector inside the band, stopping at the first
    successful witness. Their response values are not objective minima or
    maxima. A flat profile can indicate compensation, not a constant response
    to that parameter alone. An unsuccessful local search is not proof of
    infeasibility.
    Slice endpoints use bracketed root finding; profile endpoints use grid
    interpolation. Disconnected or narrow regions can need finer grids.
    Marginal profile endpoints cannot be freely combined.

    Examples
    --------
    Analyse a sampled two-parameter response without running new simulations:

    >>> import numpy as np
    >>> from coexist.sensitivity import analyse
    >>> a, b = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    >>> parameters = np.c_[a.ravel(), b.ravel()]
    >>> objective = 1 + parameters[:, 0] ** 2 + 4 * parameters[:, 1] ** 2
    >>> result = analyse(parameters, objective, verbose = False)
    >>> result.ranges
    >>> result.ranking
    >>> result.plot(others_fixed = False)
    '''
    if not np.isfinite(parameter_window) or not 0 <= parameter_window <= 1:
        raise ValueError("parameter_window must be between 0 and 1, inclusive")
    for name, value in [("objective_tolerance", objective_tolerance),
                        ("absolute_tolerance", absolute_tolerance)]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if objective_tolerance is None:
        raise ValueError("objective_tolerance must be finite and nonnegative")
    if (
        not all(isinstance(v, (int, np.integer)) for v in
                [grid_size, pair_grid_size, profile_starts]) or
        grid_size < 3 or pair_grid_size < 3 or
        profile_starts < 1
    ):
        raise ValueError(
            "Grid sizes must be >= 3 and profile_starts >= 1"
        )

    frame = pd.DataFrame(parameters).astype(float)
    if not isinstance(parameters, pd.DataFrame):
        frame.columns = [f"Parameter {i + 1}" for i in range(frame.shape[1])]
    names = list(frame.columns)
    if not names or frame.columns.duplicated().any():
        raise ValueError("At least one parameter is needed, with unique names")
    if (isinstance(parameters, pd.DataFrame) and
        isinstance(objective, (pd.Series, pd.DataFrame)) and
        not objective.index.equals(frame.index)):
        raise ValueError("Parameter and objective indices must match in order")
    values = np.asarray(objective, dtype = float)
    if values.shape == (len(frame), 1):
        values = values[:, 0]
    if values.shape != (len(frame),):
        raise ValueError("Supply one scalar objective per evaluation")

    complete = (
        np.isfinite(frame).all(axis = 1).to_numpy() & np.isfinite(values)
    )
    omitted = frame.index[~complete].tolist()
    frame, values = frame.loc[complete], values[complete]
    if len(frame) < 2:
        raise ValueError("At least two complete evaluations are needed")
    if bounds is None:
        limits = np.column_stack([frame.min(), frame.max()])
    elif isinstance(bounds, pd.DataFrame):
        limits = bounds.loc[names, ["min", "max"]].to_numpy(float)
    else:
        limits = np.asarray(bounds, dtype = float)
    if limits.shape != (len(names), 2):
        raise ValueError("bounds must contain one min/max pair per parameter")
    span = limits[:, 1] - limits[:, 0]
    if not np.isfinite(limits).all() or (span <= 0).any():
        raise ValueError("Each parameter needs finite min < max bounds")
    if reference is None:
        best = int(np.argmin(values))
        reference = frame.iloc[best]
        if reference_value is None:
            reference_value = float(values[best])
    if isinstance(reference, pd.Series):
        reference = reference.loc[names]
    center = np.asarray(reference, dtype = float)
    if (
        center.shape != (len(names),) or not np.isfinite(center).all() or
        ((center < limits[:, 0]) | (center > limits[:, 1])).any()
    ):
        raise ValueError("reference must be a finite vector within the bounds")

    matches = (frame.to_numpy() == center).all(axis = 1)
    if reference_value is None:
        responses = np.unique(values[matches])
        if len(responses) != 1:
            raise ValueError(
                "Supply reference_value for an unobserved reference or "
                "repeated reference with differing objectives"
            )
        reference_value = float(responses[0])
    if not np.isscalar(reference_value) or not np.isfinite(reference_value):
        raise ValueError("reference_value must be a finite scalar")
    observed_reference = matches & (values == reference_value)
    reference_evaluation = (str(frame.index[observed_reference][0])
                            if observed_reference.any() else None)
    if ((frame.to_numpy() < limits[:, 0]) |
        (frame.to_numpy() > limits[:, 1])).any():
        raise ValueError("Evaluated parameters fall outside original bounds")
    if (n_neighbors is not None and
        (not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 2)):
        raise ValueError("n_neighbors must be at least 2, or None")
    count = len(frame) if n_neighbors is None else min(n_neighbors, len(frame))

    tolerance = (float(objective_tolerance) * abs(float(reference_value))
                 if absolute_tolerance is None else float(absolute_tolerance))
    objective_limits = [
        reference_value - tolerance, reference_value + tolerance,
    ]
    if not np.isfinite(objective_limits).all():
        raise ValueError("The tolerance produces non-finite objective limits")

    all_points = (frame.to_numpy() - center) / span
    indices = np.argsort(
        np.linalg.norm(all_points, axis = 1), kind = "stable",
    )[:count]
    x, y = all_points[indices], values[indices]
    if interpolate:
        repeated = pd.DataFrame(x).assign(objective = y)
        variation = repeated.groupby(list(range(len(names))))
        variation = variation.objective.nunique()
        if (variation > 1).any():
            raise ValueError(
                "Repeated parameters have different objectives; "
                "use interpolate = False to fit their smoothed response"
            )
    box = np.column_stack([
        np.maximum(-parameter_window, (limits[:, 0] - center) / span),
        np.minimum(parameter_window, (limits[:, 1] - center) / span),
    ])
    observed_inside = (
        (all_points >= box[:, 0]) & (all_points <= box[:, 1])
    ).all(axis = 1)
    if verbose:
        print(
            f"Fitting GP to {len(x)} of {len(frame)} complete evaluations; "
            f"{observed_inside.sum()} observations in the reporting window.",
            flush = True,
        )
        print(
            f"Reference objective {reference_value:.6g}; "
            f"band [{objective_limits[0]:.6g}, "
            f"{objective_limits[1]:.6g}].",
            flush = True,
        )
    model = fit_gaussian_process(
        x, y, float(reference_value), random_state,
        interpolate, response_transform, kernel,
    )
    origin = np.zeros(len(names))
    predicted_reference = float(model.predict(origin[None, :])[0])
    target = float(model.transformer_.transform([[reference_value]])[0, 0])
    if verbose:
        print(f"GP prediction at reference: {predicted_reference:.6g}.",
              flush = True)
    selected = (observed_inside & (values >= objective_limits[0])
                & (values <= objective_limits[1]))
    anchors = all_points[selected]
    surface = SurfaceData(
        model, box, float(reference_value), tolerance, target, x, anchors,
        profile_starts,
    )
    single, ranges = analyse_parameters(
        surface, center, span, limits, names, grid_size, verbose,
    )
    pairs, interactions = analyse_pairs(
        surface, center, span, names, predicted_reference,
        pair_grid_size, ranges, verbose,
    )
    importance, ranking = rank_parameters(single, predicted_reference)

    if verbose:
        print("Ranking across full parameter bounds.", flush = True)
    full_tables = []
    acceptable = (
        (values >= objective_limits[0]) & (values <= objective_limits[1])
    )
    for j, name in enumerate(names):
        low, high = (limits[j] - center[j]) / span[j]
        axis = np.unique(np.r_[
            np.linspace(low, high, grid_size), 0.0,
            all_points[acceptable, j],
        ])
        points = np.zeros((len(axis), len(names)))
        points[:, j] = axis
        full_tables.append(pd.DataFrame({
            "parameter": name,
            "value": center[j] + span[j] * axis,
            "slice_objective": model.predict(points),
        }))
    importance_full, ranking_full = rank_parameters(
        pd.concat(full_tables, ignore_index = True), predicted_reference,
    )

    # Count actual observations and training rows inside the full reporting box
    training_inside = observed_inside[indices]
    training = frame.iloc[indices]
    local_training = training.loc[training_inside]
    sampling = pd.DataFrame({
        "parameter": names,
        "window_lower": center + span * box[:, 0],
        "window_upper": center + span * box[:, 1],
        "training_min": training.min().to_numpy(),
        "training_max": training.max().to_numpy(),
        "in_window_min": local_training.min().to_numpy(),
        "in_window_max": local_training.max().to_numpy(),
    })
    accepted = frame.loc[selected].copy()
    accepted.insert(0, "evaluation", accepted.index)
    accepted["observed_objective"] = values[selected]
    failures = sum(
        int((table.profile_status == "solver_failed").sum())
        for table in [single, *pairs.values()]
    )
    metadata = dict(
        n_neighbors = None if n_neighbors is None else int(n_neighbors),
        parameter_window = float(parameter_window),
        objective_tolerance = float(objective_tolerance),
        absolute_tolerance = tolerance,
        tolerance_mode = ("relative" if absolute_tolerance is None
                          else "absolute"),
        objective_limits = list(map(float, objective_limits)),
        parameter_names = list(map(str, names)),
        parameter_bounds = {str(n): v.tolist() for n, v in zip(names, limits)},
        bounds_source = "observed" if bounds is None else "supplied",
        reference = {str(k): float(v) for k, v in zip(names, center)},
        reference_value = float(reference_value),
        reference_evaluation = reference_evaluation,
        predicted_reference_value = predicted_reference,
        reference_accepted = bool(
            within_objective_band(predicted_reference, *objective_limits),
        ),
        complete_evaluations = len(frame),
        training_evaluations = len(x),
        observations_in_window = int(observed_inside.sum()),
        training_in_window = int(training_inside.sum()),
        unique_training_in_window = len(local_training.drop_duplicates()),
        omitted_incomplete_ids = list(map(str, omitted)),
        training_ids = list(map(str, frame.index[indices])),
        fitted_kernel = str(
            model.regressor_.named_steps["gaussianprocessregressor"].kernel_,
        ),
        interpolate = bool(interpolate),
        response_transform = response_transform,
        response_scale = ((abs(float(reference_value))
                           or float(np.std(y)) or 1.0)
                          if response_transform == "asinh" else None),
        failed_profile_points = failures,
        profile_starts = int(profile_starts),
        grid_size = int(grid_size), pair_grid_size = int(pair_grid_size),
        sklearn_version = sklearn.__version__,
        random_state = int(random_state),
    )
    return SensitivityResult(
        reference = pd.Series(center, index = names),
        ranges = ranges, importance = importance, interactions = interactions,
        ranking = ranking,
        importance_full = importance_full, ranking_full = ranking_full,
        single = single, pairs = pairs, sampling = sampling,
        observed_acceptable = accepted.reset_index(drop = True),
        metadata = metadata, model = model,
    )

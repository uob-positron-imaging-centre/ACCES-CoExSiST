#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_sensitivity.py
# License: GNU v3.0


'''Numerical and integration tests for local parameter sensitivity.'''


import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import coexist

pytest.importorskip("sklearn")

from coexist.sensitivity import analyse, grid_rms, _intervals


def quadratic_data():
    # Known fixed-other and compensated curvatures, with a positive minimum.
    parameters = pd.DataFrame(
        product(np.linspace(-1, 1, 9), repeat = 2), columns = ["a", "b"],
    )
    a, b = parameters.a, parameters.b
    objective = 1 + 10 * (a + 0.8 * b) ** 2 + 2 * b ** 2
    bounds = pd.DataFrame(
        {"min": [-1, -1], "max": [1, 1]}, index = parameters.columns,
    )
    settings = dict(
        reference = [0, 0], reference_value = 1,
        parameter_window = 0.4, objective_tolerance = 1,
        grid_size = 81, pair_grid_size = 11,
    )
    return parameters, objective, bounds, settings


def access_data():
    data = coexist.AccessData.empty()
    data.paths = coexist.access.AccessPaths()
    data.results = pd.DataFrame(
        product(np.linspace(-1, 1, 7), repeat = 2), columns = ["a", "b"],
    )
    data.results["error0"] = 1 + data.results.a ** 2
    data.results["error1"] = 1 + 3 * data.results.b ** 2
    data.results["error"] = data.results.error0 + data.results.error1
    data.parameters = pd.DataFrame(
        {"min": [-1., -1.], "max": [1., 1.]}, index = ["a", "b"],
    )
    data.population = 7
    return data


def test_analytic_ranges():
    parameters, objective, bounds, settings = quadratic_data()
    result = analyse(
        parameters, objective, bounds, **settings,
    )

    # Exact slice and compensated limits for an objective increase of one.
    exact = {
        ("a", "slice"): 1 / np.sqrt(10),
        ("b", "slice"): 1 / np.sqrt(8.4),
        ("a", "profile"): 1 / np.sqrt(10 - 64 / 8.4),
        ("b", "profile"): 1 / np.sqrt(2),
    }
    assert len(result.ranges) == 4
    for row in result.ranges.itertuples():
        limit = exact[(row.parameter, row.mode)]
        assert abs(row.lower + limit) < 0.012
        assert abs(row.upper - limit) < 0.012
        assert row.lower_limit == row.upper_limit == "threshold"
        if row.mode == "slice":
            points = np.zeros((2, 2))
            points[:, parameters.columns.get_loc(row.parameter)] = [
                row.lower / 2, row.upper / 2,
            ]
            np.testing.assert_allclose(
                result.model.predict(points),
                result.metadata["objective_limits"][1], atol = 1e-7,
            )

    # The anchored interaction is exactly 16 a b for this quadratic.
    pair = result.pairs[("a", "b")]
    np.testing.assert_allclose(
        pair.interaction, 16 * pair.value_a * pair.value_b, atol = 0.09,
    )
    assert result.importance.parameter.tolist() == ["a", "b"]
    assert result.ranking.parameter.tolist() == ["a", "b"]
    # Equal slice widths give RMS scores proportional to 10 and 8.4.
    np.testing.assert_allclose(
        result.ranking.relative_sensitivity, np.array([10, 8.4]) / 18.4,
        atol = 0.002,
    )
    np.testing.assert_allclose(result.ranking.relative_sensitivity.sum(), 1)
    assert result.metadata["failed_profile_points"] == 0


def test_local_and_full_rankings():
    parameters = pd.DataFrame(
        product(np.linspace(-1, 1, 9), repeat = 2), columns = ["a", "b"],
    )
    objective = 1 + 4 * parameters.a ** 2 + 20 * parameters.b ** 4
    settings = dict(grid_size = 41, pair_grid_size = 3, verbose = False)
    local = analyse(parameters, objective, parameter_window = 0.2, **settings)
    whole = analyse(parameters, objective, parameter_window = 1, **settings)

    # Quadratic a dominates near the optimum; quartic b dominates farther out.
    assert local.ranking.parameter.tolist() == ["a", "b"]
    assert local.ranking_full.parameter.tolist() == ["b", "a"]
    expected = np.array([4 * 0.4 ** 2 / np.sqrt(5), 20 * 0.4 ** 4 / 3])
    np.testing.assert_allclose(
        local.ranking.relative_sensitivity, expected / expected.sum(),
        atol = 0.03,
    )
    expected = np.array([20 / 3, 4 / np.sqrt(5)])
    np.testing.assert_allclose(
        local.ranking_full.relative_sensitivity, expected / expected.sum(),
        atol = 0.01,
    )
    np.testing.assert_allclose(local.importance.tested_parameter_span, 0.8)
    np.testing.assert_allclose(local.importance_full.tested_parameter_span, 2)
    pd.testing.assert_frame_equal(
        local.ranking_full, whole.ranking_full, rtol = 1e-10, atol = 1e-12,
    )
    pd.testing.assert_frame_equal(
        whole.ranking, whole.ranking_full, check_exact = True,
    )
    pd.testing.assert_frame_equal(
        whole.importance, whole.importance_full, check_exact = True,
    )


def test_default_bounds_and_reference():
    parameters = pd.DataFrame(
        {"friction": [-1, 0, 0.5, 1, 100, np.nan]},
        index = ["a", "best", "c", "d", "failed", "incomplete"],
    )
    objective = pd.Series(
        [2, 1, 1.25, 2, np.nan, -10], index = parameters.index,
    )
    output = StringIO()
    with redirect_stdout(output):
        result = analyse(parameters, objective, grid_size = 9)
    assert result.metadata["parameter_bounds"] == {"friction": [-1, 1]}
    assert result.metadata["bounds_source"] == "observed"
    assert result.reference.to_dict() == {"friction": 0}
    assert result.metadata["reference_value"] == 1
    assert result.metadata["reference_evaluation"] == "best"
    assert result.metadata["objective_limits"] == [0.9, 1.1]
    assert result.metadata["omitted_incomplete_ids"] == [
        "failed", "incomplete",
    ]
    np.testing.assert_allclose(
        result.sampling[["window_lower", "window_upper"]], [[-0.4, 0.4]],
    )
    assert "Fitting GP to 4 of 4 complete evaluations" in output.getvalue()
    assert "band [0.9, 1.1]" in output.getvalue()
    assert "Parameter friction" in output.getvalue()

    output = StringIO()
    with redirect_stdout(output):
        explicit = analyse(
            parameters, objective, [[-1, 1]], reference = [0],
            reference_value = 1, grid_size = 9, verbose = False,
        )
    assert not output.getvalue()
    pd.testing.assert_frame_equal(explicit.ranges, result.ranges)
    assert explicit.metadata["bounds_source"] == "supplied"


def test_selected_reference_defaults():
    # Signed objectives still default to the smallest, with stable ties.
    parameters = np.array([[0.5], [-0.5], [-1], [1]])
    objective = np.array([-2., -2., 1., 1.])
    result = analyse(parameters, objective, grid_size = 3, verbose = False)
    assert result.reference.to_dict() == {"Parameter 1": 0.5}
    assert result.metadata["reference_value"] == -2
    assert result.metadata["reference_evaluation"] == "0"
    np.testing.assert_allclose(
        result.metadata["objective_limits"], [-2.2, -1.8],
    )

    chosen = analyse(
        parameters, objective, reference = [-1],
        grid_size = 3, verbose = False,
    )
    assert chosen.metadata["reference_value"] == 1
    assert chosen.metadata["reference_evaluation"] == "2"
    with np.testing.assert_raises_regex(ValueError, "Supply reference_value"):
        analyse(parameters, objective, reference = [0])
    with np.testing.assert_raises_regex(ValueError, "differing objectives"):
        analyse([[0], [0], [1]], [1, 2, 3], reference = [0])
    with np.testing.assert_raises_regex(ValueError, "finite min < max"):
        analyse([[0], [0]], [1, 1])
    with np.testing.assert_raises_regex(ValueError, "two complete"):
        analyse([[np.nan], [1]], [0, np.nan])


def test_objective_sign():
    parameters, objective, bounds, settings = quadratic_data()
    positive = analyse(parameters, objective, bounds, **settings)
    settings["reference_value"] = -1
    negative = analyse(parameters, -objective, bounds, **settings)

    # Turning a positive minimum into a negative maximum needs no direction.
    assert negative.metadata["objective_limits"] == [-2, 0]
    np.testing.assert_allclose(
        negative.ranges[["lower", "upper"]],
        positive.ranges[["lower", "upper"]], atol = 1e-6,
    )
    np.testing.assert_array_equal(
        negative.single.profile_accepted, positive.single.profile_accepted,
    )


def test_repeatability():
    parameters, objective, bounds, settings = quadratic_data()
    settings.update(grid_size = 9, pair_grid_size = 5, verbose = False)
    state = np.random.get_state()
    try:
        np.random.seed(123)
        first = analyse(parameters, objective, bounds, **settings)
        np.random.seed(987)
        second = analyse(parameters, objective, bounds, **settings)
    finally:
        np.random.set_state(state)

    # The seed isolates fitting from NumPy's global random state. Numerical
    # linear algebra and compensated minimisers have finite precision.
    for name, value in first.metadata.items():
        if isinstance(value, float):
            assert value == pytest.approx(
                second.metadata[name], rel = 1e-10, abs = 1e-12,
            )
        else:
            assert value == second.metadata[name]
    for a, b in [(first.single, second.single),
                 (first.ranges, second.ranges),
                 (first.ranking, second.ranking),
                 (first.ranking_full, second.ranking_full),
                 (first.importance, second.importance)]:
        pd.testing.assert_frame_equal(a, b, rtol = 1e-5, atol = 1e-6)
    for names in first.pairs:
        pd.testing.assert_frame_equal(
            first.pairs[names], second.pairs[names],
            rtol = 1e-5, atol = 1e-6,
        )


@pytest.mark.parametrize("scale", [1e-12, 1, 1e12, -1])
def test_pair_range_endpoints(scale):
    parameters, objective, bounds, settings = quadratic_data()
    settings.update(
        reference_value = scale, grid_size = 9, pair_grid_size = 5,
        verbose = False,
    )
    result = analyse(parameters, scale * objective, bounds, **settings)
    pair = result.pairs[("a", "b")]
    outside = abs(pair.slice_objective / scale - 1) > 1 + 1e-8
    assert outside.any()
    assert not pair.loc[outside, "slice_accepted"].any()
    assert not pair.loc[outside, "profile_accepted"].any()

    # Slice limits are roots of the tolerance boundary. Evaluating them in
    # a batch must still accept the closed boundary despite roundoff.
    for row in result.ranges.query("mode == 'slice'").itertuples():
        other = "b" if row.parameter == "a" else "a"
        for endpoint in [row.lower, row.upper]:
            point = pair.loc[
                (pair[f"value_{row.parameter}"] == endpoint) &
                (pair[f"value_{other}"] == 0)
            ]
            assert len(point) == 1
            np.testing.assert_allclose(
                point.slice_objective / scale, 2, atol = 1e-9, rtol = 0,
            )
            assert point.slice_accepted.all()
            assert point.profile_accepted.all()


def test_parameter_and_objective_units():
    parameters, objective, bounds, settings = quadratic_data()
    original = analyse(parameters, objective, bounds, **settings)
    physical = parameters.to_numpy().copy()
    physical[:, 1] = 30000 + 10000 * physical[:, 1]
    settings.update(reference = [0, 30000], reference_value = 1024)
    converted = analyse(
        physical, 1024 * objective.to_numpy(), [[-1, 1], [20000, 40000]],
        **settings,
    )
    assert converted.reference.index.tolist() == ["Parameter 1", "Parameter 2"]
    assert original.reference.index.tolist() == ["a", "b"]
    np.testing.assert_allclose(
        converted.importance.rms_objective_change / 1024,
        original.importance.rms_objective_change, rtol = 1e-5,
    )
    np.testing.assert_allclose(
        converted.ranking.relative_sensitivity,
        original.ranking.relative_sensitivity, rtol = 1e-5,
    )
    np.testing.assert_allclose(
        converted.ranking_full.relative_sensitivity,
        original.ranking_full.relative_sensitivity, rtol = 1e-5,
    )
    expected = original.ranges.query("parameter == 'b'").upper
    actual = converted.ranges.query("parameter == 'Parameter 2'").upper
    np.testing.assert_allclose(actual, 30000 + 10000 * expected, atol = 0.1)


def test_zero_reference():
    parameters, objective, bounds, settings = quadratic_data()
    settings.update(reference_value = 0, absolute_tolerance = 1)
    shifted = analyse(parameters, objective - 1, bounds, **settings)
    assert shifted.metadata["objective_limits"] == [-1, 1]
    # Test the physical tolerance against the same analytic limits at zero.
    exact = {("a", "slice"): 1 / np.sqrt(10),
             ("b", "slice"): 1 / np.sqrt(8.4),
             ("a", "profile"): 1 / np.sqrt(10 - 64 / 8.4),
             ("b", "profile"): 1 / np.sqrt(2)}
    for row in shifted.ranges.itertuples():
        assert abs(row.lower + exact[(row.parameter, row.mode)]) < 0.02
        assert abs(row.upper - exact[(row.parameter, row.mode)]) < 0.02

    # A zero reference without an absolute tolerance means a zero-width band.
    result = analyse(
        np.linspace(-1, 1, 9)[:, None], np.zeros(9), [[-1, 1]],
        reference = [0], reference_value = 0, grid_size = 3,
    )
    assert result.metadata["absolute_tolerance"] == 0
    assert result.single.slice_accepted.all()
    assert (result.ranking_full.relative_sensitivity == 0).all()
    np.testing.assert_allclose(result.ranges.width, 0.8)


def test_tolerance_crossings():
    table = pd.DataFrame({
        "value": np.arange(-3., 4.), "response": [2, 0, 2, 3, 2, 0, 2],
    })
    ranges = _intervals(
        table, "response", [-1, 1], 0,
        "parameter_window", "parameter_window",
    )
    assert [(r["lower"], r["upper"]) for r in ranges] == [
        (-2.5, -1.5), (1.5, 2.5),
    ]
    assert not any(r["contains_reference_value"] for r in ranges)

    # Both grid nodes are outside, on opposite sides of the whole band.
    crossing = pd.DataFrame({"value": [-1., 1.], "response": [-2., 2.]})
    ranges = _intervals(
        crossing, "response", [-0.2, 0.2], 0,
        "parameter_window", "parameter_window",
    )
    np.testing.assert_allclose(
        [ranges[0]["lower"], ranges[0]["upper"]], [-0.1, 0.1],
    )


def test_endpoint_units():
    # Root finding must have the same relative precision at any input scale.
    for scale in [1e-15, 1, 1e12]:
        axis = scale * np.array([-1., 0., 1.])
        response = lambda x: 1 + 10 * (x / scale) ** 2
        table = pd.DataFrame({"value": axis, "response": response(axis)})
        ranges = _intervals(
            table, "response", [0.9, 1.1], 0, "original_bound",
            "original_bound", evaluate = response,
        )
        assert len(ranges) == 1
        np.testing.assert_allclose(
            np.array([ranges[0]["lower"], ranges[0]["upper"]]) / scale,
            [-0.1, 0.1], atol = 1e-11,
        )


def test_zero_window():
    parameters, objective, bounds, settings = quadratic_data()
    settings.update(reference_value = -1, parameter_window = 0)
    result = analyse(parameters, -objective, bounds, **settings)
    assert (result.ranges.width == 0).all()
    assert (result.importance.rms_objective_change == 0).all()
    assert (result.importance["rank"] == 1).all()
    assert (result.ranking.relative_sensitivity == 0).all()
    assert (result.ranking["rank"] == 1).all()
    assert (result.importance_full.rms_objective_change > 0).all()
    np.testing.assert_allclose(
        result.ranking_full.relative_sensitivity.sum(), 1,
    )
    with tempfile.TemporaryDirectory() as directory:
        result.save(directory)
        path = Path(directory)
        summary = json.loads((path / "summary.json").read_text())
        assert summary["objective_limits"] == [-2, 0]
        assert (path / "sensitivity_fixed.png").is_file()
        assert (path / "sampling.csv").is_file()
        saved_ranking = pd.read_csv(path / "ranking.csv")
        pd.testing.assert_frame_equal(saved_ranking, result.ranking)
        saved_full = pd.read_csv(path / "ranking_full.csv")
        pd.testing.assert_frame_equal(saved_full, result.ranking_full)
        assert (path / "importance_full.csv").is_file()
        assert (path / "sensitivity_adjusted.png").is_file()
        assert len(list(path.glob("*.png"))) == 2


def test_whole_box_predictions():
    parameters = pd.DataFrame(
        [[0, 0], [1, 0], [0, 1], [.1, .1], [.4, .2], [.2, .4]],
        columns = ["a", "b"],
    )
    result = analyse(
        parameters, 1 + parameters.a ** 2 + parameters.b ** 2,
        [[0, 1], [0, 1]], reference = [0, 0], reference_value = 1,
        parameter_window = 1, objective_tolerance = 19,
        grid_size = 5, pair_grid_size = 5,
    )
    outside = result.pairs[("a", "b")].query("value_a + value_b > 1.00001")
    assert len(outside) > 0
    assert np.isfinite(outside.slice_objective).all()
    assert outside.profile_accepted.all()
    assert result.metadata["training_in_window"] == len(parameters)


def test_pair_compensation():
    parameters = pd.DataFrame(
        product(np.linspace(-1, 1, 5), repeat = 3), columns = ["a", "b", "c"],
    )
    a, b, c = parameters.a, parameters.b, parameters.c
    objective = 1 + 2 * (a + b + c) ** 2 + 0.2 * c ** 2
    result = analyse(
        parameters, objective, [[-1, 1]] * 3,
        reference = [0, 0, 0], reference_value = 1,
        parameter_window = 0.4, objective_tolerance = 0.25,
        grid_size = 5, pair_grid_size = 5,
    )
    pair = result.pairs[("a", "b")]
    row = pair.loc[
        np.isclose(pair.value_a, 0.4) & np.isclose(pair.value_b, 0.4)
    ].iloc[0]
    assert not row.slice_accepted
    assert row.profile_accepted
    assert row.compensating_c < -0.45

    # Every reported acceptance has a GP witness within the parameter box.
    for table in [result.single, *result.pairs.values()]:
        columns = [f"compensating_{n}" for n in parameters.columns]
        witnesses = table[columns].to_numpy()
        assert (abs(witnesses) <= 0.8 + 1e-10).all()
        predicted = result.model.predict(witnesses / 2)
        accepted = table.profile_accepted.to_numpy()
        reference = result.metadata["reference_value"]
        assert (abs(predicted[accepted] - reference) <= 0.25).all()
        assert (~table.slice_accepted | table.profile_accepted).all()
    np.testing.assert_allclose(pair.compensating_a, pair.value_a)
    np.testing.assert_allclose(pair.compensating_b, pair.value_b)


def test_compensation_matches_band():
    parameters = pd.DataFrame(
        product(np.linspace(-1, 1, 5), repeat = 2), columns = ["a", "b"],
    )
    result = analyse(
        parameters, parameters.a + parameters.b, [[-1, 1]] * 2,
        reference = [0, 0], reference_value = 0,
        absolute_tolerance = 0.05, parameter_window = 0.4,
        grid_size = 5, pair_grid_size = 3,
    )
    curve = result.single.query("parameter == 'a'")
    assert curve.profile_accepted.all()
    np.testing.assert_allclose(
        curve.compensating_b, -curve.value, atol = 0.05,
    )
    assert (abs(curve.profile_objective) <= 0.05).all()


def test_access_sensitivity():
    data = access_data()
    parameters = data.parameters.copy(deep = True)
    results = data.results.copy(deep = True)
    result = data.sensitivity()
    direct = analyse(
        results[["a", "b"]], results[["error"]], parameters,
        reference = results.loc[24, ["a", "b"]], reference_value = 2,
    )
    pd.testing.assert_frame_equal(result.single, direct.single)
    assert result.metadata["objective_limits"] == [1.8, 2.2]
    assert result.metadata["training_in_window"] == 9
    assert result.metadata["unique_training_in_window"] == 9
    assert result.metadata["training_evaluations"] == len(results)
    pd.testing.assert_frame_equal(data.results, results)
    pd.testing.assert_frame_equal(data.parameters, parameters)

    # Use the existing archive reader, then analyse the loaded data.
    fixture = Path(__file__).parent / "access_data/access_seed123"
    archived = coexist.AccessData(str(fixture)).sensitivity(
        n_neighbors = 24, grid_size = 3, pair_grid_size = 3,
    )
    assert archived.metadata["training_evaluations"] == 24


def test_individual_response():
    data = access_data()
    data.results["reward"] = 10 - data.results.a - 2 * data.results.b
    result = data.sensitivity(
        objective = "reward", objective_tolerance = 0.25,
        n_neighbors = 30, grid_size = 5, pair_grid_size = 3,
    )
    assert result.metadata["reference_evaluation"] == "24"
    assert result.metadata["reference_value"] == 10
    assert result.metadata["objective_limits"] == [7.5, 12.5]
    assert data.results.reward.idxmax() != 24
    assert result.metadata["training_evaluations"] == 30


def test_exclusions_and_reference():
    data = access_data()
    data.results = data.results.rename(index = {24: 187, 25: 188})
    data.results.loc[0, "a"] = np.nan
    settings = dict(grid_size = 3, pair_grid_size = 3)
    result = data.sensitivity(**settings)
    assert "187" in result.metadata["training_ids"]
    assert "188" in result.metadata["training_ids"]
    assert result.metadata["omitted_incomplete_ids"] == ["0"]
    filtered = data.sensitivity(excluded_evaluations = [187, 188], **settings)
    assert "187" not in filtered.metadata["training_ids"]
    assert "188" not in filtered.metadata["training_ids"]
    chosen = data.sensitivity(reference = 18, **settings)
    assert chosen.metadata["reference_value"] == data.results.loc[18, "error"]
    np.testing.assert_array_equal(
        chosen.reference, data.results.loc[18, ["a", "b"]],
    )
    assert 187 in data.results.index


def test_matrix_plots_and_saving():
    data = access_data()
    data.results["error"] = -data.results.error
    result = data.sensitivity(
        reference = 24, objective_tolerance = 0.25,
        grid_size = 5, pair_grid_size = 5,
    )
    for fixed in [True, False]:
        fig = result.plot(others_fixed = fixed)
        column = "slice_objective" if fixed else "profile_objective"
        diagonal = result.single.query("parameter == 'a'")
        np.testing.assert_array_equal(
            fig.axes[0].lines[0].get_ydata(), diagonal[column],
        )
        pair = result.pairs[("a", "b")]
        grid = pair.pivot(
            index = "value_b", columns = "value_a", values = column,
        )
        np.testing.assert_allclose(
            fig.axes[1].collections[0].get_array().ravel(),
            grid.to_numpy().T.ravel(),
        )
        assert fig.axes[1].get_xlabel() == "b"
        assert fig.axes[1].get_ylabel() == "a"
        assert fig.axes[2].get_xlabel() == "a"
        assert fig.axes[2].get_ylabel() == "b"
        legends = [ax.get_legend() for ax in fig.axes
                   if ax.get_legend() is not None]
        assert len(legends) == 1
        labels = [text.get_text() for text in legends[0].get_texts()]
        assert "Observed reference (ID 24)" in labels
        assert "Reference coordinates" in labels
        assert "GP within objective band" in labels
        accepted_label = any("accepted observations" in s for s in labels)
        assert accepted_label != fixed
        plt.close(fig)

    with tempfile.TemporaryDirectory() as directory:
        result.save(directory)
        pairs = pd.read_csv(Path(directory) / "pair_1.csv")
        reference = result.metadata["reference_value"]
        np.testing.assert_array_equal(
            pairs.slice_accepted,
            abs(pairs.slice_objective - reference) <= 0.5,
        )
        assert (Path(directory) / "sensitivity_adjusted.png").is_file()


def test_invalid_inputs():
    data = access_data()
    for tolerance in [-0.1, np.inf, np.nan]:
        with np.testing.assert_raises_regex(ValueError, "objective_tolerance"):
            data.sensitivity(objective_tolerance = tolerance)
    with np.testing.assert_raises_regex(ValueError, "reference"):
        data.sensitivity(reference = 100000)
    with np.testing.assert_raises_regex(ValueError, "scalar objective"):
        analyse(
            [[0], [1]], [[1, 2], [3, 4]], [[0, 1]],
            reference = [0], reference_value = 1,
        )


def test_smoothed_reference():
    # Repeated parameters with different outcomes require a smoothed response.
    result = analyse(
        [[-1], [-0.5], [0], [0], [0.5], [1]],
        [3, 2.25, 1, 3, 2.25, 3], [[-1, 1]],
        reference = [0], reference_value = 1, grid_size = 5,
        interpolate = False,
    )
    predicted = result.metadata["predicted_reference_value"]
    assert abs(predicted - 1) > 0.1
    np.testing.assert_allclose(
        result.metadata["objective_limits"],
        [0.9, 1.1],
    )
    reference = result.single.query("value == 0").iloc[0]
    assert not reference.slice_accepted
    assert not reference.profile_accepted
    assert not result.metadata["reference_accepted"]
    assert result.ranges.empty
    assert result.observed_acceptable.observed_objective.tolist() == [1]


def test_incomplete_combined_objective():
    data = access_data()
    data.results.loc[0, "error0"] = np.nan
    data.results.loc[0, "error"] = -100
    result = data.sensitivity(grid_size = 3, pair_grid_size = 3)
    assert result.metadata["reference_evaluation"] == "24"
    assert "0" in result.metadata["omitted_incomplete_ids"]
    assert "0" not in result.metadata["training_ids"]
    assert data.results.loc[0, "error"] == -100
    component = data.sensitivity(
        objective = "error1", grid_size = 3, pair_grid_size = 3,
    )
    assert component.metadata["reference_evaluation"] == "24"
    assert "0" in component.metadata["training_ids"]


def test_log_interpolation():
    parameters = np.linspace(-1, 1, 21)[:, None]
    objective = 1 + 10 * parameters[:, 0] ** 2
    result = analyse(
        parameters, objective, [[-1, 1]], reference = [0], reference_value = 1,
        response_transform = "log", grid_size = 5,
    )
    assert result.metadata["interpolate"]
    assert result.metadata["reference_accepted"]
    assert (result.single.slice_objective > 0).all()
    assert len(result.ranges) == 2
    for row in result.ranges.itertuples():
        assert abs(row.lower + 0.1) < 0.005
        assert abs(row.upper - 0.1) < 0.005
    for fixed in [True, False]:
        fig = result.plot(others_fixed = fixed, objective_scale = "log")
        assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)
    with np.testing.assert_raises_regex(ValueError, "positive"):
        analyse(parameters, -objective, [[-1, 1]], reference = [0],
                reference_value = -1, response_transform = "log")
    with np.testing.assert_raises_regex(ValueError, "Repeated parameters"):
        analyse([[0], [0], [1]], [1, 2, 3], [[0, 1]], reference = [0],
                reference_value = 1)


def test_nonuniform_ranking_grid():
    x = np.array([0, 0.01, 0.1, 0.9, 1.0])
    y = np.array([0, 0.2, 1.0])
    np.testing.assert_allclose(grid_rms(np.sqrt(x), x), 1 / np.sqrt(2))
    np.testing.assert_allclose(
        grid_rms(np.sqrt(y[:, None] * x[None, :]), y, x), 0.5,
    )


if __name__ == "__main__":
    test_analytic_ranges()
    test_local_and_full_rankings()
    test_default_bounds_and_reference()
    test_selected_reference_defaults()
    test_objective_sign()
    test_repeatability()
    test_parameter_and_objective_units()
    test_zero_reference()
    test_tolerance_crossings()
    test_endpoint_units()
    test_zero_window()
    test_whole_box_predictions()
    test_pair_compensation()
    test_compensation_matches_band()
    test_access_sensitivity()
    test_individual_response()
    test_exclusions_and_reference()
    test_matrix_plots_and_saving()
    test_invalid_inputs()
    test_smoothed_reference()
    test_incomplete_combined_objective()
    test_log_interpolation()
    test_nonuniform_ranking_grid()

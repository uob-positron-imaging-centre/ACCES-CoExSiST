#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_access_recovery.py
# License: GNU v3.0


'''Saved ACCES results must retain exact CMA-ES coordinates after recovery.'''


import  errno
import  textwrap
from    pathlib  import  Path
from    types    import  SimpleNamespace

import  numpy    as      np
import  pandas   as      pd
import  pytest

import  coexist
from    coexist.access  import  AccessProgress


archive = Path(__file__).parent / "access_data/access_seed123"


def saved_run(tmp_path, epochs = 3):
    data = coexist.AccessData(archive)[:epochs]
    data.save(tmp_path / "run")
    return data


@pytest.mark.parametrize("epochs", [1, 3])
@pytest.mark.parametrize("empty", ["", "# No data\n\n", None])
@pytest.mark.parametrize("damaged", ["history", "epochs", "both"])
def test_repair_unscaled(tmp_path, capsys, epochs, empty, damaged):
    data = saved_run(tmp_path, epochs)

    # Error columns must never be multiplied by the parameter scaling.
    for table in [data.results, data.results_scaled]:
        table.insert(len(data.parameters), "error0", -0.)
        table.insert(len(data.parameters) + 1, "error1", np.nan)
    data.save(tmp_path / "multi_objective")
    paths = data.paths
    scaled_files = [Path(paths.history_scaled), Path(paths.epochs_scaled)]
    originals = [path.read_bytes() for path in scaled_files]

    broken = {"history": [paths.history], "epochs": [paths.epochs],
              "both": [paths.history, paths.epochs]}[damaged]
    for name in broken:
        if empty is None:
            Path(name).unlink()
        else:
            Path(name).write_text(empty)

    recovered = coexist.AccessData(paths.directory)
    output = capsys.readouterr().out
    assert output.count("Repaired missing or empty ACCES file") == len(broken)
    assert recovered.num_epochs == epochs
    assert recovered.epochs.ndim == recovered.results.ndim == 2
    assert recovered.epochs.shape == (epochs, 2 * len(data.parameters) + 1)

    for path, original in zip(scaled_files, originals):
        assert path.read_bytes() == original
    for actual, expected in [
        (recovered.results_scaled, data.results_scaled),
        (recovered.epochs_scaled, data.epochs_scaled),
    ]:
        np.testing.assert_array_equal(
            actual.to_numpy().view(np.uint64),
            expected.to_numpy().view(np.uint64),
        )

    nparams = len(data.parameters)
    expected = data.results_scaled.to_numpy().copy()
    expected[:, :nparams] *= data.scaling
    np.testing.assert_allclose(recovered.results, expected, rtol = 1e-15)
    np.testing.assert_array_equal(
        recovered.results.iloc[:, nparams:].to_numpy().view(np.uint64),
        data.results_scaled.iloc[:, nparams:].to_numpy().view(np.uint64),
    )
    expected = data.epochs_scaled.to_numpy().copy()
    expected[:, :2 * nparams] *= np.r_[data.scaling, data.scaling]
    np.testing.assert_allclose(recovered.epochs, expected, rtol = 1e-15)
    np.testing.assert_array_equal(
        recovered.epochs.overall_std.to_numpy().view(np.uint64),
        data.epochs_scaled.overall_std.to_numpy().view(np.uint64),
    )

    # Repairs persist, and a subsequent read neither writes nor reports one.
    coexist.AccessData(paths.directory)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("scaled_name", ["history_scaled", "epochs_scaled"])
@pytest.mark.parametrize("damage", ["empty", "missing", "malformed"])
def test_scaled_data_cannot_be_reconstructed(tmp_path, scaled_name, damage):
    data = saved_run(tmp_path)
    files = {
        "history_scaled": Path(data.paths.history_scaled),
        "epochs_scaled": Path(data.paths.epochs_scaled),
    }
    path = files[scaled_name]
    if damage == "empty":
        path.write_text("")
    elif damage == "missing":
        path.unlink()
    else:
        path.write_text("1 2 3\n4 5\n")
    before = {name: file.read_bytes() for name, file in files.items()
              if file.exists()}

    with pytest.raises((ValueError, FileNotFoundError)):
        coexist.AccessData(data.paths.directory)
    assert before == {name: file.read_bytes() for name, file in files.items()
                      if file.exists()}


def test_incomplete_scaled_population(tmp_path):
    data = saved_run(tmp_path)
    scaled = Path(data.paths.history_scaled)
    np.savetxt(scaled, data.results_scaled.iloc[:-1].to_numpy())
    Path(data.paths.history).write_text("")
    before = scaled.read_bytes()
    with pytest.raises(ValueError, match = "complete populations"):
        coexist.AccessData(data.paths.directory)
    assert scaled.read_bytes() == before
    assert Path(data.paths.history).read_bytes() == b""


def test_single_population_resume_tables(tmp_path):
    data = saved_run(tmp_path, epochs = 1)
    Path(data.paths.history).write_text("")
    Path(data.paths.epochs).write_text("")
    access = SimpleNamespace(
        setup = SimpleNamespace(
            parameters = data.parameters, scaling = data.scaling,
            population = data.population,
        ),
        progress = AccessProgress(), verbose = 0,
    )
    data.paths.load_history(access)
    data.paths.load_epochs(access)
    assert access.progress.epochs.shape == (1, 7)
    assert access.progress.history.shape == (8, 4)
    np.testing.assert_array_equal(
        access.progress.history_scaled.view(np.uint64),
        data.results_scaled.to_numpy().view(np.uint64),
    )


def test_uncommitted_epoch(tmp_path, capsys):
    data = saved_run(tmp_path, epochs = 3)
    # Epoch 3 was saved, but history still contains only 2 populations.
    data.results = data.results.iloc[:2 * data.population]
    data.results_scaled = data.results_scaled.iloc[:2 * data.population]
    data.save(tmp_path / "interrupted")
    Path(data.paths.history).write_text("")
    scaled_paths = [
        Path(data.paths.history_scaled), Path(data.paths.epochs_scaled),
    ]
    before = [path.read_bytes() for path in scaled_paths]

    recovered = coexist.AccessData(data.paths.directory)
    assert recovered.num_epochs == 2
    assert len(recovered.results) == 2 * recovered.population
    pd.testing.assert_frame_equal(recovered.epochs, data.epochs.iloc[:2])
    assert "Ignoring the final ACCES epoch record" in capsys.readouterr().out
    assert before == [path.read_bytes() for path in scaled_paths]


def test_all_failed_population_can_be_read(tmp_path):
    data = saved_run(tmp_path, epochs = 1)
    data.results["error"] = np.nan
    data.results_scaled["error"] = np.nan
    data.save(tmp_path / "failed")
    recovered = coexist.AccessData(data.paths.directory)
    assert recovered.num_epochs == 1
    assert recovered.results.error.isna().all()


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_atomic_write_preserves_saved_values(tmp_path, monkeypatch, failure):
    path = tmp_path / "history.csv"
    original = np.array([[np.nextafter(0.1, 1.), -0., 1e-200]])
    coexist.access.save_access_table(path, original, "a b error")
    before = path.read_bytes()

    def no_space(*args, **kwargs):
        if failure == "write":
            args[0].write("# Partly written replacement\n")
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as failed:
        if failure == "write":
            failed.setattr(np, "savetxt", no_space)
        else:
            failed.setattr(coexist.access.os, "fsync", no_space)
        with pytest.raises(OSError) as error:
            coexist.access.save_access_table(path, original + 1, "a b error")
        assert error.value.errno == errno.ENOSPC
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(
        np.loadtxt(path, ndmin = 2).view(np.uint64), original.view(np.uint64),
    )


@pytest.mark.parametrize("failed_table", [
    "epochs", "epochs_scaled", "history", "history_scaled",
])
def test_resumed_cma_history_is_exact(tmp_path, monkeypatch, failed_table):
    script = textwrap.dedent('''
        # ACCESS PARAMETERS START
        import coexist
        parameters = coexist.create_parameters(
            ["a", "b"], [-5, -5], [10, 10],
        )
        # ACCESS PARAMETERS END
        error = 0.
    ''')

    # Evaluate a deterministic toy objective in-process to exercise CMA-ES
    # replay without starting a Python interpreter for every evaluation.
    def evaluate(self, solutions, epoch):
        return (1 + np.sum(solutions ** 2, axis = 1))[:, None]

    monkeypatch.setattr(coexist.Access, "evaluate_solutions", evaluate)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    monkeypatch.chdir(baseline)
    Path("simulation.py").write_text(script)
    expected = coexist.Access("simulation.py").learn(
        num_solutions = 4, target_sigma = 0.1, random_seed = 123, verbose = 0,
    )
    assert expected.num_epochs > 3

    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    monkeypatch.chdir(interrupted)
    Path("simulation.py").write_text(script)
    original_save = coexist.access.save_access_table
    filename = ("epochs" if failed_table.startswith("epochs") else "history")
    filename += "_pop4"
    filename += "_scaled" if failed_table.endswith("scaled") else ""
    filename += ".csv"
    writes = []

    def fail_third_epoch(path, values, header):
        if Path(path).name == filename:
            writes.append(path)
            if len(writes) == 3:
                raise OSError(errno.ENOSPC, "No space left on device")
        original_save(path, values, header)

    with monkeypatch.context() as failed:
        failed.setattr(coexist.access, "save_access_table", fail_third_epoch)
        with pytest.raises(OSError):
            coexist.Access("simulation.py").learn(
                num_solutions = 4, target_sigma = 0.1, random_seed = 123,
                verbose = 0,
            )

    # Reproduce the truncated unscaled files made by older direct writers.
    run = Path("access_seed123")
    if not failed_table.endswith("scaled"):
        (run / filename).write_text("")
    scaled_paths = [
        run / "history_pop4_scaled.csv", run / "epochs_pop4_scaled.csv",
    ]
    before = [path.read_bytes() for path in scaled_paths]
    repaired = coexist.AccessData(run)
    assert repaired.num_epochs == 2
    assert before == [path.read_bytes() for path in scaled_paths]

    actual = coexist.Access("simulation.py").learn(
        num_solutions = 4, target_sigma = 0.1, random_seed = 123, verbose = 0,
    )
    # Compare every sample and objective, including all newly proposed steps.
    for actual_table, expected_table in [
        (actual.results_scaled, expected.results_scaled),
        (actual.epochs_scaled, expected.epochs_scaled),
    ]:
        np.testing.assert_array_equal(
            actual_table.to_numpy().view(np.uint64),
            expected_table.to_numpy().view(np.uint64),
        )


def test_progress_constructor():
    values = np.arange(8, dtype = float).reshape(2, 4)
    progress = AccessProgress(history = values, stdout = "finished")
    assert progress.history is values
    assert progress.stdout == "finished"


def test_single_evaluation_table(tmp_path):
    path = tmp_path / "history.csv"
    scaled_path = tmp_path / "history_scaled.csv"
    scaled = np.array([[np.nextafter(0.1, 1.), 7.]])
    np.savetxt(scaled_path, scaled)
    before = scaled_path.read_bytes()
    values, loaded = coexist.access.read_access_table(
        path, scaled_path, [6.], ["a"], population = 1,
    )
    assert values.shape == loaded.shape == (1, 2)
    np.testing.assert_array_equal(
        loaded.view(np.uint64), scaled.view(np.uint64),
    )
    np.testing.assert_array_equal(values, scaled * [6., 1.])
    assert scaled_path.read_bytes() == before

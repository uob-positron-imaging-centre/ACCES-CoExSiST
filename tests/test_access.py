#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_access.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>


'''Integration tests for ACCES runs, saved data and schedulers.'''


import  sys
import  shutil
import  textwrap
from    pathlib  import  Path

import  numpy    as      np
import  pandas   as      pd
import  pytest

import  coexist


archive = Path(__file__).parent / "access_data/access_seed123"


def test_access_data(tmp_path):
    data = coexist.AccessData(archive)
    pd.testing.assert_frame_equal(
        data.results, coexist.AccessData.read(archive).results,
    )
    assert len(data[0].results) == data.population
    assert len(data[0:1].results) == data.population
    pd.testing.assert_frame_equal(data[-1].results, data[-1:].results)
    pd.testing.assert_frame_equal(data[:].results, data.results)

    restored = tmp_path / "restored"
    data[:-1].save(str(restored))
    data2 = coexist.AccessData(restored)
    assert data2.num_epochs == data.num_epochs - 1
    pd.testing.assert_frame_equal(data2.results, data[:-1].results)

    # One-epoch files must retain their two-dimensional table shape.
    single = tmp_path / "single"
    data[0].save(str(single))
    data3 = coexist.AccessData(single)
    assert data3.epochs.shape == (1, len(data.epochs.columns))
    pd.testing.assert_frame_equal(data3.results, data[0].results)


def test_access_paths(tmp_path):
    run = tmp_path / "access_seed123"
    shutil.copytree(archive, run)
    data = coexist.AccessData(tmp_path)
    assert Path(data.paths.directory) == run

    second = tmp_path / "access_seed456"
    shutil.copytree(archive, second)
    with pytest.raises(RuntimeError, match = "Multiple ACCES directories"):
        coexist.AccessData(tmp_path)

    renamed = tmp_path / "calibration"
    run.rename(renamed)
    pd.testing.assert_frame_equal(
        coexist.AccessData(renamed).results, data.results,
    )


def test_access_requires_setup(tmp_path):
    # A history table and pickle are not a complete ACCES run.
    (tmp_path / "opt_history_10.csv").write_text("1 2 3\n")
    (tmp_path / "access_info.pickle").write_bytes(b"not a pickle")
    with pytest.raises(FileNotFoundError, match = "access_setup.toml"):
        coexist.AccessData(tmp_path)


@pytest.mark.parametrize("directive", [
    "# ACCESS PARAMETERS START",
    "#####   ACCES \t PARAMETERS    START\textra text",
])
def test_access(tmp_path, monkeypatch, directive):
    monkeypatch.chdir(tmp_path)
    code = textwrap.dedent('''
        # ACCESS PARAMETERS START
        import coexist

        parameters = coexist.create_parameters(
            ["x", "y"], [-5, -5], [10, 10],
        )
        # ACCESS PARAMETERS END

        values = parameters["value"]
        error = values["x"] ** 2 + values["y"] ** 2
    ''').replace("# ACCESS PARAMETERS START", directive)
    Path("simulation.py").write_text(code)

    access = coexist.Access("simulation.py")
    data = access.learn(
        num_solutions = 4, target_sigma = 0.8, random_seed = 123,
        verbose = 0,
    )
    assert data.results.error.min() < 2 * 2.5 ** 2
    assert len(data.results) == data.population * data.num_epochs
    assert np.isfinite(data.results.to_numpy()).all()
    pd.testing.assert_frame_equal(
        data.results, coexist.AccessData(access.paths.directory).results,
    )

    # A completed run can be resumed without evaluating its history again.
    resumed = coexist.Access("simulation.py").learn(
        num_solutions = 4, target_sigma = 0.8, random_seed = 123,
        verbose = 0,
    )
    pd.testing.assert_frame_equal(resumed.results, data.results)


@pytest.mark.parametrize("crashes", [0, 5])
def test_access_multi_objective(tmp_path, monkeypatch, crashes):
    monkeypatch.chdir(tmp_path)
    code = textwrap.dedent('''
        # ACCESS PARAMETERS START
        import coexist

        parameters = coexist.create_parameters(
            ["x", "y"], [-5, -5], [10, 10],
        )
        access_id = 0
        # ACCESS PARAMETERS END

        if access_id < CRASHES:
            raise ValueError("Simulation failed.")

        values = parameters["value"]
        error = [1 + values["x"] ** 2, 1 + values["y"] ** 2]
    ''').replace("CRASHES", str(crashes))
    Path("simulation.py").write_text(code)
    data = coexist.Access("simulation.py").learn(
        num_solutions = 4, target_sigma = 0.3, random_seed = 123,
        verbose = 0,
    )
    assert list(data.results.columns) == ["x", "y", "error0", "error1", "error"]
    assert data.results.iloc[:crashes][["error0", "error1"]].isna().all().all()
    complete = data.results.dropna()
    assert len(complete) > 0
    np.testing.assert_allclose(
        complete.error, complete.error0 * complete.error1,
    )
    np.testing.assert_allclose(
        data.parameters["value"],
        data.results.loc[data.results.error.idxmin(), ["x", "y"]],
    )


def test_access_plots():
    data = coexist.AccessData(archive)
    for source in [data, str(archive)]:
        assert len(coexist.plots.access(source).data) > 0
        assert len(coexist.plots.access2d(source).data) > 0


def test_schedulers(tmp_path):
    local = coexist.schedulers.LocalScheduler()
    assert local.schedule(str(tmp_path), 0)[0] == sys.executable

    slurm = coexist.schedulers.SlurmScheduler(
        "10:0:0",
        commands = "set -e\nmodule load python",
        qos = "normal", account = "research", mem_per_cpu = "4G",
    )
    assert slurm.schedule(str(tmp_path), 0)[0] == "sbatch"
    script = (tmp_path / slurm.script).read_text()
    assert "module load python" in script
    assert "--account research" in script

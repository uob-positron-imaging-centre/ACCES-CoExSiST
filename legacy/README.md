# Research archive

This directory preserves the microscopic CoExSiST method, particle ballistics,
the LIGGGHTS interface and the simulation-coupled `AccessCoupled` prototype.
They are research source code for further development and are excluded from the
installed `coexist` package and its source distribution.

- `optimisation.py`: microscopic experimental/simulation trajectory matching.
- `base.py`: particle simulation and experiment structures, persistence and VTK
  export.
- `liggghts.py`: the LIGGGHTS simulation interface and automatic timestep helper.
- `ballistics.py`, `prototype/`: analytical particle motion and experiments.
- `access_coupled.py`: the simulation-object-based optimisation prototype.
- `examples/`, `tests/`, `fixtures/`: associated examples and historical data.

The supported ACCES interface is `coexist.Access`, which runs user-defined
simulation scripts. Its data reader uses run directories containing
`access_setup.toml`; the historical pickle-based fixtures here are not readable
by `coexist.AccessData`.

To develop these archived interfaces, work from a repository checkout and install
the additional dependencies required by the code: `pyevtk`, `tqdm`, and `sympy`.
The LIGGGHTS interface also requires the LIGGGHTS Python bindings, imported
explicitly with `from legacy.liggghts import LiggghtsSimulation`.
These interfaces are not covered by the package's test suite.

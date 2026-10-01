[![PyPI version](https://badge.fury.io/py/coexist.svg)](https://badge.fury.io/py/coexist)
[![Documentation Status](https://readthedocs.org/projects/coexist/badge/?version=latest)](https://coexist.readthedocs.io/en/latest/?badge=latest)
[![CI Status](https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST/actions/workflows/ci.yml/badge.svg)](https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST/actions/workflows/ci.yml)
[![Colab example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p7OwnaYgENwK4DTn_6QETX4ajwVFeza0?usp=sharing)
[![License: GPL-3.0](https://img.shields.io/github/license/uob-positron-imaging-centre/ACCES-CoExSiST?style=flat-square)](https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST)


# ACCES
### Data-Driven Evolutionary Calibration & Optimisation of Simulations

ACCES (Autonomous Characterisation and Calibration via Evolutionary Simulation)
learns simulation parameters by minimising a user-defined objective. The Python
package is named `coexist`.

ACCES can run arbitrary simulation scripts, from laptops to clusters. It is
independent of the simulation engine and the quantity being optimised: examples
include DEM calibration against a powder's free-surface shape, residence time
distributions and coarse-grained digital twins.

![Calibrated GranuDrum](/docs/source/_static/calibrated.png?raw=true "Calibrated GranuDrum.")
*Example of an ACCES-calibrated DEM Digital Twin of a GranuTools GranuDrum; the calibration was done effectively against a single experimental data point - a photograph of the free surface shape yielded by MCC particles (left panel). The occupancy grid of a LIGGGHTS simulation was optimised against the free surface shape (right panel). The two superimposed grids amount to 4 mm² dissimilarity (dark blue pixels, middle panel).*


ACCES was implemented in the `coexist.Access` class, providing an interface that is easy to use, but powerful enough to **automatically parallelise arbitrary Python scripts** through code inspection and metaprogramming. It was used successfully from laptop-scale shared-memory machines to multi-node supercomputing clusters.





## Getting Started

This is a pure Python package that does not require any extra system configuration, supporting Python 3.10 and above - to install it from PyPI, simply run:

```
pip install coexist
```

Or you can install the development version from the GitHub repository:

```
pip install git+https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST
```


Sensitivity analysis uses an optional scikit-learn dependency:

```sh
python -m pip install "coexist[sensitivity]"
```

An existing run can be analysed with `coexist.AccessData("access_seed42").sensitivity()`.
The default parameter window is 20% of each bound width, and the objective
band is ±10% of the best observed objective's magnitude. See the
[sensitivity manual](https://coexist.readthedocs.io/en/latest/manual/sensitivity.html)
for interpretation, plots and the general-purpose NumPy/pandas API.

For development, from a repository checkout:

```sh
python -m pip install -e ".[test,docs,sensitivity]"
python -m pytest
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Dependencies and package metadata are declared in `pyproject.toml`. The `test`,
`docs` and `sensitivity` extras can be installed separately; building the complete
API documentation requires both `docs` and `sensitivity`.

The [research archive](https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST/tree/main/legacy)
contains the microscopic CoExSiST method and its simulation interfaces. It is
kept for future development and is not included in the installed package.




### Examples

The [documentation](https://coexist.readthedocs.io/) website contains an ACCES [tutorial](https://coexist.readthedocs.io/en/latest/tutorials/index.html) with explained code and output figures produced by `coexist`; all public functionality is fully documented in the [manual](https://coexist.readthedocs.io/en/latest/manual/index.html).

Want something more hands on? Check out the `examples` folder for example scripts using `coexist.Access`; `examples/access_simple` is a very simple, hackable example script (remember that the `simulation_script.py` can execute *anything*). For a more involved, complete calibration of a GranuTools GranuDrum digital twin - using LIGGGHTS - see our collection of peer-reviewed digital twins [repository](https://github.com/uob-positron-imaging-centre/DigitalTwins).


![GranuDrum ACCES Example](/docs/source/_static/access_example.png?raw=true "GranuDrum ACCES Example.")


The `coexist.plots` submodule can plot the convergence of ACCES onto optimally-calibrated parameters - or check intermediate results while your 50-hour simulations runs:


![Convergence Plot](/docs/source/_static/access_convergence.png?raw=true "ACCES Convergence Plot.")





### Show me some code already...

Alright, here's the gist of it: instead of having to rewrite your complex simulation into a function for calibration / optimisation (which is what virtually all optimisation frameworks require...), ACCES takes in an _entire simulation script_; here's a simple example, say in a file called `simulation_script.py`:

```python
# File: simulation_script.py

# ACCESS PARAMETERS START
import coexist

parameters = coexist.create_parameters(
    variables = ["CED", "CoR", "Epsilon", "Mu"],
    minimums = [-5, -5, -5, -5],
    maximums = [+5, +5, +5, +5],
    values = [0, 0, 0, 0],          # Optional, initial guess
)

access_id = 0                       # Optional, unique ID for each simulation
# ACCESS PARAMETERS END


# Extract the free parameters' values - ACCES will modify / optimise them.
x, y, z, t = parameters["value"]


# Define the error value in *any* way - run a simulation, analyse data, etc.
#
# For simplicity, here's an analytical 4D Himmelblau function with 8 global
# and 2 local minima - the initial guess is very close to the local one!
error = (x**2 + y - 11)**2 + (x + y**2 - 7)**2 + z * t
```

Then you can run in a separate script or Python console:

```python
# File: access_learn.py

import coexist

# Use ACCESS to learn a simulation's parameters
access = coexist.Access("simulation_script.py")
access.learn(
    num_solutions = 10,         # Number of solutions per epoch
    target_sigma = 0.1,         # Target std-dev (accuracy) of solution
    random_seed = 42,           # Reproducible / deterministic optimisation
)

# Or simply
# coexist.Access("simulation_script.py").learn(random_seed = 42)
```

That's it - ACCES will take the `simulation_script.py`, modify the free `parameters`, run the script in parallel (either on all processors of your local machine or on a distributed supercomputer) and optimise the `error` you defined; it'll look something like this:

```
================================================================================
Starting ACCES run at 08:32:23 on 01/02/2022 in directory `access_seed42`.
================================================================================
Epoch    0 | Population 10
--------------------------------------------------------------------------------
Scaled overall standard deviation: 1.0
             CED       CoR   Epsilon        Mu
estimate     0.0  0.000000  0.000000  0.000000
uncertainty  4.0  4.000050  4.000100  4.000150
scaled_std   1.0  1.000013  1.000025  1.000038

        CED       CoR   Epsilon        Mu       error
0  1.218868 -4.159988  3.001880  3.762400  331.093228
1 -3.095859 -4.967675  0.511374 -1.265018  252.732762
2 -0.067205 -3.412218  3.517680  3.111284  239.466423
3  0.264123  4.509021  1.870084 -3.437299  219.638764
4  1.475003 -3.835578  3.513889 -0.199711  243.967225
5 -0.739449 -2.723752  4.825957 -0.618141  170.752129
6 -1.713311 -1.408552  2.129290  1.461831  138.135979
7  1.650930  1.723306  2.333195 -1.625721   44.785098
8 -2.048971 -3.255132  2.463979  4.516059  114.660635
9 -0.455790 -3.360668 -3.298007  2.602469  206.454823
Total function evaluations: 10

================================================================================
Epoch    1 | Population 10
--------------------------------------------------------------------------------
...
<output truncated>
...
================================================================================
Epoch   31 | Population 10
--------------------------------------------------------------------------------
Scaled overall standard deviation: 0.1032642976669169
                  CED       CoR   Epsilon        Mu
estimate     3.621098 -1.748473  4.999445 -4.992881
uncertainty  0.052355  0.078080  0.284105  0.180196
scaled_std   0.013089  0.019520  0.071026  0.045049

          CED       CoR   Epsilon        Mu      error
310  3.574473 -1.650134  4.999374 -4.901913 -23.996813
311  3.582026 -1.756384  4.863496 -4.973417 -24.071692
312  3.628789 -1.616891  4.859678 -4.992656 -23.386000
313  3.662351 -1.782704  4.982040 -4.998059 -24.478008
314  3.594971 -1.725715  4.999877 -4.898257 -24.269162
315  3.577725 -1.744971  4.998411 -4.956502 -24.629198
316  3.613914 -1.747412  4.983253 -4.996630 -24.690880
317  3.579212 -1.774811  4.852262 -4.972910 -24.055219
318  3.634952 -1.784927  4.999971 -4.999863 -24.783959
319  3.647169 -1.872419  4.999971 -4.978640 -24.685207
Total function evaluations: 320


Optimal solution found within `target_sigma`, i.e. 10.0%:
  sigma = 0.08390460663313491 < 0.1

================================================================================
The best result was found in 32 epochs:
              CED       CoR   Epsilon        Mu      error
  value  3.569249 -1.813354  4.995112 -4.994052 -24.920092
  std    0.042168  0.060116  0.203900  0.140874

These results were found for the job:
  access_seed42/results/parameters.284.pickle
================================================================================
```

And you can access (pun intended) the results - even as ACCES is running - using:

```python
>>> import coexist
>>> data = coexist.AccessData.read("access_seed42")
>>> data
AccessData
--------------------------------------------------------------------------------
paths          ╎ AccessPaths(...)
parameters     ╎             value  min  max     sigma
               ╎ CED      3.569249 -5.0  5.0  0.052355
               ╎ CoR     -1.813354 -5.0  5.0  0.078080
               ╎ Epsilon  4.995112 -5.0  5.0  0.284105
               ╎ Mu      -4.994052 -5.0  5.0  0.180196
population     ╎ 10
num_epochs     ╎ 32
target         ╎ 0.1
seed           ╎ 42
epochs         ╎ DataFrame(CED_mean, CoR_mean, Epsilon_mean, Mu_mean, CED_std,
               ╎           CoR_std, Epsilon_std, Mu_std, overall_std)
epochs_scaled  ╎ DataFrame(CED_mean, CoR_mean, Epsilon_mean, Mu_mean, CED_std,
               ╎           CoR_std, Epsilon_std, Mu_std, overall_std)
results        ╎ DataFrame(CED, CoR, Epsilon, Mu, error)
results_scaled ╎ DataFrame(CED, CoR, Epsilon, Mu, error)
```

In this case a global optimum was found within 320 evaluations - this is of course problem-dependent, but you'll see that the optimum is often found much earlier if you check intermediate results (which you probably will when calibrating / optimising long-running simulations).

A tutorial with more detailed explanations is available [here](https://coexist.readthedocs.io/en/latest/tutorials/index.html), including generating plots, checking intermediate results and running simulations on a SLURM-managed distributed cluster.




## Parameter sensitivity

Analyse how far parameters can vary around an existing optimum while keeping
an objective within a chosen tolerance. Signed objectives and individual
responses are supported without a minimise/maximise setting:

```python
import coexist

data = coexist.AccessData("calibration_results/access_seed42")
result = data.sensitivity(
    parameter_window = 0.2,
    objective_tolerance = 0.1,
)
print(result.ranges)
print(result.importance)
print(result.ranking)
print(result.ranking_full)
print(result.interactions)
print(result.metadata["training_in_window"])
result.save("sensitivity_output")
```

The tolerance permits `abs(prediction - observed_reference)` up to
`0.1 * abs(observed_reference)`. The band remains centred on the observed
reference even when the GP predicts a different value there. Both values are
reported, and accepted ranges may be empty.
`absolute_tolerance` supplies a half-width in objective units, useful at zero.
The window permits ±20% of each original parameter-bound width. The reference
is the best complete saved combined score; `objective="error0"` studies an
individual response at that same vector.

Results include fixed-other and compensated ranges, pair maps, interaction
scores, importance rankings and the number of samples inside the window.
`result.ranking` covers the supplied local window; `result.ranking_full` covers
the complete min/max bounds. Both use the same GP and hold other parameters
fixed. Each normalises its RMS objective changes by their sum, giving
dimensionless `relative_sensitivity` scores summing to 1. All-flat responses
receive zero scores. `result.importance` and `result.importance_full` retain
objective units. These are relative response magnitudes, not variance
fractions. Saving writes all four tables. The full-bound ranking can cover
poorly sampled regions; plots and accepted ranges stay within the local window.
Predictions cover the whole box; sample counts alone do not guarantee local
accuracy. Profile ranges are numerical grid approximations with stored
compensating vectors. The default GP interpolates recorded evaluations with
a small numerical nugget and a signed `asinh` response transform.
For positive discrepancies, `response_transform="log"` keeps predictions
positive. Use `interpolate=False` to estimate a smoothed response instead.
Interpolation does not establish predictive accuracy between evaluations.

`result.plot(others_fixed=True)` returns a matrix with single responses on the
diagonal, accepted pair variations below, and objective heatmaps above. Use `False` for
adjusted parameters. `result.save(...)` writes both matrices as PNG and PDF.
Legends identify the reference, observed alternatives and fitted regions.
DataFrame column names are retained; array inputs use `Parameter 1`, etc.
The colour scale is shared across panels and figures. `objective_scale="log"`
improves readability for positive objectives spanning orders of magnitude;
`"symlog"` supports signed values. Adjusted plots distinguish actual accepted
evaluations from fitted regions. Missing component errors invalidate a saved
combined score; additional processing failures can be explicitly excluded.

All analysis lives in `coexist/sensitivity.py`. The general function
`coexist.sensitivity.analyse(samples, objectives)` accepts NumPy arrays or pandas
data independently of ACCES. Bounds default to the column minima/maxima of
complete evaluations, and the reference to the smallest observed objective.
Supply `bounds`, `reference` and `reference_value` to use other limits or study
a maximum. Progress is printed by default; `verbose=False` silences it.
`AccessData.sensitivity` forwards the stored data and original bounds to that API.
Load a directory with the existing `AccessData` reader, then call
`.sensitivity()`. The general module has no dependency on ACCES data structures.
See the [sensitivity manual](docs/source/manual/sensitivity.rst) for details.




## Contributing

This library aims to be the state-of-the-art for simulation calibration, developed in the open using modern, collaborative coding approaches - no dragons shall be dwelling in the codebase. You are more than welcome to contribute to this library in the form of code improvements, documentation or helpful examples; please submit them either as:

- GitHub issues.
- Pull requests.
- Email me at <a.l.nicusan@gmail.com>.

We are more than happy to discuss the library architecture and calibration / optimisation approach with any potential contributors and users.




## Acknowledgements and Funding

The authors gratefully acknowledge funding from the following UK funding bodies and industrial partners:

**M²E³D: Multiphase Materials Exploration via Evolutionary Equation Discovery**  
Royce Materials 4.0 Feasibility and Pilot Scheme Grant, £57,477  

**CoExSiST: Coupled Experimental-Simulational Technique**  
EPSRC MAPP Grant, Feasibility Study, £60,246  

**ACCES: Autonomous Calibration and Characterisation via Evolutionary Simulation**  
EPSRC IAA, Follow-Up Grant to CoExSiST, £52,762  

**Improving ACCES: Towards the Multi-Tool Multi-Parameter Optimisation of Complex Particulate Systems**  
EPSRC MAPP, Grant, £48,871 + £48,871 matched funding from GranuTools Belgium  

Thank you.




## Citing

If you use this library in your research, please cite the software using the
metadata in [CITATION.cff](CITATION.cff), including the version you used.


ACCES is built on top of the excellent CMA-ES evolutionary algorithm - specifically the [`pycma`](https://github.com/CMA-ES/pycma) implementation. If you use ACCES in your research, please also cite:

> Nikolaus Hansen, Youhei Akimoto, and Petr Baudis. CMA-ES/pycma on Github. Zenodo, DOI:10.5281/zenodo.2559634, February 2019.




## License and Commercial Integration

This library - in its general, domain-agnostic form - is free and open-source, published under the GNU General Public License v3.0.

If you are a company and would like to integrate ACCESS into your work - e.g. ACCESS-enabled equipment or general simulation calibration - please send an email to `a.l.nicusan@bham.ac.uk` to discuss commercial development of specific tools for your application. Relicensing for a closed-source / commercial project can be considered on an individual basis.


Copyright (C) 2020-2026 the Coexist developers. Until now, this library was built directly or indirectly through the brain-time of:

- Andrei Leonard Nicusan (University of Birmingham)
- Dominik Werner (University of Birmingham)
- Jack Sykes (University of Birmingham)
- Dr. Kit Windows-Yule (University of Birmingham)
- Prof. Jonathan Seville (University of Birmingham)
- Albert Bauer (TU Berlin)

Thank you.

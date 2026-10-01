Parameter Sensitivity (``coexist.sensitivity``)
===============================================

Find how much parameters can vary around an existing optimum while keeping a
Gaussian-process prediction within a chosen tolerance of its observed
objective value.
This works for calibration, signed optimisation objectives and individual
responses, without additional simulations.

Install the optional sensitivity dependency before using this module:

.. code-block:: console

   python -m pip install "coexist[sensitivity]"

Using ACCES results
-------------------

.. code-block:: python

   import coexist

   data = coexist.AccessData("access_seed42")
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
   result.plot().show()

The default reference is the complete evaluation with the smallest saved
combined ``error``, which is the score ACCES minimises. ``objective = "error0"``
analyses that response at the **same parameter vector**, even if the response
was maximised. ``reference = evaluation_id`` chooses another observed vector.
Only one scalar response is fitted per call; the saved combined objective is
used directly without reconstructing its combiner.
For a combined ``error``, rows missing individual error components are omitted:
a finite penalty or imputed score does not represent a complete evaluation.
Use ``excluded_evaluations`` for additional documented failures.

Using other data
----------------

All fitting, numerical analysis, saving and plotting live in ``sensitivity.py``.
``AccessData.sensitivity`` only selects and forwards the data to ``analyse``.
The general API accepts NumPy arrays or pandas tables:

.. code-block:: python

   from coexist.sensitivity import analyse

   # samples: (n_samples, n_parameters), objectives: (n_samples,)
   result = analyse(samples, objectives)

   # Alternatively supply known bounds and a specific optimum.
   # bounds: (n_parameters, 2), containing [minimum, maximum] per parameter
   result = analyse(
       samples,
       objectives,
       bounds,
       reference = samples[optimum_index],
       reference_value = objectives[optimum_index],
       parameter_window = 0.2,
       objective_tolerance = 0.1,
   )
   result.plot().show()

For pandas inputs, use a parameter DataFrame, an aligned objective Series,
and optional bounds indexed by parameter with ``min`` and ``max`` columns.
Bounds default to column minima/maxima across complete evaluations. Omit
constant columns: their variation cannot be estimated from these samples.
The reference defaults to the smallest complete objective (first in input
order for a tie). A supplied reference vector can be a Series; its value is
read from matching observations unless explicitly supplied. A reference outside
the samples provided can be provided explicitly.

Fitting counts, the band and analysis progress are printed by default.
Use ``verbose = False`` to silence these messages.

Repeatability
-------------

``random_state = 42`` fixes the random starts used when fitting the GP kernel.
Compensation searches use deterministic starts and stable ordering for ties.
Repeating an analysis with identical ordered data, settings and software
environment gives repeatable numerical tables to floating-point and optimiser
precision. Different seeds, library versions or numerical backends can produce
different fitted responses or rounding.

The procedure
-------------

1. Centre a box on the reference. ``parameter_window = 0.2`` permits +/-20% of
   each **bound width**, clipped to those bounds. ACCES supplies its original
   bounds; inferred bounds use the observed spans of complete evaluations.
   These observed extrema do not establish physical limits. It is not a
   percentage of the optimum parameter value or the CMA-ES stopping sigma.

2. Fit the scalar response with scikit-learn's GaussianProcessRegressor:
   standardised inputs and an anisotropic Matern-3/2 kernel. The default
   ``interpolate = True`` reproduces recorded evaluations, with a small
   numerical nugget. Model ``asinh(objective / scale)`` to compress large values
   while supporting both signs. The scale is the absolute reference value;
   at zero, use the sample standard deviation, or 1 for constant zero data.
   Predictions are inverse-transformed to objective units. By default all
   complete evaluations are used; ``n_neighbors`` optionally selects the
   nearest evaluations in bound-scaled coordinates.

   For strictly positive calibration discrepancies, ``response_transform =
   "log"`` guarantees positive predictions. ``interpolate = False`` instead
   fits white noise to estimate a smoothed response; use this when modelling
   the mean of noisy evaluations. Repeated parameter vectors with different
   objectives require smoothing. A sklearn ``kernel`` can be supplied to
   check sensitivity to covariance assumptions.

3. Compare the prediction with the **observed** reference objective:
   ``abs(prediction - reference_value) <= objective_tolerance *
   abs(reference_value)``. For an observed value of -100, a 10% allowance
   gives a prediction band of [-110, -90], regardless of the GP's reference
   prediction. Values beyond either side are excluded, including changes
   that would improve the original optimisation objective.

   ``absolute_tolerance`` overrides the percentage with a half-width in
   objective units. It is useful at zero, where the relative allowance is zero,
   and when the objective's zero point is arbitrary.

4. Vary each parameter alone to produce a **slice**. For a **profile**, adjust
   the remaining parameters inside the box to find a prediction within the
   band. Training vectors are projected onto the fixed coordinates and scored
   by the GP. Bounded optimisation starts from the most promising and stops
   at a feasible witness. Its response is not an objective minimum or maximum.
   Wider accepted profiles show compensation. A flat adjusted curve can reflect
   successful compensation, not insensitivity to that input alone. Store each
   compensating vector.

5. Repeat for every parameter pair, holding the rest fixed or adjusting them.
   The joint maps show which combinations satisfy the band. Pair interaction
   is the fixed-other contrast
   ``f(a, b) - f(a, b_ref) - f(a_ref, b) + f(a_ref, b_ref)``.
   It is zero for an additive response and has objective units.

6. Rank parameters by RMS objective change along their fixed-other slices.
   Rank pairs by the RMS interaction contrast. Larger scores mean more
   influence over the chosen window; allowed parameter changes retain their
   physical units. Rankings depend on the window and chosen objective.
   Trapezoidal integration gives uniform weight per parameter unit, accounting
   for the sampling. Actual acceptable coordinates and single-parameter
   range endpoints are added to pair grids to resolve narrow regions.

   Two rankings are computed using the same fitted GP and reference:

   - ``result.ranking`` covers the supplied ``parameter_window``.
   - ``result.ranking_full`` covers each parameter's complete min/max bounds.
     For ACCES these are the original run bounds; for other data they are
     the supplied bounds or the inferred observed extrema.

   Both contain ``rank``, ``parameter`` and the dimensionless
   ``relative_sensitivity`` score: each parameter's RMS change divided by
   the sum across parameters over that ranking's domain. Each set of scores
   sums to one when any response varies;
   otherwise all scores are zero and ranks are tied. Changing parameter or
   objective units does not change these relative scores for the same physical
   window. The scores describe relative fixed-other response magnitudes, not
   fractions of output variance or contributions including interactions.
   ``result.importance`` and ``result.importance_full`` retain the RMS changes
   in objective units, allowing comparisons of absolute response magnitudes.
   ``result.save`` writes all four tables as CSV files.

   The full-bound ranking varies one parameter at a time, with the others
   fixed at the reference. It uses extra GP predictions with no refitting or
   profile optimisation. It can include poorly sampled areas, particularly
   when the GP uses only ``n_neighbors`` nearby observations. It is not a
   sensitivity index averaged over joint variations of all parameters.
   Ranges and plots continue to use the supplied ``parameter_window``.

Reading the results
-------------------

``result.ranges`` contains physical endpoints, signed changes and boundary
flags for separate slice/profile intervals - these answer the question of where
the response remains within the tolerance band. ``threshold`` indicates a
tolerance crossing; ``parameter_window`` or ``original_bound``
means the analysis stopped at that boundary. Disconnected intervals remain
separate: multiple rows with the same parameter and mode are intentional.
Each row describes one accepted interval.

To select only the profile intervals containing the reference coordinate:

.. code-block:: python

   result.ranges.query("mode == 'profile' and contains_reference_value")

This selects a marginal interval, not a guarantee of a continuous accepted
path through the full parameter space. Slice crossings are refined with
bracketed root finding. Profile crossings use grid interpolation: different
feasible compensating vectors
can affect the endpoint within a grid step. Thin or disconnected regions can
require a finer grid to detect.

``result.single`` and ``result.pairs`` contain predictions, acceptance flags
and compensating vectors. A successful profile demonstrates a GP prediction
within the band. A failed local search does not prove that no such vector
exists. Marginal profile endpoints generally need different compensating
vectors and cannot be combined into an accepted parameter box. A profile
interval is a projection, not necessarily a connected path from the optimum.

Actual evaluations satisfying the same band are available separately in
``result.observed_acceptable``. The band is saved as ``objective_limits``.

Predictions cover the **whole reporting box**, including sparsely sampled
areas. ``metadata`` reports the number of complete observations, training rows,
observations inside the window, and unique training vectors inside the window.
``result.sampling`` reports the box and sampled parameter extents. These counts
describe data availability; they do not certify accuracy throughout the box.

Plotting
--------

.. code-block:: python

   fixed = result.plot(others_fixed = True)
   adjusted = result.plot(others_fixed = False)
   fixed.savefig("fixed.pdf")
   adjusted.savefig("adjusted.pdf")

   # Positive calibration objective, with readable logarithmic colour scales
   result.save("sensitivity_output", objective_scale = "log",
               objective_label = "Discrepancy (mm^4)")

Each figure includes a legend for response curves, tolerance regions and
highlighted observations. Black diamonds mark the observed reference; red
crosses mark its pair coordinates. Purple diamonds on adjusted plots mark
other accepted observations, with evaluation IDs beside the pair projections.
They indicate evaluated full vectors, not acceptance of arbitrary intermediate
combinations. Figures can be saved directly as PDF for reports.

Each figure is a parameter matrix:

- **Diagonal:** GP curves, the tolerance band and observed-reference diamonds.
- **Lower triangle:** absolute objective-departure contours; accepted GP
  regions are green. Adjusted plots also show actual accepted trials in purple.
- **Upper triangle:** objective-valued heatmaps with one common colour scale,
  shared between fixed and adjusted figures.

Each column's horizontal axis is that parameter. Pair plots use the row
parameter vertically. ``others_fixed`` chooses whether remaining parameters
are held at the reference or adjusted to satisfy the tolerance. Plotting reuses
the calculated tables. ``result.save(directory)`` writes the tables, settings,
and both matrices as ``sensitivity_fixed`` and ``sensitivity_adjusted`` in PNG
and PDF formats.
``labels`` maps parameter names to display labels. ``objective_scale`` chooses
linear, log or symlog display without changing any predictions or thresholds.
Log requires positive values; symlog also supports signed and zero objectives.

Purple points are projections of full evaluated vectors with all parameters
potentially changing. They do not validate arbitrary combinations of their
coordinates, nor intermediate points. Green regions are surrogate estimates.

DataFrame parameter names appear on the plots and in all result tables.
NumPy inputs use ``Parameter 1``, ``Parameter 2``, etc.

API
---

.. autofunction:: coexist.sensitivity.analyse

.. automethod:: coexist.AccessData.sensitivity
   :no-index:

.. autoclass:: coexist.sensitivity.SensitivityResult
   :members: save, plot

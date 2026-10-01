ACCES Structures (``coexist.access``)
=====================================

``coexist.AccessData("access_seed42")`` reads a saved run containing
``access_setup.toml`` and the corresponding history and epoch tables. A run can
be moved or renamed: stored file paths are resolved relative to its current
directory. Passing a parent directory is supported when it contains exactly
one ACCES run.

The ``results`` table stores parameter values, individual errors (when used)
and the combined ``error`` minimised by ACCES. The ``epochs`` table describes
the optimiser's sampling distribution; its spread is not an experimental
parameter confidence interval.

Reading interrupted runs
------------------------

Tables retain their two-dimensional shape even when only one epoch or one
evaluation has been saved. Both ``AccessData`` and optimisation restarts use
the same reader.

If an unscaled history or epoch file is missing or empty, the reader recreates
it from the corresponding scaled file and prints a repair message. Parameter
coordinates, means and standard deviations are multiplied by their saved
scaling factors; error columns and the overall standard deviation are unchanged.
The scaled files are never rewritten during repair. Their original floating-point
values are required to replay CMA-ES; rescaling the unscaled values can introduce
rounding differences.

Each completed epoch is saved in this order: unscaled epochs, scaled epochs,
unscaled history, scaled history. A storage failure can therefore leave an epoch
record ahead of the saved history. The reader excludes that final record in
memory, using only complete populations in scaled history. The next resumed
epoch is then recorded once. Empty, malformed or incomplete scaled data require
an intact saved copy; the reader does not guess missing optimiser values.

Table and setup writes use temporary files in the destination directory. A file
is replaced only after writing and flushing succeed, preserving its previous
contents if storage runs out. The four tables are replaced individually, so
the reader still checks for an interrupted checkpoint. Before repairing an old
run, stop its writer and ensure that space is available for the repaired files.


.. autosummary::
   :toctree: generated/

   coexist.access.AccessSetup
   coexist.access.AccessPaths
   coexist.access.AccessProgress

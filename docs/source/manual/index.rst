******
Manual
******

All public ``coexist`` subroutines are fully documented here, along with copy-pastable
examples. The `base` functionality is summarised below; the rest of the library is
organised into submodules, which you can access on the left. You can also use the
`Search` bar in the top left to go directly to what you need.

We really appreciate all help with writing useful documentation; if you feel
something can be improved, or would like to share some example code, by all means
get in contact with us - or be a superhero and click `Edit this page` on the right
and submit your changes to the GitHub repository directly!


ACCES
=====

Exported functionality related to the ACCES macro calibration suite.

.. autosummary::
   :toctree: generated/

   coexist.Access
   coexist.AccessData
   coexist.create_parameters




LIGGGHTS
========

Helpers for driving LIGGGHTS simulations with less code that is more memory efficient and
error-proof. **You need the** ``liggghts`` **Python interface to LIGGGHTS to be installed to
use** ``LiggghtsSimulation``. The `PICI-LIGGGHTS <https://github.com/uob-positron-imaging-centre/PICI-LIGGGHTS>`_ repository has instructions for this.

.. autosummary::
   :toctree: generated/

   coexist.LiggghtsSimulation
   coexist.Simulation
   coexist.to_vtk




Submodules
==========

.. toctree::
   :caption: Submodules

   access
   combiners
   plots
   schedulers
   utilities



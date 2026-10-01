***************
Getting Started
***************

ACCES is distributed as the ``coexist`` Python package. It requires Python 3.10
or newer. Your simulation engine is configured separately; ACCES runs the
simulation through your Python script.

Installation
============

Install the core optimisation, data-reading and plotting tools:

.. code-block:: console

   python -m pip install coexist

For Gaussian-process sensitivity analysis, include the optional dependency:

.. code-block:: console

   python -m pip install "coexist[sensitivity]"

The core package does not require scikit-learn. See :doc:`tutorials/index` for
an optimisation example and :doc:`manual/sensitivity` for sensitivity analysis.

Working from source
===================

.. code-block:: console

   git clone https://github.com/uob-positron-imaging-centre/ACCES-CoExSiST
   cd ACCES-CoExSiST
   python -m pip install -e ".[test,docs,sensitivity]"

An editable install uses your source changes without reinstalling. Dependencies
are declared in ``pyproject.toml``:

- The core dependencies support optimisation, saved runs and plotting.
- ``sensitivity`` adds scikit-learn for Gaussian-process analysis.
- ``test`` adds pytest. Sensitivity tests are skipped when scikit-learn is absent.
- ``docs`` adds Sphinx, numpydoc and the documentation theme. Building the complete
  API reference also requires the ``sensitivity`` extra.

Run the tests and build the documentation from the repository root:

.. code-block:: console

   python -m pytest
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

To build a source distribution and wheel:

.. code-block:: console

   python -m pip install build
   python -m build

The repository's ``legacy/`` directory preserves earlier microscopic simulation
interfaces for research. It is excluded from package distributions; its README
describes the archived code and dependencies.

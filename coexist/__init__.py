#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


# Import base data structures and algorithms
from        .base           import  save, load
from        .base           import  create_parameters
from        .base           import  Parameters, Experiment, Simulation
from        .base           import  to_vtk

try:
    from    .liggghts       import  LiggghtsSimulation, AutoTimestep
except ImportError:
    class LiggghtsNotFound:
        pass
    LiggghtsSimulation = LiggghtsNotFound
    AutoTimestep = LiggghtsNotFound

from        .optimisation   import  Coexist
from        .access         import  Access, AccessData

# Import submodules
from        .               import  schedulers
from        .               import  ballistics
from        .               import  plots

# Import package version
from        .__version__    import  __version__


__author__ = "Andrei Leonard Nicusan, "
__email__ = "a.l.nicusan@bham.ac.uk"
__license__ = "GNU v3.0"
__status__ = "Beta"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


'''ACCES simulation calibration, optimisation and analysis.'''


from    importlib           import  import_module

from    .base               import  create_parameters
from    .access             import  Access, AccessData
from    .                   import  combiners, schedulers, plots
from    .__version__        import  __version__


__author__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__license__ = "GNU v3.0"
__status__ = "Beta"




def __getattr__(name):
    '''Load optional sensitivity analysis when it is first requested.'''
    if name == "sensitivity":
        return import_module(".sensitivity", __name__)
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")

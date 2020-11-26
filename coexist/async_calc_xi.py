#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : calc_xi_async.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.11.2020


import sys
import pickle

import coexist


'''Run a simulation against a given experimental dataset and print the
accumulated error.

This script is normally called by `coexist.Coexist.optimise`. It *must* be
called with 5 arguments:

    1. A pickled `Parameters` object path.
    2. A saved simulation's path (i.e. the prefix before "restart.sim").
    3. A pickled `Experiment` object path.
    4. A start_index for the experimental positions.
    5. An end_index for the experimental positions.

The simulation parameters must already be set correctly by the `Parameters`.
This script does *not* check input parameters.
'''

params = pickle.load(open(sys.argv[1], "rb"))
sim = coexist.Simulation(sys.argv[2], params, verbose = False)
exp = pickle.load(open(sys.argv[3], "rb"))

start_index = int(sys.argv[4])
end_index = int(sys.argv[5])

xi_acc = coexist.Coexist.calc_xi_acc(sim, exp, start_index, end_index)
print(xi_acc, end = "")

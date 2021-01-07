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

This script is normally called by `coexist.Coexist.optimise` asynchronously.
It *must* be called with 5 command-line arguments:

    1. A pickled `Simulation` subclass (the class itself, not an object).
    2. A saved simulation's path that can be loaded.
    3. A pickled `Experiment` object path.
    4. A start_index for the experimental positions.
    5. An end_index for the experimental positions.

The simulation parameters must already be set correctly by the `Parameters`.
This script does *not* check input parameters.
'''

# Parse the five command-line inputs
with open(sys.argv[1], "rb") as f:
    sim_class = pickle.load(f)

sim = sim_class.load(sys.argv[2])

with open(sys.argv[3], "rb") as f:
    exp = pickle.load(f)

start_index = int(sys.argv[4])
end_index = int(sys.argv[5])

# Calculate the accumulated error for the timesteps between `Experiment`
# indices `start_index` and `end_index`. The `Coexist.calc_xi_acc` static
# method moves the simulation forward in time.
xi_acc = coexist.Coexist.calc_xi_acc(sim, exp, start_index, end_index)
print(xi_acc, end = "", flush = True)

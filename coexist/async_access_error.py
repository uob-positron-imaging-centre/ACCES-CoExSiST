#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : calc_xi_async.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.11.2020


import sys
import pickle

import numpy as np

import coexist


'''Run a simulation between `start_time` and `end_time`, saving the particle
positions at `num_checkpoints`, then print the 3D numpy array to stdout.

This script is normally called by `coexist.Access.optimise`. It *must* be
called with 5 arguments:

    1. A pickled `Parameters` object path.
    2. A saved simulation's path (i.e. the prefix before "restart.sim").
    3. A `start_time` for when to start recording particle positions.
    4. An `end_time` for when to stop the simulation.
    5. The `num_checkpoints` to save between start_time and end_time.

The simulation parameters must already be set correctly by the `Parameters`.
This script does *not* check input parameters.
'''

params = pickle.load(open(sys.argv[1], "rb"))
sim = coexist.Simulation(sys.argv[2], params, verbose = False)

start_time = float(sys.argv[3])
end_time = float(sys.argv[4])
num_checkpoints = int(sys.argv[5])

checkpoints = np.linspace(start_time, end_time, num_checkpoints)

positions = []
for t in checkpoints:
    sim.step_to_time(t)
    positions.append(sim.positions())

positions = np.array(positions, dtype = float)

print(positions.tobytes(), end = "")

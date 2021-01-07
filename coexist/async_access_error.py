#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : calc_xi_async.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.11.2020


import sys
import pickle

import numpy as np


'''Run a simulation between `start_time` and `end_time`, saving the particle
positions at `num_checkpoints`, then print the 3D numpy array to stdout.

This script is normally called by `coexist.Access.optimise`. It *must* be
called with 6 command-line arguments:

    1. A pickled `Simulation` subclass (the class itself, not an object).
    2. A saved simulation's path that can be loaded.
    3. A `start_time` for when to start recording particle positions.
    4. An `end_time` for when to stop the simulation.
    5. The `num_checkpoints` to save between start_time and end_time.
    6. A path to a `.npy` file to save positions to.

The simulation parameters must already be set correctly by the `Parameters`.
This script does *not* check input parameters.
'''

# Parse the six command-line inputs
with open(sys.argv[1], "rb") as f:
    sim_class = pickle.load(f)

sim = sim_class.load(sys.argv[2])

start_time = float(sys.argv[3])
end_time = float(sys.argv[4])
num_checkpoints = int(sys.argv[5])

positions_path = sys.argv[6]

# Run the simulation betwee start_time and end_time, saving the particle
# positions in `positions`
checkpoints = np.linspace(start_time, end_time, num_checkpoints)

positions = []
for t in checkpoints:
    sim.step_to_time(t)
    positions.append(sim.positions())

# Save the positions as a fast numpy / pickle binary file
positions = np.array(positions, dtype = float)
np.save(positions_path, positions)

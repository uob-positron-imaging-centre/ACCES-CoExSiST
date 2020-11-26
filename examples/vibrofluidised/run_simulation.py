#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run_simulation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import numpy as np
from tqdm import tqdm

import coexist


parameters = coexist.Parameters(
    ["corPP", "corPW"],
    ["fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPW} ${corPW2} \
        ${corPW} ${corPW2} ${corPW} \
        ${corPW2} ${corPW} ${corPW} ",
     "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPW} ${corPW2} \
        ${corPW} ${corPW2} ${corPW} \
        ${corPW2} ${corPW} ${corPW} "],
    [0.5, 0.5],     # Initial values
    [0.0, 0.0],     # Minimum values
    [1.0, 1.0]      # Maximum values
)

simulation = coexist.Simulation("run.sim", parameters, verbose = False)
print(simulation)
print("step size: ", simulation.step_size)


# 60 FPS
simulation_times = np.linspace(0.0, 1.0, 600)

positions = [simulation.positions()]
times = [0.0]

for t in tqdm(simulation_times):
    # Skip first timestep
    if t == 0.0:
        continue

    simulation.step_to_time(t)

    positions.append(simulation.positions())
    times.append(t)

positions = np.array(positions)
times = np.array(times)

np.save("truth/positions_short", positions)
np.save("truth/timesteps_short", times)

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


# 60 FPS
checkpoints = np.linspace(0.0, 1.0, 600)

positions = []
times = []
for time in tqdm(checkpoints):
    simulation.step_to_time(time)

    positions.append( simulation.positions() )
    times.append( simulation.time() )

positions = np.array(positions)
times = np.array(times)

np.save("truth/positions_short", positions)
np.save("truth/timesteps_short", times)



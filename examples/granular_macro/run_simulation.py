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
    ["fricPP", "corPP"],
    ["fix  m4 all property/global coefficientFriction peratomtypepair 3 \
        ${fricPP}   ${fricPW}   ${fricPSW}  \
        ${fricPW}   ${fric}     ${fric}     \
        ${fricPSW}  ${fric}     ${fric}     ",
     "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPP} ${corPP} \
        ${corPP} ${corPP} ${corPP} \
        ${corPP} ${corPP} ${corPP} "],
    [1.860605011242399132e-01, 6.838861196402620246e-01],     # Initial values
    [0.0, 0.0],     # Minimum values
    [1.0, 1.0]      # Maximum values
)

simulation = coexist.Simulation("run.sim", parameters, verbose = True)
print(simulation)


rpm = 45
num_rotations = 4

time_end = num_rotations * 60 / rpm

# Recording positions at 60 FPS
checkpoints = np.arange(1.0, time_end, 1 / 60)

positions = []
times = []
for time in tqdm(checkpoints):
    simulation.step_to_time(time)

    positions.append(simulation.positions())
    times.append(simulation.time())

positions = np.array(positions)
times = np.array(times)

np.save("truth/positions", positions)
np.save("truth/timesteps", times)

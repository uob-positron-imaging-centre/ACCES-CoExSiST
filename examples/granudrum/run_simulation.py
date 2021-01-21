#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run_simulation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import numpy as np
from tqdm import tqdm

import coexist


# Define the user-changeable / free simulation parameters
parameters = coexist.Parameters(
    variables = ["fricPP", "corPP"],
    commands = [
        "fix  m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}   ${fricPW}   ${fricPSW}  \
            ${fricPW}   ${fric}     ${fric}     \
            ${fricPSW}  ${fric}     ${fric}     ",
        "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP} ${corPP} ${corPP} \
            ${corPP} ${corPP} ${corPP} \
            ${corPP} ${corPP} ${corPP} ",
    ],
    values = [0.2, 0.7],
    minimums = [0.05, 0.05],
    maximums = [0.95, 0.95],
)

simulation = coexist.LiggghtsSimulation("granudrum.sim", parameters)
print(simulation)

# Drum speed, used to calculate the time needed for a given number of rotations
rpm = 45
num_rotations = 4

# Save particle locations at a 120 Hz sampling rate after the system reached
# steady state, from t = 1.0 s
start_time = 1.0
end_time = num_rotations * 60 / rpm
sampling_rate = 1 / 120

sampling_times = np.arange(start_time, end_time, sampling_rate)

# Save particle locations and simulation times in lists
positions = []
times = []

for time in tqdm(sampling_times):
    simulation.step_to_time(time)

    positions.append(simulation.positions())
    times.append(simulation.time())

# Save the particle locations as numpy arrays in binary format
positions = np.array(positions)
times = np.array(times)

np.save("truth/positions", positions)
np.save("truth/timesteps", times)

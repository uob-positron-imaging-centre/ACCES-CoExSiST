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
    variables = ["corPP", "corPW"],
    commands = [
        "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP} ${corPW} ${corPW2} \
            ${corPW} ${corPW2} ${corPW} \
            ${corPW2} ${corPW} ${corPW} ",
        "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP} ${corPW} ${corPW2} \
            ${corPW} ${corPW2} ${corPW} \
            ${corPW2} ${corPW} ${corPW} ",
    ],
    values = [0.5, 0.5],
    minimums = [0.05, 0.05],
    maximums = [0.95, 0.95],
)

simulation = coexist.LiggghtsSimulation("vibrofluidised.sim", parameters)
print(simulation)

# Save particle locations at a 120 Hz sampling rate up to t = 1.0 s
start_time = simulation.time()
end_time = 1.0
sampling_rate = 1 / 120

simulation_times = np.arange(start_time, end_time, sampling_rate)

# Save initial particle locations
positions = [simulation.positions()]
times = [simulation.time()]

for i, t in enumerate(tqdm(simulation_times)):
    # Skip first timestep
    if i == 0:
        continue

    simulation.step_to_time(t)

    positions.append(simulation.positions())
    times.append(t)

# Save the particle locations as numpy arrays in binary format
positions = np.array(positions)
times = np.array(times)


# Test auto timesteps
timestep = coexist.AutoTimestep(
    simulation.variable("youngmodP"),
    simulation.variable("r1") * 2,
    simulation.variable("poissP"),
    simulation.variable("densPart"),
)

simulation = coexist.LiggghtsSimulation(
    "vibrofluidised.sim",
    parameters,
    timestep = timestep,
    verbose = True,
)

print(simulation)

# Save particle locations at a 120 Hz sampling rate up to t = 1.0 s
start_time = simulation.time()
end_time = 1.0
sampling_rate = 1 / 120

simulation_times = np.arange(start_time, end_time, sampling_rate)

# Save initial particle locations
positions = [simulation.positions()]
times = [simulation.time()]

for i, t in enumerate(tqdm(simulation_times)):
    # Skip first timestep
    if i == 0:
        continue

    simulation.step_to_time(t)

    positions.append(simulation.positions())
    times.append(t)

# Save the particle locations as numpy arrays in binary format
positions = np.array(positions)
times = np.array(times)

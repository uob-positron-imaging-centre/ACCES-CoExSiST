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
    variables = [
        "dens",
        "youngmodP",
        "poissP",
        "cohPP",
        "corPP",
        "corPW",
        "fricPP",
        "fricPW",
    ],
    commands = [
        "set type 1 density ${dens}",
        "fix m1 all property/global youngsModulus peratomtype \
            ${youngmodP}    ${youngmodP}    ${youngmodP}",
        "fix m2 all property/global poissonsRatio peratomtype \
            ${poissP}       ${poissP}       ${poissP}    ",
        "fix m6 all property/global cohesionEnergyDensity peratomtypepair 3 \
            ${cohPP}        ${cohPW}        ${cohPSW}   \
            ${cohPW}        ${coh}          ${coh}      \
            ${cohPSW}       ${coh}          ${coh}      ",
        "fix m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}   \
            ${corPW}        ${cor}          ${cor}      \
            ${corPSW}       ${cor}          ${cor}      ",
        "fix m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}   \
            ${corPW}        ${cor}          ${cor}      \
            ${corPSW}       ${cor}          ${cor}      ",
        "fix m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}  \
            ${fricPW}       ${fric}         ${fric}     \
            ${fricPSW}      ${fric}         ${fric}     ",
        "fix m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}  \
            ${fricPW}       ${fric}         ${fric}     \
            ${fricPSW}      ${fric}         ${fric}     ",
    ],
    values =    [1580.0,   9.2e6, 0.30, 0,   0.61, 0.61, 0.42, 0.42],
    minimums =  [400.0,    5e6,   0.05, 0,   0.05, 0.05, 0.05, 0.05],
    maximums =  [10_000.0, 20e6,  0.49, 1e5, 0.95, 0.95, 0.95, 0.95],
)

print("Loading simulation...")
simulation = coexist.LiggghtsSimulation("granudrum.sim", parameters)
print(simulation, flush = True)

# Drum speed, used to calculate the time needed for a given number of rotations
rpm = 30
num_rotations = 3

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import numpy as np
import cma

import coexist


exp_pos = np.load("truth/positions_short.npy")
exp_tsteps = np.load("truth/timesteps_short.npy")
radius = 0.005 / 2


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
    [0.4, 0.4],     # Initial WRONG values
    [0.0, 0.0],     # Minimum values
    [1.0, 1.0]      # Maximum values
)

simulation = coexist.Simulation("run.sim", parameters, verbose = False)
print(simulation)


def compute_xi(exp_positions, sim_positions):
    xi = np.linalg.norm(exp_positions - sim_positions, axis = 1)

    # Threshold - the center of a simulated particle should always be inside
    # the equivalent experimental particle
    xi[xi < radius] = 0

    xi = xi.sum()

    return xi


for i, t in enumerate(exp_tsteps):
    simulation.step_to_time(t)
    pos = simulation.positions()

    xi = compute_xi(exp_pos[i], pos)
    print(f"---\nxi: {xi}\ni: {i}")









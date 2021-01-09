#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise_oop.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 09.11.2020


import numpy as np

import coexist


exp_timesteps = np.load("truth/timesteps_short.npy")
exp_positions = np.load("truth/positions_short.npy")

experiment = coexist.Experiment(exp_timesteps, exp_positions, 0.0002)

parameters = coexist.Parameters(
    variables = ["corPP", "corPW"],
    commands = ["fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPW} ${corPW2} \
        ${corPW} ${corPW2} ${corPW} \
        ${corPW2} ${corPW} ${corPW} ",
     "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPW} ${corPW2} \
        ${corPW} ${corPW2} ${corPW} \
        ${corPW2} ${corPW} ${corPW} "],
    values = [0.5, 0.5],     # Initial values
    minimums = [0.0, 0.0],     # Minimum values
    maximums = [1.0, 1.0]      # Maximum values
)

simulation = coexist.LiggghtsSimulation("run.sim", parameters, verbose = False)
print(simulation, "\n")

opt = coexist.Coexist(simulation, save_log = True)
opt.learn(experiment, num_solutions = 24)

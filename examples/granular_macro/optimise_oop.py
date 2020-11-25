#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import os

import numpy as np
from scipy.integrate import quad

import cma

from joblib import Parallel, delayed

import coexist


exp_positions = np.load("truth/positions_short.npy")
exp_timesteps = np.load("truth/timesteps_short.npy")
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
    [0.45, 0.55],   # Initial WRONG values
    [0.05, 0.05],   # Minimum values
    [0.95, 0.95],   # Maximum values
    #[0.1, 0.1],     # Sigma0
)

simulation = coexist.Simulation("run.sim", parameters, verbose = False)
print(simulation)


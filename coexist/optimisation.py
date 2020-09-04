#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimisation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import cma
import numpy as np


# Optimise arbitrary simulation parameters using CMA-ES
parameters = Parameters(
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

simulation = Simulation("in.sim", parameters)

print("\nInitial simulation parameters:")
print(f"corPP: {simulation.variable('corPP')}")
print(f"corPW: {simulation.variable('corPW')}")
print(simulation)






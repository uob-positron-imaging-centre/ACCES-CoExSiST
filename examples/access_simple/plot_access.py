#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot_access.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.06.2021


import coexist


# Define parameters from the `simulation_script.py` for richer plotting
parameters = coexist.create_parameters(
    variables = ["cor", "separation"],
    minimums = [-10, -15],
    maximums = [+15, +20],
    values = [5, 10],
)

# Path to either the `access_info_<hash>` folder itself, or the parent
access_path = "."

fig = coexist.plot_access(access_path, parameters)
fig.show()

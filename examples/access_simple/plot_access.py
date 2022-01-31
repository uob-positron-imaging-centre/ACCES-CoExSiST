#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot_access.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.06.2021

import coexist

# Use path to either the `access_<random_seed>` folder itself, or its
# parent directory
access_data = coexist.AccessData.read(".")

fig = coexist.plots.access(access_data)
fig.show()

# Or more tersely:
# coexist.plots.access(".").show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot_access2d.py
# License: GNU v3.0
# Author : Jack Sykes <jas653@student.bham.ac.uk>
# Date   : 09.03.2022

import coexist

access_data = coexist.AccessData.read(".")

fig = coexist.plots.access2d(access_data)
fig.show()

# Written in one line
# coexist.plots.access2d(".").show()

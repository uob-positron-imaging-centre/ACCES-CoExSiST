#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn.py
# License: GNU v3.0
# Author : Jack Sykes <jas653@student.bham.ac.uk>
# Date   : 09.03.2022

import coexist

data = coexist.Access("simulation_script.py").learn(random_seed = 123)

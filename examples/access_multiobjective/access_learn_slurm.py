#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn_slurm.py
# License: GNU v3.0
# Author : Jack Sykes <jas653@student.bham.ac.uk>
# Date   : 10.03.2022

import coexist
import textwrap
from coexist.schedulers import SlurmScheduler


scheduler = SlurmScheduler(
    "10:0:0",           # Time allocated for a single simulation
    commands = textwrap.dedent('''
        set -e
        module purge; module load bluebear
        module load SciPy-bundle/2021.05-foss-2021a
    '''),
    qos = "bbdefault",
    account = "windowcr-pept-as-a-service",
    constraint = "icelake",     # Any other #SBATCH -- <CMD> = "VAL" pair
)

# Use ACCESS to learn the simulation parameters
access = coexist.Access("simulation_script.py", scheduler)
access.learn(num_solutions = 100, target_sigma = 0.1, random_seed = 123)

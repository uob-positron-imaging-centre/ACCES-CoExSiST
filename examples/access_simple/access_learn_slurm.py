#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn_slurm.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.05.2021


import coexist
from coexist.schedulers import SlurmScheduler


scheduler = SlurmScheduler(
    "10:0:0",           # Time allocated for a single simulation
    commands = [        # Commands you'd add in the sbatch script, after `#`
        "set -e",
        "module purge; module load bluebear",
        "module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2",
    ],
    qos = "bbdefault",
    account = "windowcr-rt-royalsociety",
    constraint = "cascadelake",     # Any other #SBATCH -- <CMD> = "VAL" pair
)

# Use ACCESS to learn the simulation parameters
access = coexist.Access("simulation_script.py", scheduler)
access.learn(num_solutions = 100, target_sigma = 0.1, random_seed = 42)

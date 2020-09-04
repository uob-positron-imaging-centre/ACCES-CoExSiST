#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name = "coexist",
    version = "0.1.0",
    author = (
        "Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>, "
        "Dominik Werner <>"
    ),
    description = "Coupling experimental granular data with DEM simulations",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/D-werner-bham/pyLiggghts",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires = '>=3.6',
)



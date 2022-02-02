#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.01.2022


'''Integration tests to ensure that the base classes' behaviour is consistently
correct.
'''


import numpy as np

import coexist


def test_to_vtk():
    path = "vtk_export"

    # Generate timesteps with the same number of particles
    t = 5
    p = 10

    pos = np.random.random((t, p, 3))
    coexist.to_vtk(path, pos)

    times = np.linspace(0, 1, t)
    coexist.to_vtk(path, pos, times = times)

    radii = np.random.random((t, p))
    coexist.to_vtk(path, pos, times = times, radii = radii)

    velocities = np.random.random((t, p, 3))
    coexist.to_vtk(path, pos, times = times, radii = radii,
                   velocities = velocities)


def test_create_parameters():
    prm = coexist.create_parameters(["a", "b"], [0, 0], [1, 1])
    print(prm)

    coexist.create_parameters(["a", "b"], [0, 0], [1, 1], [0.5, 0.5])
    coexist.create_parameters(["a", "b"], [0, 0], [1, 1], [0.5, 0.5], [1, 1])


if __name__ == "__main__":
    test_to_vtk()
    test_create_parameters()

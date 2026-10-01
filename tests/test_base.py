#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>


'''Tests for constructing ACCES parameter tables.'''


import  numpy  as  np
import  pytest

import  coexist


def test_create_parameters():
    parameters = coexist.create_parameters(["a", "b"], [-1, 0], [1, 10])
    assert list(parameters.index) == ["a", "b"]
    np.testing.assert_allclose(parameters.value, [0, 5])
    np.testing.assert_allclose(parameters.sigma, [0.8, 4])

    parameters = coexist.create_parameters(
        ["a", "b"], [-1, 0], [1, 10],
        values = [0.5, 2], sigma = [0.1, 1], units = ["m", "s"],
    )
    np.testing.assert_allclose(parameters.value, [0.5, 2])
    np.testing.assert_allclose(parameters.sigma, [0.1, 1])
    assert parameters.units.to_list() == ["m", "s"]


def test_parameter_bounds():
    with pytest.raises(ValueError, match = "same length"):
        coexist.create_parameters(["a", "b"], [0], [1, 1])
    with pytest.raises(ValueError, match = "maximums"):
        coexist.create_parameters(["a"], [1], [1])

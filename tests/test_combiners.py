#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_combiners.py
# License: GNU v3.0
# Author : Jack Sykes <jas653@student.bham.ac.uk>
# Date   : 18.03.2022


'''Integration tests to ensure that the combiners classes' behaviour is
consistently correct.
'''


import numpy as np
import coexist


def test_product():
    # Test product with no weighting exponents
    comb1 = coexist.combiners.Product()
    error1 = [1, 2, 3]
    assert comb1.combine(error1) == np.prod(error1)

    error2 = 2.89
    assert comb1.combine([error2]) == error2

    # Test product with weighting exponents
    weights3 = [1, 2]
    comb3 = coexist.combiners.Product(weights = weights3)
    error3 = [2, 3]
    assert comb3.combine(error3) == np.prod(np.array(error3) ** weights3)

    weights4 = [2]
    comb4 = coexist.combiners.Product(weights = weights4)
    error4 = [2]
    assert comb4.combine(error4) == np.prod(np.array([error4]) ** weights4)

    weights5 = 2
    comb5 = coexist.combiners.Product(weights = weights5)
    error5 = [2]
    assert comb5.combine(error5) == np.prod(np.array([error5]) ** weights5)


def test_sum():
    # Test product with no weighting exponents
    comb1 = coexist.combiners.Sum()
    error1 = [1, 2, 3]
    assert comb1.combine(error1) == np.sum(error1)

    error2 = [2.89]
    assert comb1.combine([error2]) == error2

    # Test product with weighting exponents
    weights3 = [1, 2]
    comb3 = coexist.combiners.Sum(weights = weights3)
    error3 = [2, 3]
    assert comb3.combine(error3) == np.sum(np.array(error3) * weights3)

    weights4 = [2]
    comb4 = coexist.combiners.Sum(weights = weights4)
    error4 = [2]
    assert comb4.combine(error4) == np.sum(np.array([error4]) * weights4)

    weights5 = 2
    comb5 = coexist.combiners.Sum(weights = weights5)
    error5 = [2]
    assert comb5.combine(error5) == np.sum(np.array([error5]) * weights5)


def test_combiner_decorator():
    # Test user-friendly function decorator
    @coexist.combiners.combiner
    def sum_errors(errors):
        return np.sum(errors)

    errors = np.array([1, 2])
    assert sum_errors.combine(errors) == sum_errors(errors)


if __name__ == "__main__":
    test_product()
    test_sum()
    test_combiner_decorator()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : combiners.py
# License: GNU v3.0
# Author : Jack Sykes <jas653@student.bham.ac.uk>
# Date   : 17.03.2022


'''Classes for combining multiple errors into a single value to be minimised.

The evolutionary algorithm will then naturally optimise all errors at the same
time. For example, the ``Product`` class will multiply all given errors
(optionally "weighting" them by raising them to different powers) and minimise
their product.

To combine multiple errors differently, create a new class defining the method
``.combine(error)``, which will be called with the error values found for a
given simulation.

For example, a GranuDrum Digital Twin may be calibrated for two different
flowing regimes at the same time; instead of doing separate ACCES runs, one
for, say, 15 rpm, and one for 45 rpm, you can instead use the multi-objective
functionality of ACCES to optimise both cases simultaneously.
'''


import  textwrap
import  numpy as  np




def combiner(func):
    '''Make a user-defined function a multi-objective error combiner.

    Examples
    --------
    To simply multiply all error values together:

    >>> import numpy as np
    >>> from coexist.combiners import combiner
    >>>
    >>> @combiner
    >>> def multiply(errors: np.ndarray) -> float:
    >>>     return np.prod(errors)

    The input argument "errors" will always be a 1D NumPy array, even when the
    simulation error is a simple, single number.

    If you know you will have only two error values, you can sum them up like
    this:

    >>> from coexist.combiners import combiner
    >>>
    >>> @combiner
    >>> def sum_errors(errors):
    >>>     return errors[0] + errors[1]

    Then you can simply supply your function to ACCES:

    >>> import coexist
    >>> coexist.Access("<simulation_script>").learn(
    >>>     multi_objective = sum_errors,
    >>>     random_seed = 42,
    >>> )
    '''

    # Attach a method called "combine" that simply calls the function itself
    func.combine = func.__call__
    return func




class Product:
    '''Class for combining multiple errors by multiplying them.

    The multi-objective functionality allows multiple error values to be
    calibrated / optimised at the same time.

    Attributes
    ----------
    weights: float, optional
        Raise each error to a power corresponding to its respective weight.
        If unset, all errors are multiplied with no exponentiation.

    Examples
    --------
    Use the ``Product`` combiner in an ACCES run:

    >>> import coexist
    >>> access = coexist.Access("<simulation_script>")
    >>> access.learn(
    >>>     multi_objective = coexist.combiners.Product(),
    >>>     random_seed = 42,
    >>> )

    To test what a combiner will output, use its ``.combine(errors)`` method -
    this is what will be called by ACCES:

    >>> import coexist
    >>> comb = coexist.combiners.Product(weights = [2, 3])
    >>> errors = [2, 2]
    >>> comb.combine(errors)
    32.0
    '''

    def __init__(self, weights = None):
        # If `weights` values are given, put them into an array
        if weights is not None:
            if not hasattr(weights, "__iter__"):
                weights = [weights]

            weights = np.array(weights, dtype = float)

        # Initialised parameter
        self.weights = weights


    def combine(self, errors):
        '''Combine the given errors to a total error value.
        '''
        # No `weights` values, simply mutliply the errors
        if self.weights is None:
            return np.prod(errors)

        # Error-check to see if number of weights matches number of errors
        if len(errors) != len(self.weights):
            raise ValueError(textwrap.fill((
                "Number of weights given is not equal to the number of errors "
                f"set in the simulation script. Received {len(self.weights)} "
                f"weights, but found {len(errors)} errors."
            )))

        # Return the product of the errors with `weights` values as exponents
        return np.prod(errors ** self.weights)


class Sum:
    '''Class for combining errors by summing them together.

    The multi-objective functionality allows multiple error values to be
    calibrated / optimised at the same time.

    Attributes
    ----------
    weights: float, optional
        Multiply each error with its respective weight. If unset, all errors
        are multiplied with no weighting.

    Examples
    --------
    Use the ``Sum`` combiner in an ACCES run:

    >>> import coexist
    >>> access = coexist.Access("<simulation_script>")
    >>> access.learn(
    >>>     multi_objective = coexist.combiners.Sum(),
    >>>     random_seed = 42,
    >>> )

    To test what a combiner will output, use its ``.combine(errors)`` method -
    this is what will be called by ACCES:

    >>> import coexist
    >>> comb = coexist.combiners.Sum(weights = [2, 3])
    >>> errors = [2, 2]
    >>> comb.combine(errors)
    10.0
    '''

    def __init__(self, weights = None):
        # If `weights` values are given, put them into an array
        if weights is not None:
            if not hasattr(weights, "__iter__"):
                weights = [weights]

            weights = np.array(weights, dtype = float)

        # Initialised parameter
        self.weights = weights


    def combine(self, errors):
        '''Combine the given errors to a total error value.
        '''
        # No `weights` values, simply sum the errors
        if self.weights is None:
            return np.sum(errors)

        # Error-check to see if number of weights matches number of errors
        if len(errors) != len(self.weights):
            raise ValueError(textwrap.fill((
                "Number of weights given is not equal to the number of errors "
                f"set in the simulation script. Received {len(self.weights)} "
                f"weights, but found {len(errors)} errors."
            )))

        # Return the sum of the errors multiplied by their `weights` values
        return np.sum(errors * self.weights)

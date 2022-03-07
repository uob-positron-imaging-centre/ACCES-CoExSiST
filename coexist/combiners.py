#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : combiners.py
# License: GNU v3.0
# Author : Jackyboo Cuthbert pSyko <jas653@student.bham.ac.uk>
# Date   : 07.03.2022


import  textwrap
import  numpy       as  np




class Product:

    def __init__(self, weights = None):
        if weights is not None:
            weights = np.array(weights, dtype = float)

        self.weights = weights


    def combine(self, errors):
        if self.weights is None:
            return np.prod(errors)

        if len(errors) != len(self.weights):
            raise ValueError(textwrap.fill((
                "TODO: add error message"
            )))

        return np.prod(errors ** self.weights)




class Sum:

    def __init__(self, weights):
        pass


    def combine(self, errors):
        pass

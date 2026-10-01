#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_code_trees.py
# License: GNU v3.0


'''Code inspection must distinguish variables from attribute assignments.'''


from    coexist.code_trees  import  code_contains_variables


def test_assignments_to_attributes_and_items():
    code = "obj.value = 1\nparameters['value'] = 2\nx = 3\ny = 4"
    assert code_contains_variables(code, ["x", "y"])
    assert code_contains_variables(code, ["x", "y"], root = True)
    assert not code_contains_variables(code, ["value"])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : code_trees.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.04.2021


import ast
import textwrap

import astunparse




def code_contains_variables(code, variables, root = False):
    '''Check whether the given `code` (either string or list of strings)
    defines multiple `variables` names (a string or list of strings).

    If `root = True`, only check if the outermost code block contains it.
    '''
    if isinstance(variables, str):
        variables = [variables]

    tree = ast.parse("".join(code))
    vars_found = []

    if root:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    vars_found.append(target.id)

    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    vars_found.append(target.id)

    found = [False for _ in range(len(variables))]
    for i, var in enumerate(variables):
        if var in vars_found:
            found[i] = True

    return all(found)




def code_contains_variable(code, variable, root = False):
    '''Check whether the given `code` (either string or list of strings)
    defines a `variable` name (a string) in a simple assignment context; that
    is, `x = 5`, not `obj.x = 5`.

    If `root = True`, only check if the outermost code block contains it.
    '''
    tree = ast.parse("".join(code))
    if root:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == variable:
                            return True

    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == variable:
                            return True

    return False




def tree_assignment_index(tree, variable):
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == variable:
                        return i

    return -1




def code_substitute_variable(old_code, variable, new_code):
    '''Substitute the `variable` (a string) assignment in some `old_code`
    (either a string or a list of strings) with another piece of `new_code`
    (either a string or a list of strings) and return the modified code as a
    string.
    '''

    old_tree = ast.parse("".join(old_code))
    new_tree = ast.parse("".join(new_code))

    var_idx = tree_assignment_index(old_tree, variable)
    if var_idx == -1:
        raise NameError(textwrap.fill((
            "The input `old_code` does not contain an assignment to a "
            f"variable `{variable}`, so no substitution can take place."
        )))

    old_tree.body = (
        old_tree.body[:var_idx] +
        new_tree.body +
        old_tree.body[(var_idx + 1):]
    )

    return astunparse.unparse(old_tree) + "\n"

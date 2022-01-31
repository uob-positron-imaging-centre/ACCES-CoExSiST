#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_optimisation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.01.2022


import textwrap

import coexist


def test_access():
    # Simple, idiomatic, correct test
    code = textwrap.dedent('''
        # ACCESS PARAMETERS START
        import coexist

        parameters = coexist.create_parameters(
            ["fp1", "fp2", "fp3"],
            [-5, -5, -5],
            [10, 10, 10],
        )
        # ACCESS PARAMETERS END

        values = parameters["value"]
        error = values["fp1"]**2 + values["fp2"]**2
    ''')

    file = "access_script1.py"
    with open(file, "w") as f:
        f.write(code)

    access = coexist.Access(file)
    print(access)

    data = access.learn(random_seed = 123)
    print(access)
    print(data)

    data = coexist.AccessData.read(access.paths.directory)
    print(data)

    # Weird parameters directives
    code = textwrap.dedent('''
        #####   ACCES \t PARAMETERS    START\tmama mia pizzeria
        import coexist

        parameters = coexist.create_parameters(
            ["fp1", "fp2", "fp3"],
            [-5, -5, -5],
            [10, 10, 10],
        )
        ##   ACCESS   PARAMETERS   END

        values = parameters["value"]
        error = values["fp1"]**2 + values["fp2"]**2 + values["fp3"]**4
    ''')

    file = "access_script2.py"
    with open(file, "w") as f:
        f.write(code)

    coexist.Access(file).learn(random_seed = 124)

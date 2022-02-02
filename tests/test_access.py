#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_optimisation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.01.2022


import textwrap

import coexist


def test_access_data():
    # Current, modern format
    data = coexist.AccessData.read("access_data/access_seed123")
    print(data)

    # Legacy coexist-0.1.0 format
    data = coexist.AccessData.read("access_data/access_info_000042")
    print(data)


def test_access():
    # Simple, idiomatic, correct test
    code = textwrap.dedent('''
        # ACCESS PARAMETERS START
        import sys
        import coexist

        parameters = coexist.create_parameters(
            ["fp1", "fp2", "fp3"],
            [-5, -5, -5],
            [10, 10, 10],
        )
        # ACCESS PARAMETERS END

        print("Example stdout message.")
        print("Example stderr message.", file=sys.stderr)

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
        import sys
        import coexist

        parameters = coexist.create_parameters(
            ["fp1", "fp2", "fp3"],
            [-5, -5, -5],
            [10, 10, 10],
        )
        ##   ACCESS   PARAMETERS   END

        print("Example stdout message.")
        print("Example stderr message.", file=sys.stderr)

        values = parameters["value"]
        error = values["fp1"]**2 + values["fp2"]**2 + values["fp3"]**4
    ''')

    file = "access_script2.py"
    with open(file, "w") as f:
        f.write(code)

    coexist.Access(file).learn(random_seed = 124)


def test_access_plots():
    # Using AccessData
    data = coexist.AccessData.read("access_data/access_seed123")
    coexist.plots.access(data)
    coexist.plots.access2d(data)

    # Using the filepaths only
    coexist.plots.access("access_data/access_seed123")
    coexist.plots.access2d("access_data/access_seed123")


if __name__ == "__main__":
    test_access_data()
    test_access()
    test_access_plots()

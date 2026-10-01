#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : async_access_template.py
# License: GNU v3.0


'''Run a user-defined simulation script with a given set of free parameter
values, then save the `error` value and any `extra` data to disk.

AccessScript takes an arbitrary simulation script that defines its set of free
parameters between two `#### ACCESS PARAMETERS START / END` directives and
substitutes that with an ACCESS-predicted solution. After the simulation, it
saves the `error` and `extra` variables to disk.

This simulation setup is achieved via a form of metaprogramming: the user's
code is modified to change the `parameters` to what is predicted at each run,
then code is injected to save the `error` and `extra` variables. This generated
script is called in a massively parallel environment with three command-line
arguments:

    1. The path to this run's `parameters`, as predicted by ACCESS.
    2. A path to save the user-defined `error` variable to.
    3. A path to save the user-defined `extra` variable to.

The user-defined `parameters` are changed by substituting the code between the
directives with:

    ```python

    # Unpickle `parameters` from this script's first command-line argument
    with open(sys.argv[1], "rb") as f:
        parameters = pickle.load(f)

    # Set `access_id` to a unique ID from the first command-line argument
    access_id = int(sys.argv[1].split("_")[1])

    ```

The variables are saved using `pickle`, for example:

    ```python

    import pickle

    # Save variable
    with open("some_path.pickle", "wb") as f:
        pickle.dump(variable, f)

    # Load variable
    with open("some_path.pickle", "rb") as f:
        loaded_variable = pickle.load(f)

    ```

You can find them in the `access_info_<hash_code>/simulations` directory.
'''


import os
import sys
import pickle


###############################################################################
#### ACCESS INJECT USER CODE START                                         ####
'''ACCESS Example User Simulation Script

Must define one simulation whose parameters will be optimised this way:

    1. Use a variable called "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       You can set the initial guess here.

    2. The `parameters` creation should be fully self-contained between
       `#### ACCESS PARAMETERS START` and `#### ACCESS PARAMETERS END`
       blocks (i.e. it should not depend on code ran before that).

    3. By the end of the simulation script, define a variable named ``error``
       storing a single number representing this simulation's error value.

Importantly, use `parameters.at[<free parameter name>, "value"]` to get this
simulation's free / optimisable variable values.
'''

#### ACCESS PARAMETERS START

# Unpickle `parameters` from this script's first command-line argument and set
# `access_id` to a unique simulation ID

import coexist
with open(sys.argv[1], 'rb') as f:
    parameters = pickle.load(f)
access_id = 0

access_id = int(os.path.split(sys.argv[1])[1].split("_")[1])

#### ACCESS PARAMETERS END


# Extract variables
x = parameters.at["cor", "value"]
y = parameters.at["separation", "value"]


# Define the error value in any way - run a simulation, analyse data, etc.
a = 1
b = 100

error = (a - x) ** 2 + b * (y - x ** 2) ** 2
#### ACCESS INJECT USER CODE END                                           ####
###############################################################################


# Save the user-defined `error` and `extra` variables to disk.
with open(sys.argv[2], "wb") as f:
    pickle.dump(error, f)

if "extra" in locals() or "extra" in globals():
    with open(sys.argv[3], "wb") as f:
        pickle.dump(extra, f)

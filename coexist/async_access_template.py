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
#### ACCESS INJECT USER CODE END                                           ####
###############################################################################


# Save the user-defined `error` and `extra` variables to disk.
with open(sys.argv[2], "wb") as f:
    pickle.dump(error, f)

if "extra" in locals() or "extra" in globals():
    with open(sys.argv[3], "wb") as f:
        pickle.dump(extra, f)

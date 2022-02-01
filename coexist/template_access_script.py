#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : async_access_template.py
# License: GNU v3.0


'''Run a user-defined simulation script with a given set of free parameter
values, then save the `error` value to disk.

ACCES takes an arbitrary simulation script that defines its set of free
parameters between two `# ACCESS PARAMETERS START / END` directives and
substitutes them with an ACCESS-predicted solution. After the simulation, it
saves the `error` variable to disk.

This simulation setup is achieved via a form of metaprogramming: the user's
code is modified to change the `parameters` to what is predicted at each run,
then code is injected to save the `error` variable. This generated script is
called in a massively parallel environment with two command-line arguments:

    1. The path to this run's `parameters`, as predicted by ACCESS.
    2. A path to save the user-defined `error` variable to.

You can find them in the `access_seed<seed>/results` directory.
'''


import os
import sys
import pickle


###############################################################################
# ACCESS INJECT USER CODE START ###############################################
# ACCESS INJECT USER CODE END   ###############################################
###############################################################################


# Save the user-defined `error` and `extra` variables to disk.
with open(sys.argv[2], "wb") as f:
    pickle.dump(error, f)

if "extra" in locals() or "extra" in globals():
    path = os.path.split(sys.argv[2])
    path = os.path.join(path[0], path[1].replace("result", "extra"))
    with open(path, "wb") as f:
        pickle.dump(extra, f)

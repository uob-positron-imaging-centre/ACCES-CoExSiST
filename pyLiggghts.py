#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pyLiggghts.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2020


import io
import re
import textwrap
import tempfile

import numpy as np
import pandas as pd

from liggghts import liggghts



class Parameters(pd.DataFrame):
    '''Pandas DataFrame subclass with a custom constructor for LIGGGHTS
    simulation parameters.

    In order to dynamically change LIGGGHTS simulation parameters, a macro
    command must be run (e.g. `liggghts.command(fix  m1 all property/global
    youngsModulus peratomtype 0.8e9)`). This class saves in a DataFrame:

        1. The required *command template* to change a given parameter. E.g. if
           "fix  m1 all property/global youngsModulus peratomtype 0.8e9" is the
           command to set the Young's modulus, then the *command template* is
           "fix  m1 all property/global youngsModulus peratomtype {}". Notice
           the `{}`, which will be replaced dynamically with a given value.

        2. Per-parameter initial guesses.

        3. Per-parameter lower bounds (i.e. minimum valid value).

        4. Per-parameter upper bounds (i.e. maximum valid value).

    All those values are indexed by the parameter name. Below is an example
    that shows how to construct a `Parameters` class containing a hypothetical
    simulation's properties that will be dynamically changed.

    Examples
    --------

    >>> parameters = Parameters(
    >>>     ["Young's Modulus", "Poisson Ratio"],
    >>>     ["fix  m1 all property/global youngsModulus peratomtype {}",
    >>>     "fix  m2 all property/global poissonsRatio peratomtype {}"],
    >>>     [0.8e9, 0.4],
    >>>     [None, None],
    >>>     [0.0, 1.0]
    >>> )
    >>>
    >>> parameters
    >>>                                      command        value   min   max
    >>> Young's Modulus  fix  m1 all property/glob...  800000000.0  None  None
    >>> Poisson Ratio    fix  m2 all property/glob...          0.4  0.0   1.0

    Notes
    -----
    As this class inherits from `pandas.DataFrame`, all methods from it are
    available after instantiation - only the constructor is custom.
    '''

    def __init__(
        self,
        names,
        commands,
        initial_values,
        minimums = None,
        maximums = None,
    ):
        '''`Parameters` class constructor.

        Parameters
        ----------
        names: list[str]
            An iterable containing the parameter names that will be used for
            indexing / accessing each row. The names do not have to match the
            LIGGGHTS terms - it is only used by the programmer.

        commands: list[str]
            An iterable containing the macro commands required to modify the
            needed LIGGGHTS parameters, containing a `{}` as a placeholder
            for the actual value. E.g. `"fix  m1 all property/global
            youngsModulus peratomtype {}"`.

        initial_values: list[float]
            An iterable containing the initial guess for each LIGGGHTS
            parameter.

        minimums: list[float], optional
            An iterable containing the lower bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`. If unset, all
            values are `None`.

        maximums: list[float], optional
            An iterable containing the upper bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`. If unset, all
            values are `None`.
        '''

        if minimums is None:
            minimums = [None] * len(names)

        if maximums is None:
            maximums = [None] * len(names)

        if not (len(names) == len(commands) == len(initial_values) ==
            len(minimums) == len(maximums)):
            raise ValueError(textwrap.fill(
                '''The input iterables `names`, `commands`, `initial_values`
                and (if defined) `minimums` and `maximums` must all have the
                same length.'''
            ))

        for cmd in commands:
            if re.search("{.*}", cmd) is None:
                raise ValueError(textwrap.fill(
                    '''The strings in the input `commands` must contain one
                    substring `"{}"` (e.g. `"fix m1 ... {0}, {0}, {0}"`), in
                    which the right value will be substituted when running the
                    LIGGGHTS command.'''
                ))

        parameters = {
            "command": commands,
            "value": initial_values,
            "min": minimums,
            "max": maximums,
        }

        pd.DataFrame.__init__(self, parameters, index = names)




class Simulation:
    '''Class encapsulating a single LIGGGHTS simulation whose parameters will
    be modified dynamically by a driver code.

    '''

    def __init__(
        self,
        simulation,
        parameters,
    ):
        '''`Simulation` class constructor.

        Parameters
        ----------
        simulation: path-like object or str
            LIGGGHTS macro script for setting up a simulation - either a path
            (relative or absolute, e.g. "../in.sim") or a `str` containing the
            actual macro commands.

        parameters: Parameters instance
            The LIGGGHTS simulation parameters that will be dynamically
            modified, encapsulated in a `Parameters` class instance. Check its
            documentation for further information and example instantiation.
        '''

        self.simulation = liggghts()

        # Try to read in `simulation` as a file, otherwise treat it as a string
        try:
            with open(simulation) as f:
                self.simulation.file(simulation)
        except FileNotFoundError:
            self.simulation.command(str(simulation))

        if not isinstance(parameters, Parameters):
            raise TypeError(textwrap.fill(
                f'''The input `parameters` must be an instance of the
                `Parameters` class. Received {type(parameters)}.'''
            ))

        self.parameters = parameters
        self._step_size = self.simulation.extract_global("dt", 1)

        # Set simulation parameters to the values in `parameters`
        for idx in self.parameters.index:
            self[idx] = self.parameters.loc[idx, "value"]


    @property
    def step_size(self):
        # save a step_size property of the class, so we can define the
        # step_size
        return self._step_size


    @step_size.setter
    def step_size(self, new_step_size):
        # set the step_size size
        if 0 < new_step_size < 1:
            self._step_size = new_step_size
            self.simulation.command(f"timestep {new_step_size}")
        else:
            raise ValueError("Step size must be between 0 and 1 !")


    def save(self, filename = "checkpoint"):
        # write a dump file
        cmd = f"write_dump all custom {filename} id type x y z vx vy vz radius"
        self.simulation.command(cmd)


    def load(self, filename = "checkpoint"):
        # load particle positions and velocity
        cmd = f"read_dump {filename} 0 radius x y z vx vy vz"
        self.simulation.command(cmd)


    def num_atoms(self):
        return self.simulation.get_natoms()


    def positions(self):
        # get particle positions
        pos = self.simulation.gather_atoms("x", 1, 3)
        pos = np.array(list(pos)).reshape(self.num_atoms(), -1)
        return pos


    def velocities(self):
        # get particle velocities
        pos = self.simulation.gather_atoms("v", 1, 3)
        pos = np.array(list(pos)).reshape(self.num_atoms(), -1)
        return pos


    def step(self, num_steps):
        # run simulation for `num_steps` timesteps
        self.simulation.command(f"run {num_steps} ")


    def step_to(self, timestamp):
        # run simulation up to time = `timestamp`
        if timestamp < self.timestep():
            raise ValueError(textwrap.fill(
                '''Timestep is below the current timestep.\nCheck input or
                reset the timestep!'''
            ))

        self.simulation.command(f"run {timestamp} upto ")


    def reset_time(self):
        # reset the current timestep to 0
        self.simulation.command("reset_timestep 0")


    def timestep(self):
        # return the current timestep
        return self.simulation.extract_global("ntimestep", 0)


    def time(self):
        return self.simulation.extract_global("atime", 1)


    def __setitem__(self, key, value):
        # Custom key-value setter to change a parameter in the class *and*
        # during the simulation.
        # Raises an AttributeError if the key didn't exist previously.
        if not key in self.parameters.index:
            raise AttributeError(textwrap.fill(
                f'''The given parameter name (the `key`) does not exist. It
                should be set when instantiating the `Parameters`. Received
                {key}.'''
            ))

        row = self.parameters.loc[key]
        self.simulation.command(
            row["command"].format(value)
        )

        self.parameters.at[key, "value"] = value


    def __del__(self):
        self.simulation.close()


    def __str__(self):
        # Shown when calling print(class)
        sdoc = self.simulation.__str__().split(r"\\n")
        sdoc = [s.replace("', '", "") for s in sdoc]
        sdoc = "\n".join(sdoc)

        docstr = (
            f"simulation:\n{sdoc}\n\n"
            f"parameters:\n{self.parameters}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL

        docstr = (
            "Class instance that inherits from `pyLiggghts.Simulation`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}"
        )

        return docstr




parameters = Parameters(
    ["Young's Modulus", "Poisson Ratio"],
    ["fix  m1 all property/global youngsModulus peratomtype {0} {0} {0}",
     "fix  m2 all property/global poissonsRatio peratomtype {0} {0} {0}"],
    [0.8e9, 0.4],
    [None, None],
    [0.0, 1.0]
)

simulation = Simulation("in.sim", parameters)



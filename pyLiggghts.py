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
    youngsModulus peratomtype 0.8e9)`). This class saves the data needed to
    modify simulation parameters in a DataFrame:

        1. The required *command template* to change a given parameter using
           LIGGGHTS equal-style variables. E.g. "fix  m1 all property/global
           youngsModulus peratomtype ${youngmodP} ${youngmodP} ${youngmodP}" is
           a LIGGGHTS command which uses the ${youngmodP} variable.

        2. Per-parameter initial guesses.

        3. Per-parameter lower bounds (i.e. minimum valid value), optional.

        4. Per-parameter upper bounds (i.e. maximum valid value), optional.

    All those values are indexed by the parameter name. Below is an example
    that shows how to construct a `Parameters` class containing a hypothetical
    simulation's properties that will be dynamically changed.

    Examples
    --------
    In the example below, the command to change a single simulation parameter
    contains other variables which we won't modify:

    >>> parameters = Parameters(
    >>>     ["corPP", "youngmodP"],
    >>>     ["fix  m3 all property/global coefficientRestitution \
    >>>         peratomtypepair 3 ${corPP} ${corPW} ${corPW2} \
    >>>                                    ${corPW} ${corPW2} ${corPW} \
    >>>                                    ${corPW2} ${corPW} ${corPW} ",
    >>>      "fix  m1 all property/global youngsModulus peratomtype \
    >>>         ${youngmodP} ${youngmodP} ${youngmodP}"],
    >>>     [0.5, 0.8e9],
    >>>     [0.0, None],
    >>>     [1.0, None],
    >>> )
    >>>
    >>> parameters
    >>>                                              command  value   min  max
    >>> corPP  fix  m3 all property/global coefficientRes...    0.5  None  0.0
    >>> corPW  fix  m3 all property/global coefficientRes...    0.5  None  1.0

    Notes
    -----
    As this class inherits from `pandas.DataFrame`, all methods from it are
    available after instantiation - only the constructor is custom.
    '''

    def __init__(
        self,
        variables,
        commands,
        initial_values,
        minimums = None,
        maximums = None,
    ):
        '''`Parameters` class constructor.

        Parameters
        ----------
        variables: list[str]
            An iterable containing the LIGGGHTS variable names that will be
            used for changing simulation parameters.

        commands: list[str]
            An iterable containing the macro commands required to modify
            LIGGGHTS simulation parameters, containing the variable names
            as `${varname}`. E.g. `"fix  m1 all property/global youngsModulus
            peratomtype ${youngmodP} ${youngmodP} ${youngmodP}"`.

        initial_values: list[float]
            An iterable containing the initial values for each LIGGGHTS
            simulation parameter.

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
            minimums = [None] * len(variables)

        if maximums is None:
            maximums = [None] * len(variables)

        if not (len(variables) == len(commands) == len(initial_values) ==
                len(minimums) == len(maximums)):
            raise ValueError(textwrap.fill(
                '''The input iterables `variables`, `commands`,
                `initial_values` and (if defined) `minimums` and `maximums`
                must all have the same length.'''
            ))

        for cmd in commands:
            if re.search("\$\{\w+\}", cmd) is None:
                raise ValueError(textwrap.fill(
                    '''The strings in the input `commands` must contain at
                    least one substring `"${varname}"` (e.g. `"fix  m2 all
                    property/global poissonsRatio peratomtype ${poissP}
                    ${poissP} ${poissP}"`), in which the right value will be
                    substituted when running the LIGGGHTS command.'''
                ))

        parameters = {
            "command": commands,
            "value": initial_values,
            "min": minimums,
            "max": maximums,
        }

        pd.DataFrame.__init__(self, parameters, index = variables)




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
        vel = self.simulation.gather_atoms("v", 1, 3)
        vel = np.array(list(vel)).reshape(self.num_atoms(), -1)
        return vel


    def variable(self, var_name):
        return self.simulation.extract_variable(var_name, "", 0)


    def step(self, num_steps):
        # run simulation for `num_steps` timesteps
        self.simulation.command(f"run {num_steps} ")


    def step_to(self, timestamp):
        # run simulation up to timestep = `timestamp`
        if timestamp < self.timestep():
            raise ValueError(textwrap.fill(
                '''Timestep is below the current timestep.\nCheck input or
                reset the timestep!'''
            ))

        self.simulation.command(f"run {timestamp} upto ")


    def step_to_time(self, time):
        # run simulation up to sim time = `time`
        if time < self.time():
            raise ValueError(textwrap.fill(
                '''Time is below the current simulation time. Check input or
                reset the timestep!'''
            ))

        nsteps = (time - self.time()) / self.step_size
        self.step(nsteps)


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

        # Extracts variable LIGGGHTS substitutions, like ${corPP} => corPP
        variable_extractor = re.compile("\$\{|\}")

        # Substitute all occurences of ${varname} in the LIGGGHTS command with:
        #   1. `value` if `varname` == `key`
        #   2. the LIGGGHTS variable `varname` otherwise
        def replace_var(match):
            var = variable_extractor.split(match.group(0))[1]

            if var == key:
                return str(value)
            else:
                return str(self.variable(var))

        cmd = re.sub(
            "\$\{\w+\}",
            replace_var,
            self.parameters.loc[key, "command"]
        )

        # Run the command with replaced varnames
        self.simulation.command(cmd)

        # Modify the global variable name to reflect the change
        self.simulation.command(f"variable {key} equal {value}")

        # Set inner class parameter value
        self.parameters.at[key, "value"] = value


    def __del__(self):
        self.simulation.close()


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"simulation:\n{self.simulation}\n\n"
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




if __name__ == "main":
    parameters = Parameters(
        ["corPP", "corPW"],
        ["fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP} ${corPW} ${corPW2} \
            ${corPW} ${corPW2} ${corPW} \
            ${corPW2} ${corPW} ${corPW} ",
        "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP} ${corPW} ${corPW2} \
            ${corPW} ${corPW2} ${corPW} \
            ${corPW2} ${corPW} ${corPW} "],
        [0.5, 0.5],     # Initial values
        [0.0, 0.0],     # Minimum values
        [1.0, 1.0]      # Maximum values
    )

    simulation = Simulation("in.sim", parameters)

    print("\nInitial simulation parameters:")
    print(f"corPP: {simulation.variable('corPP')}")
    print(f"corPW: {simulation.variable('corPW')}")

    simulation["corPP"] = 0.75
    simulation["corPW"] = 0.25

    print("\nModified simulation parameters:")
    print(f"corPP: {simulation.variable('corPP')}")
    print(f"corPW: {simulation.variable('corPW')}")






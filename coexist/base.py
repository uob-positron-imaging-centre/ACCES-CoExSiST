#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pyLiggghts.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2020


import re
import os
import textwrap

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

    .. code-block:: python

        parameters = Parameters(
            ["corPP", "youngmodP"],
            ["fix  m3 all property/global coefficientRestitution \
                peratomtypepair 3 ${corPP} ${corPW} ${corPW2} \
                                           ${corPW} ${corPW2} ${corPW} \
                                           ${corPW2} ${corPW} ${corPW} ",
             "fix  m1 all property/global youngsModulus peratomtype \
                ${youngmodP} ${youngmodP} ${youngmodP}"],
            [0.5, 0.8e9],
            [0.0, None],
            [1.0, None],
        )

        parameters
                                                     command  value   min  max
        corPP  fix  m3 all property/global coefficientRes...    0.5  None  0.0
        corPW  fix  m3 all property/global coefficientRes...    0.5  None  1.0

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
        minimums,
        maximums,
        sigma0 = None,
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

        minimums: list[float]
            An iterable containing the lower bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`, in which case also
            define a `sigma0`.

        maximums: list[float]
            An iterable containing the upper bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`, in which case also
            define a `sigma0`.

        sigma0: list[float], optional
            The standard deviation of the first population of solutions tried
            by the CMA-ES optimisation algorithm. If unset, it is computed as
            `0.2 * (maximum - minimum)`.
        '''

        if not (len(variables) == len(commands) == len(initial_values) ==
                len(minimums) == len(maximums)):
            raise ValueError(textwrap.fill(
                '''The input iterables `variables`, `commands`,
                `initial_values`, `minimums` and `maximums` must all have the
                same length.'''
            ))

        initial_values = np.array(initial_values, dtype = float)
        minimums = np.array(minimums, dtype = float)
        maximums = np.array(maximums, dtype = float)

        if (minimums >= maximums).any():
            raise ValueError(textwrap.fill(
                '''Found value in `maximums` that was smaller or equal than the
                corresponding value in `minimums`.'''
            ))

        if sigma0 is None:
            sigma0 = 0.2 * (maximums - minimums)
        elif len(sigma0) != len(variables):
            raise ValueError(textwrap.fill(
                '''If defined, `sigma0` must have the same length as the other
                input parameters.'''
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
            "sigma": sigma0,
        }

        pd.DataFrame.__init__(self, parameters, index = variables)


    @staticmethod
    def empty():
        '''Returns an empty Parameters DataFrame; useful for simulation which
        do not have any dynamically changing parameters.

        Returns
        -------
        coexist.Parameters
            An empty Parameters DataFrame.
        '''

        return Parameters([], [], [], [], [])


    def copy(self):
        parameters_copy = Parameters(
            self.index.copy(),
            self["command"].copy(),
            self["value"].copy(),
            self["min"].copy(),
            self["max"].copy(),
            self["sigma"].copy(),
        )

        return parameters_copy




class Experiment:
    '''Class encapsulating an experiment's recorded particle positions at
    multiple timesteps.

    For a single timestep, the particles in a system can be represented as a 2D
    array with columns [x, y, z]. For multiple timesteps, the positions arrays
    can be stacked, yielding a 3D array with shape (T, N, 3), where T is the
    number of timesteps, N is the number of particles, plus three columns for
    their cartesian coordinates.

    This class contains the recorded timesteps and the stacked 3D array of
    particle positions.

    It is used for the ground truth data that another DEM simulation will learn
    the parameters of.
    '''

    def __init__(
        self,
        times,
        positions_all,
        resolution,
        **kwargs
    ):
        '''`Experiment` class constructor.

        Parameters
        ----------
        times: list[float]
            A list (or vector) of timestamp values for all positions recorded
            in the experiment.

        positions_all: (T, N, M>=3) numpy.ndarray
            A 3D array-like with dimensions (T, N, M>=3), where T is the number
            of timesteps (corresponding exactly to the length of `times`), N is
            the number of particles in the system and M is the number of data
            columns per particle (at least 3 for their [x, y, z] coordinates,
            but can contain extra columns).

        resolution: float
            The expected error on the positions recorded.

        kwargs: keyword arguments
            Extra data that should be attached to this class as new attributes
            (e.g. "exp.droplets = 5" creates a new attribute).

        Raises
        ------
        ValueError
            If `times` is not a flat list-like, or `times` contains repeated
            values, or `positions_all` does not have exactly three dimensions
            and at least 3 columns (axis 2), or
            `len(times) != len(positions_all)`.
        '''

        times = np.asarray(times, dtype = float, order = "C")
        positions_all = np.asarray(positions_all, dtype = float, order = "C")

        if times.ndim != 1:
            raise ValueError(textwrap.fill((
                "The `times` input parameter must have exactly one dimension. "
                f"Received `ndim = {times.ndim}`."
            )))

        if len(np.unique(times)) != len(times):
            raise ValueError(textwrap.fill((
                "The `times` input parameter must have only unique values. "
                f"Found {len(times) - len(np.unique(times))} repeated values."
            )))

        if positions_all.ndim != 3 or positions_all.shape[2] < 3:
            raise ValueError(textwrap.fill((
                "The `positions_all` input parameter must have exactly three "
                "dimensions and at least three columns. Received "
                f"`shape = {positions_all.shape}`."
            )))

        if len(times) != len(positions_all):
            raise ValueError(textwrap.fill((
                "The input `positions_all` parameter must be a 3D array with "
                "the particles' positions at every timestep in `times`. The "
                "shape of `positions_all` should then be (T, N, 3), where T "
                "is the number of timesteps, N is the number of particles, "
                "plus three columns for x, y, z coordinates. Received "
                f"{len(times)} timesteps in `times` and {len(positions_all)} "
                "stacked arrays in `positions_all`."
            )))

        self.times = times
        self.positions_all = positions_all
        self.resolution = float(resolution)

        for k, v in kwargs:
            self.k = v


    def positions(self, timestep):
        time_idx = np.argwhere(np.isclose(timestep, self.times))

        if len(time_idx) == 0:
            raise ValueError(textwrap.fill((
                "There are no values in `times` equal to the requested "
                f"timestep {timestep}."
            )))

        return self.positions_all[time_idx[0, 0]]


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"times:\n{self.times}\n\n"
            f"positions_all:\n{self.positions_all}"
        )

        if self.kwargs:
            docstr += f"\n\nkwargs:\n{self.kwargs}"

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL
        docstr = (
            "Class instance that inherits from `pyCoexist.Experiment`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}"
        )

        return docstr




class Simulation:
    '''Class encapsulating a single LIGGGHTS simulation whose parameters will
    be modified dynamically by a driver code.

    '''

    def __init__(
        self,
        sim_name,
        parameters = Parameters.empty(),
        verbose = True,
        log = True,
        log_file = "pyLiggghts.log",
    ):
        '''`Simulation` class constructor.

        Parameters
        ----------
        sim_name: path-like object or str
            This class can work with two types of LIGGGHTS simulation formats:
            LIGGGHTS scripts (e.g. "init.sim") or LIGGGHTS restart files (e.g.
            "vibrofluidised_restart.sim"). First, if a file with the exact name
            as `sim_name` exists, then it will be used as a LIGGGHTS script.
            Otherwise, we search for two files: `{sim_name}_restart.sim` and
            `{sim_name}_properties.sim`, which will act as restart files. Note:
            these files are saved when using the `save()` functions.

        parameters: Parameters instance
            The LIGGGHTS simulation parameters that will be dynamically
            modified, encapsulated in a `Parameters` class instance. Check its
            documentation for further information and example instantiation.

        verbose: bool, default `True`
            Show LIGGGHTS output while simulation is running.

        log: bool, default 'True'
            save all property changing commands that where exectuted in a log
            file. Does not contain commands inside the default LIGGGHTS
            macro script
            Needed for the .copy function!
            excluded commands:
                run
                write_restart
        '''

        # Type-check input parameters
        if not isinstance(parameters, Parameters):
            raise TypeError(textwrap.fill(
                f'''The input `parameters` must be an instance of the
                `Parameters` class. Received {type(parameters)}.'''
            ))

        # Create class attributes
        self._verbose = bool(verbose)
        self.log = log
        self.log_file = log_file

        # TODO: Domenico, use logging?

        if self._verbose:
            self.simulation = liggghts()
        else:
            self.simulation = liggghts(cmdargs = ["-screen", "/dev/null"])

        self.sim_name = str(sim_name)
        self.properties = []

        # A list of keywords that, if found in a LIGGGHTS script line, will
        # make the class save that script line
        self.save_keywords = [
            "variable",
            "pair_style",
            "pair_coeff",
            "neighbor",
            "neigh_modify",
            "fix",
            "timestep",
            "communicate",
            "newton",
        ]

        # A compiled Regex object for finding any of the above keywords as
        # substrings
        self.keyword_finder = re.compile(
            "|".join(self.save_keywords),
            re.IGNORECASE
        )

        # First look for a file with the same name as `sim_name`; otherwise
        # look for `sim_name_restart.sim` and `sim_name_properties.sim`
        if os.path.exists(self.sim_name):
            # It is a LIGGGHTS script!
            self.simulation.file(self.sim_name)
            self.create_properties(self.sim_name)

        elif (os.path.exists(f"{self.sim_name}_restart.sim") and
              os.path.exists(f"{self.sim_name}_properties.sim")):
            # It is a LIGGGHTS restart file + properties file
            self.load(self.sim_name)

        else:
            raise FileNotFoundError(textwrap.fill((
                f"No LIGGGHTS input file (`{self.sim_name}`) or LIGGGHTS "
                f"restart and properties file found (`{self.sim_name}_"
                f"restart.sim` and `{self.sim_name}_properties.sim`)."
            )))

        self.parameters = parameters
        self._step_size = self.simulation.extract_global("dt", 1)

        # Set simulation parameters to the values in `parameters`
        for idx in self.parameters.index:
            self[idx] = self.parameters.loc[idx, "value"]


    def create_properties(self, sim_name):
        with open(sim_name) as f:
            # Append lines without the trailing newline
            sim_input = [line.rstrip() for line in f.readlines()]

        self.properties = []

        i = 0
        while i < len(sim_input):
            # Select the command line at index i
            line = sim_input[i]

            # Concatenate next lines if the last character is "&"
            while len(line) > 0 and line[-1] == "&":
                line = line[:-1]                # Remove the last character, &
                i += 1                          # Increment index
                line += " " + sim_input[i]      # Concatenate next line

            # Remove comments from the end of the line
            line_nc = line.split("#")[0]

            # If any of the keywords is found as a substring in the command
            # line (excluding comments), append it (including comments) to the
            # properties attribute
            if self.keyword_finder.search(line_nc):
                self.properties.append(line)

            i += 1


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
            self.execute_command(f"timestep {new_step_size}")
        else:
            raise ValueError("Step size must be between 0 and 1 !")


    @property
    def log(self):
        return self._log


    @log.setter
    def log(self, log):
        self._log = bool(log)


    @property
    def verbose(self):
        self._verbose


    @verbose.setter
    def verbose(self, verbose):
        self._verbose = bool(verbose)


    def save(self, filename = None):
        '''Save a simulation's state, along with data not included in the
        standard LIGGGHTS restart file.

        This function saves two files, based on the input `filename`:

        1. "{filename}_restart.sim": the standard LIGGGHTS restart file.
        2. "{filename}_properties.sim": the additional data not included in the
           file above.

        Notice that the input `filename` is just *the prefix* of the files
        saved; do not include file extensions (i.e. instead of
        "vibrofluidised.sim", use just "vibrofluidised").

        Parameters
        ----------
        filename: str, optional
            The prefix for the two simulation files saved. If `None`, then the
            `sim_name` class attribute is used.
        '''

        # Remove trailing ".sim", if existing in `self.sim_name`
        if filename is None:
            filename = self.sim_name.split(".sim")[0]

        # Write restart file
        cmd = f"write_restart {filename}_restart.sim"
        self.execute_command(cmd)

        # Add the current time to the properties file, as a comment
        self.properties.insert(0, f"# current_time = {self.time()}")

        # Write properties file. `self.properties` is a list of strings
        # containing each command that the restart file does not save
        with open(f"{filename}_properties.sim", "w") as f:
            f.writelines("\n".join(self.properties))


    def load(self, filename = None):
        # load a new simulation based on the position data from filename and
        # the system based on self.filename
        #
        # 1st:
        # open the simulation file and search for the line where it reads the
        # restart then change the filename in this file and save

        if filename is None:
            filename = self.sim_name.split(".sim")[0]

        if not os.path.exists(f"{filename}_restart.sim"):
            raise FileNotFoundError(textwrap.fill((
                "No LIGGGHTS restart file found based on the input filename: "
                f"`{filename}_restart.sim`."
            )))

        if not os.path.exists(f"{filename}_properties.sim"):
            raise FileNotFoundError(textwrap.fill((
                "No LIGGGHTS properties file found based on the input "
                f"filename: `{filename}_properties.sim`."
            )))

        # Close previous simulation and open a new one
        self.simulation.close()

        if self._verbose:
            self.simulation = liggghts()
        else:
            self.simulation = liggghts(cmdargs = ["-screen", "/dev/null"])

        self.execute_command(f"read_restart {filename}_restart.sim")

        # Now execute all commands in the properties file
        self.simulation.file(f"{filename}_properties.sim")

        # Finally, set the time, given in a comment on the first line
        with open(f"{filename}_properties.sim") as f:
            line = f.readline()

        # Split into a list, e.g. "#  current_time =   15   " -> ["", "15   "]
        # Note: "[ ]*" means 0 or more spaces
        current_time = re.split("#[ ]*current_time[ ]*=[ ]*", line)
        current_time = float(current_time[1])

        self.simulation.set_time(current_time)


    def num_atoms(self):
        return self.simulation.get_natoms()


    def radii(self):
        # TODO Dominik - return radii of all particles in the system
        # That is, a 1D numpy array with the radius for each particle
        return np.ones(self.num_atoms()) * 0.0025


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


    def set_position(
        self,
        particle_id,    # single number of particle's ID
        position        # array or list of particle positions 1x3
    ):
        cmd = (f"set atom {particle_id + 1} x {position[0]} y {position[1]} "
               f"z {position[2]}")
        self.execute_command(cmd)


    def set_velocity(
        self,
        particle_id,    # single number of particle's ID
        velocity        # array or list of particle velocitys 1x3
    ):
        cmd = (f"set atom {particle_id + 1} vx {velocity[0]} vy {velocity[1]} "
               f"vz {velocity[2]}")
        self.execute_command(cmd)


    def variable(self, var_name):
        try:
            varb = self.simulation.extract_variable(var_name, "", 0)
        except ValueError:
            raise ValueError((
                f"[ERROR]: Tried to access non-existent variable {var_name}!"
            ))

        return varb


    def step(self, num_steps):
        # run simulation for `num_steps` timesteps
        if self.verbose:
            self.execute_command(f"run {num_steps}")
        else:
            self.execute_command(f"run {num_steps} post no")


    def step_to(self, timestamp):
        # run simulation up to timestep = `timestamp`
        if timestamp < self.timestep():
            raise ValueError(textwrap.fill(
                '''Timestep is below the current timestep.\nCheck input or
                reset the timestep!'''
            ))

        if self.verbose:
            self.execute_command(f"run {timestamp} upto")
        else:
            self.execute_command(f"run {timestamp} upto post no")


    def step_time(self, time):
        # find timestep which can run exectly to time
        # while beeing lower then self.step_size
        new_dt = time / (int(time / self.step_size) + 1)
        steps = time / new_dt

        old_dt = self.step_size
        self.step_size = new_dt
        self.step(steps)
        self.step_size = old_dt


    def step_to_time(self, time):
        # run simulation up to sim time = `time`
        if time < self.time():
            raise ValueError(textwrap.fill(
                '''Time is below the current simulation time. Check input or
                reset the timestep!'''
            ))

        rest_time = (time - self.time()) % self.step_size

        n_steps = (time - self.time() - rest_time) / self.step_size

        self.step(n_steps)
        if rest_time != 0.0:
            # Now run 1 single timestep with a smaller timestep
            old_dt = self.step_size

            # set step size to the rest time
            self.step_size = rest_time
            self.step(1)

            # reset to normal dt
            self.step_size = old_dt


    def reset_time(self):
        # reset the current timestep to 0
        self.execute_command("reset_timestep 0")


    def timestep(self):
        # return the current timestep
        return self.simulation.extract_global("ntimestep", 0)


    def time(self):
        return self.simulation.extract_global("atime", 1)


    def execute_command(self, cmd):
        self.simulation.command(cmd)

        # If the command (excluding comments) contains any of our keywords,
        # save it (with comments) in the properties attribute
        cmd_nc = cmd.split("#")[0]
        if self.keyword_finder.search(cmd_nc):
            self.properties.append(cmd)


    def copy(self, filename = None):
        """
        copy the ligghts instance
        includes:
            - particle positions /velocitys / properties
            - system
        """

        if filename is None:
            filename = f"simsave_{str(hash(self.__repr__()))}"

        self.save(filename)

        new_sim = Simulation(
            filename,
            parameters = self.parameters.copy(),
            verbose = self.verbose,
        )

        os.remove(f"{filename}_restart.sim")
        os.remove(f"{filename}_properties.sim")

        return new_sim



    def __setitem__(self, key, value):
        # Custom key-value setter to change a parameter in the class *and*
        # during the simulation.
        # Raises an AttributeError if the key didn't exist previously.
        if key not in self.parameters.index:
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
        self.execute_command(cmd)

        # Modify the global variable name to reflect the change
        self.execute_command(f"variable {key} equal {value}")

        # Set inner class parameter value
        self.parameters.at[key, "value"] = value


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

    simulation = Simulation("run.sim", parameters)

    simulation.save()
    simulation.step(200)

    simulation.save("2.save")
    simulation.step(200)

    simulation.load("2.save")

    simulation.step(200)


    print("\nInitial simulation parameters:")
    print(f"corPP: {simulation.variable('corPP')}")
    print(f"corPW: {simulation.variable('corPW')}")

    simulation["corPP"] = 0.75
    simulation["corPW"] = 0.25

    print("\nModified simulation parameters:")
    print(f"corPP: {simulation.variable('corPP')}")
    print(f"corPW: {simulation.variable('corPW')}")

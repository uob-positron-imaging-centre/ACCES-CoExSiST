#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : liggghts.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.01.2022


import  re
import  os
import  pickle
import  pathlib
import  textwrap

import  numpy       as      np

# `liggghts` is an optional dependency for LiggghtsSimulation
from    liggghts    import  liggghts        # lgtm [py/import-own-module]
from    pyevtk.hl   import  pointsToVTK

# Local imports
from    .base       import  Simulation, Parameters



class LiggghtsSimulation(Simulation):
    '''Pythonic interface to LIGGGHTS, interfacing with a given simulation
    script, optionally returning NumPy arrays to relevant particle properties.

    You need to have the ``liggghts`` Python library available; you can see the
    `https://github.com/uob-positron-imaging-centre/PICI-LIGGGHTS` repository
    for some instruction. In short: compile liggghts, then add the `python`
    directory to PYTHONPATH.

    Examples
    --------
    Simply load a LIGGGHTS simulation script's path as the first parameter:

    >>> import coexist
    >>> sim = coexist.LiggghtsSimulation("path_to_liggghts_script.sim")

    LIGGGHTS can only run simulations for a given integer number of timesteps,
    but often we want to run them up to a given physical time.

    Using this class we can run the simulation up to e.g. time 1.2 seconds
    (this will use the appropriate number of timesteps):

    >>> sim.step_to_time(1.2)

    Access the particle radii, positions and velocities as **NumPy arrays**:

    >>> radii = sim.radii()
    >>> positions = sim.positions()
    >>> velocities = sim.velocities()

    If the simulation is unstable and particles are lost, they will be set to
    NaNs.
    '''

    def __init__(
        self,
        sim_name,
        parameters = Parameters(),
        set_parameters = True,
        timestep = None,
        save_vtk = None,
        verbose = False,
    ):
        '''`Simulation` class constructor.

        Parameters
        ----------
        sim_name : path-like object or str
            This class can work with two types of LIGGGHTS simulation formats:
            LIGGGHTS scripts (e.g. "init.sim") or LIGGGHTS restart files (e.g.
            "vibrofluidised_restart.sim"). First, if a file with the exact name
            as `sim_name` exists, then it will be used as a LIGGGHTS script.
            Otherwise, we search for two files: `{sim_name}_restart.sim` and
            `{sim_name}_properties.sim`, which will act as restart files. Note:
            these files are saved when using the `save()` functions.

        parameters : Parameters instance
            The LIGGGHTS simulation parameters that will be dynamically
            modified, encapsulated in a `Parameters` class instance. Check its
            documentation for further information and example instantiation.

        set_parameters : bool, default True
            If `True`, set the simulation parameter values when instantiating
            the class. Set to `False` to change values later; this is useful if
            some parameters can only be set once, e.g. particle insertion
            command for ACCESS.

        timestep : AutoTimestep, int, float or None, optional
            Variable that determines how the LIGGGHTS simulation timestep
            is managed. If it is `AutoTimestep`, it will be computed from the
            Rayleigh formula; if it is an `int` or `float`, that value will be
            used. Otherwise, the default value from the LIGGGHTS script is
            used.

        save_vtk : str, optional
            If defined, save particle data (positions, velocity, etc.) as VTK
            files for each timestep. This parameter should be the *directory
            name* for saving files in the format
            `{save_vtk}/locations_{timestep_index}.vtk`.

        verbose : bool, default `True`
            Show LIGGGHTS output while simulation is running.
        '''

        # Type-check input parameters
        if not isinstance(parameters, Parameters):
            raise TypeError(textwrap.fill(
                f'''The input `parameters` must be an instance of the
                `Parameters` class. Received {type(parameters)}.'''
            ))

        # Create class attributes
        self._verbose = bool(verbose)

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

        # A list of keywords to ignore. They are run after `save_keywords`, so
        # for "fix ins insert/stream", if `fix` is saved and `insert/stream` is
        # ignored, then the whole line is ignored
        self.ignore_keywords = [
            r"insert\/stream",
            r"unfix",
            r"reset_timestep",
        ]

        # A compiled Regex object for finding any of the above keywords as
        # substrings
        self.finder_kw_save = re.compile(
            "|".join(self.save_keywords),
            re.IGNORECASE
        )

        self.finder_kw_ignore = re.compile(
            "|".join(self.ignore_keywords),
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
            # Read in the restart file
            self.execute_command(f"read_restart {self.sim_name}_restart.sim")

            # Now execute all commands in the properties file
            self.simulation.file(f"{self.sim_name}_properties.sim")

            # Finally, set the time, given in a comment on the first line
            with open(f"{self.sim_name}_properties.sim") as f:
                self.properties = [line.rstrip() for line in f.readlines()]
                line = self.properties[0]

            # Split into a list, e.g. "#  current_time =15  " -> ["", "15   "]
            # Note: "[ ]*" means 0 or more spaces
            current_time = re.split("#[ ]*current_time[ ]*=[ ]*", line)
            current_time = float(current_time[1])

            self.simulation.set_time(current_time)

        else:
            raise FileNotFoundError(textwrap.fill((
                f"No LIGGGHTS input file (`{self.sim_name}`) or LIGGGHTS "
                f"restart and properties file found (`{self.sim_name}_"
                f"restart.sim` and `{self.sim_name}_properties.sim`)."
            )))
        self._parameters = parameters.copy()
        self._step_size = self.simulation.extract_global("dt", 1)

        # Set simulation parameters to the values in `parameters`
        if set_parameters:
            for idx in self._parameters.index:
                self[idx] = self._parameters.loc[idx, "value"]

        # Check if timestep is auto/none/a number
        if isinstance(timestep, AutoTimestep):
            self._step_size = timestep.timestep()

            if verbose:
                print(f"Auto timestep: setting step size to {self._step_size}")

        elif isinstance(timestep, (int, float)):
            self.step_size = float(timestep)

        elif timestep is None:
            self._step_size = self.simulation.extract_global("dt", 1)

        else:
            raise TypeError(textwrap.fill((
                "The input `timestep` must be either an `AutoTimestep` "
                "instance, a number (int / float), or None (the default). "
                f"Received `{type(timestep)}`."
            )))

        if save_vtk is not None:
            self._save_vtk = str(save_vtk)

            # If there's a previous simulation saved in the folder, delete it
            if os.path.isdir(self._save_vtk):
                vtk_finder = re.compile(r"locations_.*\.vtu")

                for file in os.listdir(self._save_vtk):
                    if vtk_finder.search(file):
                        os.remove(os.path.join(self._save_vtk, file))
            else:
                # Create folder recursively if folder is nested
                pathlib.Path(self._save_vtk).mkdir(
                    parents = True,
                    exist_ok = True,
                )

            self.write_vtk()

        else:
            self._save_vtk = None


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
            if self.finder_kw_save.search(line_nc) and not \
                    self.finder_kw_ignore.search(line_nc):
                self.properties.append(line)

            i += 1


    @property
    def parameters(self):
        return self._parameters


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
    def verbose(self):
        self._verbose


    @verbose.setter
    def verbose(self, verbose):
        self._verbose = bool(verbose)


    @property
    def save_vtk(self):
        return self._save_vtk


    @save_vtk.setter
    def save_vtk(self, save_vtk):
        if save_vtk is not None:
            self._save_vtk = str(save_vtk)
        else:
            self._save_vtk = None


    def save(self, filename = None):
        '''Save a simulation's full state, along with data not included in the
        standard LIGGGHTS restart file and the internal `parameters`.

        This function saves three files, based on the input `filename`:

        1. "{filename}_restart.sim": the standard LIGGGHTS restart file.
        2. "{filename}_properties.sim": the additional data not included in the
           file above.
        3. "{filename}_parameters.pickle": the pickled `self.parameters`
           object.

        Notice that the input `filename` is just *the prefix* of the files
        saved; do not include file extensions (i.e. instead of
        "vibrofluidised.sim", use just "vibrofluidised").

        Parameters
        ----------
        filename : str, optional
            The prefix for the three simulation files saved. If `None`, then
            the `sim_name` class attribute is used (i.e. the name used when
            instantiating the class).
        '''

        # Remove trailing ".sim", if existing in `self.sim_name`
        if filename is None:
            filename = self.sim_name.split(".sim")[0]

        # Write restart file
        cmd = f"write_restart {filename}_restart.sim"
        self.execute_command(cmd)

        # Add the current time to the properties file, as a comment
        if re.match("# current_time =", self.properties[0]):
            self.properties[0] = f"# current_time = {self.time()}"
        else:
            self.properties.insert(0, f"# current_time = {self.time()}")

        # Write properties file. `self.properties` is a list of strings
        # containing each command that the restart file does not save
        with open(f"{filename}_properties.sim", "w") as f:
            f.writelines("\n".join(self.properties))

        # Save the parameters as a pickled file
        with open(f"{filename}_parameters.sim", "wb") as f:
            pickle.dump(self.parameters, f)


    @staticmethod
    def load(filename, verbose = False):
        # load a new simulation based on the position data from filename and
        # the system based on self.filename
        #
        # 1st:
        # open the simulation file and search for the line where it reads the
        # restart then change the filename in this file and save

        if not os.path.exists(f"{filename}_parameters.sim"):
            raise FileNotFoundError(textwrap.fill((
                "No pickled `coexist.Parameters` file found based on the "
                f"input filename: `{filename}_parameters.sim`."
            )))

        with open(f"{filename}_parameters.sim", "rb") as f:
            parameters = pickle.load(f)

        return LiggghtsSimulation(filename, parameters, verbose = verbose)


    def num_atoms(self):
        return self.simulation.get_natoms()


    def radii(self):
        # Get particle radii
        nlocal = self.simulation.extract_atom("nlocal", 0)[0]
        id_lig = self.simulation.extract_atom("id", 0)

        ids = np.array([id_lig[i] for i in range(nlocal)])
        radii = np.full(ids.max(), np.nan)

        radii_lig = self.simulation.extract_atom("radius", 2)

        for i in range(len(ids)):
            radii[ids[i] - 1] = radii_lig[i]

        return radii


    def set_density(self, particle_id, density):
        cmd = (f"set atom {particle_id + 1} density {density}")
        self.execute_command(cmd)


    def positions(self):
        # Get particle positions
        nlocal = self.simulation.extract_atom("nlocal", 0)[0]
        id_lig = self.simulation.extract_atom("id", 0)

        ids = np.array([id_lig[i] for i in range(nlocal)])
        pos = np.full((ids.max(), 3), np.nan)

        pos_lig = self.simulation.extract_atom("x", 3)

        for i in range(len(ids)):
            pos[ids[i] - 1, 0] = pos_lig[i][0]
            pos[ids[i] - 1, 1] = pos_lig[i][1]
            pos[ids[i] - 1, 2] = pos_lig[i][2]

        return pos


    def velocities(self):
        # Get particle velocities
        nlocal = self.simulation.extract_atom("nlocal", 0)[0]
        id_lig = self.simulation.extract_atom("id", 0)

        ids = np.array([id_lig[i] for i in range(nlocal)])
        vel = np.full((ids.max(), 3), np.nan)

        vel_lig = self.simulation.extract_atom("v", 3)

        for i in range(len(ids)):
            vel[ids[i] - 1, 0] = vel_lig[i][0]
            vel[ids[i] - 1, 1] = vel_lig[i][1]
            vel[ids[i] - 1, 2] = vel_lig[i][2]

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
                f"Tried to access non-existent variable {var_name}!"
            ))

        return varb


    def step(self, num_steps):
        # run simulation for `num_steps` timesteps
        if self.verbose:
            self.execute_command(f"run {num_steps}")
        else:
            self.execute_command(f"run {num_steps} post no")

        if self.save_vtk:
            self.write_vtk()


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

        if self.save_vtk:
            self.write_vtk()


    def step_time(self, time):
        # find timestep which can run exectly to time
        # while beeing lower then self.step_size
        new_dt = time / (int(time / self.step_size) + 1)
        steps = time / new_dt

        old_dt = self.step_size
        self.step_size = new_dt

        # Only save to VTK once (if defined), even though we're calling `step`
        # twice
        if self.save_vtk:
            old_save_vtk = self._save_vtk
            self._save_vtk = None

            self.step(steps)
            self._save_vtk = old_save_vtk
        else:
            self.step(steps)

        self.step_size = old_dt

        if self.save_vtk:
            self.write_vtk()


    def step_to_time(self, time):
        # run simulation up to sim time = `time`
        if time < self.time():
            raise ValueError(textwrap.fill(
                '''Time is below the current simulation time. Check input or
                reset the timestep!'''
            ))

        rest_time = (time - self.time()) % self.step_size

        n_steps = (time - self.time() - rest_time) / self.step_size

        def remainder_stepping():
            self.step(n_steps)
            if rest_time != 0.0:
                # Now run 1 single timestep with a smaller timestep
                old_dt = self.step_size

                # set step size to the rest time
                self.step_size = rest_time
                self.step(1)

                # reset to normal dt
                self.step_size = old_dt

        if self.save_vtk:
            old_save_vtk = self._save_vtk
            self._save_vtk = None

            remainder_stepping()
            self._save_vtk = old_save_vtk
        else:
            remainder_stepping()

        if self.save_vtk:
            self.write_vtk()


    def reset_time(self):
        # reset the current timestep to 0
        self.execute_command("reset_timestep 0")


    def timestep(self):
        # return the current timestep
        return self.simulation.extract_global("ntimestep", 0)


    def time(self):
        return self.simulation.extract_global("atime", 1)


    def execute_command(self, cmd):
        cmds = cmd.split("\n")

        for cmd in cmds:
            self.simulation.command(cmd)

        # If the command (excluding comments) contains any of our keywords,
        # save it (with comments) in the properties attribute
        for cmd in cmds:
            cmd_nc = cmd.split("#")[0]
            if self.finder_kw_save.search(cmd_nc) and not \
                    self.finder_kw_ignore.search(cmd_nc):
                self.properties.append(cmd)


    def copy(self, filename = None):
        """Create a copy the ligghts instance, including particle positions,
        velocities, properties and the simulated system.
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


    def write_vtk(self, dirname = None):
        if dirname is None and self.save_vtk is None:
            raise ValueError(textwrap.fill((
                "The input `dirname` was not set, in which case the value of "
                "`save_vtk` should be used. However, the attribute `save_vtk` "
                "was not set when instantiating this class. Set either the "
                "`dirname` input parameter or `save_vtk` class attribute."
            )))

        if dirname is not None:
            dirname = str(dirname)
        else:
            dirname = self.save_vtk

        # Find the current maximum VTK file index
        vtk_files = os.listdir(dirname)

        if len(vtk_files):
            index_splitter = re.compile(r"locations_|\.vtu")

            vtk_indices = []

            for f in vtk_files:
                vtk_split = index_splitter.split(f)

                if len(vtk_split) == 3:
                    vtk_indices.append(int(vtk_split[1]))

            if len(vtk_indices):
                vtk_index = np.array(vtk_indices).max()
            else:
                vtk_index = 0

        else:
            vtk_index = 0

        positions = np.asarray(self.positions(), order = "F")
        velocities = np.asarray(self.velocities(), order = "F")

        pointsToVTK(
            f"{dirname}/locations_{vtk_index + 1}",
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            data = dict(
                time = np.full(len(positions), self.time()),
                radius = self.radii(),
                velocity = np.linalg.norm(velocities, axis = 1),
                velocity_x = velocities[:, 0],
                velocity_y = velocities[:, 1],
                velocity_z = velocities[:, 2],
            )
        )


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

        # Modify the global variable name to reflect the change
        self.execute_command(f"variable {key} equal {value}")

        # First run any "variable ... equal ..." subcommands saved in the
        # Parameters.command column to save variables that might be substituted
        # in later
        commands = self.parameters.loc[key, "command"].split("\n")
        variable_command = re.compile(r"\s*variable.+equal.+")

        commands_nonvariable = []

        for cmd in commands:
            if variable_command.search(cmd):
                self.execute_command(cmd)
            else:
                commands_nonvariable.append(cmd)

        # Extracts variable LIGGGHTS substitutions, like ${corPP} => corPP
        variable_extractor = re.compile(r"\$\{|\}")

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
            r"\$\{\w+\}",
            replace_var,
            "\n".join(commands_nonvariable)
        )

        # Run the command with replaced varnames
        self.execute_command(cmd)

        # Set inner class parameter value
        self.parameters.at[key, "value"] = value


        docstr = (
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL
        name = self.__class__.__name__
        underline = "-" * len(name)

        return (
            f"{name}\n{underline}\n"
            f"  simulation:\n{self.simulation}\n\n"
            f"  parameters:\n{self.parameters}"
        )




class AutoTimestep():
    def __init__(
        self,
        youngs_modulus,
        particle_diameter,
        poissons_ratio,
        particle_density,
    ):
        self.youngs_modulus = float(youngs_modulus)
        self.particle_diameter = float(particle_diameter)
        self.poissons_ratio = float(poissons_ratio)
        self.particle_density = float(particle_density)


    def timestep(self):
        # Aliases
        r = 0.5 * self.particle_diameter
        pr = self.poissons_ratio
        rho = self.particle_density
        ym = self.youngs_modulus

        timestep = np.pi * r / (0.8766 + 0.163 * pr) * np.sqrt(
            2 * rho * (1 + pr) / ym
        )

        return timestep

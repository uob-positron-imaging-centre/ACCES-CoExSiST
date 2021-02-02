#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pyLiggghts.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2020


import  re
import  os
import  pickle
import  pathlib
import  textwrap
from    abc         import  ABC, abstractmethod

import  numpy       as      np
import  pandas      as      pd

from    tqdm        import  tqdm
from    pyevtk.hl   import  pointsToVTK

from    liggghts    import  liggghts




def to_vtk(
    dirname,
    positions,
    times = None,
    velocities = None,
    radii = None,
    verbose = True,
):
    # Type-checking inputs
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[2] < 3:
        raise ValueError(textwrap.fill((
            "The input `positions` must be a list of the simulated particles' "
            "locations - that is, a 3D array-like with axes (T, P, C), "
            "where T is the number of timesteps, P is the number of particles"
            ", and C are the (x, y, z) coordinates. Received an array with "
            f"shape `{positions.shape}`."
        )))

    if times is not None:
        times = np.asarray(times, order = "C")
        if times.ndim != 1 or len(times) != len(positions):
            raise ValueError(textwrap.fill((
                "The input `times` must be a 1D list of the times at which "
                "the particle locations were recorded. It should have the "
                "same length as the input `positions` array (= "
                f"{len(positions)}). Received an array with shape "
                f"`{times.shape}`."
            )))

    if velocities is not None:
        velocities = np.asarray(velocities)
        if velocities.ndim != 3 or velocities.shape[2] < 3 or \
                len(velocities) != len(positions):
            raise ValueError(textwrap.fill((
                "The input `velocities` must be a list of the simulated "
                "particles' dimension-wise velocities - that is, a 3D "
                "array-like with axes (T, P, V), where T is the number of "
                "timesteps (same as for the input `positions`), P is the "
                "number of particles, and V are the (v_x, v_y, v_z) "
                f"velocities. Received an array with shape "
                f"`{velocities.shape}`."
            )))

    if radii is not None:
        if hasattr(radii, "__iter__"):
            radii = np.asarray(radii)
        else:
            radii = np.ones(positions.shape[1]) * radii

        if radii.ndim != 1 or len(radii) != positions.shape[1]:
            raise ValueError(textwrap.fill((
                "The input `radii` must either be a single value for all the "
                "simulated particles, or a list of radii for each particle. "
                "In the latter case, it should have the same length as axis "
                f"1 of the input `positions` (= {positions.shape[1]}). "
                f"Received an array with shape {radii.shape}."
            )))

    # Create folder to save VTK files
    if not os.path.isdir(dirname):
        # Create folder recursively if folder is nested
        pathlib.Path(dirname).mkdir(parents = True, exist_ok = True)

    # Compute absolute velocities
    if velocities is not None:
        velocities_abs = np.linalg.norm(velocities, axis = 2)

    # Save each timestep in a file
    if verbose:
        positions = tqdm(positions)

    for i, pos in enumerate(positions):
        properties = dict()

        if times is not None:
            properties["time"] = np.ones(len(pos)) * times[i]

        if velocities is not None:
            vel = velocities
            properties["velocity"] = velocities_abs[i]
            properties["velocity_x"] = np.ascontiguousarray(vel[i][:, 0]),
            properties["velocity_y"] = np.ascontiguousarray(vel[i][:, 1]),
            properties["velocity_z"] = np.ascontiguousarray(vel[i][:, 2]),

        if radii is not None:
            properties["radius"] = radii

        pointsToVTK(
            os.path.join(dirname, f"locations_{i}"),
            np.ascontiguousarray(pos[:, 0]),
            np.ascontiguousarray(pos[:, 1]),
            np.ascontiguousarray(pos[:, 2]),
            data = properties,
        )




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

        parameters = coexist.Parameters(
            variables = ["fricPP", "corPP"],
            commands = [
                "fix  m4 all property/global coefficientFriction \
                    peratomtypepair 3   ${fricPP}   ${fricPW}   ${fricPSW}  \
                                        ${fricPW}   ${fric}     ${fric}     \
                                        ${fricPSW}  ${fric}     ${fric}     ",
                "fix  m3 all property/global coefficientRestitution \
                    peratomtypepair 3   ${corPP} ${corPP} ${corPP} \
                                        ${corPP} ${corPP} ${corPP} \
                                        ${corPP} ${corPP} ${corPP} ",
            ],
            values = [0.5, 0.5],
            minimums = [0.0, 0.0],
            maximums = [1.0, 1.0],
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
        *args,
        variables = [],
        commands = [],
        values = [],
        minimums = [],
        maximums = [],
        sigma = None,
        integers = [],
        **kwargs,
    ):
        '''`Parameters` class constructor.

        Parameters
        ----------
        variables: list[str], default []
            An iterable containing the LIGGGHTS variable names that will be
            used for changing simulation parameters.

        commands: list[str], default []
            An iterable containing the macro commands required to modify
            LIGGGHTS simulation parameters, containing the variable names
            as `${varname}`. E.g. `"fix  m1 all property/global youngsModulus
            peratomtype ${youngmodP} ${youngmodP} ${youngmodP}"`.

        values: list[float], default []
            An iterable containing the initial values for each LIGGGHTS
            simulation parameter.

        minimums: list[float], default []
            An iterable containing the lower bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`, in which case also
            define a `sigma0`.

        maximums: list[float], default []
            An iterable containing the upper bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`, in which case also
            define a `sigma0`.

        sigma: list[float], optional
            The standard deviation of the first population of solutions tried
            by the CMA-ES optimisation algorithm. If unset, it is computed as
            `0.2 * (maximum - minimum)`.

        integers: list[int], optional
            A list of the parameter indices that will be treated as integers.
            E.g. if the second parameter should be an integer, set `integers`
            to `[1]` (indexed from 0).
        '''

        pd.DataFrame.__init__(self, *args, **kwargs)

        if not (len(variables) == len(commands) == len(values) ==
                len(minimums) == len(maximums)):
            raise ValueError(textwrap.fill(
                '''The input iterables `variables`, `commands`,
                `values`, `minimums` and `maximums` must all have the same
                length.'''
            ))

        values = np.array(values, dtype = float)
        minimums = np.array(minimums, dtype = float)
        maximums = np.array(maximums, dtype = float)

        # Save which variables are integers as 1. and 0.
        integer_variables = np.zeros(len(values))
        for i in np.array(integers, dtype = int):
            integer_variables[i] = 1.

        if (minimums >= maximums).any():
            raise ValueError(textwrap.fill(
                '''Found value in `maximums` that was smaller or equal than the
                corresponding value in `minimums`.'''
            ))

        if sigma is None:
            sigma = 0.2 * (maximums - minimums)
        elif len(sigma) != len(variables):
            raise ValueError(textwrap.fill(
                '''If defined, `sigma` must have the same length as the other
                input parameters.'''
            ))

        for cmd in commands:
            if re.search(r"\$\{\w+\}", cmd) is None:
                raise ValueError(textwrap.fill(
                    '''The strings in the input `commands` must contain at
                    least one substring `"${varname}"` (e.g. `"fix  m2 all
                    property/global poissonsRatio peratomtype ${poissP}
                    ${poissP} ${poissP}"`), in which the right value will be
                    substituted when running the LIGGGHTS command.'''
                ))

        self["command"] = commands
        self["value"] = values
        self["min"] = minimums
        self["max"] = maximums
        self["sigma"] = sigma
        self["integer"] = integer_variables

        self.index = variables


    def copy(self, *args, **kwargs):
        parameters_copy = Parameters(
            variables = self.index.copy(),
            commands = self["command"].copy(),
            values = self["value"].copy(),
            minimums = self["min"].copy(),
            maximums = self["max"].copy(),
            sigma = self["sigma"].copy(),
            integers = [i for i, n in enumerate(self["integer"]) if n == 1.],
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




class Simulation(ABC):
    '''Abstract class defining the interface a DEM simulation engine must
    implement to be used by the `Coexist` and `Access` algorithms.

    This interface is needed in order to manage a DEM simulation whose
    parameters will be modified dynamically as the algorithms *learn* them.

    Alternatively, this interface has enough features to allow setting up,
    running, and analysing DEM simulations in Python.

    Each method required is described below. As an example of an implemented
    LIGGGHTS front-end, see `LiggghtsSimulation`.
    '''

    @property
    @abstractmethod
    def parameters(self):
        '''A `coexist.Parameters` class instance containing the free simulation
        parameters, along with their bounds.
        '''
        pass


    @property
    @abstractmethod
    def step_size(self):
        '''The time duration of a single simulated step.
        '''
        pass


    @abstractmethod
    def step(self, num_steps: int):
        '''Move the simulation forward in time for `num_steps` steps.
        '''
        pass


    @abstractmethod
    def step_to(self, step: int):
        '''Move the simulation forward in time up to timestep `step`.
        '''
        pass


    @abstractmethod
    def step_time(self, duration: float):
        '''Move the simulation forward in time for `duration` seconds.
        '''
        pass


    @abstractmethod
    def step_to_time(self, time: float):
        '''Move the simulation forward up to time `time`.
        '''
        pass


    @abstractmethod
    def time(self) -> float:
        '''Return the current simulation time.
        '''
        pass


    @abstractmethod
    def timestep(self) -> int:
        '''Return the current simulation timestep.
        '''
        pass


    @abstractmethod
    def num_atoms(self) -> int:
        '''Return the number of simulated particles.
        '''
        pass


    @abstractmethod
    def radii(self) -> np.ndarray:
        '''A 1D numpy array listing the radius of each particle.
        '''
        pass


    @abstractmethod
    def positions(self) -> np.ndarray:
        '''Return a 2D numpy array of the particle positions at the current
        timestep, where the columns represent the cartesian coordinates.
        '''
        pass


    @abstractmethod
    def velocities(self) -> np.ndarray:
        '''Return the 2D numpy array of the particle velocities at the current
        timestep, where the columns represent the velocity in the x-, y-, and
        z-dimensions.
        '''
        pass


    @abstractmethod
    def set_position(self, particle_index: int, position: np.ndarray):
        '''Set the 3D position of the particle at an index (indexed from 0).
        '''
        pass


    @abstractmethod
    def set_velocity(self, particle_index: int, velocity: np.ndarray):
        '''Set the 3D velocity of the particle at an index (indexed from 0).
        '''
        pass


    @abstractmethod
    def copy(self):
        '''Create a deep copy of the simulation, along with all its state. It
        is important to be a deep copy as it will be used in asynchronous
        contexts.
        '''
        pass


    @abstractmethod
    def save(self, filename: str):
        '''Save the full state of a simulation to a file.
        '''
        pass


    @staticmethod
    @abstractmethod
    def load(filename: str):      # -> Simulation
        '''Load the full state of a simulation from a file.
        '''
        pass


    @abstractmethod
    def __setitem__(self, key, value):
        '''A custom class attribute setter (i.e. called when using the
        subscript notation `simulation[key] = value`) that sets a simulation's
        free parameter value. The free parameter must already exist in the
        `parameters` property.
        '''
        pass




class LiggghtsSimulation(Simulation):
    '''Class encapsulating a single LIGGGHTS simulation whose parameters will
    be modified dynamically by a driver code.

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

        set_parameters: bool, default True
            If `True`, set the simulation parameter values when instantiating
            the class. Set to `False` to change values later; this is useful if
            some parameters can only be set once, e.g. particle insertion
            command for ACCESS.

        timestep: AutoTimestep, int, float or None, optional
            Variable that determines how the LIGGGHTS simulation timestep
            is managed. If it is `AutoTimestep`, it will be computed from the
            Rayleigh formula; if it is an `int` or `float`, that value will be
            used. Otherwise, the default value from the LIGGGHTS script is
            used.

        save_vtk: str, optional
            If defined, save particle data (positions, velocity, etc.) as VTK
            files for each timestep. This parameter should be the *directory
            name* for saving files in the format
            `{save_vtk}/locations_{timestep_index}.vtk`.

        verbose: bool, default `True`
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
        filename: str, optional
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
        radii = self.simulation.gather_atoms("radius", 1, 1)
        return np.array(list(radii)).reshape(self.num_atoms(), -1)


    def set_density(self, particle_id, density):
        cmd = (f"set atom {particle_id + 1} density {density}")
        self.execute_command(cmd)


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

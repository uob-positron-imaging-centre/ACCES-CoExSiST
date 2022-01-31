#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2020


import  re
import  os
import  pickle
import  textwrap
from    abc         import  ABC, abstractmethod

import  numpy       as      np
import  pandas      as      pd

from    tqdm        import  tqdm
from    pyevtk.hl   import  pointsToVTK




def save(path, obj):
    '''Save Coexist object using the binary Pickle format.
    '''
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj




def load(path):
    '''Load Coexist object from a binary Pickle-formatted file.
    '''
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj




def to_vtk(
    dirname,
    positions,
    *,
    times = None,
    velocities = None,
    radii = None,
    verbose = True,
):
    '''Export particle `positions` and optionally `times`, `velocities` and
    `radii` to a folder `dirname` in the modern binary VTK format.

    Parameters
    ----------
    dirname : str
        A path to the directory to save particle data to.

    positions : (T,) list[(P, 3) np.ndarray]
        A list-like of particle locations at each timestep T; each list entry
        represents a single timestep and should contain a 2D NumPy array with
        columns formatted as [x, y, z] - each row represents a particle P.

    times : (T,) list[float], optional
        A list of times for each timestep given in `positions`.

    velocities : (T,) list[(P, 3) np.ndarray], optional
        Same as `positions`, a list of T timesteps with each entry being a 2D
        NumPy array containing P rows and 3 columns [vx, vy, vz].

    radii : (T,) list[(P,) np.ndarray], optional
        A list of length T (for T timesteps) with each entry containing a 1D
        NumPy array containing the radii of the P particles.

    verbose : bool, default True
        Print extra information while saving.
    '''

    # Type-checking inputs
    num_timesteps = len(positions)
    for i in range(num_timesteps):
        positions[i] = np.asarray(positions[i])

        if positions[i].ndim != 2 or positions[i].shape[1] != 3:
            raise ValueError(textwrap.fill((
                "The input `positions` must be a list of the simulated "
                "particles' locations - that is, a 3D array-like with axes "
                "(T, P, C), where T is the number of timesteps, P is the "
                "number of particles, and C are the (x, y, z) coordinates. "
                f"Received an list with length {num_timesteps} where at "
                f"timestep index {i} the array of particle positions had "
                f"shape `{positions[i].shape}`."
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
        for i in range(num_timesteps):
            velocities[i] = np.asarray(velocities[i])

            if velocities[i].ndim != 2 or velocities[i].shape[1] != 3 or \
                    len(velocities) != len(positions) or \
                    len(velocities[i]) != len(positions[i]):
                raise ValueError(textwrap.fill((
                    "The input `velocities` must be a list of the simulated "
                    "particles' dimension-wise velocities - that is, a 3D "
                    "array-like with axes (T, P, V), where T is the number of "
                    "timesteps (same as for the input `positions`), P is the "
                    "number of particles, and V are the (vx, vy, vz) "
                    "velocities. "
                    f"Received a list with length {num_timesteps} where at "
                    f"timestep index {i} the array of particle velocities had "
                    f"shape `{velocities[i].shape}`."
                )))

    if radii is not None:
        for i in range(num_timesteps):
            radii[i] = np.asarray(radii[i])

            if radii[i].ndim != 1 or len(radii) != len(positions) or \
                    len(radii[i]) != len(positions[i]):
                raise ValueError(textwrap.fill((
                    "The input `radii` must be a list of the simulated "
                    "particles' radii - that is, a 2D array-like with axes "
                    "(T, R), where T is the number of timesteps (same as for "
                    "the input `positions`) and R are the radii for each "
                    "particles."
                    f"Received a list with length {num_timesteps} where at "
                    f"timestep index {i} the array of particle radii had "
                    f"shape `{radii[i].shape}`."
                )))

    # If it doesn't exist, create directory to save VTK files
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Save each timestep in a file
    if verbose:
        positions = tqdm(positions)

    for i, pos in enumerate(positions):
        properties = dict()

        if times is not None:
            properties["time"] = np.ones(len(pos)) * times[i]

        if velocities is not None:
            vel = velocities[i]
            properties["velocity"] = (
                np.ascontiguousarray(vel[:, 0]),
                np.ascontiguousarray(vel[:, 1]),
                np.ascontiguousarray(vel[:, 2]),
            )

        if radii is not None:
            properties["radius"] = radii[i]

        pointsToVTK(
            os.path.join(dirname, f"locations_{i}"),
            np.ascontiguousarray(pos[:, 0]),
            np.ascontiguousarray(pos[:, 1]),
            np.ascontiguousarray(pos[:, 2]),
            data = properties,
        )




def create_parameters(
    variables = [],
    minimums = [],
    maximums = [],
    values = None,
    sigma = None,
    **kwargs,
):
    '''Create a ``pandas.DataFrame`` storing ``Access`` free parameters' names,
    bounds, and optionally starting values and relative uncertainty.

    This is simply a helper returning a ``pandas.DataFrame`` with the format
    required by e.g. ``coexist.Access``.

    Only the `variables`, `minimums` and `maximums` are necessary. If unset,
    the initial ``values`` are set to halfway between the lower and upper
    bounds; the initial standard deviation ``sigma`` is set to 40% of this
    range, so that the entire space is explored.

    Parameters
    ----------
    variables : list[str], default []
        A list of the free parameters' names.

    minimums : list[float], default []
        A list with the same length as ``variables`` storing the lower bound
        for each corresponding variable.

    maximums : list[float], default []
        A list with the same length as ``variables`` storing the lower bound
        for each corresponding variable.

    values : list[float], optional
        The optimisation starting values; not essential as ACCES samples the
        space randomly anyways. If unset, they are set to halfway between
        ``minimums`` and ``maximums``.

    sigma : list[float], optional
        The initial standard deviation in each variable, setting how far away
        from the initial ``values`` the parameters will be sampled; the
        sampling is Gaussian. If unset, they are set to 40% of the data range
        (i.e. ``maximums`` - ``minimums``) so that the entire space is
        initially explored. ACCES will adapt and minimise this uncertainty.

    **kwargs : other keyword arguments
        Other columns to include in the returned parameters DataFrame, given
        as other lists with the same length as ``variables``.

    Returns
    -------
    pandas.DataFrame
        A table storing the intial ``value``, ``min``, ``max`` and ``sigma``
        (columns) for each free parameter (rows).

    Examples
    --------
    Create a DataFrame storing two free parameters, specifying the minimum and
    maximum bounds; notice that the starting guess and uncertainty are set
    automatically.

    >>> import coexist
    >>> parameters = coexist.create_parameters(
    >>>     variables = ["cor", "separation"],
    >>>     minimums = [-3, -7],
    >>>     maximums = [+5, +3],
    >>> )
    >>> parameters
                value  min  max  sigma
    cor           1.0 -3.0  5.0    3.2
    separation   -2.0 -7.0  3.0    4.0
    '''

    if not (len(variables) == len(minimums) == len(maximums)) or \
            (values is not None and len(values) != len(variables)) or \
            (sigma is not None and len(sigma) != len(variables)):
        raise ValueError(textwrap.fill(
            '''The input iterables `variables`, `minimums`, `maximums`,
            `values` and `sigma` (if defined), must all have the same length.
            '''
        ))

    minimums = np.array(minimums, dtype = float)
    maximums = np.array(maximums, dtype = float)

    if values is None:
        values = (maximums + minimums) / 2
    else:
        values = np.array(values, dtype = float)

    if (minimums >= maximums).any():
        raise ValueError(textwrap.fill(
            '''Found value in `maximums` that was smaller or equal than the
            corresponding value in `minimums`.'''
        ))

    if sigma is None:
        sigma = 0.4 * (maximums - minimums)

    parameters = pd.DataFrame(
        data = {
            "value": values,
            "min": minimums,
            "max": maximums,
            "sigma": sigma,
            **kwargs,
        },
        index = variables,
    )

    return parameters




class Parameters(pd.DataFrame):
    '''Pandas DataFrame subclass with a custom constructor for dynamically
    changing LIGGGHTS simulation parameters - for `Coexist`.

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
        values = None,
        minimums = [],
        maximums = [],
        sigma = None,
        **kwargs,
    ):
        '''`Parameters` class constructor.

        Parameters
        ----------
        variables : list[str], default []
            An iterable containing the LIGGGHTS variable names that will be
            used for changing simulation parameters.

        commands : list[str], default []
            An iterable containing the macro commands required to modify
            LIGGGHTS simulation parameters, containing the variable names
            as `${varname}`. E.g. `"fix  m1 all property/global youngsModulus
            peratomtype ${youngmodP} ${youngmodP} ${youngmodP}"`.

        values : list[float], default []
            An iterable containing the initial values for each LIGGGHTS
            simulation parameter.

        minimums : list[float], default []
            An iterable containing the lower bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`, in which case also
            define a `sigma0`.

        maximums : list[float], default []
            An iterable containing the upper bounds for each LIGGGHTS
            parameter. For non-existing bounds, use `None`, in which case also
            define a `sigma0`.

        sigma : list[float], optional
            The standard deviation of the first population of solutions tried
            by the CMA-ES optimisation algorithm. If unset, it is computed as
            `0.2 * (maximum - minimum)`.
        '''

        pd.DataFrame.__init__(self, *args, **kwargs)

        if not (len(variables) == len(commands) == len(minimums) ==
                len(maximums)) or \
                (values is not None and len(values) != len(variables)):
            raise ValueError(textwrap.fill(
                '''The input iterables `variables`, `commands`, `values` (if
                defined), `minimums` and `maximums` must all have the same
                length.'''
            ))

        minimums = np.array(minimums, dtype = float)
        maximums = np.array(maximums, dtype = float)

        if values is None:
            values = (maximums + minimums) / 2
        else:
            values = np.array(values, dtype = float)

        if (minimums >= maximums).any():
            raise ValueError(textwrap.fill(
                '''Found value in `maximums` that was smaller or equal than the
                corresponding value in `minimums`.'''
            ))

        if sigma is None:
            sigma = 0.4 * (maximums - minimums)
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

        self.index = variables


    def copy(self, *args, **kwargs):
        parameters_copy = Parameters(
            variables = self.index.copy(),
            commands = self["command"].copy(),
            values = self["value"].copy(),
            minimums = self["min"].copy(),
            maximums = self["max"].copy(),
            sigma = self["sigma"].copy(),
        )

        return parameters_copy




class Experiment:
    '''Class encapsulating an experiment's recorded particle positions at
    multiple timesteps for `Coexist`.

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
        times : list[float]
            A list (or vector) of timestamp values for all positions recorded
            in the experiment.

        positions_all : (T, N, M>=3) numpy.ndarray
            A 3D array-like with dimensions (T, N, M>=3), where T is the number
            of timesteps (corresponding exactly to the length of `times`), N is
            the number of particles in the system and M is the number of data
            columns per particle (at least 3 for their [x, y, z] coordinates,
            but can contain extra columns).

        resolution : float
            The expected error on the positions recorded.

        kwargs : keyword arguments
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

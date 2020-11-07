#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimisation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import  textwrap

import  numpy               as      np
from    scipy.integrate     import  solve_ivp
import  sympy

from    coexist             import  Simulation, Experiment


class Coexist:

    def __init__(
        self,
        simulation: Simulation,
    ):
        '''`Coexist` class constructor.

        Parameters
        ----------
        simulation: coexist.Simulation
            The DEM simulation, encapsulated in a `coexist.Simulation`
            instance.

        '''

        # Type-checking inputs
        if not isinstance(simulation, Simulation):
            raise TypeError(textwrap.fill((
                "The `simulation` input parameter must be an instance of "
                f"`coexist.Simulation`. Received type `{type(simulation)}`."
            )))

        # Setting class attributes
        self.simulation = simulation

        # Vector of bools corresponding to whether sim-exp particles are
        # attached or detached (i.e. sim particles are forcibly moved)
        self.attached = np.full(self.simulation.num_atoms(), True)

        # The minimum number of collisions detected in the experimental dataset
        # based on the particle positions across multiple timesteps
        self.collisions = np.zeros(self.simulation.num_atoms(), dtype = int)

        # A vector counting how many times an experimental time was used
        # for optimising DEM parameters. Will be initialised in `self.learn`
        # for the timesteps given in an experiment
        self.optimised_times = None

        self.xi = 0         # Current timestep's error
        self.xi_acc = 0     # Multiple timesteps' (accumulated) error


    def learn(
        self,
        experiment: Experiment,
        max_optimisations: 3,
        verbose = True,
    ):
        '''Start synchronising the DEM simulation against a given experimental
        dataset, learning the required DEM parameters.

        Parameters
        ----------
        experiment: coexist.Experiment
            The experimental positions recorded, encapsulated in a
            `coexist.Experiment` instance.
        '''

        # Type-checking inputs
        if not isinstance(experiment, Experiment):
            raise TypeError(textwrap.fill((
                "The `experiment` input parameter must be an instance of "
                f"`coexist.Experiment`. Received type `{type(experiment)}`."
            )))

        # Ensure there is the same number of particles in the sim and exp
        if self.simulation.num_atoms() != experiment.positions_all.shape[1]:
            raise ValueError(textwrap.fill((
                "The input `experiment` does not have the same number of "
                f"particles as the simulation ({self.simulation.num_atoms()})."
                f" The experiment has {experiment.positions_all.shape[1]} "
                "particles."
            )))

        # Setting class attributes
        self.experiment = experiment
        self.optimised_times = np.zeros(
            len(self.experiment.times),
            dtype = int,
        )

        # Aliases
        sim = self.simulation
        exp = self.experiment

        # First set the simulated particles' positions to the experimental ones
        # at t = t0. Note: in the beginning all particles are attached.
        self.move_attached(0)   # Sets positions of attached particles

        # Initial checkpoint
        sim.save()

        # First try inferring the particles' initial velocities, detaching the
        # ones we can
        self.try_detach(0)      # Sets velocities of detached particles

        # Forcibly move the attached particles' positions
        self.move_attached(0)   # Sets positions of attached particles

        # Start moving the simulation in sync with the recorded experimental
        # times in self.experiment.times. Start from the second timestep
        # because the first one was used to set the simulated particles'
        # positions and velocities
        for i, t in enumerate(exp.times):
            # Skip first timestep as it was already used for initial positions
            if i == 0:
                continue

            sim.step_to_time(t)

            sim_pos = sim.positions()
            exp_pos = exp.positions_all[i]

            self.xi = self.calc_xi(sim_pos, exp_pos)
            self.xi_acc += self.xi

            # Only save checkpoints if the simulation is not diverging. A
            # checkpoint is a saved simulation state that will be used for
            # optimisation (which is done over multiple timesteps)
            if not self.diverging():
                sim.save()

                # Keep track of the index of the last saved checkpoint for
                # future optimisations
                start_index = i
                end_index = i

            if verbose:
                print((
                    f"i: {i:>4} | "
                    f"xi: {self.xi:5.5e} | "
                    f"xi_acc: {self.xi_acc:5.5e} | "
                    f"num_checkpoints: {(end_index - start_index):>4} | "
                    f"attached: {self.attached} / {sim.num_atoms()}"
                ))

            if self.optimisable():
                pass

            # If the last saved checkpoint was used too many times for
            # optimisation, reattach all particles and start again from the
            # current timestep.
            if self.optimised_times[start_index] >= max_optimisations:
                self.attached[:] = True
                self.move_attached(i)

                sim.save()
                start_index = i
                end_index = i

                if verbose:
                    print(textwrap.fill((
                        "Maximum optimisation runs reached, reattached all "
                        "particles and moved first checkpoint to timestep "
                        f"index {i} (t = {exp.times[i]})."
                    )))

            # At the end of a timestep, try to detach the remaining particles;
            # if not possible, forcibly move the remaining attached particles
            self.try_detach(i)
            self.move_attached(i)


    def try_detach(self, time_index):
        '''Try detaching simulated particles from their experimental
        equivalents.

        In order to "detach" a particle (i.e. let it move freely, as predicted
        by the simulation, instead of setting its position to the experiment),
        we need to infer its velocity between two consecutive timesteps, in
        which *it didn't collide with anything*.

        For any single particle, if three consecutive experimental locations
        are parabolically colinear, then no collision occured. Therefore, its
        velocity between the two intervals is reliable, hence we can detach it.

        Detaching means setting the particle's index in `self.attached` to
        `False`.

        Parameters
        ----------
        time_index: int
            The time index in `self.experiment.times` for which the particles
            should be detached.

        Raises
        ------
        ValueError
            If the input `time_index` is invalid: either smaller than 0 or too
            close to the end of the simulation (we need 4 consecutive
            timesteps to compute predicted positions).
        '''

        # Three points A, B, C are colinear if AB + BC = AC; in order to allow
        # for some variance due to gravity / buoyancy, check AB + BC > 0.95 A
        pos_all = self.experiment.positions_all
        times = self.experiment.times

        # First check there are at least four recorded positions remaining
        time_index = int(time_index)
        if time_index > len(pos_all) - 4 or time_index < 0:
            raise ValueError(textwrap.fill((
                "The `time_index` input parameter must be between 0 and "
                f"`len(experiment.positions_all) - 4` = {len(pos_all) - 4}, "
                "so that at least 4 recorded positions remain to check "
                "against."
            )))

        t0 = times[time_index]
        p0 = pos_all[time_index]

        t1 = times[time_index + 1]
        p1 = pos_all[time_index + 1]

        t2 = times[time_index + 2]
        p2 = pos_all[time_index + 2]

        t3 = times[time_index + 2]
        p3 = times[time_index + 2]

        # Based on the particle positions at t0, t1 and t2, predict their
        # positions at t3; if the prediction is correct, the inferred velocity
        # at t0 is reliable => detach particle
        p3_predicted, u0 = self.predict_positions(t0, p0, t1, p1, t2, p2, t3)

        # Euclidean distance between predicted p(t3) and actual p(t3)
        error = np.linalg.norm(p3_predicted - p3, axis = 1)

        # If the positional error is smaller than measurement error on the
        # particles' position (the experiment's resolution), set the particles'
        # velocity at t0 to u0 and detach it
        resolution = self.experiment.resolution
        self.attached[error < resolution] = False

        # Only change the collisions counter if this timestep was not already
        # used for optimisation
        if self.optimised_times[time_index] == 0:
            # If the positional error is larger, at least a collision has
            # ocurred in the timesteps we looked at
            self.collisions[error >= resolution] += 1

            # We only care about collisions if particles are detached. Set the
            # collision counter for attached particles to 0. Note:
            # `self.attached` is a vector of True and False, so use it directly
            # for indexing
            self.collisions[self.attached] = 0

        # TODO: set velocity functions
        # TODO: after optimisation set collisions to 0.
        # TODO: set optimised_times to True after optimisation



    @staticmethod
    def predict_positions(
        t0, p0,
        t1, p1,
        t2, p2,
        t3,
        g = 9.81,
        rho_p = 2500,
        rho_f = 1.225,
    ):

        # Type-check input parameters
        t0 = float(t0)
        t1 = float(t1)
        t2 = float(t2)
        t3 = float(t3)

        p0 = np.asarray(p0)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        if not (p0.ndim == p1.ndim == p2.ndim and
                len(p0) == len(p1) == len(p2)) or \
                p0.shape[1] < 3 or p1.shape[1] < 3 or p2.shape[1] < 3:
            raise ValueError(textwrap.fill((
                "The positions input arrays p0, p1 and p2 should all have the "
                "same 2D shape. They represent the positions of all particles "
                "in the system at t = t0, t1 and t2, respectively. Their "
                "shape should be (N, 3), where N is the number of particles, "
                "plus three columns for their x, y, z coordinates."
            )))

        num_particles = len(p0)

        # The analytical expression was derived using SymPy in the
        # `coexist.optimisation.ballistic_approx` function
        def A(t):
            return t * t0 - (t * t + t0 * t0) / 2

        def B(t):
            return t - t0

        # The LHS of the system of 2 equations to solve *for a single particle*
        lhs_eq = np.array([
            [A(t1), B(t1)],
            [A(t2), B(t2)],
        ])

        # Stack the same equation for all particles
        lhs_eqs = np.tile(lhs_eq, (num_particles, 1, 1))

        # The RHS of the system of 2 equations is a stacked 2D array of
        # [x(t1) - x0, x(t2) - x0] for every particle
        # => shape (num_particles, 2)
        rhs_x = np.vstack((p1[:, 0] - p0[:, 0], p2[:, 0] - p0[:, 0])).T
        rhs_y = np.vstack((p1[:, 1] - p0[:, 1], p2[:, 1] - p0[:, 1])).T
        rhs_z = np.vstack((
            p1[:, 2] - p0[:, 2] - g * (1 - rho_f / rho_p) * A(t1),
            p2[:, 2] - p0[:, 2] - g * (1 - rho_f / rho_p) * A(t2),
        )).T

        # Solve the stacked equations to get drag acceleration (col 1) and
        # initial velocities (col 2)
        adx_ux0 = np.linalg.solve(lhs_eqs, rhs_x)
        ady_uy0 = np.linalg.solve(lhs_eqs, rhs_y)
        adz_uz0 = np.linalg.solve(lhs_eqs, rhs_z)

        # Now sub back into equation to find x, y, z at t3
        x3 = p0[:, 0] + adx_ux0[:, 0] * A(t3) + adx_ux0[:, 1] * B(t3)
        y3 = p0[:, 1] + ady_uy0[:, 0] * A(t3) + ady_uy0[:, 1] * B(t3)
        z3 = p0[:, 2] + adz_uz0[:, 0] * A(t3) + adz_uz0[:, 1] * B(t3)

        p3 = np.vstack((x3, y3, z3)).T
        u0 = np.vstack((adx_ux0[:, 1], ady_uy0[:, 1], adz_uz0[:, 1])).T

        return p3, u0


    def move_attached(self, time_index):
        '''Forcibly move the attached simulated particles' to the recorded
        experimental particles' positions at time `time_index`.

        Parameters
        ----------
        time_index: int
            The time index in `self.experiment.times` for which the particles
            should be detached.

        Raises
        ------
        ValueError
            If the input `time_index` is invalid: either smaller than 0 or
            larger than `len(self.experiment.times)`.
        '''
        # Check input `time_index` is valid
        time_index = int(time_index)
        if time_index >= len(self.experiment.times) or time_index < 0:
            raise ValueError(textwrap.fill((
                "The `time_index` input parameter must be between 0 and "
                f"`len(experiment.times)` = {len(self.experiment.times)}. "
                f"Received {time_index}."
            )))

        # TODO: set positions functions


    @staticmethod
    def calc_xi(sim_pos, exp_pos):
        return np.linalg.norm(sim_pos - exp_pos, axis = 1).sum()


    def diverging(self):




def ballistic():
    '''Derive the analytical expression of a single particle's trajectory
    when only gravity, buoyancy and drag act on it.

    For the z-dimension, its solution is implicit and computationally expensive
    to find - you can copy and run this code with `uzt = ...` commented out.
    '''

    from sympy import Symbol, Function, Eq

    # Nice printing to terminal
    sympy.init_printing()

    # Define symbols and functions used in integration
    t = Symbol("t")
    t0 = Symbol("t0")

    ux = Function('ux')
    ux0 = Symbol("ux0")
    dux_dt = ux(t).diff(t)

    uy = Function('uy')
    uy0 = Symbol("uy0")
    duy_dt = uy(t).diff(t)

    uz = Function('uz')
    uz0 = Symbol("uz0")
    duz_dt = uz(t).diff(t)

    cd = Symbol("cd")           # Drag coefficient
    d = Symbol("d")             # Particle diameter
    g = Symbol("g")             # Gravitational acceleration
    rho_p = Symbol("rho_p")     # Particle density
    rho_f = Symbol("rho_f")     # Fluid density

    # Define expressions to integrate for velocities in each dimension
    # Eq: d(ux) / dt = expr...
    ux_eq = Eq(dux_dt, -3 / (4 * d) * cd * ux(t) ** 2)
    uy_eq = Eq(duy_dt, -3 / (4 * d) * cd * uy(t) ** 2)
    uz_eq = Eq(duz_dt, -3 / (4 * d) * cd * uz(t) ** 2 - g + rho_f * g / rho_p)

    uxt = sympy.dsolve(
        ux_eq,
        ics = {ux(t0): ux0},
    )

    uyt = sympy.dsolve(
        uy_eq,
        ics = {uy(t0): uy0},
    )

    # This takes a few minutes and returns a nasty implicit solution!
    uzt = sympy.dsolve(
        uz_eq,
        ics = {uz(t0): uz0},
    )

    return uxt, uyt, uzt




def ballistic_approx():
    '''Derive the analytical expression of a single particle's trajectory
    when only gravity, buoyancy and *constant drag* act on it.

    For small changes in velocity, the drag is effectively constant,
    simplifying the solution tremendously. Given three points, it would be
    possible to infer the value of drag.
    '''

    from sympy import Symbol, Function, Eq

    # Nice printing to terminal
    sympy.init_printing()

    # Define symbols and functions used in integration
    t = Symbol("t")
    t0 = Symbol("t_0")

    ux = Function('u_x')
    ux0 = Symbol("u_x0")
    dux_dt = ux(t).diff(t)

    uy = Function('u_y')
    uy0 = Symbol("u_y0")
    duy_dt = uy(t).diff(t)

    uz = Function('u_z')
    uz0 = Symbol("u_z0")
    duz_dt = uz(t).diff(t)

    adx = Symbol("a_dx")        # Acceleration due to drag in the x-direction
    ady = Symbol("a_dy")        # Acceleration due to drag in the y-direction
    adz = Symbol("a_dz")        # Acceleration due to drag in the z-direction

    g = Symbol("g")             # Gravitational acceleration
    rho_p = Symbol("rho_p")     # Particle density
    rho_f = Symbol("rho_f")     # Fluid density

    # Define expressions to integrate for velocities in each dimension
    # Eq: d(ux) / dt = expr...
    ux_eq = Eq(dux_dt, -adx)        # Some constant acceleration due to drag
    uy_eq = Eq(duy_dt, -ady)
    uz_eq = Eq(duz_dt, -adz - g + rho_f * g / rho_p)

    # Solve equations of motion to find analytical expressions for velocities
    # in each dimension as functions of time
    uxt = sympy.dsolve(
        ux_eq,
        ics = {ux(t0): ux0},
    )

    uyt = sympy.dsolve(
        uy_eq,
        ics = {uy(t0): uy0},
    )

    uzt = sympy.dsolve(
        uz_eq,
        ics = {uz(t0): uz0},
    )

    # Use expressions for velocities to derive positions as functions of time
    x = Function("x")
    x0 = Symbol("x_0")
    dx_dt = x(t).diff(t)

    y = Function("y")
    y0 = Symbol("y_0")
    dy_dt = y(t).diff(t)

    z = Function("z")
    z0 = Symbol("z_0")
    dz_dt = z(t).diff(t)

    # Define expressions to integrate for positions in each dimension
    # Eq: d(x) / dt = expr...
    x_eq = Eq(dx_dt, uxt.rhs)
    y_eq = Eq(dy_dt, uyt.rhs)
    z_eq = Eq(dz_dt, uzt.rhs)

    # Solve to find particle positions wrt. time
    xt = sympy.dsolve(
        x_eq,
        ics = {x(t0): x0},
    )

    yt = sympy.dsolve(
        y_eq,
        ics = {y(t0): y0},
    )

    zt = sympy.dsolve(
        z_eq,
        ics = {z(t0): z0},
    )

    return xt, yt, zt




if __name__ == "__main__":
    pass

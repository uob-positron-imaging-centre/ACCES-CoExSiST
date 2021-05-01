#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import  os
import  sys
import  textwrap
import  subprocess
import  pickle
from    datetime            import  datetime
from    concurrent.futures  import  ProcessPoolExecutor

import  numpy               as      np
import  pandas              as      pd
import  cma

import  coexist
from    coexist             import  Simulation, Experiment
from    coexist             import  schedulers

from    .code_trees         import  code_contains_variable
from    .code_trees         import  code_substitute_variable




class Coexist:

    def __init__(
        self,
        simulation: Simulation,
        save_log = False,
    ):
        '''`Coexist` class constructor.

        Parameters
        ----------
        simulation: coexist.Simulation
            The DEM simulation, encapsulated in a `coexist.Simulation`
            instance.

        save_log: bool, default False
            If true, save details about the optimisations runs executed,
            including the parameter combinations tried.

        '''

        # Type-checking inputs
        if not isinstance(simulation, Simulation):
            raise TypeError(textwrap.fill((
                "The `simulation` input parameter must be an instance of "
                f"`coexist.Simulation`. Received type `{type(simulation)}`."
            )))

        self.save_log = bool(save_log)
        self.log = [] if self.save_log else None

        # Setting class attributes
        self.simulation = simulation
        self.experiment = None          # Set in `self.learn()`

        self.rng = None

        # Vector of bools corresponding to whether sim-exp particles are
        # attached or detached (i.e. sim particles are forcibly moved or not)
        self.attached = np.full(self.simulation.num_atoms(), True)

        # The minimum number of collisions detected in the experimental dataset
        # based on the particle positions across multiple timesteps.
        # Have a counter at each timestep
        self.collisions = None      # Set in `self.learn()`

        # A vector counting how many times an experimental time was used
        # for optimising DEM parameters. Will be initialised in `self.learn`
        # for the timesteps given in an experiment
        self.optimised_times = None
        self.optimisation_runs = 0

        self.xi = 0                 # Current timestep's error
        self.xi_acc = 0             # Multiple timesteps' (accumulated) error

        self.diverged = False
        self.num_solutions = None       # Set in `self.learn()`
        self.target_sigma = None        # Set in `self.learn()`
        self.min_checkpoints = None     # Set in `self.learn()`

        # Capture what is printed to stderr by spanwed OS processes
        self._stderr = None


    def learn(
        self,
        experiment: Experiment,
        max_optimisations = 3,
        num_solutions = 8,
        target_sigma = 0.1,
        min_checkpoints = 5,
        random_seed = None,
        verbose = True,
    ):
        '''Start synchronising the DEM simulation against a given experimental
        dataset, learning the required DEM parameters.

        Parameters
        ----------
        experiment: coexist.Experiment
            The experimental positions recorded, encapsulated in a
            `coexist.Experiment` instance.

        max_optimisations: int, default 3
            The maximum number of optimisation runs to execute before
            reattaching particles.

        num_solutions: int, default 8
            The number of parameter combinations to try at each optimisation
            step. A larger number corresponds to a more "global" parameter
            space search, at the expense of longer runtime. The more
            parameters you optimise at the same time, the larger this
            value should be (~ 4 * num_parameters).

        target_sigma: float, default 0.1
            The target overall scaled standard deviation in the solution - i.e.
            the trust interval at which to stop the optimisation. The smaller
            the `target_sigma`, the more solutions are tried.

        verbose: bool, default True
            If True, print information about the optimisation to the terminal.

        '''

        # Type-checking inputs and setting class attributes
        if not isinstance(experiment, Experiment):
            raise TypeError(textwrap.fill((
                "The `experiment` input parameter must be an instance of "
                f"`coexist.Experiment`. Received type `{type(experiment)}`."
            )))

        self.experiment = experiment

        # Ensure there is the same number of particles in the sim and exp
        if self.simulation.num_atoms() != experiment.positions_all.shape[1]:
            raise ValueError(textwrap.fill((
                "The input `experiment` does not have the same number of "
                f"particles as the simulation ({self.simulation.num_atoms()})."
                f" The experiment has {experiment.positions_all.shape[1]} "
                "particles."
            )))

        self.max_optimisations = int(max_optimisations)
        self.optimisation_runs = 0

        self.num_solutions = int(num_solutions)
        self.target_sigma = float(target_sigma)
        self.min_checkpoints = int(min_checkpoints)

        self.rng = np.random.default_rng(random_seed)
        self.verbose = bool(verbose)

        self.collisions =  np.zeros(
            (len(self.experiment.times), self.simulation.num_atoms()),
            dtype = int,
        )

        self.optimised_times = np.zeros(
            len(self.experiment.times),
            dtype = int,
        )

        # Aliases
        sim = self.simulation
        exp = self.experiment

        # Save the current simulation state in a `restarts` folder
        if not os.path.isdir("coexist_info"):
            os.mkdir("coexist_info")

        with open("coexist_info/experiment.pickle", "wb") as f:
            pickle.dump(exp, f)

        with open("coexist_info/simulation_class.pickle", "wb") as f:
            pickle.dump(sim.__class__, f)

        # First set the simulated particles' positions to the experimental ones
        # at t = t0. Note: in the beginning all particles are attached.
        self.move_attached(0)   # Sets positions of attached particles

        # Initial checkpoint
        sim.save("coexist_info/main")

        # First try inferring the particles' initial velocities, detaching the
        # ones we can
        self.try_detach(0)      # Sets velocities of detached particles

        # Forcibly move the attached particles' positions
        self.move_attached(0)   # Sets positions of attached particles

        # Optimisation start / end timestep indices
        start_index = 0
        end_index = 0

        # Flag to keep track of divergence between sim and exp
        self.diverged = False

        # Nice printing of optimisation parameters
        if verbose:
            print((
                "    i |    time    |     xi     |   xi_acc   | checkpoints | "
                "  attached   | max_collisions \n"
                "------+------------+------------+------------+-------------+-"
                "-------------+----------------"
            ))

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
            if not self.diverging(i):
                sim.save("coexist_info/main")

                # Keep track of the index of the last saved checkpoint for
                # future optimisations
                start_index = i
                end_index = i

            if verbose:
                print((
                    f" {i:>4} | "
                    f" {sim.time():5.3e} | "
                    f" {self.xi:5.3e} | "
                    f" {self.xi_acc:5.3e} | "
                    f" {(i - start_index):>10} | "
                    f" {self.attached.sum():>4} / {sim.num_atoms():>4} | "
                    f" {self.collisions[i].max():>10}"
                ))

            if self.optimisable(i, start_index):
                # Load previous checkpoint's simulation
                parameters_save = self.simulation.parameters

                self.simulation = self.simulation.load("coexist_info/main")
                sim = self.simulation

                sim.parameters["value"] = parameters_save["value"]
                sim.parameters["sigma"] = parameters_save["sigma"]

                # Optimise against experimental positions from
                # start_index:end_index (excluding end_index)
                end_index = i + 1

                # Re-iterate the steps since the last checkpoint, optimising
                # the DEM parameters. This function will also run the sim up to
                # the current time t and set the new xi_acc
                self.optimise(start_index, end_index, verbose = verbose)

                # After optimisation, reset `collisions`, increment
                # `optimised_times` and reset the `diverged` flag
                self.collisions[i:, :] = 0
                self.optimised_times[start_index:end_index] += 1
                self.diverged = False

                self.optimisation_runs += 1

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
                        "Maximum optimisation runs reached; reattached all "
                        "particles and moved first checkpoint to the timestep "
                        f"at index {i} (t = {exp.times[i]})."
                    )))

            # At the end of a timestep, try to detach the remaining particles;
            # if not possible, forcibly move the remaining attached particles
            self.try_detach(i)
            self.move_attached(i)


    def diverging(self, time_index):
        '''Check whether the simulation diverges from the recorded particle
        positions at the timestep at index `time_index`.

        Parameters
        ----------
        time_index: int
            The time index in `self.experiment.times` for which the particles
            should be detached.

        Returns
        -------
        bool
            True if the distance between any simulated particle and its
            corresponding experimental particle is greater than
            `self.experiment.resolution`. False otherwise.

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

        # Once the sim diverged, it should keep returning True until the
        # optimiser is run and resets it to False
        if self.diverged is True:
            return True

        sim_positions = self.simulation.positions()
        exp_positions = self.experiment.positions_all[time_index]

        distances = np.linalg.norm(sim_positions - exp_positions, axis = 1)
        self.diverged = (distances > self.experiment.resolution).any()

        return self.diverged


    def optimisable(self, time_index, save_index):
        '''Check whether the simulation is in the optimum state to learn the
        DEM parameters.

        A simulation is optimisable if:

        1. All particles collided at least twice or any particle collided at
           least four times.
        2. The accumulated error (`xi_acc`) is larger than the sum of particle
           radii.
        3. More than `min_checkpoints` checkpoints were saved.

        Because the collisions are checked three timesteps in advance, only
        start optimising three timesteps after the conditions above are true.

        Parameters
        ----------
        time_index: int
            The time index in `self.experiment.times` for which to check
            whether the simulation is optimisable.

        save_index: int
            The time index for the last saved simulation state (one before the
            first checkpoint).

        Returns
        -------
        bool
            True if simulation is optimisable, False otherwise.
        '''

        # Check input `time_index` is valid
        time_index = int(time_index)
        if time_index >= len(self.experiment.times) or time_index < 0:
            raise ValueError(textwrap.fill((
                "The `time_index` input parameter must be between 0 and "
                f"`len(experiment.times)` = {len(self.experiment.times)}. "
                f"Received {time_index}."
            )))

        # Aliases
        col = self.collisions[time_index]

        # If all particles collided twice or any particle collided four times
        collided = ((col >= 2).all() or (col >= 6).any())

        return (self.diverged and collided and
                time_index - save_index >= self.min_checkpoints)


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

        t3 = times[time_index + 3]
        p3 = pos_all[time_index + 3]

        # Based on the particle positions at t0, t1 and t2, predict their
        # positions at t3; if the prediction is correct, the inferred velocity
        # at t0 is reliable => detach particle
        p3_predicted, u0 = self.predict_positions(t0, p0, t1, p1, t2, p2, t3)

        # Euclidean distance between predicted p(t3) and actual p(t3)
        error = np.linalg.norm(p3_predicted - p3, axis = 1)

        # If the positional error is smaller than measurement error on an
        # attached particle's position (i.e. the experiment's resolution),
        # detach the particle and set its velocity at t0 to u0
        resolution = self.experiment.resolution
        particle_indices = np.arange(self.simulation.num_atoms())
        to_detach = (self.attached & (error < resolution))

        for pid in particle_indices[to_detach]:
            self.simulation.set_velocity(pid, u0[pid])

        self.attached[to_detach] = False

        # Only change the collisions counter if this timestep was not already
        # used for optimisation
        if self.optimised_times[time_index] == 0:
            # If the positional error is large, at least a collision has
            # ocurred in the timesteps we looked at. We only care about
            # collisions for the detached particles.
            self.collisions[time_index + 3] = self.collisions[time_index + 2]
            self.collisions[
                time_index + 3,
                (error >= resolution) & (~self.attached),
            ] += 1


    @staticmethod
    def predict_positions(
        t0, p0,
        t1, p1,
        t2, p2,
        t3,
        g = 9.81,
        rho_p = 1000,
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

        if not (p0.ndim == p1.ndim == p2.ndim == 2 and
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
            return -0.5 * (t - t0) ** 2

        def B(t):
            return t - t0

        # The LHS of the system of 2 equations to solve *for a single particle*
        # to get the acceleration (ad) and initial velocity (u0)
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
        z3 = p0[:, 2] + adz_uz0[:, 0] * A(t3) + adz_uz0[:, 1] * B(t3) + \
            g * (1 - rho_f / rho_p) * A(t3)

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

        # Aliases
        sim = self.simulation
        exp = self.experiment

        particle_indices = np.arange(sim.num_atoms())

        for pid in particle_indices[self.attached]:
            sim.set_position(pid, exp.positions_all[time_index, pid])


    def optimise(
        self,
        start_index,
        end_index,
        verbose = True,
    ):
        '''Start optimisation of DEM parameters against the recorded
        experimental positions between timestep indices `start_index` and
        `end_index` (excluding `end_index`).

        This function also runs to simulation up to the current timestep (i.e.
        `end_index - 1`) and sets the new accumulated error (`xi_acc`)
        appropriately.

        Parameters
        ----------
        start_index: int
            The time index in `self.experiment.times` at which the optimisation
            starts (inclusive). Note that the simulation is currently at this
            timestep.

        end_index: int
            The time index in `self.experiment.times` at which the optimisation
            ends (exclusive). When the function returns, the simulation will be
            at the timestep index `end_index - 1`.

        verbose: bool, default True
            Print extra information to the terminal.

        Raises
        ------
        ValueError
            If the input `start_index` or `end_index` is invalid: either
            smaller than 0 or larger than `len(self.experiment.times)`.
        '''

        if verbose:
            print((
                "\nStarting optimisation of simulation between timestep "
                f"indices {start_index}:{end_index}."
            ))
            print(self.simulation, "\n")

        # If logging is on, create a new entry for this optimisation as a dict
        if self.save_log:
            self.log.append(
                dict(
                    start_index = start_index,
                    end_index = end_index,
                    num_solutions = self.num_solutions,
                    scaling = None,
                    solutions = [],
                    standard_deviations = [],
                    standard_deviation = [],
                    results = [],
                    best_solution = None,
                    best_result = None,
                )
            )

        # Aliases
        sim = self.simulation
        exp = self.experiment
        rng = self.rng

        # Save current checkpoint and extra data for parallel computation
        rand_hash = str(round(abs(rng.random() * 1e8)))

        if not os.path.isdir(f"coexist_info/run_{self.optimisation_runs}"):
            os.mkdir(f"coexist_info/run_{self.optimisation_runs}")

        # Minimum and maximum possible values for the DEM parameters
        params_mins = sim.parameters["min"].to_numpy()
        params_maxs = sim.parameters["max"].to_numpy()

        # If any `sigma` value is smaller than 5% (max - min), clip it
        sim.parameters["sigma"].clip(
            lower = 0.05 * (sim.parameters["max"] - sim.parameters["min"]),
            inplace = True,
        )

        # Scale sigma, bounds, solutions, results to unit variance
        scaling = sim.parameters["sigma"].to_numpy()

        if self.save_log:
            self.log[-1]["scaling"] = scaling

        # First guess, scaled
        x0 = sim.parameters["value"].to_numpy() / scaling
        sigma0 = 1.0
        bounds = [
            params_mins / scaling,
            params_maxs / scaling
        ]

        # Instantiate CMA-ES optimiser
        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds = bounds,
            ftarget = 0.0,
            popsize = self.num_solutions,
            randn = lambda *args: rng.standard_normal(args),
            verbose = 3 if verbose else -9,
        ))

        # Start optimisation: ask the optimiser for parameter combinations
        # (solutions), run the simulation between `start_index:end_index` and
        # feed the results back to CMA-ES.
        while not es.stop():
            solutions = es.ask()

            if verbose:
                print((
                    f"Scaled overall standard deviation: {es.sigma}\n"
                    f"Scaled individual standard deviations:\n{es.result.stds}"
                    f"\n\nTrying {len(solutions)} parameter combinations..."
                ))

            results = self.try_solutions(
                rand_hash,
                solutions * scaling,
                start_index,
                end_index,
            )

            es.tell(solutions, results)

            if verbose:
                cols = list(sim.parameters.index) + ["xi_acc"]
                sols_results = np.hstack((
                    solutions * scaling,
                    results[:, np.newaxis],
                ))

                sols_results = pd.DataFrame(
                    data = sols_results,
                    columns = cols,
                    index = None,
                )

                print(f"{sols_results}")
                print(f"Function evaluations: {es.result.evaluations}\n---",
                      flush = True)

            # If logging is on, save the solutions tried and their results
            if self.save_log:
                self.log[-1]["solutions"].extend(solutions * scaling)
                self.log[-1]["standard_deviation"].extend([es.sigma])
                self.log[-1]["standard_deviations"].extend(
                    es.result.stds * scaling
                )

                self.log[-1]["results"].extend(results)

                with open("coexist_info/optimisation_log.pickle", "wb") as f:
                    pickle.dump(self.log, f)

            if es.sigma < self.target_sigma:
                if verbose:
                    print((
                        "Optimal solution found within `target_sigma`, i.e. "
                        f"{self.target_sigma * 100}%:\n"
                        f"overall_sigma = {es.sigma} < {self.target_sigma}\n"
                    ))
                break

        solutions = es.result.xbest * scaling
        param_names = sim.parameters.index

        # Change sigma, min and max based on optimisation results
        sim.parameters["sigma"] = es.result.stds * scaling

        if verbose:
            print((
                f"Best results for solutions:\n{solutions}\n\n"
                f"Scaled individual standard deviations:\n{es.result.stds}\n"
            ))

        # Change parameters to the best solution
        for i, sol_val in enumerate(solutions):
            sim[param_names[i]] = sol_val

        # Compute xi_acc for found solutions and move simulation forward in
        # time
        self.xi_acc = self.calc_xi_acc(sim, exp, start_index, end_index)

        if verbose:
            print(f"Accumulated error (xi_acc) for solution: {self.xi_acc}")

        if self.save_log:
            self.log[-1]["best_solution"] = solutions
            self.log[-1]["best_result"] = self.xi_acc

            with open("coexist_info/optimisation_log.pickle", "wb") as f:
                pickle.dump(self.log, f)

            if verbose:
                print((
                    "Saved optimisation log as a pickled list of `dict`s in "
                    "coexist_info/optimisation_log.pickle.\n"
                    "---\n"
                ))


    @staticmethod
    def calc_xi(sim_pos, exp_pos):
        return np.linalg.norm(sim_pos - exp_pos, axis = 1).sum()


    @staticmethod
    def calc_xi_acc(
        simulation: Simulation,
        experiment: Experiment,
        start_index,
        end_index
    ):
        xi_acc = Coexist.calc_xi(
            simulation.positions(),
            experiment.positions_all[start_index],
        )

        for i in range(start_index + 1, end_index):
            simulation.step_to_time(experiment.times[i])
            xi = Coexist.calc_xi(
                simulation.positions(),
                experiment.positions_all[i],
            )
            xi_acc += xi

        return xi_acc


    def try_solutions(self, rand_hash, solutions, start_index, end_index):

        # Aliases
        sim = self.simulation
        param_names = sim.parameters.index

        # Path to `async_coexist_error.py`
        async_xi = os.path.join(
            os.path.split(coexist.__file__)[0],
            "async_coexist_error.py"
        )

        # For every solution to try, start a separate OS process that runs the
        # `async_coexist_error.py` file and captures the output value
        processes = []
        sim_paths = []

        results = np.full(len(solutions), np.finfo(float).max)

        # Catch the KeyboardInterrupt (Ctrl-C) signal to shut down the spawned
        # processes before aborting.
        try:
            for i, sol in enumerate(solutions):

                # Change simulation parameters
                for j, sol_val in enumerate(sol):
                    sim[param_names[j]] = sol_val

                # Save current simulation (i.e. with current parameters)
                sim_paths.append((
                    f"coexist_info/run_{self.optimisation_runs}/"
                    f"opt_{rand_hash}_{i}"
                ))
                sim.save(sim_paths[i])

                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,     # The Python interpreter path
                            async_xi,           # The `async_calc_xi.py` path
                            "coexist_info/simulation_class.pickle",
                            sim_paths[i],
                            "coexist_info/experiment.pickle",
                            str(start_index),
                            str(end_index),
                        ],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                    )
                )

            for i, proc in enumerate(processes):
                stdout, stderr = proc.communicate()

                # If we had new errors, write them to `error.log`
                if len(stderr) and stderr != self._stderr:
                    self._stderr = stderr

                    print((
                        "An error ocurred while running a simulation "
                        "asynchronously:\n"
                        f"{stderr.decode('utf-8')}\n\n"
                    ))

                results[i] = float(stdout)

        except KeyboardInterrupt:
            for proc in processes:
                proc.kill()

            sys.exit(130)

        return results




class Access:

    def __init__(
        self,
        simulations: Simulation,
        scheduler = [sys.executable],
        max_workers = None,
    ):
        '''`Access` class constructor.

        Parameters
        ----------
        simulations: Simulation or list[Simulation]
            The particle simulation object, implementing the
            `coexist.Simulation` interface (e.g. `LiggghtsSimulation`).
            Alternatively, this can be a *list of simulations*, in which case
            multiple simulations will be run **for the same parameter
            combination tried**. This allows the implementation of error
            functions using multiple simulation regimes.

        scheduler: list[str], default [sys.executable]
            The executable that will run individual simulations as separate
            Python scripts. In the simplest case, it would be just
            `["python3"]`. Alternatively, this can be used to schedule
            simulations on a cluster / supercomputer, e.g.
            `["srun", "-n", "1", "python3"]` for SLURM. Each list element is
            a piece of a shell command that will be run.

        max_workers: int, optional
            The maximum number of threads used by the main Python script to
            run the error function on each simulation result. If `None`, the
            output from `len(os.sched_getaffinity(0))` is used.

        '''

        # Type-checking inputs
        def error_simulations(simulations):
            # Print error message if the input `simulations` does not have the
            # required type
            raise TypeError(textwrap.fill((
                "The `simulation` input parameter must be an instance of "
                "`coexist.Simulation` (or a subclass thereof) or a list "
                f"of simulations. Received type `{type(simulations)}`."
            )))


        if isinstance(simulations, Simulation):
            self.simulations = [simulations]

        elif hasattr(simulations, "__iter__"):
            if not len(simulations) >= 1:
                error_simulations(simulations)

            params = simulations[0].parameters

            for sim in simulations:
                if not isinstance(sim, Simulation):
                    error_simulations(simulations)

                if not params.equals(sim.parameters):
                    raise ValueError((
                        "The simulation parameters (`Simulation.parameters` "
                        "attribute) must be the same across all simulations "
                        "in the input list. The two unequal parameters are:\n"
                        f"{params}\n\n"
                        f"{sim.parameters}\n"
                    ))

            self.simulations = simulations

        else:
            error_simulations(simulations)

        # Setting class attributes
        self.scheduler = list(scheduler)

        if max_workers is None:
            self.max_workers = len(os.sched_getaffinity(0))
        else:
            self.max_workers = int(max_workers)

        # These are the constant class attributes relevant for a single
        # ACCESS run. They will be set in the `learn` method
        self.rng = None
        self.error = None

        self.start_times = None
        self.end_times = None

        self.num_checkpoints = None
        self.num_solutions = None
        self.target_sigma = None

        self.use_historical = None
        self.save_positions = None

        self.save_path = None
        self.simulations_path = None
        self.outputs_path = None
        self.classes_path = None
        self.sim_classes_paths = None
        self.sim_paths = None

        self.history_path = None
        self.history_scaled_path = None

        self.random_seed = None
        self.verbose = None

        # Message printed to the stdout and stderr by spawned OS processes
        self._stdout = None
        self._stderr = None


    def learn(
        self,
        error,
        start_times,
        end_times,
        num_checkpoints = 100,
        num_solutions = 10,
        target_sigma = 0.1,
        use_historical = True,
        save_positions = True,
        random_seed = None,
        verbose = True,
    ):
        # Type-checking inputs
        if not callable(error):
            raise TypeError(textwrap.fill((
                "The input `error` must be a function (i.e. callable). "
                f"Received `{error}` with type `{type(error)}`."
            )))

        self.error = error

        if hasattr(start_times, "__iter__"):
            self.start_times = [float(st) for st in start_times]
        else:
            self.start_times = [float(start_times) for _ in self.simulations]

        if hasattr(end_times, "__iter__"):
            self.end_times = [float(et) for et in end_times]
        else:
            self.end_times = [float(end_times) for _ in self.simulations]

        if len(self.start_times) != len(self.simulations) or \
                len(self.end_times) != len(self.simulations):
            raise ValueError(textwrap.fill((
                "The input `start_times` and `end_times` must have the same "
                "number of elements as the number of simulations. Received "
                f"{len(self.start_times)} start times / {len(self.end_times)} "
                f"end times for {len(self.simulations)} simulations."
            )))

        if hasattr(num_checkpoints, "__iter__"):
            self.num_checkpoints = [int(nc) for nc in num_checkpoints]
        else:
            self.num_checkpoints = [
                int(num_checkpoints)
                for _ in self.simulations
            ]

        if len(self.num_checkpoints) != len(self.simulations):
            raise ValueError(textwrap.fill((
                "The input `num_checkpoints` must be either a single value or "
                "a list with the same number of elements as the number of "
                f"simulations. Received {len(self.num_checkpoints)} "
                f"checkpoints for {len(self.simulations)} simulations."
            )))

        self.num_solutions = int(num_solutions)
        self.target_sigma = float(target_sigma)

        self.use_historical = bool(use_historical)
        self.save_positions = bool(save_positions)

        self.random_seed = random_seed
        self.verbose = bool(verbose)

        # Setting constant class attributes
        self.rng = np.random.default_rng(self.random_seed)

        # Aliases
        sims = self.simulations
        rng = self.rng

        # The random hash that represents this optimisation run. If a random
        # seed was specified, this will make the simulation and optimisations
        # fully deterministic (i.e. repeatable)
        rand_hash = str(round(abs(rng.random() * 1e6)))

        self.save_path = f"access_info_{rand_hash}"
        self.simulations_path = f"{self.save_path}/simulations"
        self.classes_path = f"{self.save_path}/classes"
        self.outputs_path = f"{self.save_path}/outputs"

        self.sim_classes_paths = [
            f"{self.classes_path}/simulation_{i}_class.pickle"
            for i in range(len(self.simulations))
        ]

        self.sim_paths = [
            f"{self.classes_path}/simulation_{i}"
            for i in range(len(self.simulations))
        ]

        # Check if we have historical data about the optimisation - these are
        # pre-computed values for this exact simulation, random seed, and
        # number of solutions
        self.history_path = (
            f"{self.save_path}/opt_history_{self.num_solutions}.csv"
        )

        self.history_scaled_path = (
            f"{self.save_path}/opt_history_{self.num_solutions}_scaled.csv"
        )

        # Create all required paths above if they don't exist already
        self.create_directories()

        # History columns: [param1, param2, ..., stddev_param1, stddev_param2,
        # ..., stddev_all, error_value]
        if use_historical and os.path.isfile(self.history_path):
            history = np.loadtxt(self.history_path, dtype = float)
        elif use_historical:
            history = []
        else:
            history = None

        # Scaling and unscaling parameter values introduce numerical errors
        # that confuse the optimiser. Thus save unscaled values separately
        if self.use_historical and os.path.isfile(self.history_scaled_path):
            history_scaled = np.loadtxt(
                self.history_scaled_path, dtype = float
            )
        elif self.use_historical:
            history_scaled = []
        else:
            history_scaled = None

        # Minimum and maximum possible values for the DEM parameters
        params_mins = sims[0].parameters["min"].to_numpy()
        params_maxs = sims[0].parameters["max"].to_numpy()

        # If any `sigma` value is smaller than 5% (max - min), clip it
        for sim in sims:
            sim.parameters["sigma"].clip(
                lower = 0.05 * (params_maxs - params_mins),
                inplace = True,
            )

        # Scale sigma, bounds, solutions, results to unit variance
        scaling = sims[0].parameters["sigma"].to_numpy()

        # First guess, scaled
        x0 = sims[0].parameters["value"].to_numpy() / scaling
        sigma0 = 1.0
        bounds = [
            params_mins / scaling,
            params_maxs / scaling
        ]

        # Instantiate CMA-ES optimiser
        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds = bounds,
            popsize = self.num_solutions,
            randn = lambda *args: rng.standard_normal(args),
            verbose = 3 if self.verbose else -9,
        ))

        # Start optimisation: ask the optimiser for parameter combinations
        # (solutions), run the simulations between `start_index:end_index` and
        # feed the results back to CMA-ES.
        epoch = 0

        while not es.stop():
            solutions = es.ask()

            # If we have historical data, inject it for each epoch
            if self.use_historical and \
                    epoch * self.num_solutions < len(history_scaled):

                self.inject_historical(es, history_scaled, epoch)
                epoch += 1

                if self.finished(es):
                    break

                continue

            if self.verbose:
                self.print_before_eval(es, solutions)

            results = self.try_solutions(solutions * scaling, epoch)

            es.tell(solutions, results)
            epoch += 1

            # Save every step's historical data as function evaluations are
            # very expensive. Save columns [param1, param2, ..., stdev_param1,
            # stdev_param2, ..., stdev_all, error_val].
            if self.use_historical:
                if not isinstance(history, list):
                    history = list(history)

                for sol, res in zip(solutions, results):
                    history.append(
                        list(sol * scaling) +
                        list(es.result.stds * scaling) +
                        [es.sigma, res]
                    )

                np.savetxt(self.history_path, history)

                # Save scaled values separately to avoid numerical errors
                if not isinstance(history_scaled, list):
                    history_scaled = list(history_scaled)

                for sol, res in zip(solutions, results):
                    history_scaled.append(
                        list(sol) +
                        list(es.result.stds) +
                        [es.sigma, res]
                    )

                np.savetxt(self.history_scaled_path, history_scaled)

            if self.verbose:
                self.print_after_eval(es, solutions, scaling, results)

            if self.finished(es):
                break

        solutions = es.result.xbest * scaling
        stds = es.result.stds * scaling

        if self.verbose:
            print(f"Best results for solutions: {solutions}", flush = True)

        # Run the simulation with the best parameters found and save the
        # results to disk. Return the paths to the saved values
        radii_paths, positions_paths, velocities_paths = \
            self.run_simulation_best(solutions, stds)

        return radii_paths, positions_paths, velocities_paths


    def run_simulation_best(self, solutions, stds):
        # Path for saving the simulations with the best parameters
        best_path = f"{self.save_path}/best"

        if not os.path.isdir(best_path):
            os.mkdir(best_path)

        # Paths for saving the best simulations' outputs
        best_radii_paths = [
            f"{best_path}/best_radii_{i}.npy"
            for i in range(len(self.simulations))
        ]

        best_positions_paths = [
            f"{best_path}/best_positions_{i}.npy"
            for i in range(len(self.simulations))
        ]

        best_velocities_paths = [
            f"{best_path}/best_velocities_{i}.npy"
            for i in range(len(self.simulations))
        ]

        # Change sigma, min and max based on optimisation results
        param_names = self.simulations[0].parameters.index

        for sim in self.simulations:
            sim.parameters["sigma"] = stds

        # Change parameters to the best solution
        for i, sol_val in enumerate(solutions):
            for sim in self.simulations:
                sim[param_names[i]] = sol_val

        # Run each simulation with the best parameters found. This cannot be
        # done in parallel as the simulation library might be thread-unsafe
        for i, sim in enumerate(self.simulations):
            if self.verbose:
                print((
                    f"Running the simulation at index {i} with the best "
                    "parameter values found..."
                ))

            checkpoints = np.linspace(
                self.start_times[i],
                self.end_times[i],
                self.num_checkpoints[i],
            )

            positions = []
            velocities = []

            for t in checkpoints:
                sim.step_to_time(t)
                positions.append(sim.positions())
                velocities.append(sim.velocities())

            radii = sim.radii()
            positions = np.array(positions, dtype = float)
            velocities = np.array(velocities, dtype = float)

            np.save(best_radii_paths[i], radii)
            np.save(best_positions_paths[i], positions)
            np.save(best_velocities_paths[i], velocities)

        error = self.error(
            best_radii_paths,
            best_positions_paths,
            best_velocities_paths,
        )

        if self.verbose:
            print((f"Error (computed by the `error` function) for solution: "
                   f"{error}\n---"), flush = True)

        return best_radii_paths, best_positions_paths, best_velocities_paths


    def create_directories(self):
        # Save the current simulation state in a `restarts` folder
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        # Save positions and parameters in a new folder inside `restarts`
        if not os.path.isdir(self.simulations_path):
            os.mkdir(self.simulations_path)

        # Save classes objects in a new folder inside `restarts`
        if not os.path.isdir(self.classes_path):
            os.mkdir(self.classes_path)

        # Save simulation outputs (stderr and stdout) in a new folder
        if not os.path.isdir(self.outputs_path):
            os.mkdir(self.outputs_path)

        # Serialize the simulations' concrete class so that it can be
        # reconstructed even if it is a `coexist.Simulation` subclass
        for i in range(len(self.simulations)):
            with open(self.sim_classes_paths[i], "wb") as f:
                pickle.dump(self.simulations[i].__class__, f)

            # Save current checkpoint and extra data for parallel computation
            self.simulations[i].save(self.sim_paths[i])

        with open(f"{self.save_path}/opt_run_info.txt", "a") as f:
            now = datetime.now().strftime("%H:%M:%S - %D")
            f.writelines([
                "--------------------------------------------------------\n",
                f"Starting ACCESS run at {now}\n\n",
                "Simulations:\n"
                f"{self.simulations}\n\n",
                f"start_times =         {self.start_times}\n",
                f"end_times =           {self.end_times}\n",
                f"num_checkpoints =     {self.num_checkpoints}\n",
                f"target_sigma =        {self.target_sigma}\n",
                f"num_solutions =       {self.num_solutions}\n",
                f"random_seed =         {self.random_seed}\n",
                f"use_historical =      {self.use_historical}\n",
                f"save_positions =      {self.save_positions}\n\n",
                f"save_path =           {self.save_path}\n",
                f"simulations_path =    {self.simulations_path}\n",
                f"classes_path =        {self.classes_path}\n",
                f"outputs_path =        {self.outputs_path}\n",
                f"sim_classes_paths =   {self.sim_classes_paths}\n",
                f"sim_paths =           {self.sim_paths}\n\n",
                f"history_path =        {self.history_path}\n",
                f"history_scaled_path = {self.history_scaled_path}\n",
                "--------------------------------------------------------\n\n",
            ])


    def inject_historical(self, es, history_scaled, epoch):
        '''Inject the CMA-ES optimiser with pre-computed (historical) results.
        The solutions must have a Gaussian distribution in each problem
        dimension - though the standard deviation can vary for each of them.
        Ideally, this should only use historical values that CMA-ES asked for
        in a previous ACCESS run.
        '''

        ns = self.num_solutions
        num_params = len(self.simulations[0].parameters)

        results_scaled = history_scaled[(epoch * ns):(epoch * ns + ns)]
        es.tell(results_scaled[:, :num_params], results_scaled[:, -1])

        if self.verbose:
            print((
                f"Injected {(epoch + 1) * len(results_scaled)} / "
                f"{len(history_scaled)} historical solutions."
            ))


    def print_before_eval(self, es, solutions):
        print((
            f"Scaled overall standard deviation: {es.sigma}\n"
            f"Scaled individual standard deviations:\n{es.result.stds}"
            f"\n\nTrying {len(solutions)} parameter combinations..."
        ), flush = True)


    def print_after_eval(
        self,
        es,
        solutions,
        scaling,
        results,
    ):
        # Display evaluation results: solutions, error values, etc.
        cols = list(self.simulations[0].parameters.index) + ["error"]
        sols_results = np.hstack((
            solutions * scaling,
            results[:, np.newaxis],
        ))

        # Store solutions and results in a DataFrame for easy pretty printing
        sols_results = pd.DataFrame(
            data = sols_results,
            columns = cols,
            index = None,
        )

        # Display all the DataFrame columns and rows
        old_max_columns = pd.get_option("display.max_columns")
        old_max_rows = pd.get_option("display.max_rows")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        print((
            f"{sols_results}\n"
            f"Function evaluations: {es.result.evaluations}\n---"
        ), flush = True)

        pd.set_option("display.max_columns", old_max_columns)
        pd.set_option("display.max_rows", old_max_rows)


    def finished(self, es):
        if es.sigma < self.target_sigma:
            if self.verbose:
                print((
                    "Optimal solution found within `target_sigma`, i.e. "
                    f"{self.target_sigma * 100}%:\n"
                    f"sigma = {es.sigma} < {self.target_sigma}"
                ), flush = True)

            return True

        return False


    def std_outputs(self, run_index, sim_index, stdout, stderr):
        # If we had new errors, write them to `error.log`
        if len(stderr) and stderr != self._stderr:
            self._stderr = stderr.decode("utf-8")

            error_path = (
                f"{self.outputs_path}/"
                f"error_{run_index}_{sim_index}.log"
            )

            print((
                "A new error ocurred while running simulation (run "
                f"{run_index} / index {sim_index}):\n"
                f"{self._stderr}\n\n"
                f"Writing error message to `{error_path}`\n"
            ))

            with open(error_path, "w") as f:
                f.write(self._stderr)

        # If we had new outputs, write them to `output.log`
        if len(stdout) and stdout != self._stdout:
            self._stdout = stdout.decode("utf-8")

            output_path = (
                f"{self.outputs_path}/"
                f"output_{run_index}_{sim_index}.log"
            )

            print((
                "A new message was outputted while running simulation (run "
                f"{run_index} / index {sim_index}):\n"
                f"{self._stdout}\n\n"
                f"Writing output message to `{output_path}`\n"
            ))

            with open(output_path, "w") as f:
                f.write(self._stdout)


    def simulations_save_paths(self, epoch):
        sim_paths = []
        radii_paths = []
        positions_paths = []
        velocities_paths = []

        start_index = epoch * self.num_solutions

        # For every parameter combination...
        for i in range(self.num_solutions):
            sim_run = []
            radii_run = []
            positions_run = []
            velocities_run = []

            # For every simulation run...
            for j in range(len(self.simulations)):
                sim_run.append((
                    f"{self.simulations_path}/"
                    f"opt_{start_index + i}_{j}"
                ))

                radii_run.append((
                    f"{self.simulations_path}/"
                    f"opt_{start_index + i}_{j}_radii.npy"
                ))

                positions_run.append((
                    f"{self.simulations_path}/"
                    f"opt_{start_index + i}_{j}_positions.npy"
                ))

                velocities_run.append((
                    f"{self.simulations_path}/"
                    f"opt_{start_index + i}_{j}_velocities.npy"
                ))

            sim_paths.append(sim_run)
            radii_paths.append(radii_run)
            positions_paths.append(positions_run)
            velocities_paths.append(velocities_run)

        return sim_paths, radii_paths, positions_paths, velocities_paths


    def try_solutions(self, solutions, epoch):
        # Aliases
        param_names = self.simulations[0].parameters.index

        # Path to `async_access_error.py`
        async_xi = os.path.join(
            os.path.split(coexist.__file__)[0],
            "async_access_error.py"
        )

        # For every solution to try and simulation run, start a separate OS
        # process that runs the `async_access_error.py` file and saves the
        # positions in a `.npy` file
        processes = []

        # These are all lists of lists: axis 0 is the parameter combination to
        # try, while axis 1 is the particular simulation to run
        sim_paths, radii_paths, positions_paths, velocities_paths = \
            self.simulations_save_paths(epoch)

        # Catch the KeyboardInterrupt (Ctrl-C) signal to shut down the spawned
        # processes before aborting.
        try:
            # Run all simulations in `self.simulations` for each parameter
            # combination in `solutions`
            for i, sol in enumerate(solutions):
                single_run_processes = []

                # For every simulation in `self.simulations`, start a new proc
                for j, sim in enumerate(self.simulations):

                    # Change parameter values to save them along with the
                    # full simulation state
                    for k, sol_val in enumerate(sol):
                        sim.parameters.at[param_names[k], "value"] = sol_val

                    sim.save(sim_paths[i][j])

                    single_run_processes.append(
                        subprocess.Popen(
                            self.scheduler + [  # Python interpreter path
                                async_xi,       # async_access_error.py path
                                self.sim_classes_paths[j],
                                sim_paths[i][j],
                                str(self.start_times[j]),
                                str(self.end_times[j]),
                                str(self.num_checkpoints[j]),
                                radii_paths[i][j],
                                positions_paths[i][j],
                                velocities_paths[i][j],
                            ],
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE,
                        )
                    )

                processes.append(single_run_processes)

            # Compute the error function values in a parallel environment.
            with ProcessPoolExecutor(max_workers = self.max_workers) \
                    as executor:
                futures = []

                # Get the output from each OS process / simulation
                for i, run_procs in enumerate(processes):
                    # Check simulations didn't crash
                    crashed = False

                    for j, proc in enumerate(run_procs):
                        stdout, stderr = proc.communicate()

                        proc_index = epoch * self.num_solutions + i
                        self.std_outputs(proc_index, j, stdout, stderr)

                        # Only load simulations if they exist - i.e. no errors
                        # occurred
                        if not (os.path.isfile(radii_paths[i][j]) and
                                os.path.isfile(positions_paths[i][j]) and
                                os.path.isfile(velocities_paths[i][j])):

                            print((
                                "At least one of the simulation files "
                                f"{radii_paths[i][j]}, "
                                f"{positions_paths[i][j]} or "
                                f"{velocities_paths[i][j]}, was not found; "
                                f"the simulation (run {i} / index {j}) most "
                                "likely crashed. Check the error, output and "
                                "LIGGGHTS logs for what went wrong. The error "
                                "value for this simulation run is set to NaN."
                            ))

                            crashed = True

                    # If no simulations crashed in this run, execute the error
                    # function in parallel
                    if not crashed:
                        futures.append(
                            executor.submit(
                                self.error,
                                radii_paths[i],
                                positions_paths[i],
                                velocities_paths[i],
                            )
                        )
                    else:
                        futures.append(None)

                # Crashed solutions will have np.nan as a value.
                results = np.full(len(solutions), np.nan)

                for i, f in enumerate(futures):
                    if f is not None:
                        results[i] = f.result()


        except KeyboardInterrupt:
            for proc_run in processes:
                for proc in proc_run:
                    proc.kill()

            sys.exit(130)

        # If `save_positions` is not True, remove all simulation files
        if not self.save_positions:
            for radii_run in radii_paths:
                [os.remove(rp) for rp in radii_run]

            for positions_run in positions_paths:
                [os.remove(pp) for pp in positions_run]

            for velocities_run in velocities_paths:
                [os.remove(vp) for vp in velocities_run]

        return results




class AccessScript:
    '''Optimise an arbitrary user-defined script's parameters in parallel.

    A minimal user script - saved in a separate file - would be:

    ```python

    #### ACCESS PARAMETERS START
    import coexist

    parameters = coexist.create_parameters(
        variables = ["fp1", "fp2"],
        minimums = [-5, -10],
        maximums = [+5, +10],
    )

    access_id = 1                           # Optional
    #### ACCESS PARAMETERS END

    x = parameters.at["fp1", "value"]
    y = parameters.at["fp2", "value"]

    error = x ** 2 + y ** 2
    extra = dict(info = "Some info")        # Optional

    ```

    This script defines two free parameters to optimise "fp1" and "fp2" with
    ranges [-5, +5] and [-10, +10] and saves an error value to be optimised
    in the variable `error`. To optimise it, run in another file:

    ```python

    import coexist

    access = coexist.AccessScript("script_filepath.py")
    access.learn()

    ```

    One you run `access.learn()`, a folder named "access_info_<hashcode>" is
    generated which stores all information about this access run, including all
    simulation data. Feel free to look through it and extract anything, but
    don't manually edit the files in there.

    In general, an ACCESS user script must define one simulation whose
    parameters will be optimised this way:

    1. Use a variable named "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       An initial guess can also be set here.

    2. The `parameters` creation should be fully self-contained between two
       `#### ACCESS PARAMETERS START` and `#### ACCESS PARAMETERS END` blocks.
       Self-contained means it should not depend on code ran before.

    3. By the end of the simulation script, define two variables:
        a. `error` - one number representing this simulation's error value.
        b. `extra` - (optional) *any* python data structure storing extra
                     information you want to save for a simulation run (e.g.
                     particle positions).

    Notice that there is no limitation on how the error value is calculated. It
    can be any simulation, executed in any way - even externally; just launch
    a separate process from the Python script, run the simulation, extract data
    back into the Python script and set `error` to what you need optimised.

    If you need to save data to disk, use the `access_id` variable which is set
    to a unique ID for each simulation, so that you don't overwrite existing
    files when simulations are executed in parallel.

    For more information on the implementation details and how parallel
    execution of your user script is achieved, check out the generated file
    "access_info_<hashcode>/access_code.py" after running `access.learn()`.

    Attributes
    ----------
    parameters: pandas.DataFrame
        The free / optimisable parameters extracted from the user script.

    scheduler: coexist.schedulers.Scheduler subclass
        Scheduler used to spawn function evaluations / simulations in parallel.
        The default `LocalScheduler` simply starts new Python interpreters on
        the local machine for executing the user's script. See the other
        schedulers in `coexist.schedulers` for e.g. spawning jobs on a cluster.

    Methods
    -------
    learn(num_solutions = 10, target_sigma = 0.1, random_seed = None,
          verbose = True)
        Learn the free `parameters` from the user script that minimise the
        `error` variable by trying `num_solutions` parameter combinations at
        a time until the uncertainty in each parameter becomes lower than
        `target_sigma`.

    '''

    def __init__(
        self,
        script_path: str,
        scheduler = schedulers.LocalScheduler(),
        max_workers = None,
    ):
        '''`Access` class constructor.

        Parameters
        ----------
        script_path: str
            A path to a user-defined script that runs one simulation. It should
            use the free / optimisable parameters saved in a pandas.DataFrame
            named exactly `parameters`, defined between two comments
            "#### ACCESS PARAMETERS START" and "#### ACCESS PARAMETERS END".
            By the end of the script, one variable named `error` must be
            defined containing the error value, a number.

        scheduler: coexist.schedulers.Scheduler subclass
            Scheduler used to spawn function evaluations / simulations in
            parallel. The default `LocalScheduler` simply starts new Python
            interpreters on the local machine for executing the user's script.
            See the other schedulers in `coexist.schedulers` for e.g. spawning
            jobs on a supercomputing cluster.

        max_workers: int, optional
            [TODO] Only spawn `max_workers` processes at a time.

        '''

        # Setting class attributes
        self.access_code, self.parameters = self.generate_code(script_path)
        if not isinstance(scheduler, schedulers.Scheduler):
            raise TypeError(textwrap.fill((
                "The input `scheduler` must be a subclass of `coexist."
                f"schedulers.Scheduler`. Received {type(scheduler)}."
            )))

        self.scheduler = scheduler.generate()

        if max_workers is not None:
            self.max_workers = int(max_workers)

        # These are the constant class attributes relevant for a single
        # ACCESS run. They will be set in the `learn` method
        self.rng = None

        self.num_solutions = None
        self.target_sigma = None

        self.save_path = None
        self.simulations_path = None
        self.outputs_path = None

        self.history_path = None
        self.history_scaled_path = None

        self.random_seed = None
        self.verbose = None

        # Message printed to the stdout and stderr by spawned OS processes
        self._stdout = None
        self._stderr = None


    def generate_code(self, script_path):
        '''Generate the ACCESS code from a user's script at `script_path`.

        The user script is modified such that:

        1. The `parameters` assignment expression is substituted with *loading*
           the parameters from an ACCESS-defined location; those parameters are
           varied according to the optimisation procedure.
        2. At the end of the script, the `error` and (if defined) `extra`
           variables are saved to an ACCESS-defined location.

        The ACCESS-defined locations are given to the script as three
        command-line arguments; the script itself will be run in a
        massively-parallel environment.

        Parameters
        ----------
        script_path : str
            A path to a user's Python script.

        Returns
        -------
        generated_code : str
            The complete, modified ACCESS code, returned as a single string
            containing the correct newlines.

        parameters : pandas.DataFrame
            The `parameters` variable defined in the user script; that portion
            of the script was executed separately; this is the created object.
        '''
        # Read in the user's Python script at `script_path`
        with open(script_path, "r") as f:
            user_code = f.readlines()

        # Find the two parameter definitions directives
        params_start_line = None
        params_end_line = None

        for i, line in enumerate(user_code):
            if line.startswith("#### ACCESS PARAMETERS START"):
                params_start_line = i

            if line.startswith("#### ACCESS PARAMETERS END"):
                params_end_line = i

        if params_start_line is None or params_end_line is None:
            raise NameError(textwrap.fill((
                f"The user script found in file `{script_path}` did not "
                "contain the blocks `#### ACCESS PARAMETERS START` and "
                "`#### ACCESS PARAMETERS END`. Please define your simulation "
                "free parameters between these two comments / directives."
            )))

        # Execute the code between the two directives to get the initial
        # `parameters`. `exec` saves all the code's variables in the
        # `parameters_exec` dictionary
        user_params_code = "".join(
            user_code[params_start_line:params_end_line]
        )
        user_params_exec = dict()
        exec(user_params_code, user_params_exec)

        if "parameters" not in user_params_exec:
            raise NameError(textwrap.fill((
                "The code between the user script's directives "
                "`#### ACCESS PARAMETERS START` and "
                "`#### ACCESS PARAMETERS END` does not define a variable "
                "named exactly `parameters`."
            )))

        self.validate_parameters(user_params_exec["parameters"])

        if not code_contains_variable(user_code, "error"):
            raise NameError(textwrap.fill((
                f"The user script found in file `{script_path}` does not "
                "define the required variable `error`."
            )))

        # Substitute the `parameters` creation in the user code with loading
        # them from an ACCESS-defined location
        parameters_code = [
            "\n# Unpickle `parameters` from this script's first " +
            "command-line argument and set\n",
            '# `access_id` to a unique simulation ID\n',
            code_substitute_variable(
                user_code[params_start_line:params_end_line],
                "parameters",
                ('with open(sys.argv[1], "rb") as f:\n'
                 '    parameters = pickle.load(f)\n')
            )
        ]

        # Also define a unique ACCESS ID for each simulation
        parameters_code += (
            'access_id = int(os.path.split(sys.argv[1])[1].split("_")[1])\n\n'
        )

        # Read in the `async_access_template.py` code template and find the
        # code injection directives
        template_code_path = os.path.join(
            os.path.split(coexist.__file__)[0],
            "async_access_template.py"
        )

        with open(template_code_path, "r") as f:
            template_code = f.readlines()

        for i, line in enumerate(template_code):
            if line.startswith("#### ACCESS INJECT USER CODE START"):
                inject_start_line = i

            if line.startswith("#### ACCESS INJECT USER CODE END"):
                inject_end_line = i

        generated_code = "".join((
            template_code[:inject_start_line + 1] +
            user_code[:params_start_line + 1] +
            parameters_code +
            user_code[params_end_line:] +
            template_code[inject_end_line:]
        ))

        return generated_code, user_params_exec["parameters"]


    def validate_parameters(self, parameters):
        if not isinstance(parameters, pd.DataFrame):
            raise ValueError(textwrap.fill((
                "The `parameters` variable defined in the user script is "
                "not a pandas.DataFrame instance (or subclass thereof)."
            )))

        if len(parameters) < 2:
            raise ValueError(textwrap.fill((
                "The `parameters` DataFrame defined in the user script must "
                "have at least two free parameters defined. Found only"
                f"`len(parameters) = {len(parameters)}`."
            )))

        columns_needed = ["value", "min", "max", "sigma"]
        if not all([c in parameters.columns for c in columns_needed]):
            raise ValueError(textwrap.fill((
                "The `parameters` DataFrame defined in the user script must "
                "have at least four columns defined: ['value', 'min', "
                f"'max', 'sigma']. Found these: `{parameters.columns}`. You "
                "can use the `coexist.create_parameters` function for this."
            )))


    def learn(
        self,
        num_solutions = 10,
        target_sigma = 0.1,
        random_seed = None,
        verbose = True,
    ):
        # Type-checking inputs
        self.num_solutions = int(num_solutions)
        self.target_sigma = float(target_sigma)

        if random_seed is None:
            self.random_seed = np.random.randint(np.iinfo(np.int64).max)
        else:
            self.random_seed = random_seed

        self.verbose = bool(verbose)

        # Setting constant class attributes
        self.rng = np.random.default_rng(self.random_seed)

        # Aliases
        rng = self.rng

        # The random hash that represents this optimisation run. If a random
        # seed was specified, this will make the simulation and optimisations
        # fully deterministic (i.e. repeatable)
        rand_hash = str(round(abs(rng.random() * 1e6)))

        self.save_path = f"access_info_{rand_hash}"
        self.simulations_path = f"{self.save_path}/simulations"
        self.outputs_path = f"{self.save_path}/outputs"

        # Check if we have historical data about the optimisation - these are
        # pre-computed values for this exact simulation, random seed, and
        # number of solutions
        self.history_path = (
            f"{self.save_path}/opt_history_{self.num_solutions}.csv"
        )

        self.history_scaled_path = (
            f"{self.save_path}/opt_history_{self.num_solutions}_scaled.csv"
        )

        self.access_code_path = f"{self.save_path}/access_code.py"

        # Create all required paths above if they don't exist already
        self.create_directories()

        # Save the generated ACCESS code to a file
        with open(self.access_code_path, "w") as f:
            f.write(self.access_code)

        # History columns: [param1, param2, ..., stddev_param1, stddev_param2,
        # ..., stddev_all, error_value]
        if os.path.isfile(self.history_path):
            history = np.loadtxt(self.history_path, dtype = float)
        else:
            history = []

        # Scaling and unscaling parameter values introduce numerical errors
        # that confuse the optimiser. Thus save unscaled values separately
        if os.path.isfile(self.history_scaled_path):
            history_scaled = np.loadtxt(
                self.history_scaled_path, dtype = float
            )
        else:
            history_scaled = []

        # Minimum and maximum possible values for the DEM parameters
        params_mins = self.parameters["min"].to_numpy()
        params_maxs = self.parameters["max"].to_numpy()

        # If any `sigma` value is smaller than 5% (max - min), clip it
        self.parameters["sigma"].clip(
            lower = 0.05 * (params_maxs - params_mins),
            inplace = True,
        )

        # Scale sigma, bounds, solutions, results to unit variance
        scaling = self.parameters["sigma"].to_numpy()

        # First guess, scaled
        x0 = self.parameters["value"].to_numpy() / scaling
        sigma0 = 1.0
        bounds = [
            params_mins / scaling,
            params_maxs / scaling
        ]

        # Instantiate CMA-ES optimiser
        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds = bounds,
            popsize = self.num_solutions,
            randn = lambda *args: rng.standard_normal(args),
            verbose = 3 if self.verbose else -9,
        ))

        # Start optimisation: ask the optimiser for parameter combinations
        # (solutions), run the simulations between `start_index:end_index` and
        # feed the results back to CMA-ES.
        epoch = 0

        while not es.stop():
            solutions = es.ask()

            # If we have historical data, inject it for each epoch
            if epoch * self.num_solutions < len(history_scaled):

                self.inject_historical(es, history_scaled, epoch)
                epoch += 1

                if self.finished(es):
                    break

                continue

            if self.verbose:
                self.print_before_eval(es, solutions)

            results = self.try_solutions(solutions * scaling, epoch)

            es.tell(solutions, results)
            epoch += 1

            # Save every step's historical data as function evaluations are
            # very expensive. Save columns [param1, param2, ..., stdev_param1,
            # stdev_param2, ..., stdev_all, error_val].
            if not isinstance(history, list):
                history = list(history)

            for sol, res in zip(solutions, results):
                history.append(
                    list(sol * scaling) +
                    list(es.result.stds * scaling) +
                    [es.sigma, res]
                )

            np.savetxt(self.history_path, history)

            # Save scaled values separately to avoid numerical errors
            if not isinstance(history_scaled, list):
                history_scaled = list(history_scaled)

            for sol, res in zip(solutions, results):
                history_scaled.append(
                    list(sol) +
                    list(es.result.stds) +
                    [es.sigma, res]
                )

            np.savetxt(self.history_scaled_path, history_scaled)

            if self.verbose:
                self.print_after_eval(es, solutions, scaling, results)

            if self.finished(es):
                break

        if es.result.xbest is None:
            raise ValueError(textwrap.fill((
                "No parameter combination was evaluated successfully. All "
                "simulations crashed - please check the error logs in the "
                "`access_info_<hash_code>/outputs` folder."
            )))

        solutions = es.result.xbest * scaling
        stds = es.result.stds * scaling

        if self.verbose:
            print((
                "\n---\n"
                "The best result was achieved for these parameter values:\n"
                f"{solutions}\n\n"
                "The standard deviation / uncertainty in each parameter is:\n"
                f"{stds}\n\n"
                "For these parameters, the error value found was: "
                f"{history[es.result.evals_best - 1][-1]}\n\n"
                "These results were found for the simulation at index "
                f"{es.result.evals_best - 1}, which can be found in:\n"
                f"{self.simulations_path}\n"
            ), flush = True)


    def create_directories(self):
        # Save the current simulation state in a `restarts` folder
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        # Save positions and parameters in a new folder inside `restarts`
        if not os.path.isdir(self.simulations_path):
            os.mkdir(self.simulations_path)

        # Save simulation outputs (stderr and stdout) in a new folder
        if not os.path.isdir(self.outputs_path):
            os.mkdir(self.outputs_path)

        with open(f"{self.save_path}/opt_run_info.txt", "a") as f:
            now = datetime.now().strftime("%H:%M:%S - %D")
            f.writelines([
                "--------------------------------------------------------\n",
                f"Starting ACCESS run at {now}\n\n",
                f"Generated ACCESS code:    {self.access_code_path}\n"
                f"ACCESS parameters:\n{self.parameters}\n\n"
                f"target_sigma =            {self.target_sigma}\n",
                f"num_solutions =           {self.num_solutions}\n",
                f"random_seed =             {self.random_seed}\n\n",
                f"save_path =               {self.save_path}\n",
                f"simulations_path =        {self.simulations_path}\n",
                f"outputs_path =            {self.outputs_path}\n\n",
                f"history_path =            {self.history_path}\n",
                f"history_scaled_path =     {self.history_scaled_path}\n",
                "--------------------------------------------------------\n\n",
            ])


    def inject_historical(self, es, history_scaled, epoch):
        '''Inject the CMA-ES optimiser with pre-computed (historical) results.
        The solutions must have a Gaussian distribution in each problem
        dimension - though the standard deviation can vary for each of them.
        Ideally, this should only use historical values that CMA-ES asked for
        in a previous ACCESS run.
        '''

        ns = self.num_solutions
        num_params = len(self.parameters)

        results_scaled = history_scaled[(epoch * ns):(epoch * ns + ns)]
        es.tell(results_scaled[:, :num_params], results_scaled[:, -1])

        if self.verbose:
            print((
                f"Injected {(epoch + 1) * len(results_scaled):>6} / "
                f"{len(history_scaled):>6} historical solutions."
            ))


    def print_before_eval(self, es, solutions):
        print((
            f"Scaled overall standard deviation: {es.sigma}\n"
            f"Scaled individual standard deviations:\n{es.result.stds}"
            f"\n\nTrying {len(solutions)} parameter combinations..."
        ), flush = True)


    def print_after_eval(
        self,
        es,
        solutions,
        scaling,
        results,
    ):
        # Display evaluation results: solutions, error values, etc.
        cols = list(self.parameters.index) + ["error"]
        sols_results = np.hstack((
            solutions * scaling,
            results[:, np.newaxis],
        ))

        # Store solutions and results in a DataFrame for easy pretty printing
        sols_results = pd.DataFrame(
            data = sols_results,
            columns = cols,
            index = None,
        )

        # Display all the DataFrame columns and rows
        old_max_columns = pd.get_option("display.max_columns")
        old_max_rows = pd.get_option("display.max_rows")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        print((
            f"{sols_results}\n"
            f"Function evaluations: {es.result.evaluations}\n---"
        ), flush = True)

        pd.set_option("display.max_columns", old_max_columns)
        pd.set_option("display.max_rows", old_max_rows)


    def finished(self, es):
        if es.sigma < self.target_sigma:
            if self.verbose:
                print((
                    "Optimal solution found within `target_sigma`, i.e. "
                    f"{self.target_sigma * 100}%:\n"
                    f"sigma = {es.sigma} < {self.target_sigma}"
                ), flush = True)

            return True

        return False


    def try_solutions(self, solutions, epoch):
        # Aliases
        param_names = self.parameters.index

        # For every solution to try, start a separate OS process that runs the
        # `access_info_<hash>/access_code.py` script, which computes and saves
        # an error value
        processes = []

        # These are this optimisation run's paths to save the simulation
        # outputs to; they will be given to `self.access_code` as command-line
        # arguments
        start_index = epoch * self.num_solutions

        parameters_paths = [
            f"{self.simulations_path}/opt_{start_index + i}_parameters.pickle"
            for i in range(self.num_solutions)
        ]

        error_paths = [
            f"{self.simulations_path}/opt_{start_index + i}_error.pickle"
            for i in range(self.num_solutions)
        ]

        extra_paths = [
            f"{self.simulations_path}/opt_{start_index + i}_extra.pickle"
            for i in range(self.num_solutions)
        ]

        # Catch the KeyboardInterrupt (Ctrl-C) signal to shut down the spawned
        # processes before aborting.
        try:
            # Spawn a separate process for every solution to try / sim to run
            for i, sol in enumerate(solutions):
                # Change the `self.parameters` values and save them to disk
                for j, sol_val in enumerate(sol):
                    self.parameters.at[param_names[j], "value"] = sol_val

                with open(parameters_paths[i], "wb") as f:
                    pickle.dump(self.parameters, f)

                processes.append(
                    subprocess.Popen(
                        self.scheduler + [          # Python interpreter path
                            self.access_code_path,
                            parameters_paths[i],
                            error_paths[i],
                            extra_paths[i],
                        ],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                    )
                )

            # Gather results
            results = []
            errored = []
            outputted = []
            crashed = []

            for i, proc in enumerate(processes):
                stdout, stderr = proc.communicate()

                proc_index = epoch * self.num_solutions + i

                # If a new error message was recorded in stderr, log it
                if len(stderr) and stderr != self._stderr:
                    errored.append(proc_index + i)

                    self._stderr = stderr.decode("utf-8")
                    with open(f"{self.outputs_path}/error_{proc_index}.log",
                              "w") as f:
                        f.write(self._stderr)

                # If a new output message was recorded in stdout, log it
                if len(stdout) and stdout != self._stdout:
                    outputted.append(proc_index + i)

                    self._stdout = stdout.decode("utf-8")
                    with open(f"{self.outputs_path}/output_{proc_index}.log",
                              "w") as f:
                        f.write(self._stdout)

                if os.path.isfile(error_paths[i]):
                    with open(error_paths[i], "rb") as f:
                        results.append(float(pickle.load(f)))
                else:
                    results.append(np.nan)
                    crashed.append(proc_index + i)

        except KeyboardInterrupt:
            for proc in processes:
                proc.kill()

            sys.exit(130)

        # Print messages for errors, outputs and simulation crashes
        if len(errored):
            print((
                "New error messages were recorded in `stderr` during "
                f"these simulations:\n{errored}\n\n"
                "The error messages were logged in the "
                f"`{self.outputs_path}` directory.\n"
            ), flush = True)

        if len(outputted):
            print((
                "New output messages were recorded in `stdout` during "
                f"these simulations:\n{outputted}\n\n"
                "The output messages were logged in the "
                f"`{self.outputs_path}` directory.\n"
            ), flush = True)

        if len(crashed):
            print((
                "The expected `error` values were not found for these "
                f"simulations:\n{crashed}\n\n"
                "They most likely crashed; check the error and output "
                "logs for what happened. The error values for these "
                "simulations were set to NaN.\n"
            ), flush = True)

        return np.array(results)

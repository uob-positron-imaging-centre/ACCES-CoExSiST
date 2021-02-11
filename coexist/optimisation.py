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
        simulation: Simulation,
        scheduler = [sys.executable],
        max_workers = None,
    ):

        # Type-checking inputs
        if not isinstance(simulation, Simulation):
            raise TypeError(textwrap.fill((
                "The `simulation` input parameter must be an instance of "
                f"`coexist.Simulation`. Received type `{type(simulation)}`."
            )))

        # Setting class attributes
        self.simulation = simulation
        self.scheduler = list(scheduler)

        if max_workers is None:
            self.max_workers = len(os.sched_getaffinity(0))
        else:
            self.max_workers = int(max_workers)

        self.rng = None
        self.error = None

        self.start_time = None
        self.end_time = None

        self.num_checkpoints = None
        self.num_solutions = None

        self.verbose = None

        # Message printed to the stdout and stderr by spawned OS processes
        self._stdout = None
        self._stderr = None


    def learn(
        self,
        error,
        start_time,
        end_time,
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

        self.start_time = float(start_time)
        self.end_time = float(end_time)

        self.num_checkpoints = int(num_checkpoints)
        self.num_solutions = int(num_solutions)
        self.target_sigma = float(target_sigma)

        self.rng = np.random.default_rng(random_seed)

        if random_seed is None:
            use_historical = False
        else:
            use_historical = bool(use_historical)

        self.verbose = bool(verbose)

        save_positions = bool(save_positions)

        # Aliases
        sim = self.simulation
        rng = self.rng

        # The random hash that represents this optimisation run. If a random
        # seed was specified, this will make the simulation and optimisations
        # fully deterministic (i.e. repeatable)
        rand_hash = str(round(abs(rng.random() * 1e8)))

        self.create_directories(
            random_seed, rand_hash, use_historical, save_positions
        )

        # Check if we have historical data about the optimisation - these are
        # pre-computed values for this exact simulation, random seed, and
        # number of solutions
        history_path = (
            f"access_info_{rand_hash}/"
            f"opt_history_{self.num_solutions}.csv"
        )

        history_scaled_path = (
            f"access_info_{rand_hash}/"
            f"opt_history_{self.num_solutions}_scaled.csv"
        )

        # History columns: [param1, param2, ..., stddev_param1, stddev_param2,
        # ..., stddev_all, error_value]
        if use_historical and os.path.isfile(history_path):
            history = np.loadtxt(history_path, dtype = float)
        elif use_historical:
            history = []
        else:
            history = None

        # Scaling and unscaling parameter values introduce numerical errors
        # that confuses the optimiser. Thus save unscaled values separately
        if use_historical and os.path.isfile(history_scaled_path):
            history_scaled = np.loadtxt(history_scaled_path, dtype = float)
        elif use_historical:
            history_scaled = []
        else:
            history_scaled = None

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
            popsize = self.num_solutions,
            randn = lambda *args: rng.standard_normal(args),
            verbose = 3 if self.verbose else -9,
        ))

        # Start optimisation: ask the optimiser for parameter combinations
        # (solutions), run the simulation between `start_index:end_index` and
        # feed the results back to CMA-ES.
        epoch = 0

        while not es.stop():
            solutions = es.ask()

            # If we have historical data, inject it for each epoch
            if use_historical and \
                    epoch * self.num_solutions < len(history_scaled):

                self.inject_historical(es, history_scaled, epoch)
                epoch += 1

                if self.finished(es):
                    break

                continue

            if self.verbose:
                self.print_before_eval(es, solutions)

            results = self.try_solutions(
                rand_hash,
                solutions * scaling,
                save_positions = epoch if save_positions else None,
            )

            es.tell(solutions, results)
            epoch += 1

            # Save every step's historical data as function evaluations are
            # very expensive. Save columns [param1, param2, ..., stdev_param1,
            # stdev_param2, ..., stdev_all, error_val].
            if use_historical:
                if not isinstance(history, list):
                    history = list(history)

                for sol, res in zip(solutions, results):
                    history.append(
                        list(sol * scaling) +
                        list(es.result.stds * scaling) +
                        [es.sigma, res]
                    )

                np.savetxt(history_path, history)

                # Save scaled values separately to avoid numerical errors
                if not isinstance(history_scaled, list):
                    history_scaled = list(history_scaled)

                for sol, res in zip(solutions, results):
                    history_scaled.append(
                        list(sol) +
                        list(es.result.stds) +
                        [es.sigma, res]
                    )

                np.savetxt(history_scaled_path, history_scaled)

            if self.verbose:
                self.print_after_eval(es, solutions, scaling, results)

            if self.finished(es):
                break

        solutions = es.result.xbest * scaling
        param_names = sim.parameters.index

        # Change sigma, min and max based on optimisation results
        sim.parameters["sigma"] = es.result.stds * scaling

        if self.verbose:
            print(f"Best results for solutions: {solutions}", flush = True)

        # Change parameters to the best solution
        for i, sol_val in enumerate(solutions):
            sim[param_names[i]] = sol_val

        # Compute error for found solutions and move simulation forward in time
        checkpoints = np.linspace(
            self.start_time, self.end_time, self.num_checkpoints
        )

        positions = []
        for t in checkpoints:
            sim.step_to_time(t)
            positions.append(sim.positions())

        positions = np.array(positions, dtype = float)
        err = self.error(positions)

        if self.verbose:
            print((f"Error (computed by the `error` function) for solution: "
                   f"{err}\n---"), flush = True)

        return positions


    def create_directories(
        self,
        random_seed,
        rand_hash,
        use_historical,
        save_positions,
    ):
        sim = self.simulation

        # Save the current simulation state in a `restarts` folder
        if not os.path.isdir(f"access_info_{rand_hash}"):
            os.mkdir(f"access_info_{rand_hash}")

        # Save positions and parameters in a new folder inside `restarts`
        if not os.path.isdir(f"access_info_{rand_hash}/positions"):
            os.mkdir(f"access_info_{rand_hash}/positions")

        # Serialize the simulation's concrete class so that it can be
        # reconstructed even if it is a `coexist.Simulation` subclass
        with open(f"access_info_{rand_hash}/simulation_class.pickle",
                  "wb") as f:
            pickle.dump(sim.__class__, f)

        # Save current checkpoint and extra data for parallel computation
        sim.save(f"access_info_{rand_hash}/opt")
        with open(f"access_info_{rand_hash}/opt_run_info.txt", "a") as f:
            now = datetime.now().strftime("%H:%M:%S - %D")
            f.writelines([
                f"Starting ACCESS run at {now}\n",
                f"{sim}\n\n",
                f"start_time =          {self.start_time}\n",
                f"end_time =            {self.end_time}\n",
                f"target_sigma =        {self.target_sigma}\n",
                f"num_checkpoints =     {self.num_checkpoints}\n",
                f"num_solutions =       {self.num_solutions}\n",
                f"random_seed =         {random_seed}\n",
                f"use_historical =      {use_historical}\n",
                f"save_positions =      {save_positions}\n",
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
        num_params = len(self.simulation.parameters)

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
        cols = list(self.simulation.parameters.index) + ["error"]
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


    def std_outputs(self, rand_hash, proc_index, stdout, stderr):
        # If we had new errors, write them to `error.log`
        if len(stderr) and stderr != self._stderr:
            self._stderr = stderr.decode("utf-8")

            print((
                "A new error ocurred while running a simulation "
                "asynchronously:\n"
                f"{self._stderr}\n\n"
                "Writing error message to "
                f"`restarts_{rand_hash}/error_{proc_index}.log`\n"
            ))

            with open(
                f"access_info_{rand_hash}/error_{proc_index}.log", "w"
            ) as f:
                f.write(self._stderr)

        # If we had new outputs, write them to `output.log`
        if len(stdout) and stdout != self._stdout:
            self._stdout = stdout.decode("utf-8")

            print((
                "A new message was outputted while running a "
                "simulation asynchronously:\n"
                f"{self._stdout}\n\n"
                "Writing output message to "
                f"`restarts_{rand_hash}/output_{proc_index}.log`\n"
            ))

            with open(
                f"access_info_{rand_hash}/output_{proc_index}.log", "w"
            ) as f:
                f.write(self._stdout)


    def save_paths(self, rand_hash, save_positions = None):
        epoch = save_positions          # Either the epoch (int) or None

        sim_paths = []
        radii_paths = []
        positions_paths = []
        velocities_paths = []

        # Save current parameter values. If `save_positions` is given, then
        # save them to unique paths. Otherwise reuse the same ones
        if save_positions is not None:
            start_index = epoch * self.num_solutions
        else:
            start_index = 0

        for i in range(self.num_solutions):
            sim_paths.append((
                f"access_info_{rand_hash}/positions/"
                f"opt_{start_index + i}"
            ))

            radii_paths.append((
                f"access_info_{rand_hash}/positions/"
                f"opt_{start_index + i}_radii.npy"
            ))

            positions_paths.append((
                f"access_info_{rand_hash}/positions/"
                f"opt_{start_index + i}_positions.npy"
            ))

            velocities_paths.append((
                f"access_info_{rand_hash}/positions/"
                f"opt_{start_index + i}_velocities.npy"
            ))

        return sim_paths, radii_paths, positions_paths, velocities_paths


    def try_solutions(self, rand_hash, solutions, save_positions = None):
        # `save_positions` is either None or an int => optimisation solution
        # iteration (i.e. epoch)
        epoch = save_positions

        # Aliases
        sim = self.simulation
        param_names = sim.parameters.index

        # Path to `async_access_error.py`
        async_xi = os.path.join(
            os.path.split(coexist.__file__)[0],
            "async_access_error.py"
        )

        # For every solution to try, start a separate OS process that runs the
        # `async_access_error.py` file and saves the positions in a `.npy` file
        processes = []

        sim_paths, radii_paths, positions_paths, velocities_paths = \
            self.save_paths(rand_hash, save_positions)

        # Catch the KeyboardInterrupt (Ctrl-C) signal to shut down the spawned
        # processes before aborting.
        try:
            for i, sol in enumerate(solutions):

                # Change parameter values to save them along with the full
                # simulation state
                for j, sol_val in enumerate(sol):
                    sim.parameters.at[param_names[j], "value"] = sol_val

                sim.save(sim_paths[i])

                processes.append(
                    subprocess.Popen(
                        self.scheduler + [  # The Python interpreter path
                            async_xi,       # The `async_access_error.py` path
                            f"access_info_{rand_hash}/simulation_class.pickle",
                            sim_paths[i],
                            str(self.start_time),
                            str(self.end_time),
                            str(self.num_checkpoints),
                            radii_paths[i],
                            positions_paths[i],
                            velocities_paths[i],
                        ],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                    )
                )

            # Compute the error function values in a parallel environment.
            with ProcessPoolExecutor(max_workers = self.max_workers) \
                    as executor:
                futures = []

                # Get the output from each OS process / simulation
                for i, proc in enumerate(processes):
                    stdout, stderr = proc.communicate()

                    if epoch is not None:
                        proc_index = epoch * self.num_solutions + i
                    else:
                        proc_index = i

                    self.std_outputs(rand_hash, proc_index, stdout, stderr)

                    # Only load simulations if they exist - i.e. no errors
                    # occurred
                    if os.path.isfile(radii_paths[i]) and \
                            os.path.isfile(positions_paths[i]) and \
                            os.path.isfile(velocities_paths[i]):

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

                        print((
                            "At least one of the simulation files "
                            f"{radii_paths[i]}, {positions_paths[i]} or "
                            f"{velocities_paths[i]}, was not found; the "
                            "simulation most likely crashed. Check the error, "
                            "output and LIGGGHTS logs for what went wrong. "
                            "The error value for this simulation is set to "
                            "`NaN`."
                        ))

                # Crashed solutions will have np.nan as a value.
                results = np.full(len(solutions), np.nan)

                for i, f in enumerate(futures):
                    if f is not None:
                        results[i] = f.result()


        except KeyboardInterrupt:
            for proc in processes:
                proc.kill()

            sys.exit(130)

        return results

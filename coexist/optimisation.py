#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimisation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import  os
import  sys
import  textwrap
import  subprocess
import  pickle

import  numpy               as      np
import  pandas              as      pd
import  cma

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

        self.collisions = np.zeros(
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




class AccessInfo:
    '''Legacy class here only for backwards-compatibility in AccessData.legacy.
    Needs to be in `coexist/optimisation.py`.
    '''

    def __init__(
        self,
        parameters: pd.DataFrame,
        scheduler: list,
        num_solutions: int = None,
        target_sigma: float = None,
        random_seed: int = None,
    ):
        self.parameters = parameters
        self.scheduler = scheduler
        self.num_solutions = num_solutions
        self.target_sigma = target_sigma
        self.random_seed = random_seed

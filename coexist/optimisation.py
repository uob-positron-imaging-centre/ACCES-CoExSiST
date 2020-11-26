#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimisation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import  os
import  sys
import  textwrap
import  time
import  subprocess
import  pickle
import  warnings

import  numpy               as      np
import  pandas              as      pd
import  sympy
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

        self.xi = 0                 # Current timestep's error
        self.xi_acc = 0             # Multiple timesteps' (accumulated) error

        self.diverged = False
        self.num_solutions = None   # Set in `self.learn()`

        # A list of simulations that will be run in parallel when trying
        # different parameter combinations for the optimisation steps
        self.simulation_instances = []
        self.max_workers = None


    def learn(
        self,
        experiment: Experiment,
        max_optimisations = 3,
        num_solutions = 8,
        max_workers = None,
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

        max_workers: int, optional
            The maximum number of threads that can be used to run simulations
            in parallel during the optimisation step. If undefined, the return
            value from `os.cpu_count()` is used.

        verbose: bool, default True
            If True, print information about the optimisation to the terminal.

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

        max_optimisations = int(max_optimisations)

        # Setting class attributes
        self.experiment = experiment
        self.optimised_times = np.zeros(
            len(self.experiment.times),
            dtype = int,
        )

        self.num_solutions = int(num_solutions)

        if max_workers is None:
            self.max_workers = os.cpu_count()
        else:
            self.max_workers = int(max_workers)

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

        # Optimisation start / end timestep indices
        start_index = 0
        end_index = 0

        # Flag to keep track of divergence between sim and exp
        self.diverged = False

        # Nice printing of optimisation parameters
        if verbose:
            print((
                "    i |     xi     |   xi_acc   | checkpoints | "
                "  attached   | max_collisions \n"
                "------+------------+------------+-------------+-"
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

            if len(sim_pos) != 100:
                print("Sim pos not 100!")
                print(sim_pos)

            self.xi = self.calc_xi(sim_pos, exp_pos)
            self.xi_acc += self.xi

            # Only save checkpoints if the simulation is not diverging. A
            # checkpoint is a saved simulation state that will be used for
            # optimisation (which is done over multiple timesteps)
            if not self.diverging(i):
                sim.save()

                # Keep track of the index of the last saved checkpoint for
                # future optimisations
                start_index = i
                end_index = i

            if verbose:
                print((
                    f" {i:>4} | "
                    f" {self.xi:5.3e} | "
                    f" {self.xi_acc:5.3e} | "
                    f" {(i - start_index):>10} | "
                    f" {self.attached.sum():>4} / {sim.num_atoms():>4} | "
                    f" {self.collisions.max():>10}"
                ))

            if self.optimisable():
                # Load previous checkpoint's simulation
                sim.load()

                # Optimise against experimental positions from
                # start_index:end_index (excluding end_index)
                end_index = i + 1

                # Re-iterate the steps since the last checkpoint, optimising
                # the DEM parameters. This function will also run the sim up to
                # the current time t and set the new xi_acc
                self.optimise(start_index, end_index, verbose = verbose)

                # After optimisation, reset `collisions`, increment
                # `optimised_times` and reset the `diverged` flag
                self.collisions[:] = 0
                self.optimised_times[start_index:end_index] += 1
                self.diverged = False

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

        self.diverged = (self.xi > 1.6e-05)

        return self.diverged


    def optimisable(self):
        '''Check whether the simulation is in the optimum state to learn the
        DEM parameters.

        A simulation is optimisable if:

        1. All particles collided at least twice or any particle collided at
           least four times.
        2. The accumulated error (`xi_acc`) is larger than the sum of particle
           radii.

        Returns
        -------
        bool
            True if simulation is optimisable, False otherwise.
        '''

        # Aliases
        col = self.collisions

        # If all particles collided twice or any particle collided four times
        collided = ((col >= 2).all() or (col >= 6).any())

        return (collided and self.diverged)


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
            self.collisions[(error >= resolution) & (~self.attached)] += 1


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
            print("\nStarting optimisation of simulation:")
            print(self.simulation, "\n")

        # If logging is on, create a new entry for this optimisation as a tuple
        # containing:
        #   0. A list of [start_index, end_index]
        #   1. A list of solutions tried
        #   2. A list of results (xi_acc) for those solutions
        #   3. A list of [best_solution, best_result]
        if self.save_log:
            self.log.append(([start_index, end_index], [], [], []))

        # Aliases
        sim = self.simulation
        exp = self.experiment

        # Save the current simulation state in a `restarts` folder
        if not os.path.isdir("restarts"):
            os.mkdir("restarts")

        # Save current checkpoint and extra data for parallel computation
        rand_hash = str(round(np.random.random() * 1e8))
        sim.save(f"restarts/simopt_{rand_hash}")

        pickle.dump(
            exp,
            open(f"restarts/simopt_{rand_hash}_experiment.pickle", "wb")
        )

        # Minimum and maximum possible values for the DEM parameters
        params_mins = sim.parameters["min"].to_numpy()
        params_maxs = sim.parameters["max"].to_numpy()

        # If any `sigma` value is smaller than 5% (max - min), clip it
        p = sim.parameters
        sigma_clipped = p["sigma"].clip(lower = 0.05 * (p["max"] - p["min"]))

        # Scale sigma, bounds, solutions, results to unit variance
        scaling = sigma_clipped.to_numpy()

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
            verbose = 3 if verbose else -9,
        ))

        # Start optimisation: ask the optimiser for parameter combinations
        # (solutions), run the simulation between `start_index:end_index` and
        # feed the results back to CMA-ES.
        while not es.stop():
            solutions = es.ask()

            if verbose:
                print(f"Scaled sigma: {es.sigma}")
                print(f"Trying {len(solutions)} parameter combinations...")

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
                print(f"Function evaluations: {es.result.evaluations}\n---")

            # If logging is on, save the solutions tried and their results
            if self.save_log:
                self.log[-1][1].extend(solutions * scaling)
                self.log[-1][2].extend(results)

            if es.sigma < 0.05:
                if verbose:
                    print("Optimal solution found within 5%:")
                    print(f"sigma = {es.sigma} < 0.05")
                break

        solutions = es.result.xbest * scaling
        param_names = sim.parameters.index

        # Change sigma, min and max based on optimisation results
        sim.parameters["sigma"] = es.result.stds * scaling

        if verbose:
            print(f"Best results for solutions: {solutions}")

        # Change parameters to the best solution
        for i, sol_val in enumerate(solutions):
            sim[param_names[i]] = sol_val

        # Compute xi_acc for found solutions and move simulation forward in
        # time
        self.xi_acc = self.calc_xi_acc(sim, exp, start_index, end_index)

        # Delete the saved checkpoint data
        os.remove(f"restarts/simopt_{rand_hash}_restart.sim")
        os.remove(f"restarts/simopt_{rand_hash}_properties.sim")
        os.remove(f"restarts/simopt_{rand_hash}_experiment.pickle")

        if verbose:
            print((f"Accumulated error (xi_acc) for solution: "
                   f"{self.xi_acc}\n---"))

        if self.save_log:
            self.log[-1][3].extend([solutions, self.xi_acc])


    @staticmethod
    def sim_hash(solutions):
        h = hash(np.array(solutions).tobytes())
        return str(h)


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

        # Path to `async_calc_xi.py`
        async_xi = os.path.join(
            os.path.split(coexist.__file__)[0],
            "async_calc_xi.py"
        )

        # For every solution to try, start a separate OS process that runs the
        # `async_calc_xi.py` file and captures the output value
        processes = []
        for i, sol in enumerate(solutions):

            # Change parameters
            for j, sol_val in enumerate(sol):
                sim[param_names[j]] = sol_val

            # Save current parameter values
            pickle.dump(sim.parameters, open(
                f"restarts/simopt_{rand_hash}_{i}_parameters.pickle", "wb"
            ))

            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,     # The Python interpreter path
                        async_xi,           # The `async_calc_xi.py` path
                        f"restarts/simopt_{rand_hash}_{i}_parameters.pickle",
                        f"restarts/simopt_{rand_hash}",
                        f"restarts/simopt_{rand_hash}_experiment.pickle",
                        str(start_index),
                        str(end_index),
                    ],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                )
            )

        results = np.full(len(solutions), np.finfo(float).max)
        for i, proc in enumerate(processes):
            stdout, stderr = proc.communicate()
            results[i] = float(stdout)

        for i in range(len(solutions)):
            os.remove(f"restarts/simopt_{rand_hash}_{i}_parameters.pickle")

        return results







class Access:

    def __init__(
        self,
        simulation: Simulation,
        save_log = False,
    ):

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

        self.rng = None
        self.error = None

        self.start_time = None
        self.end_time = None

        self.num_checkpoints = None
        self.num_solutions = None


    def learn(
        self,
        error,
        start_time,
        end_time,
        num_checkpoints = 100,
        num_solutions = 10,
        use_historical = True,
        random_seed = None,
        verbose = True,
    ):
        # Type-checking input
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

        self.rng = np.random.default_rng(random_seed)

        if random_seed is None:
            use_historical = False
        else:
            use_historical = bool(use_historical)

        # Aliases
        sim = self.simulation
        rng = self.rng

        # If logging is on, create a new entry for this optimisation as a tuple
        # containing:
        #   0. A list of [start_time, end_time]
        #   1. A list of solutions tried
        #   2. A list of results (output from `error`) for those solutions
        #   3. A list of [best_solution, best_result]
        if self.save_log:
            self.log.append(([self.start_time, self.end_time], [], [], []))

        # Save the current simulation state in a `restarts` folder
        if not os.path.isdir("restarts"):
            os.mkdir("restarts")

        # Save current checkpoint and extra data for parallel computation
        rand_hash = str(round(rng.random() * 1e8))
        sim.save(f"restarts/simacc_{rand_hash}")

        # Check if we have historical data about the optimisation - these are
        # pre-computed values for this exact simulation, random seed, and
        # number of solutions
        history_path = f"restarts/history_{rand_hash}_{self.num_solutions}.csv"

        if use_historical and os.path.isfile(history_path):
            history = np.loadtxt(history_path, dtype = float)
        elif use_historical:
            history = []
        else:
            history = None

        # Minimum and maximum possible values for the DEM parameters
        params_mins = sim.parameters["min"].to_numpy()
        params_maxs = sim.parameters["max"].to_numpy()

        # If any `sigma` value is smaller than 5% (max - min), clip it
        p = sim.parameters
        sigma_clipped = p["sigma"].clip(lower = 0.05 * (p["max"] - p["min"]))

        # Scale sigma, bounds, solutions, results to unit variance
        scaling = sigma_clipped.to_numpy()

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
        i = 0

        while not es.stop():
            solutions = es.ask()

            if use_historical and i * self.num_solutions < len(history):
                ns = self.num_solutions
                results = history[(i * ns):(i * ns + ns)]

                solutions = np.array(solutions)

                if not np.isclose(solutions, results[:, :-1]).all():
                    warnings.warn(textwrap.fill((
                        "Historical data is different to solutions tried.\n"
                        f"Solutions:\n{solutions}\n"
                        f"Results:\n{results}\n---\n"
                    )))

                es.tell(solutions, results[:, -1])
                i += 1

                continue

            if verbose:
                print(f"Scaled sigma: {es.sigma}")
                print(f"Trying {len(solutions)} parameter combinations...")

            results = self.try_solutions(
                rand_hash,
                solutions * scaling,
            )

            es.tell(solutions, results)
            i += 1

            if use_historical:
                if not isinstance(history, list):
                    history = list(history)

                for sol, res in zip(solutions, results):
                    history.append(list(sol * scaling) + [res])

                np.savetxt(history_path, history)

            if verbose:
                cols = list(sim.parameters.index) + ["error"]
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
                print(f"Function evaluations: {es.result.evaluations}\n---")

            # If logging is on, save the solutions tried and their results
            if self.save_log:
                self.log[-1][1].extend(solutions * scaling)
                self.log[-1][2].extend(results)

            if es.sigma < 0.05:
                if verbose:
                    print("Optimal solution found within 5%:")
                    print(f"sigma = {es.sigma} < 0.05")
                break

        solutions = es.result.xbest * scaling
        param_names = sim.parameters.index

        # Change sigma, min and max based on optimisation results
        sim.parameters["sigma"] = es.result.stds * scaling

        if verbose:
            print(f"Best results for solutions: {solutions}")

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
            positions.append(sim.positions)

        positions = np.array(positions, dtype = float)
        err = self.error(positions)

        if verbose:
            print((f"Error (computed by the `error` function) for solution: "
                   f"{err}\n---"))

        if self.save_log:
            self.log[-1][3].extend([solutions, err])

        # Delete the saved checkpoint data
        os.remove(f"restarts/simacc_{rand_hash}_restart.sim")
        os.remove(f"restarts/simacc_{rand_hash}_properties.sim")

        return positions


    def try_solutions(self, rand_hash, solutions):
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
        for i, sol in enumerate(solutions):

            # Change parameters
            for j, sol_val in enumerate(sol):
                sim[param_names[j]] = sol_val

            # Save current parameter values
            pickle.dump(sim.parameters, open(
                f"restarts/simacc_{rand_hash}_{i}_parameters.pickle", "wb"
            ))

            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,     # The Python interpreter path
                        async_xi,           # The `async_access_error.py` path
                        f"restarts/simacc_{rand_hash}_{i}_parameters.pickle",
                        f"restarts/simacc_{rand_hash}",
                        str(self.start_time),
                        str(self.end_time),
                        str(self.num_checkpoints),
                        f"restarts/simacc_{rand_hash}_{i}_positions.npy",
                    ],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                )
            )

        positions_all = []
        for i, proc in enumerate(processes):
            stdout, stderr = proc.communicate()

            positions_all.append(
                np.load(f"restarts/simacc_{rand_hash}_{i}_positions.npy")
            )

        results = np.array([self.error(pos) for pos in positions_all])

        for i in range(len(solutions)):
            os.remove(f"restarts/simacc_{rand_hash}_{i}_parameters.pickle")
            os.remove(f"restarts/simacc_{rand_hash}_{i}_positions.npy")

        return results







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

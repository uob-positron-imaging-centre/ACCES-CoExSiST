#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_coupled.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.01.2022


import  os
import  sys
import  textwrap
import  subprocess
import  pickle
from    datetime            import  datetime, timedelta
from    concurrent.futures  import  ProcessPoolExecutor
import  numpy               as      np
import  pandas              as      pd
import  cma
from    .base               import  Simulation




class AccessCoupled:

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
            `legacy.Simulation` interface (e.g. `LiggghtsSimulation`).
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
                "`legacy.Simulation` (or a subclass thereof) or a list "
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
        '''Save the radii, positions, and velocities for the simulations with
        the best parameter values.
        '''
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
        # reconstructed even if it is a `legacy.Simulation` subclass
        for i in range(len(self.simulations)):
            with open(self.sim_classes_paths[i], "wb") as f:
                pickle.dump(self.simulations[i].__class__, f)

            # Save current checkpoint and extra data for parallel computation
            self.simulations[i].save(self.sim_paths[i])

        infofile = f"{self.save_path}/opt_run_info.txt"
        with open(infofile, "a", encoding = "utf-8") as f:
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
        '''Print the individual and overal scaled standard deviations along
        with the parameter combinations to try.
        '''
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
        '''If the overal sigma value is less than the target sigma value,
        finish the simulation.
        '''
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
        '''If new errors and outputs are produced, write them to the correct
        olders.
        '''
        # If we had new errors, write them to `error.log`.
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
        '''For every simulation run in every parameter combination, append the
        simulation, radii, position and velocity path to a corresponding list.
        '''
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
        '''For every solution to try and simulate a run, start a separate OS
        process that runs the `async_access_error.py` file and saves the
        positions in a `.npy` file.
        '''

        # Aliases
        param_names = self.simulations[0].parameters.index

        # Path to `async_access_error.py`
        async_xi = os.path.join(
            os.path.dirname(__file__),
            "async_access_error.py"
        )

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.01.2022


import  re
import  os
import  sys
import  time
import  textwrap
import  contextlib
import  subprocess
import  pickle
from    datetime            import  datetime
from    concurrent.futures  import  ProcessPoolExecutor

import  numpy               as      np
import  pandas              as      pd
import  toml
import  cma

import  coexist

from    .base               import  Simulation
from    .                   import  schedulers

from    .combiners          import  Product

from    .code_trees         import  code_contains_variable
from    .code_trees         import  code_substitute_variable

from    .utilities          import  autorepr
from    .utilities          import  SignalHandlerKI




signal_handler = SignalHandlerKI()




@autorepr(short = {"script", "scheduler_cmd"})
class AccessSetup:
    '''Structure storing constant attributes for an ACCES optimisation run.

    Code validation and generation are handled too.

    Attributes
    ----------
    parameters : pd.DataFrame
        The free parameters extracted from the user script.

    parameters_scaled : pd.DataFrame
        The free parameters scaled to the phenotype space, such that the
        initial variance (`sigma`) is unity.

    scaling : np.ndarray
        A vector of values that the free parameters are scaled by; it is the
        initial variance (`sigma`) given by the user.

    script : str
        The modified user script that will be executed.

    scheduler_cmd : list
        The list of commands the will be prepended to each script execution,
        defined by the `Scheduler`.

    population : int
        The number of simulations to be run in parallel in each epoch.

    target : float
        The target scaled variance - i.e. decrease the uncertainty from the
        initial 1 down to `target`.

    seed: int
        The random seed defining a single ACCES run.

    rng: np.random.Generator
        The random number generator used, seeded with `seed`.
    '''

    def __init__(self, script_path, scheduler):
        '''Given a path to a user-defined simulation script, extract the free
        parameters and generate the ACCES script.
        '''
        # Uninitialised parameters (will be set later)
        self.population = None
        self.target = None
        self.rng = None
        self.seed = None

        # Type-check and generate scheduler commands
        if not isinstance(scheduler, schedulers.Scheduler):
            raise TypeError(textwrap.fill((
                "The input `scheduler` must be a subclass of `coexist."
                f"schedulers.Scheduler`. Received {type(scheduler)}."
            )))

        self.scheduler_cmd = scheduler.generate()

        # Extract parameters and generate ACCES script
        with open(script_path, "r") as f:
            user_code = f.readlines()

        # Find the two parameter definition directives
        params_start_line = None
        params_end_line = None

        regex_prefix = r"#+\s*ACCES{1,2}\s+PARAMETERS"
        params_start_finder = re.compile(regex_prefix + r"\s+START")
        params_end_finder = re.compile(regex_prefix + r"\s+END")

        for i, line in enumerate(user_code):
            if params_start_finder.match(line):
                params_start_line = i

            if params_end_finder.match(line):
                params_end_line = i

        if params_start_line is None or params_end_line is None:
            raise NameError(textwrap.fill((
                f"The user script found in file `{script_path}` did not "
                "contain the blocks `# ACCESS PARAMETERS START` and "
                "`# ACCESS PARAMETERS END`. Please define your simulation "
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
                "`# ACCESS PARAMETERS START` and "
                "`# ACCESS PARAMETERS END` does not define a variable "
                "named exactly `parameters`."
            )))

        self.validate_parameters(user_params_exec["parameters"])
        self.parameters = user_params_exec["parameters"]

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
            'access_id = int(sys.argv[1].split(".")[-2])\n'
        )

        # Read in the `async_access_template.py` code template and find the
        # code injection directives
        template_code_path = os.path.join(
            os.path.split(coexist.__file__)[0],
            "template_access_script.py"
        )

        with open(template_code_path, "r") as f:
            template_code = f.readlines()

        for i, line in enumerate(template_code):
            if line.startswith("# ACCESS INJECT USER CODE START"):
                inject_start_line = i

            if line.startswith("# ACCESS INJECT USER CODE END"):
                inject_end_line = i

        generated_code = "".join((
            template_code[:inject_start_line + 1] +
            user_code[:params_start_line + 1] +
            parameters_code +
            user_code[params_end_line:] +
            template_code[inject_end_line:]
        ))

        self.script = generated_code

        # Scale free parameters (+ bounds and sigma) to unit variance
        self.scaling = self.parameters["sigma"].to_numpy().copy()
        self.parameters_scaled = self.parameters.copy()
        for i in range(len(self.parameters_scaled.columns)):
            self.parameters_scaled.iloc[:, i] /= self.scaling


    @staticmethod
    def validate_parameters(parameters):
        '''Validate the free parameters extracted from a user script (a
        ``pandas.DataFrame``).
        '''
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
        if not all(c in parameters.columns for c in columns_needed):
            raise ValueError(textwrap.fill((
                "The `parameters` DataFrame defined in the user script must "
                "have at least four columns defined: ['value', 'min', "
                f"'max', 'sigma']. Found these: `{parameters.columns}`. You "
                "can use the `coexist.create_parameters` function for this."
            )))


    def setup_complete(self, population, target, seed):
        '''Set up the final attributes before starting the ACCES run - i.e.
        the ones set in the ``Access.learn`` method.
        '''
        # Type-checking inputs and setting attributes
        self.population = int(population)
        self.target = float(target)

        if seed is None:
            self.seed = np.random.randint(1000, 10_000)
        else:
            self.seed = int(seed)

        self.rng = np.random.default_rng(self.seed)


    def starting_guess(self):
        '''Return the initial parameter combinations to start CMA-ES with.
        '''
        # First guess, scaled
        x0 = self.parameters_scaled["value"].to_numpy()
        bounds = [
            self.parameters_scaled["min"].to_numpy(),
            self.parameters_scaled["max"].to_numpy(),
        ]

        return x0, bounds




@autorepr
class AccessPaths:
    '''Structure handling IO and storing all paths relevant for an ACCES run.

    Loading and saving epochs and history are handled too.

    Attributes
    ----------
    directory : str
        Path to the ACCES directory, e.g. ``access_seed123``.

    results : str
        Path to the results directory.

    outputs : str
        Path to the outputs directory.

    script : str
        Path to the ACCES-modified user script.

    setup : str
        Path to the saved ACCES setup.

    epochs : str
        Path to the epochs CSV data file.

    epochs_scaled : str
        Path to the scaled epochs CSV data file.

    history : str
        Path to the historical CSV data file.

    history_scaled : str
        Path to the scaled historical CSV data file.
    '''

    def __init__(
        self,
        directory: str = None,
        results: str = None,
        outputs: str = None,
        script: str = None,
        setup: str = None,
        epochs: str = None,
        epochs_scaled: str = None,
        history: str = None,
        history_scaled: str = None,
    ):

        self.directory = directory
        self.results = results
        self.outputs = outputs
        self.script = script
        self.setup = setup
        self.epochs = epochs
        self.epochs_scaled = epochs_scaled
        self.history = history
        self.history_scaled = history_scaled


    def create_directories(self, access):
        '''Given a ``coexist.Access`` instance, create the required directory
        hierarchy for a single ACCES run.
        '''
        # Include the random seed used in the `access_seed<seed>` dirpath
        self.directory = f"access_seed{access.setup.seed}"
        self.results = os.path.join(self.directory, "results")
        self.outputs = os.path.join(self.directory, "outputs")

        if access.verbose >= 3:
            now = datetime.now().strftime(r"%H:%M:%S on %d/%m/%Y")
            print(
                "\n" + "=" * 80 + "\n" +
                f"Starting ACCES run at {now} in directory "
                f"`{self.directory}`.",
                flush = True,
            )

        # Include the population size in the history filename to ensure future
        # runs don't accidentally use wrong numbers of solutions per epoch
        #
        # Results history
        self.history = os.path.join(
            self.directory,
            f"history_pop{access.setup.population}.csv",
        )

        self.history_scaled = os.path.join(
            self.directory,
            f"history_pop{access.setup.population}_scaled.csv",
        )

        # Per-epoch data
        self.epochs = os.path.join(
            self.directory,
            f"epochs_pop{access.setup.population}.csv",
        )

        self.epochs_scaled = os.path.join(
            self.directory,
            f"epochs_pop{access.setup.population}_scaled.csv",
        )

        self.script = os.path.join(self.directory, "access_script.py")
        self.setup = os.path.join(self.directory, "access_setup.toml")

        # Create directories
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        elif access.verbose >= 3:
            pass

        if not os.path.isdir(self.results):
            os.mkdir(self.results)

        if not os.path.isdir(self.outputs):
            os.mkdir(self.outputs)

        # Save information about the run
        now = datetime.now().strftime("%H:%M:%S on %d/%m/%Y")

        logfile = os.path.join(self.directory, "loginfo.txt")
        with open(logfile, "a", encoding = "utf-8") as f:
            f.write((
                80 * "-" + "\n" +
                f"Starting ACCESS run at {now}\n\n" +
                access.__repr__() + "\n"
            ))

        readmefile = os.path.join(self.directory, "readme.rst")
        with open(readmefile, "w", encoding = "utf-8") as f:
            f.write(textwrap.dedent(f'''
                ACCES Optimisation Run Directory
                --------------------------------

                This directory was generated by ACCES at {now}.

                You can load the data saved by ACCES into a tidy Python object
                using ``coexist.AccessData(<dirpath>)``; alternatively, the
                calibration / optimisation results may be read from the CSV
                files (see below) in any external program.

                All data can be accessed as the calibration progresses to check
                intermediate results. This `{self.directory}` directory
                is self-contained and may be archived / moved outside the
                initial ACCES folder.

                You are welcome to read and use the data saved here, but
                modifying or removing files is not recommended (except for the
                files inside `outputs` and `results`, see below).


                File Hierarchy
                --------------

                For a given simulation, ACCES runs are uniquely determined by a
                seed for the stochastic algorithms and the number of parameter
                combinations to try in one epoch (i.e. the population size).
                The resulting hierarchy looks like this:

                ::

                    access_seed<seed_number>
                    ├── access_script.py
                    ├── access_setup.toml
                    ├── epochs_pop<population_size>.csv
                    ├── epochs_pop<population_size>_scaled.csv
                    ├── history_pop<population_size>.csv
                    ├── history_pop<population_size>_scaled.csv
                    ├── loginfo.txt
                    ├── readme.rst
                    ├── outputs
                    │   ├── stderr.0.log
                    │   ...
                    │   ├── stdout.0.log
                    │   ...
                    └── results
                        ├── parameters.0.pickle
                        ...
                        ├── result.0.pickle
                        ...

                The `access_seed<seed_number>` is the current directory where
                ACCES saves simulation results; naturally, don't execute
                multiple ACCES runs with the same random seeds in the same
                directory.

                The `access_script.py` Python script is the modified user-code
                that will be executed for each function evaluation.

                The `access_setup.toml` file saves the constant ACCES objects
                for a given run: relevant paths, parameters, population size,
                target uncertainty and random seed. The TOML format it uses is
                human-readable and may be loaded into a Python ``dict`` using:

                .. code-block:: python

                    import toml
                    with open("filepath.toml") as f:
                        obj = toml.load(f)

                The `epochs_pop<population_size>.csv` file stores relevant data
                for each epoch: the current parameter optimum estimates and
                uncertainties; the other `_scaled.csv` file stores the same,
                but scaled to the internal CMA-ES "phenotype space". Notably,
                the uncertainties are scaled between 0 (perfect estimate) and
                1 (the initial estimate); if the parameter response is very
                nonlinear / convolved / inexistent, the uncertainty may
                increase beyond 1. The column names are given in the file; the
                number of rows is equal to the number of epochs completed.

                The `history_pop<population_size>.csv` file stores the history
                of ACCES results: the parameter combinations tried and the
                evaluated error values for each epoch.  The other `_scaled.csv`
                file stores the same, but scaled to the internal CMA-ES
                "phenotype space". The column names are given in the file; all
                epochs are concatenated, so the number of rows will be the
                population size multiplied by the number of executed epochs.

                The `loginfo.txt` file logs when ACCES runs are (re)started.
                This `readme.rst` file is self-explanatory - you're reading it!

                The `outputs` directory stores the captured `stdout` and
                `stderr` messages that were emitted during each simulation.

                The `results` directory stores the parameter values tried for
                each simulation (e.g. `parameters.0.pickle`) and the
                corresponding error value found (e.g. `result.0.pickle`).

                The `outputs` and `results` directories are only needed for
                logging purposes; you may safely remove the files inside *only
                for the completed ACCES epochs*.
            '''))


    def update_paths(self, prefix):
        '''Translate all paths saved in this class relative to a new `prefix`
        (which will replace the `directory` attribute).

        Please ensure that the `prefix` directory contains the required ACCES
        files.
        '''

        self.directory = prefix
        for attr in ["results", "outputs", "script", "setup", "epochs",
                     "epochs_scaled", "history", "history_scaled"]:
            prev = getattr(self, attr)
            setattr(self, attr, os.path.join(prefix, os.path.split(prev)[1]))


    def save_history(self, setup, progress):
        '''Given an ``AccessSetup`` and ``AccessProgress`` instance, save the
        results history.
        '''
        np.savetxt(
            self.history,
            progress.history,
            header = " ".join(setup.parameters.index.to_list() + ["error"]),
        )

        np.savetxt(
            self.history_scaled,
            progress.history_scaled,
            header = " ".join(setup.parameters.index.to_list() + ["error"]),
        )


    def load_history(self, access):
        '''Load previous results into ``access.progress``.
        '''
        # History columns: [param1, param2, ..., error_value] for each function
        # evaluation
        if os.path.isfile(self.history):
            access.progress.history = np.loadtxt(
                self.history, dtype = float
            )
        else:
            access.progress.history = None

        # Scaling and unscaling parameter values introduce numerical errors
        # that confuse the optimiser. Thus save unscaled values separately
        if os.path.isfile(self.history_scaled):
            if access.verbose >= 3:
                print(
                    "Found previous ACCES results in " +
                    f"`{self.history_scaled}`.\n" + "=" * 80 + "\n",
                    flush = True,
                )

            access.progress.history_scaled = np.loadtxt(
                self.history_scaled, dtype = float
            )
        else:
            access.progress.history_scaled = None


    def save_epochs(self, setup, progress):
        '''Given an ``AccessSetup`` and ``AccessProgress`` instance, save the
        optimisation epochs.
        '''
        np.savetxt(
            self.epochs,
            progress.epochs,
            header = " ".join(
                [f"{p}_mean" for p in setup.parameters.index] +
                [f"{p}_std" for p in setup.parameters.index] +
                ["overall_std"]
            ),
        )

        np.savetxt(
            self.epochs_scaled,
            progress.epochs_scaled,
            header = " ".join(
                [f"{p}_mean" for p in setup.parameters.index] +
                [f"{p}_std" for p in setup.parameters.index] +
                ["overall_std"]
            ),
        )


    def load_epochs(self, access):
        '''Given an ``Access`` instance, load previous ACCES runs' `epochs` and
        `epochs_scaled` into ``access.progress``.
        '''
        # Epochs columns: [param1_mean, param2_mean, ..., param1_std,
        # param2_std,..., overall_std] for each epochs
        if os.path.isfile(self.epochs):
            access.progress.epochs = np.loadtxt(
                self.epochs, dtype = float
            )
        else:
            access.progress.epochs = np.empty(
                (0, 2 * len(access.setup.parameters) + 1)
            )

        if os.path.isfile(self.epochs_scaled):
            access.progress.epochs_scaled = np.loadtxt(
                self.epochs_scaled, dtype = float
            )
        else:
            access.progress.epochs_scaled = np.empty(
                (0, 2 * len(access.setup.parameters) + 1)
            )




@autorepr(short = True)
class AccessProgress:
    '''Structure saving the current ACCES optimisation progress.

    The `epochs` array has columns [mean_param1, mean_param2, ..., std_param1,
    std_param2, ..., std_overall] for each epoch.

    The `history` array has columns [param1, param2, ..., error] for each
    function evaluation.

    Attributes
    ----------
    epochs: np.ndarray
        Matrix with columns [mean_param1, mean_param2, ..., std_param1,
        std_param2, ..., std_overall] with one row per epoch.

    epochs_scaled: np.ndarray
        Same as ``epochs``, scaled such that the initial variance (``sigma``)
        becomes unity.

    history: np.ndarray = None
        Matrix with columns [param1, param2, ..., error] for each parameter
        combination tried - i.e. ``population * num_epochs``.

    history_scaled: np.ndarray = None
        Same as ``history``, scaled such that the initial variance (``sigma``)
        becomes unity.

    stdout: str = None
        The latest unique recorded stdout message.

    stderr: str = None
        The latest unique recorded stderr message.
    '''

    def __init__(
        self,
        epochs: np.ndarray = None,
        epochs_scaled: np.ndarray = None,
        history: np.ndarray = None,
        history_scaled: np.ndarray = None,
        stdout: str = None,
        stderr: str = None,
    ):
        self.epochs = None
        self.epochs_scaled = None
        self.history = None
        self.history_scaled = None
        self.stdout = None
        self.stderr = None


    def update_epochs(self, es, scaling):
        '''Update each epoch array after an ACCES run has been completed.
        '''
        self.epochs = np.vstack((
            self.epochs,
            np.hstack((
                es.result.xfavorite * scaling,
                es.result.stds * scaling,
                es.sigma,
            )),
        ))

        self.epochs_scaled = np.vstack((
            self.epochs_scaled,
            np.hstack([es.result.xfavorite, es.result.stds, es.sigma]),
        ))


    def update_history(self, es, scaling, solutions, results):
        '''Update the ACCES history with the latest simulation solutions and
        results.
        '''
        solutions = np.asarray(solutions)
        current = np.c_[solutions * scaling, results]
        current_scaled = np.c_[solutions, results]
        # Check that we have the correct number of columns - if all simulations
        # in an epoch crashed, we'll have a single error column filled with NaN
        history = self.history
        history_scaled = self.history_scaled

        if self.history is None:
            # First epoch
            self.history = current
            self.history_scaled = current_scaled
            return

        if np.isnan(self.history[:, -1]).all():
            # History does not have enough columns - pad with NaNs
            pad = np.full((
                self.history.shape[0],
                current.shape[1] - self.history.shape[1],
            ), np.nan)
            history = np.c_[self.history, pad]
            history_scaled = np.c_[self.history_scaled, pad]

        if np.isnan(current[:, -1]).all():
            # Current does not have enough columns - pad with NaNs
            pad = np.full((
                current.shape[0],
                self.history.shape[1] - current.shape[1],
            ), np.nan)
            current = np.c_[current, pad]
            current_scaled = np.c_[current_scaled, pad]

        self.history = np.vstack((history, current))
        self.history_scaled = np.vstack((history_scaled, current_scaled))


    def gather_results(
        self,
        processes,
        paths,
        result_paths,
        multi_objective,
        verbose,
    ):
        '''Check whether the jobs have finished and retrieve the standard
        deviation, errors and the combined total error.
        '''
        results = []
        stdout_rec = []
        stderr_rec = []
        crashed = []

        # Occasionally check if jobs finished
        wait = 0.1          # Time between checking results
        waited = 0.         # Total time waited
        logged = 0          # Number of times logged remaining simulations
        tlog = 30 * 60      # Time until logging remaining simulations again

        while wait != 0:
            done = sum((p.poll() is not None for p in processes))

            if done == len(processes):
                wait = 0
                for i, proc in enumerate(processes):
                    proc_index = int(proc.args[-1].split(".")[-2])
                    stdout, stderr = proc.communicate()

                    # If a *new* output message was recorded in stdout, log it
                    if len(stdout) and stdout != self.stdout:
                        stdout_rec.append(proc_index)
                        self.stdout = stdout

                        stdout_log = os.path.join(
                            paths.outputs,
                            f"stdout.{proc_index}.log",
                        )
                        with open(stdout_log, "w") as f:
                            f.write(self.stdout)

                    # If a *new* error message was recorded in stderr, log it
                    if len(stderr) and stderr != self.stderr:
                        stderr_rec.append(proc_index)
                        self.stderr = stderr

                        stderr_log = os.path.join(
                            paths.outputs,
                            f"stderr.{proc_index}.log",
                        )
                        with open(stderr_log, "w") as f:
                            f.write(self.stderr)

                    # Load result if the file exists, otherwise set it to NaN
                    if os.path.isfile(result_paths[i]):
                        with open(result_paths[i], "rb") as f:
                            errors = pickle.load(f)
                            if hasattr(errors, "__iter__"):
                                errors = np.array(errors, dtype = float)
                            else:
                                errors = np.array([errors], dtype = float)

                            combined = multi_objective.combine(errors)
                            results.append(np.append(errors, combined))
                    else:
                        results.append(None)
                        crashed.append(proc_index)

            # Every `remaining` seconds print remaining jobs
            if verbose >= 4 and wait != 0 and waited > (logged + 1) * tlog:
                logged += 1
                tlog *= 1.5

                remaining = " ".join([
                    p.args[-1].split(".")[-2]
                    for p in processes if p.poll() is None
                ])

                minutes = int(waited / 60)
                if minutes > 60:
                    timer = f"{minutes // 60} h {minutes % 60} min"
                else:
                    timer = f"{minutes} min"

                print((
                    f"  * Remaining jobs after {timer}:\n" +
                    textwrap.indent(textwrap.fill(remaining), "  * ")
                ), flush = True)

            # Wait for increasing numbers of seconds until checking for results
            # again - at most 1 minute
            time.sleep(wait)
            waited += wait
            wait = min(wait * 1.5, 60)

        return results, stdout_rec, stderr_rec, crashed




@autorepr
class Access:
    '''Optimise an arbitrary user-defined script's parameters in parallel.

    A minimal user script - saved in a separate file - would be:

    ::

        # ACCESS PARAMETERS START
        import coexist

        parameters = coexist.create_parameters(
            variables = ["fp1", "fp2"],
            minimums = [-3, -7],
            maximums = [+5, +3],
        )

        access_id = 0                           # Optional
        # ACCESS PARAMETERS END

        x = parameters.at["fp1", "value"]
        y = parameters.at["fp2", "value"]

        error = x ** 2 + y ** 2

    This script defines two free parameters to optimise "fp1" and "fp2" with
    ranges [-3, +5] and [-7, +3] and saves an error value to be optimised
    in the variable `error`. To optimise it, run in another file:

    ::

        import coexist

        access = coexist.Access("script_filepath.py")
        access.learn(num_solutions = 10, target_sigma = 0.1, random_seed = 42)

    Once you run `access.learn()`, a folder named "access_42" is
    generated which stores all information about this access run, including all
    simulation data. You can load this data using ``coexist.AccessData.read``
    even while the optimisation is still running.

    In general, an ACCESS user script must define one simulation whose
    parameters will be optimised this way:

    1. Use a variable named "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       An initial guess can also be set here.

    2. The `parameters` creation should be **fully self-contained** between two
       ``# ACCESS PARAMETERS START`` and ``# ACCESS PARAMETERS END`` comments -
       i.e. it should not depend on code ran before the block.

    3. By the end of the simulation script, define a variable named ``error``
       storing a single number representing this simulation's error value.

    Notice that there is no limitation on how the error value is calculated. It
    can be any simulation, executed in any way - even externally; just launch
    a separate process from the Python script, run the simulation, extract data
    back into the Python script and set ``error`` to what you need optimised.

    If you need to save data to disk, use file names containing the
    ``access_id`` variable which is set to a unique integer ID for each
    simulation, so that you don't overwrite existing files when simulations are
    executed in parallel.

    For more information on the implementation details and how parallel
    execution of your user script is achieved, check out the generated file
    "access_seed<seed>/access_script.py" after running `access.learn()`.

    Attributes
    ----------
    script : str
        The ACCES code generated from the user-supplied script.

    setup : coexist.access.AccessSetup
        Structure storing given ACCES configuration, containing the free
        ``parameters``, scheduler commands ``scheduler_cmd``, number of
        solutions to try in parallel ``population``, target uncertainty
        ``target``, seeded random number generator ``rng`` and the ``seed``.

    paths : coexist.access.AccessPaths
        Structure storing paths to the ACCES ``directory``, saved ``state``,
        ``simulations`` tried, captured ``outputs`` directory, and paths to
        the previous results ``history`` and ``history_scaled``.

    progress : coexist.access.AccessProgress
        Structure storing ACCES optimisation run progress - ``epochs``,
        ``history`` (and CMA-scaled versions of them) and latest ``stdout`` and
        ``stderr`` messages.

    verbose : int
        Integer denoting the level of verbosity, where 0 is quiet and 5 is
        maximally verbose.
    '''

    def __init__(
        self,
        script_path: str,
        scheduler = schedulers.LocalScheduler(),
    ):
        '''`Access` class constructor.

        Parameters
        ----------
        script_path : str
            A path to a user-defined script that runs one simulation. It should
            use the free / optimisable parameters saved in a pandas.DataFrame
            named exactly ``parameters``, defined between two comments
            ``# ACCESS PARAMETERS START`` and ``# ACCESS PARAMETERS END``.
            By the end of the script, one variable named `error` must be
            defined containing the error value, a number.

        scheduler : coexist.schedulers.Scheduler subclass
            Scheduler used to spawn function evaluations / simulations in
            parallel. The default ``LocalScheduler`` simply starts new Python
            interpreters on the local machine for executing the user's script.
            See the other schedulers in `coexist.schedulers` for e.g. spawning
            jobs on a supercomputing cluster.
        '''

        # Creating class attributes
        self.setup = AccessSetup(script_path, scheduler)
        self.paths = AccessPaths()
        self.progress = AccessProgress()
        self.multi_objective = None
        self.verbose = None


    def learn(
        self,
        num_solutions = 8,
        target_sigma = 0.1,
        random_seed = None,
        multi_objective = Product(),
        verbose = 4,
    ):
        '''Learn the free `parameters` from the user script that minimise the
        `error` variable by trying `num_solutions` parameter combinations at
        a time until the overall uncertainty becomes lower than `target_sigma`.

        For `multi_objective` optimisation, use a `coexist.combiner` to combine
        multiple error values into a single one.
        '''

        # Type-checking inputs
        if not hasattr(multi_objective, "combine"):
            raise TypeError(textwrap.fill((
                "The input `mulit_objective` has no attribute `combine`. "
                "Check you are using a `coexist.combiner` to combine "
                "multiple error values into a single combined error."
            )))

        # Set last setup attributes and create ACCES directories
        self.verbose = int(verbose)
        self.setup.setup_complete(num_solutions, target_sigma, random_seed)
        self.paths.create_directories(self)
        self.multi_objective = multi_objective

        # Save this ACCES run's files
        self.save_setup()

        # Load previous history and epochs into self.progress
        self.paths.load_history(self)
        self.paths.load_epochs(self)

        # Scale sigma, bounds, solutions, results to unit variance
        scaling = self.setup.scaling
        x0, bounds = self.setup.starting_guess()
        sigma0 = 1.

        # Instantiate CMA-ES optimiser; silence initial CMA-ES message
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
                bounds = bounds,
                popsize = self.setup.population,
                randn = lambda *args: self.setup.rng.standard_normal(args),
                verbose = 3 if self.verbose >= 3 else -9,
            ))
            es.logger = cma.CMADataLogger(
                os.path.join(self.paths.directory, "cache", "")
            )

        # Start optimisation: ask the optimiser for parameter combinations
        # (solutions), run the simulations between `start_index:end_index` and
        # feed the results back to CMA-ES.
        epoch = 0

        while not es.stop():
            solutions = es.ask()

            # If we have historical data, inject it for each epoch
            if self.has_historical(epoch):
                self.inject_historical(es, epoch)
                epoch += 1

                if self.finished(es):
                    break
                continue

            if self.verbose >= 2:
                self.print_before_eval(es, epoch, scaling)

            # Save current epoch's mean, sigma
            self.progress.update_epochs(es, scaling)

            # Evaluate each solution - i.e. run simulations in parallel
            results = self.evaluate_solutions(solutions * scaling, epoch)
            es.tell(solutions, results[:, -1])
            epoch += 1

            # Save historical data as function evaluations are very expensive
            self.progress.update_history(es, scaling, solutions, results)

            self.paths.save_epochs(self.setup, self.progress)
            self.paths.save_history(self.setup, self.progress)

            if self.verbose >= 2:
                self.print_after_eval(es, epoch, solutions, scaling, results)

            if self.finished(es):
                break

        if es.result.xbest is None:
            raise ValueError(textwrap.fill((
                "No parameter combination was evaluated successfully. All "
                "simulations crashed - please check the error logs in the "
                f"`{self.paths.outputs}` folder."
            )))

        if self.verbose >= 1:
            self.print_finished(es, scaling)

        return AccessData.read(self.paths.directory)


    def save_setup(self):
        '''Save current ACCES run's modified script (py) and state (toml).
        '''

        with open(self.paths.script, "w") as f:
            f.write(self.setup.script)

        setup_dict = dict(
            paths = self.paths.__dict__,
            setup = dict(
                parameters = self.setup.parameters.to_dict(),
                parameters_scaled = self.setup.parameters_scaled.to_dict(),
                scaling = self.setup.scaling.tolist(),
                population = self.setup.population,
                target = self.setup.target,
                seed = self.setup.seed,
            ),
        )

        with open(self.paths.setup, "w") as f:
            toml.dump(setup_dict, f)


    def has_historical(self, epoch):
        '''Check ACCES still has historical solutions to inject.
        '''

        if (
            self.progress.history_scaled is not None and
            epoch * self.setup.population < len(self.progress.history_scaled)
        ):
            return True
        return False


    def inject_historical(self, es, epoch):
        '''Inject the CMA-ES optimiser with pre-computed (historical) results.
        The solutions must have a Gaussian distribution in each problem
        dimension - though the standard deviation can vary for each of them.
        Ideally, this should only use historical values that CMA-ES asked for
        in a previous ACCESS run.
        '''

        pop = self.setup.population
        num_params = len(self.setup.parameters)

        results_scaled = self.progress.history_scaled[
            (epoch * pop):(epoch * pop + pop)
        ]
        es.tell(results_scaled[:, :num_params], results_scaled[:, -1])

        if self.verbose >= 1:
            ns = len(self.progress.history_scaled)
            maxlen = len(str(ns))
            print((
                f"Injected {(epoch + 1) * len(results_scaled):>{maxlen}} / "
                f"{ns} historical solutions"
            ))


    def print_before_eval(self, es, epoch, scaling):
        '''Print current estimates before evaluating current epoch.
        '''
        info = pd.DataFrame(
            np.vstack((
                es.result.xfavorite * scaling,
                es.result.stds * scaling,
                es.result.stds,
            )),
            index = ["estimate", "uncertainty", "scaled_std"],
            columns = self.setup.parameters.index,
        )

        # Display all the DataFrame columns and rows
        old_max_columns = pd.get_option("display.max_columns")
        old_max_rows = pd.get_option("display.max_rows")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        head = "=" * 80
        line = "-" * 80
        print((
            f"{head}\n"
            f"Epoch {epoch:>4} | Population {self.setup.population}\n"
            f"{line}\n"
            f"Scaled overall standard deviation: {es.sigma}\n"
            f"{info}\n"
        ), flush = True)

        pd.set_option("display.max_columns", old_max_columns)
        pd.set_option("display.max_rows", old_max_rows)


    def print_status_eval(self, stdout_rec, stderr_rec, crashed):
        '''Print logged stdout and stderr messages and crashed simulations
        after evaluating an epoch.
        '''
        line = "-" * 80
        if len(stderr_rec):
            stderr_rec_str = textwrap.fill(" ".join(
                str(s) for s in stderr_rec
            ))
            print(
                line + "\n" +
                "New stderr messages were recorded while running jobs:\n" +
                textwrap.indent(stderr_rec_str, "  ") + "\n" +
                f"The error messages were logged in:\n  {self.paths.outputs}",
                flush = True,
            )

        if len(stdout_rec):
            stdout_rec_str = textwrap.fill(" ".join(
                str(s) for s in stdout_rec
            ))
            print(
                line + "\n" +
                "New stdout messages were recorded while running jobs:\n" +
                textwrap.indent(stdout_rec_str, "  ") + "\n" +
                f"The output messages were logged in:\n  {self.paths.outputs}",
                flush = True,
            )

        if len(crashed):
            crashed_str = textwrap.fill(" ".join(
                str(c) for c in crashed
            ))

            print(
                line + "\n" +
                "No results were found for these jobs:\n" +
                textwrap.indent(crashed_str, "  ") + "\n" +
                "They crashed or terminated early; for details, check the "
                f"output logs in:\n  {self.paths.outputs}\n"
                "The error values for these simulations were set to NaN.",
                flush = True,
            )

        if len(stderr_rec) or len(stdout_rec) or len(crashed):
            print(line + "\n")


    def print_after_eval(
        self,
        es,
        epoch,
        solutions,
        scaling,
        results,
    ):
        '''Display parameter combinations evaluated in the current epoch and
        the corresponding errors found.
        '''
        # Display evaluation results: solutions, error values, etc.
        sols_results = np.c_[solutions * scaling, results]
        cols = self.setup.parameters.index.to_list() + [
            f"error{i}" for i in range(results.shape[1] - 1)
        ] + ["error"]

        # Store solutions and results in a DataFrame for easy pretty printing
        pop = len(results)
        sols_results = pd.DataFrame(
            data = sols_results,
            columns = cols,
            index = range(epoch * pop - pop, epoch * pop),
        )

        # Display all the DataFrame columns and rows
        old_max_columns = pd.get_option("display.max_columns")
        old_max_rows = pd.get_option("display.max_rows")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        print((
            f"{sols_results}\n"
            f"Total function evaluations: {es.result.evaluations}\n"
        ), flush = True)

        pd.set_option("display.max_columns", old_max_columns)
        pd.set_option("display.max_rows", old_max_rows)


    def print_finished(self, es, scaling):
        '''Display final message after successful convergence on optimum
        parameters.
        '''
        solutions = list(es.result.xbest * scaling) + [es.result.fbest]
        stds = list(es.result.stds * scaling) + [" "]
        proc = os.path.join(
            self.paths.results,
            f"parameters.{es.result.evals_best - 1}.pickle",
        )

        info = pd.DataFrame(
            [solutions, stds],
            index = ["value", "sigma"],
            columns = self.setup.parameters.index.to_list() + ["error"],
        )

        line = "=" * 80
        print((
            f"\n{line}\n"
            f"The best result was found in {es.result.iterations} epochs:\n"
            f"{textwrap.indent(str(info), '  ')}\n\n"
            "These results were found for the job:\n"
            f"  {proc}\n"
            f"{line}"
        ), flush = True)


    def finished(self, es):
        '''Check if the optimisation run is done and display the best solution
        found for the target sigma value.
        '''
        if es.sigma < self.setup.target:
            if self.verbose >= 1:
                print((
                    "\nOptimal solution found within `target_sigma`, i.e. "
                    f"{self.setup.target * 100}%:\n"
                    f"  sigma = {es.sigma} < {self.setup.target}"
                ), flush = True)

            return True
        return False


    def evaluate_solutions(self, solutions, epoch):
        '''Evaluate the parameter combinations given in `solutions` for the
        current `epoch` in parallel.
        '''

        # Aliases
        param_names = self.setup.parameters.index
        pop = self.setup.population
        start_index = epoch * pop

        # For every solution to try, start a separate OS process that runs the
        # `access_seed<seed>/access_code.py` script, which computes and saves
        # an error value
        processes = []

        # These are this epoch's paths to save the simulation outputs to; they
        # will be given to `self.paths.script` as command-line arguments
        parameters_paths = [
            os.path.join(
                self.paths.results,
                f"parameters.{start_index + i}.pickle",
            ) for i in range(pop)
        ]

        result_paths = [
            os.path.join(
                self.paths.results,
                f"result.{start_index + i}.pickle",
            ) for i in range(pop)
        ]

        # Catch the KeyboardInterrupt (Ctrl-C) signal to shut down the spawned
        # processes before aborting.
        try:
            signal_handler.set()

            # Spawn a separate process for every solution to try / sim to run
            for i, sol in enumerate(solutions):
                # Create new set of parameters and save them to disk
                parameters = self.setup.parameters.copy()

                for j, sol_val in enumerate(sol):
                    parameters.at[param_names[j], "value"] = sol_val

                with open(parameters_paths[i], "wb") as f:
                    pickle.dump(parameters, f)

                processes.append(
                    subprocess.Popen(
                        self.setup.scheduler_cmd + [
                            self.paths.script,
                            parameters_paths[i],
                            result_paths[i],
                        ],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        universal_newlines = True,      # outputs in text mode
                    )
                )

            # Gather results
            results, stdout_rec, stderr_rec, crashed = \
                self.progress.gather_results(
                    processes,
                    self.paths,
                    result_paths,
                    self.multi_objective,
                    self.verbose,
                )

        except KeyboardInterrupt:
            for proc in processes:
                proc.kill()

            raise

        finally:
            signal_handler.unset()

        self.print_status_eval(stdout_rec, stderr_rec, crashed)

        # Find number of error values returned for one successful simulation
        num_errors = 1
        for i, res in enumerate(results):
            if res is not None:
                if num_errors == 1:
                    num_errors = len(res)
                    continue

                if len(res) != num_errors:
                    raise ValueError(textwrap.fill((
                        f"The simulation at index {start_index + i} returned "
                        f"{len(res) - 1} error values, while previous "
                        f"simulations had {num_errors - 1} error values."
                    )))

        # Substitute results that are None (i.e. crashed) with rows of NaNs
        for i in range(len(results)):
            if results[i] is None:
                results[i] = np.full(num_errors, np.nan)

        return np.array(results)




class AccessFileNotFoundLegacy:
    pass




class AccessData:
    '''Access (pun intended) data generated by a ``coexist.Access`` run; read
    it in using ``coexist.AccessData.read("access_seed<seed>")``.

    Attributes
    ----------
    paths : AccessPaths
        Struct-like object storing relevant paths in the given ACCES directory.

    parameters : pd.DataFrame
        The optimum free parameters found (final or intermediate if ACCES is
        still running).

    parameters_scaled : pd.DataFrame
        The optimum free parameters found, divided by ``scaling`` such that the
        initial variance in the parameter values was unity.

    scaling : np.ndarray
        A vector with the values to scale each parameter by - they are the
        initial variances (``sigma``).

    population : int
        The number of simulations to run in parallel within a single epoch, or
        number of parameter combinations to try at once.

    num_epochs : int
        The number of epochs that were successfully executed.

    target : float
        The target variance, where the initial parameter uncertainty must be
        decreased from 1 to ``target``.

    seed : int
        The random number generator seed uniquely defining this ACCES run.

    epochs : np.ndarray
        Matrix with columns [mean_param1, mean_param2, ..., std_param1,
        std_param2, ..., std_overall] with one row per epoch.

    epochs_scaled : np.ndarray
        Same as ``epochs``, scaled such that the initial variance (``sigma``)
        becomes unity.

    results : np.ndarray
        Matrix with columns [param1, param2, ..., error] for each parameter
        combination tried - i.e. ``population * num_epochs``.

    results_scaled : np.ndarray
        Same as ``results``, scaled such that the initial variance (``sigma``)
        becomes unity.

    Examples
    --------
    Suppose you run ``coexist.Access.learn(random_seed = 123)`` - then a
    directory "access_seed123/" would be generated. Access (yes, still
    intended) all data generated in a Python-friendly format using:

    >>> import coexist
    >>> data = coexist.AccessData("access_123")
    >>> data
    AccessData
    --------------------------------------------------------------------------
    paths          ╎ AccessPaths(...)
    parameters     ╎         value  min   max     sigma
                   ╎ fp1 -0.005312 -5.0  10.0  0.024483
                   ╎ fp2  0.003409 -5.0  10.0  0.034576
                   ╎ fp3  0.296074 -5.0  10.0  2.078181
    num_epochs     ╎ 31
    target         ╎ 0.1
    seed           ╎ 123
    epochs         ╎ DataFrame(fp1_mean, fp2_mean, fp3_mean, fp1_std, fp2_std,
                   ╎           fp3_std, overall_std)
    epochs_scaled  ╎ DataFrame(fp1_mean, fp2_mean, fp3_mean, fp1_std, fp2_std,
                   ╎           fp3_std, overall_std)
    results        ╎ DataFrame(fp1, fp2, fp3, error)
    results_scaled ╎ DataFrame(fp1, fp2, fp3, error)
    '''

    def __init__(self, access_path = "."):
        '''Read in data generated by ``coexist.Access``; the `access_path` can
        be either the "`access_info_<hash>`" directory itself, or its
        parent directory.
        '''

        access_path = find_access_path(access_path)

        # Check for data in the legacy coexist-0.1.0 format
        legacy_finder = re.compile(r"opt_history_[0-9]+.csv")
        if any(legacy_finder.match(f) for f in os.listdir(access_path)):
            self.legacy(access_path)
            return

        setup_path = os.path.join(access_path, "access_setup.toml")
        with open(setup_path) as f:
            setup_dict = toml.load(f)

        # Update paths prefix in case files are read from a different location
        # than they were created at
        paths = AccessPaths(**setup_dict["paths"])
        paths.update_paths(access_path)

        parameters = pd.DataFrame.from_dict(
            setup_dict["setup"]["parameters"]
        )
        parameters_scaled = pd.DataFrame.from_dict(
            setup_dict["setup"]["parameters_scaled"]
        )
        scaling = np.array(setup_dict["setup"]["scaling"])
        population = setup_dict["setup"]["population"]
        target = setup_dict["setup"]["target"]
        seed = setup_dict["setup"]["seed"]

        history = np.loadtxt(paths.history)
        columns = parameters.index.to_list() + [
            f"error{i}"
            for i in range(history.shape[1] - len(parameters) - 1)
        ] + ["error"]
        results = pd.DataFrame(history, columns = columns, dtype = float)

        history_scaled = np.loadtxt(paths.history_scaled)
        columns_scaled = parameters.index.to_list() + [
            f"error{i}"
            for i in range(history_scaled.shape[1] - len(parameters) - 1)
        ] + ["error"]
        results_scaled = pd.DataFrame(
            history_scaled,
            columns = columns_scaled,
            dtype = float,
        )

        epochs = pd.DataFrame(
            np.loadtxt(paths.epochs),
            columns = (
                [f"{p}_mean" for p in parameters.index] +
                [f"{p}_std" for p in parameters.index] +
                ["overall_std"]
            ),
            dtype = float,
        )

        epochs_scaled = pd.DataFrame(
            np.loadtxt(paths.epochs_scaled),
            columns = (
                [f"{p}_mean" for p in parameters.index] +
                [f"{p}_std" for p in parameters.index] +
                ["overall_std"]
            ),
            dtype = float,
        )

        num_epochs = len(epochs)

        # Set parameters' values to the best results
        ns = len(parameters)
        parameters["value"] = results.iloc[results["error"].idxmin()][:-1]
        parameters["sigma"] = epochs.iloc[-1, ns:ns + ns].to_numpy()

        parameters_scaled["value"] = results_scaled.iloc[
            results_scaled["error"].idxmin()
        ][:-1]
        parameters_scaled["sigma"] = epochs_scaled.iloc[
            -1, ns:ns + ns
        ].to_numpy()

        # Set class attributes
        self.paths = paths
        self.parameters = parameters
        self.parameters_scaled = parameters_scaled
        self.scaling = scaling
        self.population = population
        self.num_epochs = num_epochs
        self.target = target
        self.seed = seed
        self.epochs = epochs
        self.epochs_scaled = epochs_scaled
        self.results = results
        self.results_scaled = results_scaled


    @staticmethod
    def empty():
        '''Create an empty `AccessData` object that you can set attributes
        to directly.

        Examples
        --------
        Create an empty `AccessData` object:

        >>> import coexist
        >>> data = coexist.AccessData.empty()
        '''
        return AccessData.__new__(AccessData)


    @staticmethod
    def read(access_path = "."):
        '''Read in data generated by ``coexist.Access``; the `access_path` can
        be either the "`access_seed<hash>`" directory itself, or its
        parent directory.

        Here for backwards-compatibility; you can instantiate the class
        directly with the ``access_path``, e.g. ``AccessData(".")``.
        '''

        return AccessData(access_path)


    def legacy(self, access_path):
        '''Read in data from legacy coexist-0.1.0 ACCES format; this is
        normally called automatically by ``AccessData.read``.
        '''

        # Check all legacy files exist
        legacy_files = ["access_code.py", "access_info.pickle"]
        if any(not os.path.isfile(os.path.join(access_path, f))
               for f in legacy_files):
            raise FileNotFoundError(textwrap.fill((
                f"The legacy AccessData files `{legacy_files}` were not found "
                f"in `{access_path}`."
            )))

        # Find legacy history file
        history_finder = re.compile(r"opt_history_[0-9]+\.csv")

        for f in os.listdir(access_path):
            if history_finder.search(f):
                history_path = os.path.join(access_path, f)
                history_scaled_path = (
                    history_path.split(".csv")[0] + "_scaled.csv"
                )
                num_solutions = int(
                    re.split(r"opt_history_|\.csv", history_path)[1]
                )
                break
        else:
            raise FileNotFoundError(textwrap.fill((
                f"No legacy history file was found in `{access_path}`."
            )))

        with open(os.path.join(access_path, "access_info.pickle"), "rb") as f:
            access_info = pickle.load(f)

        history = np.loadtxt(history_path)
        history_scaled = np.loadtxt(history_scaled_path)

        # Translate legacy data into modern format
        notfound = AccessFileNotFoundLegacy()

        paths = AccessPaths(
            directory = access_path,
            results = os.path.join(access_path, "simulations"),
            outputs = os.path.join(access_path, "outputs"),
            script = os.path.join(access_path, "access_code.py"),
            setup = notfound,
            epochs = notfound,
            epochs_scaled = notfound,
            history = history_path,
            history_scaled = history_scaled_path,
        )

        parameters = access_info.parameters
        population = num_solutions
        num_epochs = len(history) // population
        target = access_info.target_sigma
        seed = access_info.random_seed

        # Infer scaled parameters
        pop = population
        nparams = len(parameters)

        scaling = np.mean(
            history[:, :nparams] / history_scaled[:, :nparams],
            axis = 0,
        )
        parameters_scaled = parameters.copy()
        for i in range(len(parameters_scaled.columns)):
            parameters_scaled.iloc[:, i] /= scaling

        # Infer epochs data
        means = np.array([
            history[i * pop:i * pop + pop, :nparams].mean(axis = 0)
            for i in range(num_epochs)
        ])
        stds = history[::pop, nparams:2 * nparams]
        overall_stds = history[::pop, -2]
        epochs = pd.DataFrame(
            np.c_[means, stds, overall_stds],
            columns = (
                [f"{p}_mean" for p in parameters.index] +
                [f"{p}_std" for p in parameters.index] +
                ["overall_std"]
            ),
            dtype = float,
        )

        means = np.array([
            history_scaled[i * pop:i * pop + pop, :nparams].mean(axis = 0)
            for i in range(num_epochs)
        ])
        stds = history_scaled[::pop, nparams:2 * nparams]
        overall_stds = history_scaled[::pop, -2]
        epochs_scaled = pd.DataFrame(
            np.c_[means, stds, overall_stds],
            columns = (
                [f"{p}_mean" for p in parameters.index] +
                [f"{p}_std" for p in parameters.index] +
                ["overall_std"]
            ),
            dtype = float,
        )

        # Translate history data
        results = pd.DataFrame(
            history[:, list(range(nparams)) + [-1]],
            columns = parameters.index.to_list() + ["error"],
            dtype = float,
        )

        results_scaled = pd.DataFrame(
            history_scaled[:, list(range(nparams)) + [-1]],
            columns = parameters.index.to_list() + ["error"],
            dtype = float,
        )

        # Set parameters' values to the best results
        ns = len(parameters)
        parameters["value"] = results.iloc[results["error"].idxmin()][:-1]
        parameters["sigma"] = epochs.iloc[-1, ns:ns + ns].to_numpy()

        parameters_scaled["value"] = results_scaled.iloc[
            results_scaled["error"].idxmin()
        ][:-1]
        parameters_scaled["sigma"] = epochs_scaled.iloc[
            -1, ns:ns + ns
        ].to_numpy()

        # Set class attributes
        self.paths = paths
        self.parameters = parameters
        self.parameters_scaled = parameters_scaled
        self.scaling = scaling
        self.population = population
        self.num_epochs = num_epochs
        self.target = target
        self.seed = seed
        self.epochs = epochs
        self.epochs_scaled = epochs_scaled
        self.results = results
        self.results_scaled = results_scaled


    def __repr__(self):
        name = self.__class__.__name__
        underline = "-" * 80

        def wrap(text, prep = 30):
            return textwrap.fill(
                text, width = 80,
                initial_indent = prep * " ",
                subsequent_indent = prep * " ",
            )[prep:]
            '''Combine the column data for the relevant parameters.
            '''
        cols = wrap(", ".join(self.epochs.columns))
        epochs = f"DataFrame({cols})"

        cols = wrap(", ".join(self.epochs_scaled.columns))
        epochs_scaled = f"DataFrame({cols})"

        cols = wrap(", ".join(self.results.columns))
        results = f"DataFrame({cols})"

        cols = wrap(", ".join(self.results_scaled.columns))
        results_scaled = f"DataFrame({cols})"

        parameters = str(self.parameters).split("\n")
        parameters = "\n".join(
            parameters[0:1] +
            [20 * " " + p for p in parameters[1:]]
        )

        parameters_scaled = str(self.parameters_scaled).split("\n")
        parameters_scaled = "\n".join(
            parameters_scaled[0:1] +
            [20 * " " + p for p in parameters_scaled[1:]]
        )

        docstr = (
            f"{name}\n"
            f"{underline}\n"
            f"paths               {self.paths.__class__.__name__}(...)\n"
            f"parameters          {parameters}\n"
            f"parameters_scaled   {parameters_scaled}\n"
            f"scaling             {self.scaling}\n"
            f"population          {self.population}\n"
            f"num_epochs          {self.num_epochs}\n"
            f"target              {self.target}\n"
            f"seed                {self.seed}\n"
            f"epochs              {epochs}\n"
            f"epochs_scaled       {epochs_scaled}\n"
            f"results             {results}\n"
            f"results_scaled      {results_scaled}\n"
        )

        # Add vertical line
        docstr = docstr.split("\n")
        for i in range(2, len(docstr) - 1):
            d = docstr[i]
            docstr[i] = d[:18] + "╎" + d[19:]

        return "\n".join(docstr)




def find_access_path(path):
    '''Locate the `access_seed<seed>` directory.
    '''
    finder = re.compile(r"access_seed[0-9]+")
    # The directory itself
    if finder.match(path):
        return path

    # The parent
    matched = [f for f in os.listdir(path) if finder.match(f)]

    if len(matched) == 1:
        return os.path.join(path, matched[0])
    elif len(matched) > 1:
        raise RuntimeError((
            f"Multiple ACCES directories were found at `{path}`:\n"
            f"{matched}\n\n"
            "Use the full path to the ACCES directory you want."
        ))

    # If no `access_seed<seed>` was found, return the path as is in case it was
    # renamed but ACCES files are inside still
    return path




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
        # reconstructed even if it is a `coexist.Simulation` subclass
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
            os.path.split(coexist.__file__)[0],
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

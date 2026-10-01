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
import  shutil
import  warnings
import  tempfile
from    datetime            import  datetime

import  numpy               as      np
import  pandas              as      pd
import  toml
import  cma

import  coexist

from    .                   import  schedulers

from    .combiners          import  Product

from    .code_trees         import  code_contains_variable
from    .code_trees         import  code_substitute_variable

from    .utilities          import  autorepr
from    .utilities          import  SignalHandlerKI




signal_handler = SignalHandlerKI()




@autorepr(short = {"script"})
class AccessSetup:
    '''Structure storing constant attributes for an ACCES optimisation run.

    Code validation and generation are handled too.

    Attributes
    ----------
    parameters : pd.DataFrame
        The free parameters extracted from the user script.

    parameters_scaled : pd.DataFrame
        The free parameters scaled to the phenotype space, such that the
        initial standard deviation (`sigma`) is unity.

    scaling : np.ndarray
        A vector of values that the free parameters are scaled by; it is the
        initial standard deviation (`sigma`) given by the user.

    script : str
        The modified user script that will be executed.

    population : int
        The number of simulations to be run in parallel in each epoch.

    target : float
        The target scaled standard deviation - decrease the uncertainty
        from the initial 1 down to `target`.

    seed: int
        The random seed defining a single ACCES run.

    rng: np.random.Generator
        The random number generator used, seeded with `seed`.
    '''

    def __init__(self, script_path):
        '''Given a path to a user-defined simulation script, extract the free
        parameters and generate the ACCES script.
        '''
        # Uninitialised parameters (will be set later)
        self.population = None
        self.target = None
        self.rng = None
        self.seed = None

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

        prefix = os.fspath(prefix)
        self.directory = prefix
        for attr, prev in [
            ("results", self.results), ("outputs", self.outputs),
            ("script", self.script), ("setup", self.setup),
            ("epochs", self.epochs), ("epochs_scaled", self.epochs_scaled),
            ("history", self.history), ("history_scaled", self.history_scaled),
        ]:
            if prev is not None:
                current = os.path.join(prefix, os.path.basename(prev))
                setattr(self, attr, current)



    def save_history(self, setup, progress):
        '''Given an ``AccessSetup`` and ``AccessProgress`` instance, save the
        results history without truncating an existing file on write failure.
        '''
        count = progress.history.shape[1] - len(setup.parameters) - 1
        columns = setup.parameters.index.to_list() + [
            f"error{i}" for i in range(count)
        ] + ["error"]

        save_access_table(self.history, progress.history, " ".join(columns))
        save_access_table(
            self.history_scaled, progress.history_scaled, " ".join(columns),
        )


    def load_history(self, access):
        '''Load previous results into ``access.progress``, retaining the exact
        saved CMA-ES coordinates. Repair missing or empty unscaled history.
        '''
        history, scaled = read_access_table(
            self.history, self.history_scaled, access.setup.scaling,
            access.setup.parameters.index.to_list(),
            population = access.setup.population,
        )
        access.progress.history = history
        access.progress.history_scaled = scaled

        if scaled is not None and access.verbose >= 3:
            print(
                "Found previous ACCES results in " +
                f"`{self.history_scaled}`.\n" + "=" * 80 + "\n",
                flush = True,
            )


    def save_epochs(self, setup, progress):
        '''Given an ``AccessSetup`` and ``AccessProgress`` instance, save the
        optimisation epochs without truncating existing files on write failure.
        '''
        header = " ".join(
            [f"{p}_mean" for p in setup.parameters.index] +
            [f"{p}_std" for p in setup.parameters.index] +
            ["overall_std"]
        )
        save_access_table(self.epochs, progress.epochs, header)
        save_access_table(self.epochs_scaled, progress.epochs_scaled, header)


    def load_epochs(self, access):
        '''Load the epoch tables into ``access.progress``. Only include epochs
        with a complete population in the saved scaled history.
        '''
        names = access.setup.parameters.index
        columns = (
            [f"{p}_mean" for p in names] + [f"{p}_std" for p in names] +
            ["overall_std"]
        )
        epochs, scaled = read_access_table(
            self.epochs, self.epochs_scaled,
            np.r_[access.setup.scaling, access.setup.scaling, 1.], columns,
        )
        history = access.progress.history_scaled
        count = (0 if history is None
                 else len(history) // access.setup.population)
        if epochs is None:
            if count:
                raise ValueError("Saved history has no corresponding epochs.")
            epochs = np.empty((0, len(columns)))
            scaled = epochs.copy()

        epochs, scaled = completed_access_epochs(epochs, scaled, count)
        access.progress.epochs = epochs
        access.progress.epochs_scaled = scaled


    def copy(self):
        '''Create a copy of an `AccessPaths` object.
        '''

        return AccessPaths(
            directory = self.directory,
            results = self.results,
            outputs = self.outputs,
            script = self.script,
            setup = self.setup,
            epochs = self.epochs,
            epochs_scaled = self.epochs_scaled,
            history = self.history,
            history_scaled = self.history_scaled,
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
        Same as ``epochs``, scaled such that the initial standard deviation
        (``sigma``) becomes unity.

    history: np.ndarray = None
        Matrix with columns [param1, param2, ..., error] for each parameter
        combination tried - i.e. ``population * num_epochs``.

    history_scaled: np.ndarray = None
        Same as ``history``, scaled such that the initial standard deviation
        (``sigma``) becomes unity.

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
        self.epochs = epochs
        self.epochs_scaled = epochs_scaled
        self.history = history
        self.history_scaled = history_scaled
        self.stdout = stdout
        self.stderr = stderr


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
        # stdout_rec = []
        # stderr_rec = []
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
                    proc.communicate()

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

        return results, crashed




@autorepr
class Access:
    '''Optimise an arbitrary user-defined script's parameters in parallel.

    A minimal user script - saved in a separate file - would be:

    ::

        # In file "script_filepath.py"

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

        # In file "access_learn.py"
        import coexist

        access = coexist.Access("script_filepath.py")
        access.learn(num_solutions = 10, target_sigma = 0.1, random_seed = 42)

    Once you run `access.learn()`, a folder named "access_seed42" is
    generated which stores all information about this access run, including all
    simulation data. You can load this data using ``coexist.AccessData``
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
    setup : coexist.access.AccessSetup
        Structure storing given ACCES configuration, containing the free
        ``parameters``, number of solutions to try in parallel ``population``,
        target uncertainty ``target``, seeded random number generator ``rng``
        and the ``seed``.

    paths : coexist.access.AccessPaths
        Structure storing paths to the ACCES ``directory``, saved ``state``,
        ``simulations`` tried, captured ``outputs`` directory, and paths to
        the previous results ``history`` and ``history_scaled``.

    progress : coexist.access.AccessProgress
        Structure storing ACCES optimisation run progress - ``epochs``,
        ``history`` (and CMA-scaled versions of them) and latest ``stdout`` and
        ``stderr`` messages.

    scheduler : coexist.schedulers.Scheduler subclass
        The scheduler used to launch each simulation in parallel.

    multi_objective : object, optional
        An object defining the method `combine`, combining multiple errors into
        a single value.

    verbose : int, optional
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
        self.setup = AccessSetup(script_path)
        self.paths = AccessPaths()
        self.progress = AccessProgress()

        # Type-check scheduler
        if not isinstance(scheduler, schedulers.Scheduler):
            raise TypeError(textwrap.fill((
                "The input `scheduler` must be a subclass of `coexist."
                f"schedulers.Scheduler`. Received {type(scheduler)}."
            )))
        self.scheduler = scheduler

        # Will be set in `learn`
        self.multi_objective = None
        self.verbose = None
        self._elapsed = None


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
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), \
                warnings.catch_warnings():

            # CMA-ES sometimes warns about changing the initial standard
            # deviation; ACCES users don't control that, so we can hide it
            warnings.simplefilter("ignore", UserWarning)

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

            # We already warn about crashed simulations, so hide CMA-ES ones
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
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

        return AccessData(self.paths.directory)


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

        with atomic_access_file(self.paths.setup) as f:
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

        # Current time, formatted
        now_str = datetime.now().strftime(r"%H:%M:%S")

        # Save and print time elapsed since last epoch
        now = time.time()
        elapsed = 0 if self._elapsed is None else now - self._elapsed
        self._elapsed = now

        if elapsed == 0:
            since_str = ""
        else:
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)

            elapsed_str = f"{minutes:02}:{seconds:02}"
            if hours > 0:
                elapsed_str = f"{hours:02}:" + elapsed_str

            since_str = f" | Since Last {elapsed_str}"

        print((
            f"{head}\n"
            f"Epoch {epoch:>4} | Population {self.setup.population:>4} | "
            f"Time {now_str}{since_str}\n"
            f"{line}\n"
            f"Scaled overall standard deviation: {es.sigma}\n"
            f"{info}\n"
        ), flush = True)

        pd.set_option("display.max_columns", old_max_columns)
        pd.set_option("display.max_rows", old_max_rows)


    def print_status_eval(self, crashed):
        '''Print logged stdout and stderr messages and crashed simulations
        after evaluating an epoch.
        '''

        if len(crashed):
            line = "-" * 80

            crashed_str = textwrap.fill(" ".join(
                str(c) for c in crashed
            ))

            print(
                line + "\n" +
                "No results were found for these jobs:\n" +
                textwrap.indent(crashed_str, "  ") + "\n" +
                "They crashed or terminated early; for details, check the "
                f"output logs in:\n  {self.paths.outputs}\n"
                "The error values for these simulations were set to NaN.\n" +
                line + "\n",
                flush = True,
            )


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

        # If overall sigma went below target
        if es.sigma < self.setup.target:
            if self.verbose >= 1:
                print((
                    "\nOptimal solution found within `target_sigma`, i.e. "
                    f"{self.setup.target * 100}%:\n"
                    f"  sigma = {es.sigma} < {self.setup.target}"
                ), flush = True)
            return True

        # If all individual sigmas went below target
        if np.all(es.result.stds < self.setup.target):
            if self.verbose >= 1:
                print((
                    "\nAll parameters found within `target_sigma`, i.e. "
                    f"{self.setup.target * 100}%:\n"
                    f"  scaled_std = {es.result.stds} < {self.setup.target}"
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

        output_files = [
            open(
                os.path.join(
                    self.paths.outputs,
                    f"output.{start_index + i}.log"
                ),
                "w",
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

                # Get job scheduling command
                scheduler_cmd = self.scheduler.schedule(
                    self.paths.directory,
                    start_index + i,
                )

                processes.append(
                    subprocess.Popen(
                        scheduler_cmd + [
                            self.paths.script,
                            parameters_paths[i],
                            result_paths[i],
                        ],
                        stdout = output_files[i],
                        stderr = subprocess.STDOUT,
                    )
                )

            # Gather results and crashed simulations
            results, crashed = self.progress.gather_results(
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

            for of in output_files:
                of.close()

        if self.verbose >= 1:
            self.print_status_eval(crashed)

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




class AccessData:
    '''Access (pun intended) data generated by a ``coexist.Access`` run; read
    it in using ``coexist.AccessData("access_seed<seed>")``.

    Attributes
    ----------
    paths : AccessPaths
        Struct-like object storing relevant paths in the given ACCES directory.

    parameters : pd.DataFrame
        The optimum free parameters found (final or intermediate if ACCES is
        still running).

    parameters_scaled : pd.DataFrame
        The optimum free parameters found, divided by ``scaling`` such that the
        initial standard deviation in the parameter values was unity.

    scaling : np.ndarray
        A vector with the values to scale each parameter by - they are the
        initial standard deviations (``sigma``).

    population : int
        The number of simulations to run in parallel within a single epoch, or
        number of parameter combinations to try at once.

    num_epochs : int
        The number of epochs that were successfully executed.

    target : float
        The target scaled standard deviation, decreasing the initial
        sampling scale from 1 to ``target``.

    seed : int
        The random number generator seed uniquely defining this ACCES run.

    epochs : pd.DataFrame
        Matrix with columns [mean_param1, mean_param2, ..., std_param1,
        std_param2, ..., std_overall] with one row per epoch.

    epochs_scaled : pd.DataFrame
        Same as ``epochs``, scaled such that the initial standard deviation
        (``sigma``) becomes unity.

    results : pd.DataFrame
        Matrix with columns [param1, param2, ..., error] for each parameter
        combination tried - i.e. ``population * num_epochs``.

    results_scaled : pd.DataFrame
        Same as ``results``, scaled such that the initial standard deviation
        (``sigma``) becomes unity.

    Notes
    -----
    Missing or empty unscaled history and epoch files are automatically
    reconstructed from their scaled counterparts, with a message printed for
    each repair. Only the unscaled files are written. Original scaled values
    are needed to replay CMA-ES without changing floating-point precision.

    Epoch files can contain one extra record after an interrupted save. This
    record is excluded in memory until its population exists in scaled history.
    An empty or damaged scaled file requires an intact saved copy; it cannot
    be reconstructed exactly from the unscaled data.

    Examples
    --------
    Suppose you run ``coexist.Access.learn(random_seed = 123)`` - then a
    directory "access_seed123/" would be generated. Access (yes, still
    intended) all data generated in a Python-friendly format using:

    >>> import coexist
    >>> data = coexist.AccessData("access_seed123")
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
        be either the "`access_seed<seed>`" directory itself, or its
        parent directory.
        '''

        access_path = find_access_path(access_path)

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

        names = parameters.index.to_list()
        history, history_scaled = read_access_table(
            paths.history, paths.history_scaled, scaling, names,
            population = population,
        )
        if history is None:
            raise ValueError(
                f"No saved ACCES history was found at `{access_path}`."
            )

        columns = names + [
            f"error{i}" for i in range(history.shape[1] - len(names) - 1)
        ] + ["error"]
        results = pd.DataFrame(history, columns = columns, dtype = float)
        results_scaled = pd.DataFrame(
            history_scaled, columns = columns, dtype = float,
        )

        columns = (
            [f"{p}_mean" for p in names] + [f"{p}_std" for p in names] +
            ["overall_std"]
        )
        epochs, epochs_scaled = read_access_table(
            paths.epochs, paths.epochs_scaled,
            np.r_[scaling, scaling, 1.], columns,
        )
        if epochs is None:
            raise ValueError("Saved history has no corresponding epochs.")
        num_epochs = len(history_scaled) // population
        epochs, epochs_scaled = completed_access_epochs(
            epochs, epochs_scaled, num_epochs,
        )
        epochs = pd.DataFrame(epochs, columns = columns, dtype = float)
        epochs_scaled = pd.DataFrame(
            epochs_scaled, columns = columns, dtype = float,
        )

        # Set parameter estimates from a successful observation, if available.
        ns = len(parameters)
        valid = np.isfinite(results_scaled["error"])
        if valid.any():
            best = results_scaled.loc[valid, "error"].idxmin()
            parameters["value"] = results.loc[best, names]
            parameters_scaled["value"] = results_scaled.loc[best, names]
        parameters["sigma"] = epochs.iloc[-1, ns:ns + ns].to_numpy()
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
        be either the "`access_seed<seed>`" directory itself, or its
        parent directory.

        Equivalent to constructing ``AccessData(access_path)``.
        '''

        return AccessData(access_path)


    def sensitivity(
        self,
        parameter_window = 0.2,
        objective_tolerance = 0.1,
        *,
        objective = "error",
        reference = None,
        excluded_evaluations = None,
        **kwargs,
    ):
        '''Analyse local parameter variations using existing evaluations.

        Forward the stored parameters, scalar response and original bounds to
        ``coexist.sensitivity.analyse``. Install ``coexist[sensitivity]``
        to use this optional functionality.

        Parameters
        ----------
        parameter_window : float in [0, 1], default 0.2
            Box half-width as a fraction of each original parameter-bound
            width, clipped to those bounds.

        objective_tolerance : float >= 0, default 0.1
            Symmetric allowance around the observed reference response. A
            value of 0.1 allows +/-10% of its absolute magnitude.

        objective : str, default "error"
            Scalar response column to analyse. For the combined "error",
            evaluations with missing individual error components are omitted,
            even if their combined score is a finite penalty.

        reference : index label, optional
            Evaluation to use as the reference. Defaults to the complete
            evaluation with the smallest combined error, including when
            analysing an individual response such as "error0".

        excluded_evaluations : iterable, optional
            Additional evaluation index labels to omit as invalid.

        **kwargs : other keyword arguments
            Options forwarded to ``coexist.sensitivity.analyse``, including
            ``absolute_tolerance``, ``interpolate``, ``response_transform``
            and grid sizes. Progress is printed by default; set
            ``verbose = False`` to silence it.

        Returns
        -------
        coexist.sensitivity.SensitivityResult
            Parameter ranges, rankings, paired responses and fitted model.
            The saved ACCES data are not modified by this analysis.

        Examples
        --------
        Analyse an ACCES run and save its tables and figures:

        >>> import coexist
        >>> data = coexist.AccessData("access_seed42")
        >>> result = data.sensitivity(parameter_window = 0.2)
        >>> result.ranges
        >>> result.save("sensitivity_output")
        '''
        from .sensitivity import analyse

        if not isinstance(objective, str):
            raise TypeError("objective must name one scalar response column")
        excluded = ([] if excluded_evaluations is None
                    else list(excluded_evaluations))
        rows = self.results.drop(index = excluded, errors = "ignore")
        if not rows.index.is_unique:
            raise ValueError("Evaluation index labels must be unique")
        names = self.parameters.index.tolist()
        values = rows[objective].copy()
        components = [name for name in rows.columns
                      if name.startswith("error") and name[5:].isdigit()]
        complete_components = np.isfinite(rows[components]).all(axis = 1)
        if objective == "error":
            values.loc[~complete_components] = np.nan
        complete = np.isfinite(rows[names]).all(axis = 1)
        complete &= np.isfinite(values)
        if reference is None:
            eligible = rows.loc[
                complete & complete_components & np.isfinite(rows.error)
            ]
            if eligible.empty:
                raise ValueError("No complete evaluation defines a reference")
            reference = eligible.error.idxmin()
        if reference not in rows.index or not complete.loc[reference]:
            raise ValueError(
                "reference must identify a complete, included evaluation"
            )

        result = analyse(
            rows[names], values, self.parameters[["min", "max"]],
            reference = rows.loc[reference, names],
            reference_value = float(rows.loc[reference, objective]),
            parameter_window = parameter_window,
            objective_tolerance = objective_tolerance,
            **kwargs,
        )
        result.metadata.update(
            archive = self.paths.directory,
            objective = objective, reference_evaluation = str(reference),
            excluded_evaluations = list(map(str, excluded)),
        )
        return result


    def copy(self):
        '''Return copy of `AccessData` object.
        '''
        data = AccessData.empty()
        data.paths = self.paths.copy()
        data.parameters = self.parameters.copy()
        data.parameters_scaled = self.parameters_scaled.copy()
        data.scaling = self.scaling.copy()
        data.population = self.population
        data.num_epochs = self.num_epochs
        data.target = self.target
        data.seed = self.seed
        data.epochs = self.epochs.copy()
        data.epochs_scaled = self.epochs_scaled.copy()
        data.results = self.results.copy()
        data.results_scaled = self.results_scaled.copy()
        return data


    def save(self, dirname):
        '''Save `AccessData` to a new directory at `dirname`.
        '''
        # Copy previous folder to new location
        shutil.copytree(self.paths.directory, dirname)

        self.paths.update_paths(dirname)

        # Save history
        to_pad = len(self.results.columns) - len(self.parameters) - 1
        columns = self.parameters.index.to_list() + [
            f"error{i}" for i in range(to_pad)
        ] + ["error"]

        save_access_table(
            self.paths.history,
            self.results.to_numpy(),
            header = " ".join(columns),
        )

        save_access_table(
            self.paths.history_scaled,
            self.results_scaled.to_numpy(),
            header = " ".join(columns),
        )

        # Save epochs
        save_access_table(
            self.paths.epochs,
            self.epochs.to_numpy(),
            header = " ".join(
                [f"{p}_mean" for p in self.parameters.index] +
                [f"{p}_std" for p in self.parameters.index] +
                ["overall_std"]
            ),
        )

        save_access_table(
            self.paths.epochs_scaled,
            self.epochs_scaled.to_numpy(),
            header = " ".join(
                [f"{p}_mean" for p in self.parameters.index] +
                [f"{p}_std" for p in self.parameters.index] +
                ["overall_std"]
            ),
        )

        # Save setup
        setup_dict = dict(
            paths = self.paths.__dict__,
            setup = dict(
                parameters = self.parameters.to_dict(),
                parameters_scaled = self.parameters_scaled.to_dict(),
                scaling = self.scaling.tolist(),
                population = self.population,
                target = self.target,
                seed = self.seed,
            ),
        )

        with atomic_access_file(self.paths.setup) as f:
            toml.dump(setup_dict, f)


    def __getitem__(self, index):
        # Select AccessData epochs
        if isinstance(index, int):
            # Allow negative indices
            while index < 0:
                index += self.num_epochs

            if index >= self.num_epochs:
                raise IndexError(textwrap.fill((
                    f"The index=`{index}` is out of bounds for AccessData "
                    f"with {self.num_epochs} epochs."
                )))

            data = self.copy()
            data.num_epochs = 1
            data.epochs = self.epochs.iloc[index:index + 1]
            data.epochs_scaled = self.epochs_scaled.iloc[index:index + 1]
            data.results = self.results.iloc[
                index * self.population:(index + 1) * self.population
            ]
            data.results_scaled = self.results_scaled.iloc[
                index * self.population:(index + 1) * self.population
            ]
            return data

        elif isinstance(index, slice):
            if index.step is not None and index.step != 1:
                raise ValueError(textwrap.fill((
                    "Indexing with a `slice.step` is not yet available. "
                    "If this would be useful for you please get in touch."
                )))

            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.num_epochs

            # Allow negative indices
            while start < 0:
                start += self.num_epochs

            while stop < 0:
                stop += self.num_epochs

            if stop > self.num_epochs or start >= self.num_epochs or \
                    start >= stop:
                raise IndexError(textwrap.fill((
                    f"The slice=`{start}:{stop}` is out of bounds for "
                    f"AccessData with {self.num_epochs} epochs."
                )))

            data = self.copy()
            data.num_epochs = stop - start
            data.epochs = self.epochs.iloc[start:stop]
            data.epochs_scaled = self.epochs_scaled.iloc[start:stop]
            data.results = self.results.iloc[
                start * self.population:stop * self.population
            ]
            data.results_scaled = self.results_scaled.iloc[
                start * self.population:stop * self.population
            ]
            return data

        else:
            raise TypeError(textwrap.fill((
                "Epoch selection via subscripting is only possible with "
                "integer / slice indices (e.g. `access_data[5]` or "
                "`access_data[2:5]`). Received index with type "
                f"`{type(index)}`."
            )))


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




@contextlib.contextmanager
def atomic_access_file(path):
    '''Yield a temporary text file and replace `path` after a successful flush.

    A failed write leaves the previous file intact. The temporary file is on
    the same filesystem as the destination so replacement is atomic.
    '''
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    with tempfile.TemporaryDirectory(
        prefix = ".access-", dir = parent,
    ) as temp:
        pending = os.path.join(temp, os.path.basename(path))
        with open(pending, "w") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())

        if os.path.exists(path):
            shutil.copymode(path, pending)
        os.replace(pending, path)




def save_access_table(path, values, header):
    '''Save an ACCES table without truncating an existing file on failure.

    NumPy's default ``%.18e`` format preserves float64 values when read back.
    '''
    with atomic_access_file(path) as stream:
        np.savetxt(stream, values, header = header)




def read_access_table(path, scaled_path, scaling, columns, population = None):
    '''Read paired ACCES tables as matrices, recovering missing or empty
    unscaled data from the saved scaled values.

    `scaling` multiplies only the leading parameter columns. For history,
    `population` also requires complete generations and appends error column
    names. For epochs, `columns` lists all columns, including the unchanged
    overall standard deviation.

    Scaled files are never reconstructed or rewritten here: dividing recovered
    unscaled values by `scaling` can change the float64 values used by CMA-ES.
    '''
    if not os.path.isfile(path) and not os.path.isfile(scaled_path):
        return None, None
    if not os.path.isfile(scaled_path):
        raise FileNotFoundError(
            f"Missing scaled ACCES file `{scaled_path}`. Exact CMA-ES history "
            "cannot be reconstructed from unscaled values."
        )

    # Empty files are handled explicitly instead of emitting loadtxt warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message = "loadtxt: input contained no data",
            category = UserWarning,
        )
        scaled = np.loadtxt(scaled_path, ndmin = 2)
        if not scaled.size:
            raise ValueError(
                f"Scaled ACCES file `{scaled_path}` is empty. Restore it "
                "from an intact copy; unscaled values cannot reproduce it "
                "at the exact precision required by CMA-ES."
            )

        columns = list(columns)
        if population is not None:
            if population < 1 or len(scaled) % population:
                raise ValueError(
                    f"Scaled history `{scaled_path}` does not contain "
                    "complete populations. Restore an intact saved history."
                )
            columns += [
                f"error{i}"
                for i in range(scaled.shape[1] - len(columns) - 1)
            ] + ["error"]

        scaling = np.asarray(scaling, dtype = float)
        if (
            scaled.shape[1] != len(columns) or
            not np.isfinite(scaled[:, :len(scaling)]).all()
        ):
            raise ValueError(f"Invalid scaled ACCES table `{scaled_path}`.")
        if not np.isfinite(scaling).all() or (scaling <= 0).any():
            raise ValueError(
                "ACCES scaling values must be finite and positive."
            )

        values = (np.loadtxt(path, ndmin = 2)
                  if os.path.isfile(path) else np.empty((0, 0)))

    if not values.size:
        values = scaled.copy()
        values[:, :len(scaling)] *= scaling
        save_access_table(path, values, " ".join(columns))
        print(
            f"Repaired missing or empty ACCES file `{path}` from "
            f"`{scaled_path}`. The scaled file was left unchanged.",
            flush = True,
        )
    elif values.shape[1] != scaled.shape[1] or len(values) < len(scaled):
        raise ValueError(
            f"ACCES tables `{path}` and `{scaled_path}` have inconsistent "
            "shapes. Automatic repair requires missing or empty unscaled data."
        )
    elif len(values) > len(scaled):
        print(
            f"Reading the first {len(scaled)} rows of `{path}` to match "
            f"the saved scaled table `{scaled_path}`.",
            flush = True,
        )
        values = values[:len(scaled)]

    return values, scaled




def completed_access_epochs(epochs, scaled, count):
    '''Select the epoch records corresponding to complete saved populations.

    Epochs are written before history, so one extra record can remain after
    an interrupted save. Exclude it in memory without changing the files.
    '''
    if len(scaled) < count or len(scaled) > count + 1:
        raise ValueError(
            "Saved ACCES epoch and history counts are inconsistent."
        )
    if len(scaled) > count:
        print(
            "Ignoring the final ACCES epoch record: its population was not "
            "saved in the scaled history. The files were left unchanged.",
            flush = True,
        )
    return epochs[:count], scaled[:count]




def find_access_path(path):
    '''Locate a run containing ``access_setup.toml``, directly or in a parent.
    '''
    path = os.fspath(path)
    if os.path.isfile(os.path.join(path, "access_setup.toml")):
        return path

    finder = re.compile(r"access_seed[0-9]+")
    matched = sorted(
        name for name in os.listdir(path)
        if finder.fullmatch(name) and os.path.isfile(
            os.path.join(path, name, "access_setup.toml")
        )
    )

    if len(matched) == 1:
        return os.path.join(path, matched[0])
    if len(matched) > 1:
        raise RuntimeError((
            f"Multiple ACCES directories were found at `{path}`:\n"
            f"{matched}\n\n"
            "Use the full path to the ACCES directory you want."
        ))

    raise FileNotFoundError(
        f"No ACCES run containing `access_setup.toml` was found at `{path}`."
    )

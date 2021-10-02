**************
ACCES Tutorial
**************


Introduction
============

ACCES - or *autonomous calibration and characterisation via evolutionary software* -
can calibrate virtually any simulation parameters against a user-defined cost
function, quantifying and then minimising the "difference" between the simulated
system and experimental reality  – in essence, autonomously ‘learning’ the physical
properties of the particles within the system, without the need for human input.
This cost function is completely general, allowing ACCESS to calibrate simulations
against measurements as simple as photographed occupancy plots, or complex system
properties captured through e.g. Lagrangian particle tracking.

.. image:: ../_static/cost_function.png
   :alt: Example cost function quantifying the disparity between a simulation and reality.


But what does this "difference between simulation and reality" look like? Damn awful:

.. image:: ../_static/cost_surface.png
   :alt: Surface of an example cost function.

We ran the same DEM simulation of a vibrofluidised bed on a 100x100 grid of
parameters (10,000 simulations indeed, thank you BlueBEAR @Birmingham), then
compared central simulation's occupancy profile with the other simulations.

Fundamentally, this is an optimisation problem: *what parameters do I need such
that my simulation reproduces an experimental measurement*? But it is a horrible
one, with the error function typically being noisy, non-smooth and having many
false, local minima. In mathematical terms, it is non-convex and non-smooth -
so gradient-based optimisers go out the window. Plus, a single simulation (a
single point on the graph above) can take hours or days to run.

Enter CMA-ES [CMAES]_, a gradient-free global optimisation strategy that consistently
ranks amongst the best optimisers for tough, global problems [OptBench]_ which
forms the core of ACCES.

ACCES can take arbitrary Python scripts defining a simulation, automatically
parallelise them to execute efficiently on multithreaded laptops or even distributed
computing clusters and deterministically optimise the user-defined free parameters.
It is fault-tolerant and can return to the latest optimisation state even after e.g.
a system crash.


.. [CMAES] Hansen N, Müller SD, Koumoutsakos P. Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES). Evolutionary computation. 2003 Mar;11(1):1-8.

.. [OptBench] Rios LM, Sahinidis NV. Derivative-free optimization: a review of algorithms and comparison of software implementations. Journal of Global Optimization. 2013 Jul;56(3):1247-93.




Example ACCES Run
=================

Simulations are oftentimes huge beasts that are hard to set up and run correctly /
efficiently; ask a modeller to re-write their simulation as a function (which is
what virtually all optimisers expect) and you might not be friends anymore. ACCES
takes a different approach: it accepts *entire simulation scripts*; so you can
take even pre-existing simulations and just define the free parameters ACCES can
vary at the top, e.g.:


::

    # In file `simulation_script.py`

    #### ACCESS PARAMETERS START
    import coexist
    parameters = coexist.create_parameters(
        variables = ["cor", "separation"],
        minimums = [-3, -7],
        maximums = [+5, +3],
        values = [3, 2],                # Optional, initial guess
    )

    access_id = 0                       # Optional, unique ID for each simulation
    #### ACCESS PARAMETERS END


    # Extract variables
    x = parameters.at["cor", "value"]
    y = parameters.at["separation", "value"]


    # Define the error value in any way - run a simulation, analyse data, etc.
    a = 1
    b = 100

    error = (a - x) ** 2 + b * (y - x ** 2) ** 2


All you need to do is to create a variable called ``parameters`` (a simple
``pandas.DataFrame``) storing the free parameters' names and bounds, and
optionally starting values and relative uncertainty. Then, by the end of
the simulation script, just define a variable named exactly ``error``,
storing a number that will need to be minimised.

Importantly, **the above simulation script can be run on its own** - so
while prototyping your simulation, just run the script and see if a single
simulation works fine; if it does, then you can pass it to ACCES to
optimise the ``error``:


::

    # In file `access_learn.py`, same folder as `simulation_script.py`

    import coexist

    # Use ACCESS to learn a simulation's parameters
    access = coexist.Access("simulation_script.py")
    access.learn(
        num_solutions = 10,         # Number of solutions per epoch
        target_sigma = 0.1,         # Target std-dev (accuracy) of solution
        random_seed = 42,           # Reproducible / deterministic optimisation
    )


You will get an output showing the parameter combinations being tried and
corresponding error values:

::

    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 2 (seed=<module 'time' (built-in)>, Sat Oct  2 20:10:32 2021)
    Scaled overall standard deviation: 1.0
    Scaled individual standard deviations:
    [1.       1.000025]

    Trying 10 parameter combinations...
            cor  separation         error
    0  3.975095   -2.160040  32270.104639
    1  4.999955    0.937647  57913.534250
    2 -2.996414   -3.208848  14869.102695
    3  3.409089    0.734998  11858.244979
    4  2.946236   -1.412261  10189.783399
    5  4.900442    1.588754  50305.857496
    6  3.211298    0.190922  10249.394181
    7  4.496030   -1.437256  46891.143262
    8  4.180003   -1.835626  37290.181348
    9  4.901931    1.800291  49426.433170
    Function evaluations: 10
    ---
    Scaled overall standard deviation: 0.7917725529475667
    Scaled individual standard deviations:
    [0.67378042 0.74292858]

    [...output truncated...]

    ---
    Scaled overall standard deviation: 0.12014944702373544
    Scaled individual standard deviations:
    [0.03222829 0.04423768]

    Trying 10 parameter combinations...
            cor  separation     error
    0  0.921384    0.843603  0.009038
    1  1.019462    1.022632  0.028168
    2  0.736744    0.523327  0.107192
    3  0.844403    0.735111  0.073025
    4  0.917605    0.861830  0.046117
    5  0.923034    0.847540  0.007905
    6  1.085804    1.166795  0.022186
    7  0.902186    0.815508  0.009814
    8  0.791441    0.644249  0.075431
    9  0.881143    0.784338  0.020410
    Function evaluations: 230
    ---
    Optimal solution found within `target_sigma`, i.e. 10.0%:
    sigma = 0.09046951690975955 < 0.1

    ---
    The best result was achieved for these parameter values:
    [0.9484971  0.89783533]

    The standard deviation / uncertainty in each parameter is:
    [0.06539225 0.11303058]

    For these parameters, the error value found was: 0.002980675258852438

    These results were found for the simulation at index 200, which can be found in:
    access_info_000042/simulations



Each ACCES run creates a folder "access_info_<random_seed>" saving the optimisation
state. You can access (pun intended) it using ``coexist.AccessData.read()``,
even while the optimisation is still running for intermediate results:

::

    >>> access_data = coexist.AccessData.read("access_info_000042")
    >>> access_data

    AccessData(
      parameters:
                    value  min  max  sigma
        cor           3.0 -3.0  5.0    3.2
        separation    2.0 -7.0  3.0    4.0

      num_solutions:
        10

      target_sigma:
        0.1

      random_seed:
        42

      results:
                  cor  separation   cor_std  separation_std  overall_std         error
        0    3.975095   -2.160040  2.156097        2.971714     0.791773  32270.104639
        1    4.999955    0.937647  2.156097        2.971714     0.791773  57913.534250
        2   -2.996414   -3.208848  2.156097        2.971714     0.791773  14869.102695
        3    3.409089    0.734998  2.156097        2.971714     0.791773  11858.244979
        4    2.946236   -1.412261  2.156097        2.971714     0.791773  10189.783399
        ..        ...         ...       ...             ...          ...           ...
        225  0.923034    0.847540  0.065392        0.113031     0.090470      0.007905
        226  1.085804    1.166795  0.065392        0.113031     0.090470      0.022186
        227  0.902186    0.815508  0.065392        0.113031     0.090470      0.009814
        228  0.791441    0.644249  0.065392        0.113031     0.090470      0.075431
        229  0.881143    0.784338  0.065392        0.113031     0.090470      0.020410

        [230 rows x 6 columns]

      results_scaled:
                  cor  separation   cor_std  separation_std  overall_std         error
        0    1.242217   -0.540010  0.673780        0.742929     0.791773  32270.104639
        1    1.562486    0.234412  0.673780        0.742929     0.791773  57913.534250
        2   -0.936379   -0.802212  0.673780        0.742929     0.791773  14869.102695
        3    1.065340    0.183750  0.673780        0.742929     0.791773  11858.244979
        4    0.920699   -0.353065  0.673780        0.742929     0.791773  10189.783399
        ..        ...         ...       ...             ...          ...           ...
        225  0.288448    0.211885  0.020435        0.028258     0.090470      0.007905
        226  0.339314    0.291699  0.020435        0.028258     0.090470      0.022186
        227  0.281933    0.203877  0.020435        0.028258     0.090470      0.009814
        228  0.247325    0.161062  0.020435        0.028258     0.090470      0.075431
        229  0.275357    0.196085  0.020435        0.028258     0.090470      0.020410

        [230 rows x 6 columns]

      num_epochs:
        23
    )


You can create a "convergence plot" showing the evolution of the ACCES run using
``coexist.plots.plot_access``:

::

    import coexist

    # Use path to either the `access_info_<random_seed>` folder itself, or its
    # parent directory
    access_data = coexist.AccessData.read(".")

    fig = coexist.plots.access(access_data)
    fig.show()


Which will produce the following interactive Plotly figure:

.. image:: ../_static/convergence.png
   :alt: Convergence plot.


If you zoom into the error value, you'll see that ACCES effectively found the
optimum in less than 10 epochs; while this particular error function is
well-behaved and a gradient-based optimiser may be quicker and more accurate,
this can never be assumed with physical simulations and noisy measurements (see
the image at the top of the page).

If you have only two free parameters (or take a slice through your larger parameter
space) you can see which parts ACCES explored:

::

    import coexist

    # Use path to either the `access_info_<random_seed>` folder itself, or its
    # parent directory
    access_data = coexist.AccessData.read(".")

    fig = coexist.plots.access2d(access_data)
    fig.show()


Which will produce the following interactive Plotly figure:

.. image:: ../_static/explored.png
   :alt: Explored parameter space.

The dots are the parameter combinations tried; the cells' colours represent the
closest simulation's error value (darker = smaller). The smaller the cell, the
more simulations were run in that region - notice how ACCES spends minimum
computational time on unimportant, high-error areas and more around the global
optimum.


ACCES on Supercomputing Clusters
================================

ACCES can also run each simulation as separate, massively-parallel jobs on distributed
supercomputing clusters using the ``coexist.schedulers`` interface. For example, for
executing simulations as ``sbatch`` jobs on a SLURM-managed supercomputer (such as
the awesome BlueBEAR @Birmingham):

::

    import coexist
    from coexist.schedulers import SlurmScheduler


    scheduler = SlurmScheduler(
        "10:0:0",           # Time allocated for a single simulation
        commands = [        # Commands you'd add in the sbatch script, after `#`
            "set -e",
            "module purge; module load bluebear",
            "module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2",
        ],
        qos = "bbdefault",
        account = "windowcr-rt-royalsociety",
        constraint = "cascadelake",     # Any other #SBATCH -- <CMD> = "VAL" pair
    )

    # Use ACCESS to learn the simulation parameters
    access = coexist.Access("simulation_script.py", scheduler)
    access.learn(num_solutions = 100, target_sigma = 0.1, random_seed = 12345)


Same script as before, except for the ``scheduler`` second argument to
``coexist.Access``. For full details - and extra possible settings - do check out
the "Manual" tab at the top of the page.



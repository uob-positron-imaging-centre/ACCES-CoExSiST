# Ground Truth Data - Experimental Data Placeholder

There are 6 numpy archives (pickles?) in this directory, coming from 3
different simulations. The `positions` files contain a 3D numpy array of the
particle positions in the simulation: a (600, 100, 3) array, for 600 saved
timesteps with 100 particles that have 3 coordinates.

    1. `positions.npy` and `timesteps.npy` are from a simulation with corPP and
       corPW = 0.5. The simulation ran up to t = 10.0 seconds, with 600
       checkpoints (it's as if we have 60 FPS of data).

    2. `positions_short` and `timesteps_short` are from a simulation with corPP
       and corPW = 0.5 (like above). The simulation ran up to t = 1.0 seconds,
       with 600 checkpoints (so it has 10 times finer data than the previous).

    3. `positions_short_opt` and `timesteps_short_opt` are from a simulation
       with corPP and corPW = 0.4. The simulation ran up to t = 1.0 seconds,
       with 600 checkpoints (like above). It's useful to see the differences
       between the simulations (check view_opt.py for a plotly visualisation).






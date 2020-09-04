#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import numpy as np
import cma

import coexist


exp_pos = np.load("truth/positions_short.npy")
exp_tsteps = np.load("truth/timesteps_short.npy")
radius = 0.005 / 2


parameters = coexist.Parameters(
    ["corPP", "corPW"],
    ["fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPW} ${corPW2} \
        ${corPW} ${corPW2} ${corPW} \
        ${corPW2} ${corPW} ${corPW} ",
    "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
        ${corPP} ${corPW} ${corPW2} \
        ${corPW} ${corPW2} ${corPW} \
        ${corPW2} ${corPW} ${corPW} "],
    [0.4, 0.4],     # Initial WRONG values
    [0.0, 0.0],     # Minimum values
    [1.0, 1.0]      # Maximum values
)

simulation = coexist.Simulation("run.sim", parameters)#, verbose = False)
print(simulation)


def compute_xi(exp_positions, sim_positions):
    xi = np.linalg.norm(exp_positions - sim_positions, axis = 1)

    # Threshold - the center of a simulated particle should always be inside
    # the equivalent experimental particle
    xi[xi < radius] = 0.

    xi = xi.sum()

    return xi


def sim_hash(solutions):
    h = (np.array(solutions) * 10000).sum()
    return str(h)[:4]


def simulate_xi(simulation, num_steps, solutions, truth):
    print(f"Trying solutions: {solutions}")

    global_save_name = f"restarts/simopt_{sim_hash(solutions)}.restart"
    local_save_name = "restarts/simopt.restart"


    print("SAVING\n")
    simulation.save(global_save_name)
    print("SAVED\n")

    param_names = simulation.parameters.index

    results = []
    for sol in solutions:
        # Save before running
        simulation.save(local_save_name)

        # Change parameters
        for i, sol_val in enumerate(sol):
            simulation[param_names[i]] = sol_val

        # Run simulation for `num_steps`
        simulation.step(num_steps)

        # Save xi value
        results.append(compute_xi(truth, simulation.positions()))

        # Load back simulation
        simulation.load(local_save_name)

    simulation.load(global_save_name)
    return results


def optimise(simulation, num_steps, truth):
    print("\nStarting optimisation of simulation:")
    print(simulation, "\n")

    x0 = simulation.parameters["value"].to_numpy()
    mins = simulation.parameters["min"].to_numpy()
    maxs = simulation.parameters["max"].to_numpy()

    sigma0 = 0.1
    bounds = [
        mins,   # lower bounds
        maxs,   # upper bounds
    ]

    es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
        bounds = bounds,
        ftarget = 0.,
        # tolflatfitness = 10,                # Try 10 times if f is flat
        # tolfun = -np.inf,                   # Only stop when tolx < 1
        # tolx = 1,                           # Stop if changes in x < 1
        # verbose = 3 if verbose else -9
    ))

    while not es.stop():
        solutions = es.ask()

        results = simulate_xi(simulation, num_steps, solutions, truth)

        es.tell(solutions, results)
        es.disp()


# Number of steps between checkpoints
num_steps = int((exp_tsteps[1] - exp_tsteps[0]) / simulation.step_size)
simulation.save()

for i, t in enumerate(exp_tsteps):
    simulation.step_to_time(t)
    pos = simulation.positions()

    xi = compute_xi(exp_pos[i], pos)
    print(f"i: {i:>4} | xi: {xi}")

    if xi > radius:
        # Load previous timestep's simulation
        simulation.load()

        # Re-iterate this timestep, optimising the sim parameters
        optimise(simulation, num_steps, exp_pos[i])

    simulation.save()






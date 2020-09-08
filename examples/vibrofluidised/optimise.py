#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import numpy as np
import cma

import coexist


exp_positions = np.load("truth/positions_short.npy")
exp_timesteps = np.load("truth/timesteps_short.npy")
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
    [0.45, 0.45],       # Initial WRONG values
    [0.05, 0.05],     # Minimum values
    [0.95, 0.95]      # Maximum values
)

simulation = coexist.Simulation("run.sim", parameters, verbose = False)
print(simulation)

global_solutions = []
global_results = []

###
### Optimisation Functions
###

def compute_xi(sim_positions, exp_positions):
    xi = np.linalg.norm(exp_positions - sim_positions, axis = 1)

    # Threshold - the center of a simulated particle should always be inside
    # the equivalent experimental particle
    #xi[xi < radius] = 0.

    xi = xi.sum()

    return xi


def compute_xi_acc(simulation, num_steps, num_checkpoints, truth):
    xi_acc = 0.

    for i in range(num_checkpoints):
        simulation.step(num_steps)
        xi_acc += compute_xi(simulation.positions(), truth[i + 1])

    return xi_acc


def sim_hash(solutions):
    h = (np.array(solutions) * 10000).sum()
    return str(h)[:4]


def try_solutions(simulation, num_steps, num_checkpoints, solutions, truth):
    # Save current checkpoint
    save_name = f"restarts/simopt_{sim_hash(solutions)}.restart"
    simulation.save(save_name)

    param_names = simulation.parameters.index


    # TEST
    simulation["corPP"] = 0.5
    simulation["corPW"] = 0.5
    xiacc = compute_xi_acc(simulation, num_steps, num_checkpoints, truth)
    print(f"\n\n\nTEST xiacc: {xiacc}\n\n\n")
    simulation.load(save_name)


    results = []
    for sol in solutions:
        # Change parameters
        for i, sol_val in enumerate(sol):
            simulation[param_names[i]] = sol_val

        # Compute and save the xi_acc value
        results.append(
            compute_xi_acc(simulation, num_steps, num_checkpoints, truth)
        )

        # Load the  simulation
        simulation.load(save_name)

    global_solutions.extend(solutions)
    global_results.extend(results)

    return results


def optimise(
    simulation,
    num_steps,
    num_checkpoints,
    exp_positions
):
    print("\nStarting optimisation of simulation:")
    print(simulation, "\n")

    x0 = simulation.parameters["value"].to_numpy()
    mins = simulation.parameters["min"].to_numpy()
    maxs = simulation.parameters["max"].to_numpy()

    sigma0 = 0.2
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
        print(f"Trying solutions {solutions}")

        results = try_solutions(
            simulation,
            num_steps,
            num_checkpoints,
            solutions,
            exp_positions,
        )

        es.tell(solutions, results)
        es.disp()

    solutions = es.result.xbest
    param_names = simulation.parameters.index

    print(f"Best results for solutions {solutions}")

    # Change parameters
    for i, sol_val in enumerate(solutions):
        simulation[param_names[i]] = sol_val

    # Compute xi_acc for found solutions and move simulation forward in time
    xi_acc = compute_xi_acc(simulation, num_steps, num_checkpoints, exp_positions)
    return xi_acc


###
### Optimisation Driver Script
###


# Number of steps between checkpoints
num_steps = int((exp_timesteps[1] - exp_timesteps[0]) / simulation.step_size)
simulation.save()

# Number of particles
num_particles = len(exp_positions[0])

# Cummulative xi
xi_acc = 0.

# Number of checkpoints with non-zero xi
num_checkpoints = 0

# Number of LIGGGHTS steps per checkpoint
num_steps = 10_000

# Erroneous experimental positions at checkpoints from start to end
start_index = 0
end_index = 0


for i in range(100):
    # Run simulation up to experimental timestep t
    simulation.step(num_steps)
    sim_positions = simulation.positions()

    xi = compute_xi(sim_positions, exp_positions[i])
    xi_acc += xi

    print(f"i: {i:>4} | xi: {xi:5.5e} | xi_acc: {xi_acc:5.5e} | num_checkpoints: {num_checkpoints:>4}")
    #print(np.hstack((sim_positions, exp_positions[i])))

    # Only save checkpoints if experiment is perfectly synchronised
    if xi_acc == 0.:
        simulation.save()
        num_checkpoints = 0

        start_index = i
        end_index = i
    else:
        num_checkpoints += 1

    # If the accummulated erorr (Xi) is large enough, start optimisation
    if xi_acc > radius * num_particles:
        if xi_acc > 2 * radius * num_particles:
            print("Might be chaotic")

        # Load previous checkpoint's simulation
        simulation.load()

        # Optimise against experimental positions from start_index:end_index
        end_index = i + 1

        # Re-iterate the steps since the last checkpoints, optimising the sim
        # parameters. This function will also run the sim up to the current t
        xi_acc = optimise(
            simulation,
            num_steps,
            num_checkpoints,
            exp_positions[start_index:end_index],
        )





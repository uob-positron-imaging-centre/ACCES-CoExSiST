#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import os

import numpy as np
import cma

from joblib import Parallel, delayed

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
    [0.45, 0.55],   # Initial WRONG values
    [0.05, 0.05],   # Minimum values
    [0.95, 0.95],   # Maximum values
    #[0.1, 0.1],     # Sigma0
)

simulation = coexist.Simulation("run.sim", parameters, verbose = False)
print(simulation)

global_solutions = []
global_results = []

###
### Optimisation Functions
###


def diverging(sim_positions, exp_positions):
    dist = np.linalg.norm(exp_positions - sim_positions, axis = 1)
    return (dist > radius).any()


def compute_xi(sim_positions, exp_positions):
    return np.linalg.norm(exp_positions - sim_positions, axis = 1).sum()


def compute_xi_acc(simulation, num_steps, num_checkpoints, truth):
    xi_acc = 0.

    for i in range(num_checkpoints):
        simulation.step(num_steps)
        xi_acc += compute_xi(simulation.positions(), truth[i + 1])

    return xi_acc


def sim_hash(solutions):
    h = hash(np.array(solutions).tobytes())
    return str(h)


def try_solutions(simulation, num_steps, num_checkpoints, solutions, truth):
    # Save current checkpoint
    save_name = f"restarts/simopt_{sim_hash(solutions)}.restart"
    simulation.save(save_name)

    param_names = simulation.parameters.index

    results = []
    def try_solution(sol):
        simulation.load(save_name)

        # Change parameters
        for i, sol_val in enumerate(sol):
            simulation[param_names[i]] = sol_val

        # Compute and save the xi_acc value
        xi_acc = compute_xi_acc(simulation, num_steps, num_checkpoints, truth)

        # Load the simulation
        simulation.load(save_name)

        return xi_acc

    # results = Parallel(n_jobs = 8)(
    #     delayed(try_solution)(sol)
    #     for sol in solutions
    # )
    results = [try_solution(sol) for sol in solutions]

    global_solutions.extend(solutions)
    global_results.extend(results)

    # Delete the saved checkpoint
    os.remove(save_name)

    return results


def optimise(
    simulation,
    num_steps,
    num_checkpoints,
    exp_positions,
):
    print("\nStarting optimisation of simulation:")
    print(simulation, "\n")

    # Track solutions tried and the corresponding results
    global global_solutions
    global global_results
    global_solutions = []
    global_results = []
    # Global track end

    mins = simulation.parameters["min"].to_numpy()
    maxs = simulation.parameters["max"].to_numpy()

    # If any `sigma` value is smaller than 5% (max - min), clip it
    p = simulation.parameters
    sigma_clipped = p["sigma"].clip(lower = 0.05 * (p["max"] - p["min"]))

    # Scale sigma, bounds, solutions, results to unit variance
    scaling = sigma_clipped.to_numpy()

    x0 = simulation.parameters["value"].to_numpy() / scaling
    sigma0 = 1.0
    bounds = [
        mins / scaling,
        maxs / scaling
    ]

    es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
        bounds = bounds,
        ftarget = 0.0,
        popsize = 8 * len(simulation.parameters),
        # verbose = 3 if verbose else -9
    ))

    while not es.stop():
        solutions = es.ask()

        print(f"Scaled sigma: {es.sigma}")
        print(f"Trying solutions:\n{solutions * scaling}")

        results = try_solutions(
            simulation,
            num_steps,
            num_checkpoints,
            solutions * scaling,
            exp_positions,
        )

        es.tell(solutions, results)

        print(f"Function evaluations: {es.result.evaluations}\n---")

        if es.sigma < 0.05:
            print("Optimal solution found within 5%:")
            print(f"sigma = {es.sigma} < 0.05")
            break

    solutions = es.result.xbest * scaling
    param_names = simulation.parameters.index

    # Change sigma, min and max based on optimisation results
    simulation.parameters["sigma"] = es.result.stds * scaling

    print(f"Best results for solutions {solutions}\n---")

    # Change parameters
    for i, sol_val in enumerate(solutions):
        simulation[param_names[i]] = sol_val

    # Compute xi_acc for found solutions and move simulation forward in time
    xi_acc = compute_xi_acc(
        simulation,
        num_steps,
        num_checkpoints,
        exp_positions
    )

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

# Minimum number of diverging checkpoints for optimisation to start
min_checkpoints = 5

# Maximum number of optimisation runs before binding sim_pos to exp_pos
max_optimisations = 3

# Number of LIGGGHTS steps per checkpoint
num_steps = 10_000

num_optimisations = 0

for i in range(100):
    # Run simulation up to experimental timestep t
    simulation.step(num_steps)
    sim_positions = simulation.positions()

    xi = compute_xi(sim_positions, exp_positions[i])
    xi_acc += xi

    # Only save checkpoints if experiment is perfectly synchronised
    if not diverging(sim_positions, exp_positions[i]):
        simulation.save()
        num_checkpoints = 0

        # Number of optimisation runs done for this starting time window
        num_optimisations = 0

        # Erroneous experimental positions at checkpoints from start to end
        start_index = i
        end_index = i
    else:
        num_checkpoints += 1

    print(f"i: {i:>4} | xi: {xi:5.5e} | xi_acc: {xi_acc:5.5e} | " +
          f"num_checkpoints: {num_checkpoints:>4}")

    # If the accummulated erorr (Xi) is large enough, start optimisation
    if (xi_acc > radius * num_particles and
        num_checkpoints >= min_checkpoints * (num_optimisations + 1)):

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

        num_optimisations += 1

    # If too many optimisation runs were done, bind the sim_pos to exp_pos
    if num_optimisations > max_optimisations:
        pass




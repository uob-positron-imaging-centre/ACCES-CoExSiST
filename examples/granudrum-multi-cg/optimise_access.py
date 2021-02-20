#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise_access.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import sys
import shutil

import numpy as np

import cv2

import pept
import pept.processing as pp
import coexist


rpms = [10, 30, 60]
scripts = [f"granudrum_{rpm}rpm.sim" for rpm in rpms]
images = [f"{rpm}rpm_avg.jpg" for rpm in rpms]


class GranuDrum:
    '''Class encapsulating a GranuTools GranuDrum system dimensions.

    The GranuDrum is assumed to be centred at (0, 0). The default values are
    given **in millimetres**.

    Attributes
    ----------
    xlim : (2,) numpy.ndarray
        The system limits in the x-dimension (i.e. horizontally).

    ylim : (2,) numpy.ndarray
        The system limits in the y-dimension (i.e. vertically).

    radius : float
        The GranuDrum's radius, typically 42 mm.

    '''

    def __init__(
        self,
        xlim = [-42, +42],
        ylim = [-42, +42],
        radius = None,
    ):
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)

        if radius is None:
            self.radius = xlim[1]
        else:
            self.radius = float(radius)


def encode_u8(image: np.ndarray) -> np.ndarray:
    '''Convert occupancy from doubles to uint8 - i.e. encode real values to
    the [0-255] range.
    '''

    u8min = np.iinfo(np.uint8).min
    u8max = np.iinfo(np.uint8).max

    img_min = float(image.min())
    img_max = float(image.max())

    img_bw = (image - img_min) / (img_max - img_min) * (u8max - u8min) + u8min
    img_bw = np.array(img_bw, dtype = np.uint8)

    return img_bw


def image_thresh(granudrum: GranuDrum, image_path: str) -> pept.Pixels:
    '''Return the thresholded GranuDrum image in the `pept.Pixels` format.
    '''

    # Load the image in grayscale and ensure the orientation is:
    #    - x is downwards
    #    - y is rightwards
    image = 255 - cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[::-1]
    image = pept.Pixels(image.T, xlim = granudrum.xlim, ylim = granudrum.ylim)

    # Pixellise / discretise the GranuDrum circular outline
    xgrid = np.linspace(granudrum.xlim[0], granudrum.xlim[1], image.shape[0])
    ygrid = np.linspace(granudrum.ylim[0], granudrum.ylim[1], image.shape[1])

    # Physical coordinates of all pixels
    xx, yy = np.meshgrid(xgrid, ygrid)

    # Remove the GranuDrum's circular outline
    image[xx ** 2 + yy ** 2 > 0.98 * granudrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((21, 21), np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Global thresholding + binarisation
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img[xx ** 2 + yy ** 2 > 0.98 * granudrum.radius ** 2] = 0

    img = pept.Pixels(img, xlim = granudrum.xlim, ylim = granudrum.ylim)

    return img


def simulation_thresh(
    granudrum: GranuDrum,
    image_shape: tuple,
    positions: np.ndarray,
    particle_radii: np.ndarray,
) -> pept.Pixels:
    '''Return the thresholded occupancy grid of the GranuDrum DEM simulation in
    the `pept.Pixels` format.
    '''

    # Compute occupancy grid from the XZ plane (i.e. granular drum side)
    sim_occupancy = pp.occupancy2d(
        positions[:, [0, 2]],
        image_shape,
        radii = particle_radii,
        xlim = granudrum.xlim,
        ylim = granudrum.ylim,
        verbose = False,
    )

    # Pixellise / discretise the GranuDrum circular outline
    xgrid = np.linspace(granudrum.xlim[0], granudrum.xlim[1], image_shape[0])
    ygrid = np.linspace(granudrum.ylim[0], granudrum.ylim[1], image_shape[1])

    # Physical coordinates of all pixels
    xx, yy = np.meshgrid(xgrid, ygrid)

    # Remove the GranuDrum's circular outline
    sim_occupancy[xx ** 2 + yy ** 2 > 0.98 * granudrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((21, 21), np.uint8)
    sim = cv2.morphologyEx(
        encode_u8(sim_occupancy),
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Global thresholding + binarisation
    _, sim = cv2.threshold(sim, 30, 255, cv2.THRESH_BINARY)
    sim[xx ** 2 + yy ** 2 > 0.98 * granudrum.radius ** 2] = 0

    sim = pept.Pixels(sim, xlim = granudrum.xlim, ylim = granudrum.ylim)

    return sim


def subtract_occupancies(
    granudrum: GranuDrum,
    image_path: str,
    radii_path: str,
    positions_path: str,
) -> float:
    '''Error function for a GranuTools GranuDrum, quantifying the difference
    between an experimental free surface shape (from an image) and a simulated
    one (from an occupancy grid).

    The error value represents the difference in the XZ-projected area of the
    granular drum (i.e. the side view) between the GranuDrum image and the
    occupancy grid of the corresponding simulation.

    Parameters
    ----------
    granudrum: GranuDrum
        The physical dimensions of the GranuTools GranuDrum, encapsulated in a
        class (defined in this file).

    image_path: str
        The path to an aligned ".jpg" image of the GranuDrum side view.

    radii_path: str
        The path to a `.npy` NumPy binary file (saved with `numpy.save`)
        containing the radius of each particle in the system. If needed, load
        it with `numpy.load(radii_path)`.

    positions_path: str
        The path to a `.npy` NumPy binary file (saved with `numpy.save`)
        containing the stacked particle positions arrays (num_particles *
        3 columns for [x, y, z]) for all timesteps; i.e. shape (T, P, 3), where
        T is the number of timesteps, P is the number of particles. If needed,
        load it with `numpy.load(positions_path)`.

    Returns
    -------
    float
        The integrated absolute difference between the fitted 3rd order
        polynomials of the image and the simulated positions.

    '''

    # Load the simulated particles' radiis and positions
    positions_all = np.load(positions_path) * 1000                      # mm

    radii = np.tile(np.load(radii_path), len(positions_all)) * 1000     # mm
    positions = np.concatenate(positions_all)

    # The thresholded GranuDrum image and occupancy plot, as `pept.Pixels`,
    # containing only 0s and 1s
    img = image_thresh(granudrum, image_path)
    sim = simulation_thresh(granudrum, img.shape, positions, radii)

    # Pixel physical dimensions, in mm
    dx = (granudrum.xlim[1] - granudrum.xlim[0]) / img.shape[0]
    dy = (granudrum.ylim[1] - granudrum.ylim[0]) / img.shape[1]

    # The error is the total different area, i.e. the number of pixels with
    # different values times the area of a pixel
    error = float((img != sim).sum()) * dx * dy

    return error


def error_function(
    radii_paths: list[str],
    positions_paths: list[str],
    velocities_paths: list[str],
) -> float:
    '''An ACCESS error function quantifying the difference between multiple
    GranuDrum images (at different RPMs) and the corresponding DEM simulations,
    all with the same particle parameters that are being optimised.

    The error value is computed for each simulation as the difference in the
    XZ-projected area of the granular drum (i.e. the side view) between the
    GranuDrum image and the occupancy grid of the corresponding simulation.
    The final error value is simply the sum of the individual errors.

    The input parameters are all lists of paths to the corresponding `.npy`
    NumPy array binary archive containing the simulation data; each can be
    loaded with `numpy.load(path)`.
    '''

    # Rotating drum system dimensions
    granudrum = GranuDrum()
    errors = np.zeros(len(rpms))

    # Sum all error values from each simulation. Velocities are not needed.
    for i in range(len(rpms)):
        errors[i] = subtract_occupancies(
            granudrum,
            images[i],
            radii_paths[i],
            positions_paths[i],
        )

    return errors.sum()


# Define the user-changeable / free simulation parameters
parameters = coexist.Parameters(
    variables = [
        "N",
        "fricPP",
        "fricPW",
        "fricPSW",
        "fricRollPP",
        "fricRollPW",
        "fricRollPSW",
        "corPP",
        "corPW",
        "corPSW",
    ],
    commands = [
        "fix ins all insert/stream seed 67867967 distributiontemplate pdd \
            nparticles ${N} particlerate 1000000 overlapcheck yes all_in no \
            vel constant 0.0 0.0 -1.0 insertion_face inface extrude_length \
            0.03                                            ",
        "fix m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}      \
            ${fricPW}       ${fric}         ${fric}         \
            ${fricPSW}      ${fric}         ${fric}         ",
        "fix m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}      \
            ${fricPW}       ${fric}         ${fric}         \
            ${fricPSW}      ${fric}         ${fric}         ",
        "fix m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}      \
            ${fricPW}       ${fric}         ${fric}         \
            ${fricPSW}      ${fric}         ${fric}         ",
        "fix m7 all property/global coefficientRollingFriction peratomtypepair 3 \
            ${fricRollPP}   ${fricRollPW}   ${fricRollPSW}  \
            ${fricRollPW}   ${fricRoll}     ${fricRoll}     \
            ${fricRollPSW}  ${fricRoll}     ${fricRoll}     ",
        "fix m7 all property/global coefficientRollingFriction peratomtypepair 3 \
            ${fricRollPP}   ${fricRollPW}   ${fricRollPSW}  \
            ${fricRollPW}   ${fricRoll}     ${fricRoll}     \
            ${fricRollPSW}  ${fricRoll}     ${fricRoll}     ",
        "fix m7 all property/global coefficientRollingFriction peratomtypepair 3 \
            ${fricRollPP}   ${fricRollPW}   ${fricRollPSW}  \
            ${fricRollPW}   ${fricRoll}     ${fricRoll}     \
            ${fricRollPSW}  ${fricRoll}     ${fricRoll}     ",
        "fix m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}       \
            ${corPW}        ${cor}          ${cor}          \
            ${corPSW}       ${cor}          ${cor}          ",
        "fix m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}       \
            ${corPW}        ${cor}          ${cor}          \
            ${corPSW}       ${cor}          ${cor}          ",
        "fix m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}       \
            ${corPW}        ${cor}          ${cor}          \
            ${corPSW}       ${cor}          ${cor}          ",
    ],
    values =    [2000, 0.75, 0.75, 0.75, 0.50, 0.50, 0.50, 0.60, 0.60, 0.60],
    minimums =  [1500, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    maximums =  [2500, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 0.95, 0.95, 0.95],
)

simulations = []
for i, script in enumerate(scripts):
    print(f"Loading simulation {i + 1} from {script}...")

    simulations.append(
        coexist.LiggghtsSimulation(
            script,
            parameters,
            set_parameters = False,
        )
    )

# Simulate GD for 3 rotations (it usually reaches steady state within one
# rotation) and use the last one for computing the surface. Choose the slowest
# RPM for computing the simulation time
num_rotations = 3

start_times = [(num_rotations - 1) * 60 / rpm for rpm in rpms]
end_times = [num_rotations * 60 / rpm for rpm in rpms]

# If we're on a SLURM multi-node cluster, use `sbatch` for running simulations.
# Otherwise just use the normal Python executable
if shutil.which("sbatch"):
    scheduler = ["sbatch", "batch_single_sim.sh"]
else:
    scheduler = [sys.executable]

# Use ACCESS to learn the simulation parameters
access = coexist.Access(simulations, scheduler = scheduler)

best_paths = access.learn(
    error = error_function,
    start_times = start_times,
    end_times = end_times,
    num_solutions = 100,
    target_sigma = 0.1,
    use_historical = True,
    save_positions = True,
    random_seed = 19937,
)

print("\n\n------------\nFinished!!!\n------------\n\n")
print(f"Paths to the simulation data of the best parameters: {best_paths}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import numpy as np

from numpy.polynomial import Polynomial
from scipy.integrate import quad

import cv2

import pept.processing as pp
import coexist


class GranuDrum:
    '''Class encapsulating a GranuTools GranuDrum system dimensions.

    The GranuDrum is assumed to be centred at (0, 0).

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
        xlim = [-0.042, +0.042],
        ylim = [-0.042, +0.042],
        radius = None,
    ):
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)

        if radius is None:
            self.radius = xlim[1]
        else:
            self.radius = float(radius)


def encode_u8(image):
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


def surface_image(image_path, granudrum):
    '''Extract the free surface from a granular drum's image and return a
    fitted 3rd order polynomial.
    '''

    # Read in image and run the Canny edge detection algorithm
    img = 255 - cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[::-1]
    img_edges = cv2.Canny(img, 200, 400)

    # Extract edges as [x, y] point pairs
    img_edges = np.argwhere(img_edges)
    img_edges = img_edges[:, [1, 0]]

    # Physical dimension of each pixel
    xlim = granudrum.xlim
    ylim = granudrum.ylim
    radius = granudrum.radius

    dx = (xlim[1] - xlim[0]) / img.shape[0]
    dy = (ylim[1] - ylim[0]) / img.shape[1]

    # Transform edges from pixels to physical dimensions
    img_edges = img_edges * [dx, dy]
    img_edges += [xlim[0], ylim[0]]

    # Remove enclosing circle to extract free surface
    img_surface = img_edges[
        (img_edges ** 2).sum(axis = 1) < 0.95 * radius ** 2
    ]

    # Fit third order polynomial to free surface extracted from image
    img_poly = Polynomial.fit(img_surface[:, 0], img_surface[:, 1], 3)

    return img_poly


def surface_simulation(positions, particle_radii, granudrum):
    '''Extract the free surface from a granular drum simulation and return a
    fitted 3rd order polynomial.
    '''

    # Concatenate positions from multiple timesteps
    positions = np.concatenate(positions)

    # Physical dimension of each pixel
    xlim = granudrum.xlim
    ylim = granudrum.ylim
    radius = granudrum.radius

    # Number of pixels in the image
    shape = (808, 808)

    dx = (xlim[1] - xlim[0]) / shape[0]
    dy = (ylim[1] - ylim[0]) / shape[1]

    # Compute occupancy grid from the XZ plane (i.e. granular drum side)
    sim_occupancy = pp.occupancy2d(
        positions[:, [0, 2]],
        shape,
        radii = particle_radii,
        xlim = xlim,
        ylim = ylim,
    )

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((10, 10), np.uint8)
    sim = cv2.morphologyEx(
        encode_u8(sim_occupancy),
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Otsu thresholding + binarisation
    _, sim = cv2.threshold(sim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    sim_edges = cv2.Canny(sim, 30, 40)

    # Extract edges as [x, y] point pairs
    sim_edges = np.argwhere(sim_edges)

    # Transform edges from pixels to physical dimensions
    sim_edges = sim_edges * [dx, dy]
    sim_edges += [xlim[0], ylim[0]]

    # Remove enclosing circle to extract free surface
    sim_surface = sim_edges[
        (sim_edges ** 2).sum(axis = 1) < 0.95 * radius ** 2
    ]

    # Fit third order polynomial
    sim_poly = Polynomial.fit(sim_surface[:, 0], sim_surface[:, 1], 3)

    return sim_poly


def integrate_surfaces(image_path, positions, particle_radii = 0):
    '''Error function for a GranuTools GranuDrum, quantifying the difference
    between an experimental free surface shape (from an image) and a simulated
    one (from an occupancy grid).

    The error function represents the integrated area between the two free
    surfaces, both computed as 3rd order polynomials fitted to OpenCV-processed
    images.

    Parameters
    ----------
    image_path: str
        A path to a GranuTools GranuDrum image.

    positions: (T, P, 3) numpy.ndarray
        A numpy array of stacked 2D particle positions arrays (num_particles *
        3 columns for [x, y, z]) for all timesteps.

    particle_radii: float or (P,) list-like, default 0
        The radius of each particle. If zero, every particle is considered as a
        discrete point. If a single float, all particles are considered to have
        the same radius. If it is a numpy array, it specifies each particle’s
        radius, and must have the same length as positions.

    Returns
    -------
    float
        The integrated absolute difference between the fitted 3rd order
        polynomials of the image and the simulated positions.

    '''

    # Rotating drum system dimensions
    granudrum = GranuDrum()

    img_poly = surface_image(image_path, granudrum)
    sim_poly = surface_simulation(positions, particle_radii, granudrum)

    # Integrate the absolute difference over the image's interval
    x0 = img_poly.domain[0]
    x1 = img_poly.domain[1]

    err = quad(lambda x: np.abs(img_poly(x) - sim_poly(x)), x0, x1)
    return err[0]


# Define the user-changeable / free simulation parameters
parameters = coexist.Parameters(
    variables = [
        "dens", "youngmodP", "cohPP", "corPP", "corPW", "fricPP", "fricPW"
    ],
    commands = [
        "set type 1 density ${dens}",
        "fix  m1 all property/global youngsModulus peratomtype \
            ${youngmodP}    ${youngmodP}    ${youngmodP}",
        "fix  m6 all property/global cohesionEnergyDensity peratomtypepair 3 \
            ${cohPP}        ${cohPW}        ${cohPSW}   \
            ${cohPW}        ${coh}          ${coh}      \
            ${cohPSW}       ${coh}          ${coh}      ",
        "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}   \
            ${corPW}        ${cor}          ${cor}      \
            ${corPSW}       ${cor}          ${cor}      ",
        "fix  m3 all property/global coefficientRestitution peratomtypepair 3 \
            ${corPP}        ${corPW}        ${corPSW}   \
            ${corPW}        ${cor}          ${cor}      \
            ${corPSW}       ${cor}          ${cor}      ",
        "fix  m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}  \
            ${fricPW}       ${fric}         ${fric}     \
            ${fricPSW}      ${fric}         ${fric}     ",
        "fix  m4 all property/global coefficientFriction peratomtypepair 3 \
            ${fricPP}       ${fricPW}       ${fricPSW}  \
            ${fricPW}       ${fric}         ${fric}     \
            ${fricPSW}      ${fric}         ${fric}     ",
    ],
    values =    [1580.0,   9.2e6, 0,   0.61, 0.61, 0.42, 0.42],
    minimums =  [100.0,    5e6,   0,   0.05, 0.05, 0.05, 0.05],
    maximums =  [10_000.0, 1e9,   1e5, 0.95, 0.95, 0.95, 0.95],
)

print("Loading simulation...", flush = True)
simulation = coexist.Simulation("granudrum.sim", parameters)
print(simulation, flush = True)

# Simulate GD for 3 rotations (it usually reaches steady state within one
# rotation) and use the last one for computing the surface
rpm = 30
num_rotations = 3

start_time = (num_rotations - 1) * 60 / rpm
end_time = num_rotations * 60 / rpm


access = coexist.Access(simulation)

positions = access.learn(
    error = lambda pos: integrate_surfaces("30rpm_avg.jpg", pos, 1.2e-3 / 2),
    start_time = start_time,
    end_time = end_time,
    num_solutions = 14,
    target_sigma = 0.1,
    use_historical = True,
    save_positions = True,
    random_seed = 19937,
)

print("\n\n------------\nFinished!!!\n------------\n\n")
np.save("positions_best.npy", positions)

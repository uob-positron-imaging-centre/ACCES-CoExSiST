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

import pept
import pept.processing as pp

import coexist


def extract_surface(occupancy):
    '''Extract the surface from a 2D occupancy plot (which is given as cells)
    and return a fitted 3rd order polynomial with the zeroth order coefficient
    set to 0.

    Parameters
    ----------
    occupancy: pept.Pixels
        Input occupancy plot.

    Returns
    -------
    xpos
        X coordinates of the surface.

    ypos
        Y coordinates of the surface.

    numpy.polynomial.Polynomial
        The fitted 3rd order Polynomial.

    '''

    # Aliases
    img = occupancy

    # Threshold against 10% of largest value
    maxval = float(img.max())
    img[img < maxval * 0.1] = 0.

    # Loop over occupancy and extract highest value in height (along columns)
    # Find indices of last nonzero elements
    ypos = np.zeros(img.shape[1])

    for i, row in enumerate(img):
        nz = row.nonzero()[0]
        ypos[i] = nz[-1] if len(nz) else np.nan

    # Multiply them by the cell length to get x and y points
    xpos = img.xlim[0] + img.pixel_size[0] * (0.5 + np.arange(img.shape[0]))
    ypos = img.ylim[0] + img.pixel_size[1] * (0.5 + ypos)

    # Take the highest point on the occupancy and clip its sides
    max_idx = np.nanargmax(ypos)

    xpos_clip = xpos[max_idx:-max_idx]
    ypos_clip = ypos[max_idx:-max_idx]

    # Fit a 3rd order polynomial to the clipped occupancy
    occupancy_polynomial = Polynomial.fit(xpos_clip, ypos_clip, 3)

    return xpos, ypos, occupancy_polynomial


def error(positions):
    '''Error function for granular drum, quantifying the difference between an
    experimental free surface shape (from an image) and a simulated one (from
    an occupancy grid).

    Parameters
    ----------
    positions: (T, P, 3) numpy.ndarray
        A numpy array of stacked 2D particle positions arrays (num_particles *
        3 columns for [x, y, z]) for all timesteps.

    Returns
    -------
    float
        The integrated absolute difference between the fitted 3rd order
        polynomials of the image and the simulated positions.

    '''

    # GranuDrum side picture path
    image_path = "test.bmp"

    # Read in experimental image and threshold it using OpenCV
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    thresh, bw_image = cv2.threshold(
        gray_image,
        40,
        255,
        cv2.THRESH_BINARY
    )

    # Extract circle to get the GranuDrum diameter and shape
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1.2, 100)

    if len(circles) > 1:
        raise ValueError("Too many circles detected!")

    x, y, r = circles[0][0]
    image = 255 - bw_image

    xlim = [-0.042, 0.042]
    ylim = [-0.042, 0.042]

    # Save image in a pixel grid (that also stores system dimensions) from the
    # `pept` package, then extract the free surface shape (as x-y pairs) and
    # a fitted 3rd order polynomial
    image = pept.Pixels(
        np.rot90(image, 3),
        xlim = xlim,
        ylim = ylim,
    )

    img_xpos, img_ypos, img_poly = extract_surface(image)

    # Extract the free surface shapes from the recorded simulated positions by
    # creating a 2D occupancy grid
    radii = 1.5e-3
    number_of_pixels = (1000, 1000)

    pixels = pp.occupancy2d(
        positions[0][:, [0, 2]],
        number_of_pixels,
        radii,
        xlim = xlim,
        ylim = ylim,
    )

    # Superimpose the pixel grids from multiple simulation timesteps
    for i in range(1, len(positions)):
        pixels += pp.occupancy2d(
            positions[i][:, [0, 2]],
            number_of_pixels,
            radii,
            xlim = xlim,
            ylim = ylim,
            verbose = False
        )

    # Extract free surface and fitted polynomial
    pix_xpos, pix_ypos, pix_poly = extract_surface(pixels)

    # The two fitted free surface polynomials
    p1 = img_poly
    p2 = pix_poly

    # Integrate the difference over the largest common interval
    x0 = max(p1.domain[0], p2.domain[0])
    x1 = min(p1.domain[1], p2.domain[1])

    err = quad(lambda x: np.abs(p1(x) - p2(x)), x0, x1)
    return err[0]



parameters = coexist.Parameters(
    parameters = ["youngmodP", "cohPP", "corPP", "corPW", "fricPP", "fricPW"],
    commands = [
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
    values =    [5.2e8, 1e2, 0.79, 0.58, 0.20, 0.71],
    minimums =  [5e6,   0,   0.05, 0.05, 0.05, 0.05],
    maximums =  [1e9,   1e5, 0.95, 0.95, 0.95, 0.95],
)

print("Loading simulation...")
simulation = coexist.Simulation(
    "run.sim",
    parameters,
    verbose = False
)

print(simulation)

# Simulate GD for 4 rotations and use the last one for computing the surface
rpm = 45
num_rotations = 3

time_start = (num_rotations - 1) * 60 / rpm
time_end = num_rotations * 60 / rpm


access = coexist.Access(simulation)

positions = access.learn(
    error,
    time_start,
    time_end,
    num_solutions = 20,
    target_sigma = 0.05,
    use_historical = True,
    save_positions = True,
    random_seed = 19937,
)

print("\n\n------------\nFinished!!!\n------------\n\n")
np.save("positions_best.npy", positions)

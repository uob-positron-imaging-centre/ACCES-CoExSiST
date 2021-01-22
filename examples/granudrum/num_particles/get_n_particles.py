#! /usr/bin/python3

"""
Programm to get the number of particles from a rotating drum image


"""
import cv2
import numpy as np
from numpy.polynomial import Polynomial

import plotly.graph_objects as go

def particlenumber_from_image(
    image,
    radius_drum,
    depth_drum,
    radius_particle,
    particle_fraction = 0.7
):
    img = cv2.imread(image)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    threshold = 30
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
	    circles = np.round(circles[0, :]).astype("int")
	    # loop over the (x, y) coordinates and radius of the circles
	    for (x, y, r) in circles:
		    # draw the circle in the output image, then draw a rectangle
		    # corresponding to the center of the circle
		    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		    #print(x,y,r)
        # show the output image
	    gray[gray > threshold] = 255
	    gray[gray < threshold] = 0
	    cv2.imwrite("output.png", gray)#np.hstack([gray, output]))

    volume_particles = (np.sum(gray)/255)  #in pixel**2

    volume_drum = 2 * np.pi * r**2 # in pixel**2

    #print(volume_particles/volume_drum)

    Volume_drum_real = 2 * np.pi * radius_drum**2 * depth_drum
    print("volume of drum = ", Volume_drum_real)
    real_particle_volume = Volume_drum_real *  volume_particles/volume_drum
    print("volume of particles in drum = ", real_particle_volume)
    single_particle =  4/3 * np.pi* radius_particle **3

    n_particles = int( particle_fraction * real_particle_volume / single_particle)

    return n_particles


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def surface_polynom_from_image(image):
    threshold =50
    img = cv2.imread(image)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray < threshold] = 0
    gray[gray > threshold] = 255
    gray = 255-gray
    non_zero = first_nonzero(gray, 0, -1)
    idx =np.vstack((np.arange(0,len(non_zero),dtype = int), non_zero.astype(int))).T
    gray[:] = 0
    for x,y in idx:
        if y == -1:
            continue
        gray[y][x]= 255
    cv2.imwrite("output2.png", gray)
    return extract_surface(gray.T,1)







def extract_surface(occupancy, cell_length):
    '''Extract the surface from a 2D occupancy plot (which is given as cells)
    and return a fitted 3rd order polynomial with the zeroth order coefficient
    set to 0.

    Parameters
    ----------
    occupancy: (M, N) numpy.ndarray
        Input occupancy plot - a 2D numpy array where 0 represents an empty
        cell, and a 1 represents the surface.

    cell_length: float
        The length of a single cell in the x- and y-dimensions.

    Returns
    -------
    numpy.polynomial.Polynomial
        The fitted 3rd order Polynomial, **with the 0th order coefficient set
        to zero**.
    '''

    # Extract the coordinates of the non-zero elements in `occupancy` and
    # multiply them by the cell length to get x and y points
    occupancy = np.flip(occupancy,axis=1)
    surface = np.argwhere(occupancy > 0.5)
    surface = surface * cell_length

    # Sort the points by the x coordinates, then y coordinates
    surface = surface[
        np.lexsort(
            (surface[:, 1], surface[:, 0])
        )
    ]


    fig = go.Figure()
    fig.add_trace(go.Scatter( x= surface[:,0],y =surface[:,1]))


    # Take the highest point on the surface and clip its sides
    max_idx = surface[:, 1].argmax()

    surface_clipped = np.delete(surface, np.s_[:max_idx], axis = 0)
    surface_clipped = np.delete(surface_clipped, np.s_[-max_idx:], axis = 0)
    #print(surface_clipped)
    # Fit a 3rd order polynomial to the clipped surface
    surface_poly = Polynomial.fit(
        surface_clipped[:, 0],
        surface_clipped[:, 1],
        3
    )

    # Set the zeroth-order coefficient to 0 to "lower" the surface
    #surface_poly.coef[0] = 0


    plot_polynom(surface_poly, fig = fig)
    return surface_poly, surface




def plot_polynom( surface_poly, surface = None, fig=None, plot = True):
    if fig is  None:
        fig = go.Figure()
    x = np.linspace(surface_poly.domain[0], surface_poly.domain[1], 1000)
    fig.add_trace(go.Scatter(x = x, y = surface_poly(x)))
    if surface is not None:
        fig.add_trace(go.Scatter(x = surface[:, 0], y = surface[:, 1]))
    if plot:
        fig.show()
    return fig



if __name__ == "__main__":
    n = particlenumber_from_image(
        image = "mcc-30rpm.jpg",
        radius_drum= 42e-3,
        depth_drum = 20.1e-3,
        radius_particle = 1.2e-3 / 2,
        particle_fraction = 0.3
    )
    print("Number of particles required = ", n)

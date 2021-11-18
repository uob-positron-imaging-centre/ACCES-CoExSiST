'''ACCESS Example User Simulation Script

Must define one simulation whose parameters will be optimised this way:

    1. Use a variable called "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       You can set the initial guess here.

    2. The `parameters` creation should be fully self-contained between
       `#### ACCESS PARAMETERS START` and `#### ACCESS PARAMETERS END`
       blocks (i.e. it should not depend on code ran before that).

    3. By the end of the simulation script, define two variables:
        a. `error` - one number representing this simulation's error value.
        b. `extra` - (optional) any python data structure storing extra
                     information you want to save for a simulation run (e.g.
                     particle positions).

Importantly, use `parameters.at[<free parameter name>, "value"]` to get this
simulation's free / optimisable variable values.
'''

# Either run the actual GranuDrum simulation (takes ~40 minutes) or extract
# pre-computed example data and instantly show error value and plots
run_simulation = False      # Run simulation (True) or use example data (False)
save_data = False           # Save particle positions, radii and timestamps
show_plots = True           # Show plots of simulated & experimental GranuDrum


#### ACCESS PARAMETERS START
from typing import Tuple

import numpy as np
import cv2

import pept
import coexist

parameters = coexist.create_parameters(
    variables = ["sliding", "rolling", "nparticles"],
    minimums = [0., 0., 15494],
    maximums = [2., 1., 36152],
    values = [0.4, 0.1, 30000],
)

access_id = 0                   # Unique ID for each ACCESS run
#### ACCESS PARAMETERS END

# Extract current free parameters' values
sliding = parameters.at["sliding", "value"]
rolling = parameters.at["rolling", "value"]
nparticles = parameters.at["nparticles", "value"]

# Create a new LIGGGHTS simulation script with the parameter values above; read
# in the simulation template and change the relevant lines
with open("granudrum_mcc.sim", "r") as f:
    sim_script = f.readlines()

sim_script[0] = f"log simulation_inputs/granudrum_mcc_{access_id:0>6}.log\n"

sim_script[16] = f"variable fricPP equal {sliding}\n"
sim_script[17] = f"variable fricPW equal {sliding}\n"
sim_script[18] = f"variable fricPSW equal {sliding}\n"

sim_script[21] = f"variable fricRollPP equal {rolling}\n"
sim_script[22] = f"variable fricRollPW equal {rolling}\n"
sim_script[23] = f"variable fricRollPSW equal {rolling}\n"

sim_script[9] = f"variable N equal {nparticles}\n"

# Save the simulation template with the changed free parameters
filepath = f"simulation_inputs/granudrum_mcc_{access_id:0>6}.sim"
with open(filepath, "w") as f:
    f.writelines(sim_script)

# Load simulation and run it for two GranuDrum rotations. Use the last quarter
# rotation to compute the error value
rpm = 45
rotations = 2
start_time = (rotations - 0.25) / (rpm / 60)
end_time = rotations / (rpm / 60)


if run_simulation:
    sim = coexist.LiggghtsSimulation(filepath, verbose = True)

    # Record times, radii and positions at 120 FPS
    checkpoints = np.arange(start_time, end_time, 1 / 120)

    times = []
    positions = []
    radii = []

    for t in checkpoints:
        sim.step_to_time(t)

        times.append(sim.time())
        radii.append(sim.radii())
        positions.append(sim.positions())

    times = np.hstack(times)
    radii = np.hstack(radii)
    positions = np.vstack(positions)

    if save_data:
        np.save(f"example_positions/time_{access_id}.npy", times)
        np.save(f"example_positions/radii_{access_id}.npy", radii)
        np.save(f"example_positions/positions_{access_id}.npy", positions)
else:
    # Load example simulated data
    times = np.load(f"example_positions/time_{access_id}.npy")
    radii = np.load(f"example_positions/radii_{access_id}.npy")
    positions = np.load(f"example_positions/positions_{access_id}.npy")


# Filter NaNs from missing atoms
cond = (~np.isnan(radii))
radii = radii[cond]
positions = positions[cond]


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


def image_thresh(
    granudrum: GranuDrum,
    image_path: str,
    trim: float = 0.7,
) -> pept.Pixels:
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
    image[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((11, 11), np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Global thresholding + binarisation
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    img = pept.Pixels(img, xlim = granudrum.xlim, ylim = granudrum.ylim)

    return image, img


def simulation_thresh(
    granudrum: GranuDrum,
    image_shape: Tuple[int, int],
    positions: np.ndarray,
    particle_radii: np.ndarray,
    trim: float = 0.7,
) -> pept.Pixels:
    '''Return the thresholded occupancy grid of the GranuDrum DEM simulation in
    the `pept.Pixels` format.
    '''

    # Compute occupancy grid from the XZ plane (i.e. granular drum side)
    sim_occupancy = pept.processing.circles2d(
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
    sim_occupancy[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    # Colour GranuDrum's background
    sim_occupancy[
        (xx ** 2 + yy ** 2 < trim * granudrum.radius ** 2) &
        (sim_occupancy == 0.)
    ] = 7 / 255 * float(sim_occupancy.max())

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((11, 11), np.uint8)
    sim = cv2.morphologyEx(
        encode_u8(sim_occupancy),
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Global thresholding + binarisation
    _, sim = cv2.threshold(sim, 30, 255, cv2.THRESH_BINARY)
    sim[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] = 0

    sim = pept.Pixels(sim, xlim = granudrum.xlim, ylim = granudrum.ylim)

    return sim_occupancy, sim


# Error function for a GranuTools GranuDrum, quantifying the difference
# between an experimental free surface shape (from an image) and a simulated
# one (from an occupancy grid).
#
# The error value represents the difference in the XZ-projected area of the
# granular drum (i.e. the side view) between the GranuDrum image and the
# occupancy grid of the corresponding simulation.
image_path = "granudrum_45rpm_mcc.bmp"

# Rotating drum system dimensions
granudrum = GranuDrum()

# The thresholded GranuDrum image and occupancy plot, as `pept.Pixels`,
# containing only 0s and 1s
trim = 0.6
img_raw, img_pixels = image_thresh(granudrum, image_path, trim)
sim_raw, sim_pixels = simulation_thresh(granudrum, img_pixels.shape,
                                        positions, radii, trim)

# Pixel physical dimensions, in mm
dx = (granudrum.xlim[1] - granudrum.xlim[0]) / img_pixels.shape[0] * 1000
dy = (granudrum.ylim[1] - granudrum.ylim[0]) / img_pixels.shape[1] * 1000

# The error is the total different area, i.e. the number of pixels with
# different values times the area of a pixel
error = float((img_pixels != sim_pixels).sum()) * dx * dy
extra = dict(radii = radii, positions = positions)


# Plot the simulated and imaged GranuDrums and the difference between them
if show_plots:
    # Create colorscale starting from white
    import plotly
    cm = plotly.colors.sequential.Blues
    cm[0] = 'rgb(255,255,255)'

    grapher = pept.plots.PlotlyGrapher2D(cols = 3)

    # Plot "raw", untrimmed images
    img_raw, img_pixels = image_thresh(granudrum, image_path, trim = 1.)
    sim_raw, sim_pixels = simulation_thresh(
        granudrum, img_pixels.shape, positions, radii, trim = 1.
    )

    grapher.add_pixels(img_raw, colorscale = cm)

    # Plot both simulation and experiment, colour-coding differences in the middle
    diff = pept.Pixels.empty(img_pixels.shape, img_pixels.xlim, img_pixels.ylim)
    diff[(img_pixels == 255) & (sim_pixels == 255)] = 64
    diff[(img_pixels == 255) & (sim_pixels == 0)] = 128
    diff[(img_pixels == 0) & (sim_pixels == 255)] = 192

    # "Whiten" / blur the areas not used
    xgrid = np.linspace(granudrum.xlim[0], granudrum.xlim[1], img_pixels.shape[0])
    ygrid = np.linspace(granudrum.ylim[0], granudrum.ylim[1], img_pixels.shape[1])
    xx, yy = np.meshgrid(xgrid, ygrid)
    diff[xx ** 2 + yy ** 2 > trim * granudrum.radius ** 2] *= 0.2

    grapher.add_pixels(diff, colorscale = cm, col = 2)

    # Plot "raw", untrimmed simulation on the right
    grapher.add_pixels(sim_raw, colorscale = cm, col = 3)

    grapher.show()

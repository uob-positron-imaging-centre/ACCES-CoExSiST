# GranuDrum ACCESS run - MCC particles, coarse grained, 10 / 30 / 60 RPM

DEM free parameters:

- "N" - the number of particles inserted in the system
- "fricPP" - particle-particle friction coefficient
- "fricPW" - particle-wall friction coefficient
- "fricPSW" - particle-side wall friction coefficient
- "fricRollPP" - particle-particle rolling friction coefficient
- "fricRollPW" - particle-wall rolling friction coefficient
- "fricRollPSW" - particle-side wall rolling friction coefficient
- "corPP" - particle-particle restitution coefficient
- "corPW" - particle-wall restitution coefficient
- "corPSW" - particle-side wall restitution coefficient

These parameters are optimised against GranuDrum (TM) images generously shared with us by [GranuTools](https://www.granutools.com/en/). They can be found in the `10rpm_avg.jpg`, `30rpm_avg.jpg` and `60rpm_avg.jpg` files.
The error function can be found in the `optimise_access.py` file; it corresponds to the total area between the experimental free surface (from an OpenCV-processed GranuDrum (TM) image) and the simulation free surface (from an occupancy grid computed from the simulated particle locations at 100 timesteps).

Simulation settings:

- Three GranuDrum (TM) simulations at 10, 30 and 60 RPM for 3 rotations.
- Last rotation is used to compute the occupancy grid for each simulation.
- Variable number of MCC particles with 5 particle fractions between 2.6 mm and 3.4 mm, split according to a log-normal distribution.

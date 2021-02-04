# GranuDrum ACCESS run - MCC particles, coarse grained, 30 RPM

DEM free parameters:

- "N" - the number of particles inserted in the system
- "fricPP" - particle-particle friction coefficient
- "fricPW" - particle-wall friction coefficient
- "fricRollPP" - particle-particle rolling friction coefficient
- "fricRollPW" - particle-wall rolling friction coefficient

These parameters are optimised against a GranuDrum (TM) image generously shared with us by [GranuTools](https://www.granutools.com/en/). It can be found in the `mcc-30rpm.jpg` file.
The error function can be found in the `optimise_access.py` file; it corresponds to the total area between the experimental free surface (a 3rd order polynomial fitted against the OpenCV-processed GranuDrum (TM) image) and the simulation free surface (a 3rd order polynomial fitted against an occupancy grid computed from the simulated particle locations at 100 timesteps).

Simulation settings:

- GranuDrum (TM) simulated at 30 RPM for 3 rotations.
- Last rotation is used to compute the occupancy grid.
- Variable number of MCC particles with 5 particle fractions between 2.6 mm and 3.4 mm, split according to a log-normal distribution.

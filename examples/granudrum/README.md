# GranuDrum ACCESS run - MCC particles, non-coarse grained, 30 RPM

DEM free parameters:

- "cohPP" - particle-particle cohesion
- "corPP" - particle-particle coefficient of restitution
- "corPW" - particle-wall coefficient of restitution
- "fricPP" - particle-particle friction coefficient
- "fricPW" - particle-wall friction coefficient
- "fricRollPP" - particle-particle rolling friction coefficient
- "fricRollPW" - particle-wall rolling friction coefficient

These parameters are optimised against a GranuDrum (TM) image generously shared with us by [GranuTools](https://www.granutools.com/en/). It can be found in the `mcc-30rpm.jpg` file.
The error function can be found in the `optimise_access.py` file; it corresponds to the total area between the experimental free surface (a 3rd order polynomial fitted against the OpenCV-processed GranuDrum (TM) image) and the simulation free surface (a 3rd order polynomial fitted against an occupancy grid computed from the simulated particle locations at 100 timesteps).

Simulation settings:

- GranuDrum (TM) simulated at 30 RPM for 3 rotations.
- Last rotation is used to compute the occupancy grid.
- 37,000 MCC particles with 1.2 mm diameter.
- No coarse graining.

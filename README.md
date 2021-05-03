# CoExSiST & ACCESS
### Coupling Experimental Granular Data with DEM Simulations
#### *With automatic parallelisation on supercomputing clusters*

Autonomously *learn* a given set of free parameters, such that an experiment is *synchronised* with an equivalent simulation. This synchronisation can be done in one of two ways:

- **Microscopically**: in a Discrete Element Method (DEM) context, all simulated particles follow their experimental counterparts *exactly*. Naturally, this technique is limited to dilute systems and experimental imaging techniques that can capture the 3D position of *all* moving particles (e.g. PIV) - however, it provides information about the fundamental aspects of particle collision.
- **Macroscopically**: a given simulation reproduces a system-specific *macroscopic* quantity (e.g. residence time distribution, angle of repose). This technique is completely agnostic to the simulation method and the quantity to be reproduced. In a DEM context, it can train *coarse-grained* simulations, using larger meso-particles to model multiple smaller ones.

The two corresponding algorithms are:
- **CoExSiST**: Coupled Experimental-Simulational Study Tool.
- **ACCESS**: Autonomous Characterisation and Calibration via Evolutionary Simulation. 

ACCESS was implemented in the `coexist.AccessScript` class, providing an interface that is easy to use, but powerful enough to automatically parallelise arbitrary Python scripts automatically through code inspection and metaprogramming. It was used successfully from laptop-scale shared-memory machines to multi-node supercomputing clusters.


## Installation

Before the package is published to PyPI, you can install it directly from this GitHub repository: 

```
pip install git+https://github.com/uob-positron-imaging-centre/Coexist
```

Alternatively, you can download all the code and run `pip install .` inside its directory:

```
git clone https://github.com/uob-positron-imaging-centre/Coexist
cd Coexist
pip install .
```

If you would like to modify the source code and see your changes without reinstalling the package, use the `-e` flag for a *development installation*:

```
pip install -e .
```


## Examples

Check out the `examples` folder for example scripts using `coexist.Coexist` and `coexist.AccessScript`.


### Requirements
A shared library from LIGGGHTS is required to run CoExSiSt. To ensure compability and 
avoid any bugs it is recommended to use the forked LIGGGHTS repository:
https://github.com/D-werner-bham/LIGGGHTS-PUBLIC.git


## License and Commercial Integration
This library - in its general, domain-agnostic form - is free and open-source, published under the GNU General Public License v3.0.

If you are a company and would like to integrate ACCESS into your work - e.g. ACCESS-enabled equipment or general simulation calibration - please send an email to `a.l.nicusan@bham.ac.uk` to discuss commercial development of specific tools for your application. Relicensing for a closed-source / commercial project can be considered on an individual basis.
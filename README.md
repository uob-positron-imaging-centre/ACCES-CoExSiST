# CoExSiST & ACCES
### Coupling Experimental Granular Data with DEM Simulations

A Python library for autonomously learning simulation parameters from experimental data, from the *micro* to the *macro*, from laptops to clusters. This is done using either of two closely related frameworks:

- **CoExSiST**: Coupled Experimental-Simulational Study Tool.
- **ACCES**: Autonomous Characterisation and Calibration via Evolutionary Simulation. 

Both libraries learn a given set of free parameters, such that an experiment is synchronised with an equivalent simulation; this synchronisation is done in one of two ways:

- CoExSiST calibrates **microscopically**: in a Discrete Element Method (DEM) context, all simulated particles follow their experimental counterparts *exactly*. Naturally, this technique is limited to dilute systems and experimental imaging techniques that can capture the 3D position of *all* moving particles (e.g. PIV) - however, it provides information about the fundamental aspects of particle collision.
- ACCES calibrates / optimises **macroscopically**: a given simulation reproduces a system-specific *macroscopic* quantity (e.g. residence time distribution, angle of repose). This technique is completely agnostic to the simulation method and the quantity to be reproduced. In a DEM context, it can train *coarse-grained* simulations, using larger meso-particles to model multiple smaller ones.


ACCES is ready for production use; it was successfully used to calibrate coarse-grained DEM digital twins of [GranuTools](https://www.granutools.com/en/) equipment (Andrei Leonard Nicusan and Dominik Werner, *paper under review*), CFDEM fluidised beds (Hanqiao Cha, *paper under review*) and even signal processing parameters in a PET scanner model (Matthew Herald, *paper under review*).


![Calibrated GranuDrum](/docs/source/_static/calibrated.png?raw=true "Calibrated GranuDrum.")
*Example of an ACCES-calibrated DEM Digital Twin of a GranuTools GranuDrum; the calibration was done effectively against a single experimental data point - a photograph of the free surface shape yielded by MCC particles (left panel). The occupancy grid of a LIGGGHTS simulation was optimised against the free surface shape (middle panel).*


ACCES was implemented in the `coexist.Access` class, providing an interface that is easy to use, but powerful enough to **automatically parallelise arbitrary Python scripts** through code inspection and metaprogramming. It was used successfully from laptop-scale shared-memory machines to multi-node supercomputing clusters.





## Getting Started

These instructions will help you get started with Coexist. This is a pure Python package that does not require any extra system configuration, supporting Python 3.6 and above (though it might work with even older versions).

Before the package is published to PyPI, you can install it directly from this GitHub repository: 

```
pip install git+https://github.com/uob-positron-imaging-centre/Coexist
```




### Examples

The [documentation](https://coexist.readthedocs.io/) website contains an ACCES [tutorial](https://coexist.readthedocs.io/en/latest/tutorials/index.html) with explained code and output figures produced by `coexist`; all public functionality is fully documented in the [manual](https://coexist.readthedocs.io/en/latest/manual/index.html).

Want something more hands on? Check out the `examples` folder for example scripts using `coexist.Coexist` and `coexist.Access`; `examples/access_simple` is a very simple, hackable example script (remember that the `simulation_script.py` can execute *anything*), while `examples/access_granudrum` contains a more involved LIGGGHTS digital twin of a GranuTools GranuDrum, calibrated against an experimental image.




## Contributing

This library aims to be the state-of-the-art for simulation calibration, developed in the open using modern, collaborative coding approaches - no dragons shall be dwelling in the codebase. You are more than welcome to contribute to this library in the form of code improvements, documentation or helpful examples; please submit them either as:

- GitHub issues.
- Pull requests.
- Email me at <a.l.nicusan@bham.ac.uk>.

We are more than happy to discuss the library architecture and calibration / optimisation approach with any potential contributors and user.




## Acknowledgements and Funding

The authors gratefully acknowledge funding from the EPSRC Future Manufacturing Hub in Manufacture using Advanced Powder Processes, grant number 944885 and the University of Birmingham's BlueBEAR supercomputing service which was used extensively while developing the ACCES algorithm.

[TODO: list other funding & support we received]




## Citing

If you use this library in your research, you are kindly asked to cite:

> [Paper after publication]

Until the ACCES paper is published, you may cite this repository:

> Nicusan AL, Werner D, Seville JPK, Windows-Yule CR. ACCES: Autonomous Characterisation and Calibration via Evolutionary Simulation. GitHub repository. 2021 December 1.




## License and Commercial Integration

This library - in its general, domain-agnostic form - is free and open-source, published under the GNU General Public License v3.0.

If you are a company and would like to integrate ACCESS into your work - e.g. ACCESS-enabled equipment or general simulation calibration - please send an email to `a.l.nicusan@bham.ac.uk` to discuss commercial development of specific tools for your application. Relicensing for a closed-source / commercial project can be considered on an individual basis.


Copyright (C) 2020-2021 the Coexist developers. Until now, this library was built directly or indirectly through the brain-time of:

- Andrei Leonard Nicusan (University of Birmingham)
- Dominik Werner (University of Birmingham)
- Dr. Kit Windows-Yule (University of Birmingham)
- Prof. Jonathan Seville (University of Birmingham)

Thank you.
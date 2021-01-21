# CoExSiST & ACCESS
### Coupling experimental granular data with DEM simulations

Learn a given set of Discrete Element Method free parameters, such that an experiment is synchronised with an equivalent simulation.

- **CoExSiST**: Coupled Experimental-Simulational Study Tool
- **ACCESS**: Autonomous Characterisation and Calibration via Evolutionary Simulation 



## Installation
In the same directory as `setup.py`, run:

```
pip install -e .
```

The `-e` flag means "development installation", so that any changes you make to
the files in `coexist` will be immediately available to your code making use of
it, without needing to reinstall the package.

### Requirements
A shared library from LIGGGHTS is required to run CoExiSt. To ensure compability and 
avoid any bugs it is recommended to use the forked LIGGGHTS repository:
https://github.com/D-werner-bham/LIGGGHTS-PUBLIC.git
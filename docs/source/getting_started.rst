***************
Getting Started
***************
These instructions will help you get started with Coexist. This is a pure Python
package that does not require any extra system configuration.


Prerequisites
-------------
This package supports Python 3.6 and above (though it might work with even older
versions).


Installation
------------
Before the package is published to PyPI, you can install it directly from this GitHub
repository: 

```
pip install git+https://github.com/uob-positron-imaging-centre/Coexist
```

Alternatively, you can download all the code and run `pip install .` inside its
directory:

```
git clone https://github.com/uob-positron-imaging-centre/Coexist
cd Coexist
pip install .
```

If you would like to modify the source code and see your changes without reinstalling
the package, use the `-e` flag for a *development installation*:

```
pip install -e .
```


Optional Dependencies
---------------------
The ``coexist`` library can offer some extra functionality if optional dependencies
are found:

- **SymPy**: for the ``coexist.ballistics`` subpackages for analytically inferring the
  equations of motion of a particle travelling without collisions.
- **liggghts**: for the ``coexist.LiggghtsSimulation`` high-level interface to the
  LIGGGHTS DEM engine.


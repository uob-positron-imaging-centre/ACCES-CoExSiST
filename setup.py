#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.09.2020


import os
import sys
import shutil
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


def requirements(filename):
    # The dependencies are the same as the contents of requirements.txt
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]


# What packages are required for this module to be executed?
required = requirements("requirements.txt")
extras = {
    "docs": requirements("requirements_extra.txt")
}

# Load the package's __version__.py module as a dictionary.
here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, "coexist", "__version__.py")) as f:
    exec(f.read(), about)


class UploadCommand(setuptools.Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print(f'\033[1m{s}\033[0m')

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(f'{sys.executable} setup.py sdist bdist_wheel --universal')

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system(f'git tag v{about["__version__"]}')
        os.system('git push --tags')

        sys.exit()


setuptools.setup(
    name = "coexist",
    version = about["__version__"],
    author = (
        "Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>, "
        "Dominik Werner <d.werner.1@bham.ac.uk>, "
        "Jack Sykes <jas653@student.bham.ac.uk>"
    ),

    keywords = (
        "simulation optimization calibration parameter-estimation "
        "parameter-tuning physics-simulation computational-fluid-dynamics "
        "granular DEM discrete-element-method medical-imaging"
    ),
    description = (
        "Learning simulation parameters from experimental data, from the "
        "micro to the macro, from the laptop to the cluster."
    ),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/uob-positron-imaging-centre/Coexist",

    install_requires = required,
    extras_require = extras,
    include_package_data = True,
    packages = setuptools.find_packages(),

    license = 'GNU',
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: GNU General Public License v3 or later " +
        "(GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Clustering",
    ],

    python_requires = '>=3.6',

    # $ setup.py publish support.
    cmdclass = {
        'upload': UploadCommand,
    },
)

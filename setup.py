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
required = requirements('requirements.txt')

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
        print('\033[1m{0}\033[0m'.format(s))

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
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setuptools.setup(
    name = "coexist",
    version = "0.1.0",
    author = (
        "Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>, "
        "Dominik Werner <d.werner.1@bham.ac.uk>"
    ),

    keywords = "simulation optimisation DEM",
    description = "Coupling experimental granular data with DEM simulations",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/D-werner-bham/pyLiggghts",

    install_requires = required,
    include_package_data = True,
    packages = setuptools.find_packages(),

    license = 'GNU',
    classifiers = [
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later " +
        "(GPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    python_requires = '>=3.6',
    # $ setup.py publish support.
    cmdclass = {
        'upload': UploadCommand,
    },
)

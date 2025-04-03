# Pyrobopath
[![Python package](https://github.com/alexarbogast/pyrobopath/actions/workflows/build.yml/badge.svg)](https://github.com/alexarbogast/pyrobopath/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/pyrobopath/badge/?version=latest)](https://pyrobopath.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrobopath?logo=python&logoColor=white)


PyRoboPath is a Python package for working with robotic toolpaths. The
target functionalities include:
* G-code interpretation and Python interfaces for working with G-code toolpaths
  in Python - including tools for path modification and visualization
* Path smoothing and trajectory parameterization
* Collision checking with
  [python-fcl](https://github.com/BerkeleyAutomation/python-fcl) and custom
  interfaces
* A scheduling library providing interfaces for robotic toolpath scheduling
* Planning algorithms for scheduling multi-agent toolpaths 

> [!NOTE]
> _This project is under heavy development and subject to changes in API and
> functionality._

## Installation

Install pyrobopath from the PyPI package index.

```sh
pip install pyrobopath 
```
To install the python package locally from source, clone the repository and
install with pip.

```sh
git clone git@github.com:alexarbogast/pyrobopath.git
cd pyrobopath
pip install -e .
```

If you would like to run the tests or build the docs locally.

```sh
pip install -e .[dev,docs]
```

To verify the installation, run tests with pytest or unittest

```sh
pytest # pytest
python3 -m unittest # unittest
```

## ROS interfaces
See the [pyrobopath_ros](https://github.com/alexarbogast/pyrobopath_ros) package
for ROS interfaces to pyrobopath.



## Documentation
Checkout the [Pyrobopath
Documentation](https://pyrobopath.readthedocs.io/en/latest/) for installation
help, examples, and API reference. 


## Usage
See [examples.py](./examples/examples.py) for python usage demos.

# PyRoboPath
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

This project contains python and ROS packages for working with robotic toolpaths. The target functionalities include:
* G-code interpretation and Python interfaces for working with G-code toolpaths in Python - including manipulation and visualization tools
* Collision checking with [python-fcl](https://github.com/BerkeleyAutomation/python-fcl) and custom interfaces
* Scheduling tools for collision-free multi-agent robotic toolpaths

_This project is under heavy development and subject to changes in API and functionality._ 

## Dependencies
- [NetworkX](https://networkx.org/)
- [GcodeParser](https://github.com/AndyEveritt/GcodeParser)
- [python-fcl](https://github.com/BerkeleyAutomation/python-fcl/releases)

## Usage
See [examples.py](./examples/examples.py) for usage demos.

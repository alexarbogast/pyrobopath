===============
Python Examples
===============

Creating a toolpath
-------------------

A toolpath is created from a collection of :class:`.Contour` s.
Contours represent a contiguous path that is traversed by a specified tool.
The toolpath can either be created manually, or parsed from a standard
Gcode file.

The `tool` representation is up to the user, but a good choice is to define each
tool with an enum.

.. code-block:: python
  :caption: Example tool represenation

  from enum import Enum

  class Materials(Enum):
      MATERIAL_A = 1
      MATERIAL_B = 2

Manual Toolpath Creation
^^^^^^^^^^^^^^^^^^^^^^^^
To create a toolpath manually, we must first define a set of contours.

.. code-block:: python

  from pyrobopath.toolpath import Contour, Toolpath

  path1 = [np.array(1.0, 0.0, 0.0), np.array(0.0, 0.0, 0.0), np.array(0.0, 1.0, 0.0)]
  path2 = [np.array(1.0, 0.0, 0.0), np.array(0.0, 0.0, 0.0), np.array(0.0, 1.0, 0.0)]
  path3 = [np.array(1.0, 0.0, 0.0), np.array(0.0, 0.0, 0.0), np.array(0.0, 1.0, 0.0)]

  contour1 = Contour(path=path1, tool=Materials.MATERIAL_A)
  contour2 = Contour(path=path2, tool=Materials.MATERIAL_A)
  contour3 = Contour(path=path2, tool=Materials.MATERIAL_A)

  toolpath = Toolpath()
  toolpath.contours = [contour1, contour2, contour3]

Gcode Toolpath Creation
^^^^^^^^^^^^^^^^^^^^^^^
A toolpath can be created from standard Gcode flavors.

.. Caution::
  Only the reprap flavor from slic3r has been tested thus far. It should be
  relatively simple to write a parser for other representations.

The `gcodeparser <https://pypi.org/project/gcodeparser/>`_ package is used to
read a Gcode file to a python representation.

.. code-block:: python

  from gcodeparser import GCodeParser

  filepath = "<path to gcode>"
  with open(filepath, "r") as f:
      gcode = f.read()
  parsed_gcode = GCodeParser(gcode)


Then, the parsed Gcode is transformed to a pyrobopath :class:`.Toolpath`

.. code-block:: python

  from pyrobopath.toolpath import *

  toolpath = Toolpath.from_gcode(parsed_gcode.lines)


Creating a Multi-robot System
-----------------------------

A toolpath planner requires a system definition that defines the robot base
frame position, home position, collision model, and others. This system
definition is provided as a dictionary with the keys as agent IDs and the
values as :class:`.AgentModel`.

We will create a simple two robot system.

.. code-block:: python

  from pyrobopath.collision_detection import FCLRobotBBCollisionModel
  from pyrobopath.toolpath_scheduling import *

  bf1 = np.array([-350.0, 0.0, 0.0])
  bf2 = np.array([350.0, 0.0, 0.0])

  # create agent collision models
  agent1 = AgentModel(
      base_frame_position=bf1,
      home_position=np.array([-250.0, 0.0, 0.0]),
      capabilities=[Materials.MATERIAL_A],
      velocity=50.0,
      travel_velocity=50.0,
      collision_model=FCLRobotBBCollisionModel(200.0, 50.0, 300.0, bf1),
  )
  agent2 = AgentModel(
      base_frame_position=bf2,
      home_position=np.array([250.0, 0.0, 0.0]),
      capabilities=[Materials.MATERIAL_B],
      velocity=50.0,
      travel_velocity=50.0,
      collision_model=FCLRobotBBCollisionModel(200.0, 50.0, 300.0, bf2),
  )
  agent_models = {"robot1": agent1, "robot2": agent2}


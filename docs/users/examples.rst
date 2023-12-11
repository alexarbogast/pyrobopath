
Examples
========

Creating a toolpath
-------------------

A toolpath is created from a collection of :py:class:`Contour` s. 
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

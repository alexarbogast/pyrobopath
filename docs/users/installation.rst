.. _installation:

Installation
============

Pyrobopath can be installed as a standalone Python package. A ROS package is
available if you plan to use ROS to execute Pyrobopath schedules.

Step 1: Installing the Python Package
-------------------------------------

The Python package can be installed locally or via the PyPI package index.

Using pip
^^^^^^^^^
Install pyrobopath from the PyPI package index.

.. code-block:: sh

    pip install pyrobopath

Local installation
^^^^^^^^^^^^^^^^^^

To install the python package locally from source, clone the repository and
install with pip.

.. code-block:: sh

    git clone git@github.com:alexarbogast/pyrobopath.git
    cd pyrobopath
    pip install -e .

If you would like to run the tests or build the docs locally.

.. code-block:: sh

    pip install -e .[dev,docs]

To verify the installation, run tests with pytest or unittest

.. code-block:: sh

    pytest # pytest
    python3 -m unittest # unittest

If you plan to only use the Python package provided by Pyrobopath, you can stop
here and skip the following steps installing the ROS dependencies.


Step 2: Installing ROS Package
------------------------------

Create a catkin workspace

.. code-block:: sh

    mkdir -p pyrobopath_ws/src && cd pyrobopath_ws/src

The Pyrobopath ROS interface depends on the `cartesian_planning
<https://github.com/alexarbogast/cartesian_planning>`_ package for executing
toolpath schedules.

To use the package, clone the
https://github.com/alexarbogast/cartesian_planning and
https://github.com/alexarbogast/pyrobopath_ros repositories into your catkin
workspace and build the packages.

.. code-block:: sh

    git clone git@github.com:alexarbogast/cartesian_planning.git
    git clone git@github.com:alexarbogast/pyrobopath_ros.git
    cd ../
    catkin build


Step 3: Multi-robot system packages
-----------------------------------

Specific robot and multi-robot system layouts ( `urdf
<https://wiki.ros.org/urdf>`_, `ros_control
<https://github.com/ros-controls/ros_control>`_, and `parameter files
<https://wiki.ros.org/rosparam>`_) are required by `cartesian_planning
<https://github.com/alexarbogast/cartesian_planning>`_ and pyrobopath _ros.

.. note:: Pyrobopath is configured to work with system configurations similar
   to the one pictured below.

.. figure:: https://user-images.githubusercontent.com/46149643/221465891-7995e74a-185d-49c6-80c6-7d63d122b182.png

    hydra_ros

See the examples below for reference when creating a system:

- `cartesian_planning examples <https://github.com/alexarbogast/cartesian_planning/tree/master/cartesian_planning_examples>`_
- `za_ros <https://github.com/alexarbogast/za_ros>`_
- `hydra_ros <https://github.com/alexarbogast/hydra_ros>`_

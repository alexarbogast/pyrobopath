Installation
============

Step 1: Install dependencies
----------------------------

Install the **Python** dependencies

- `NetworkX <https://networkx.org/>`_
- `GcodeParser <https://github.com/AndyEveritt/GcodeParser>`_
- `python-fcl <https://github.com/BerkeleyAutomation/python-fcl>`_
- `Quaternion <https://pypi.org/project/Quaternion>`_

.. code-block:: sh

    $ pip install 
    $ git clone git@github.com:alexarbogast/pyrobopath.git

Install the **ROS** dependencies

- `cartesian_planning <https://github.com/alexarbogast/cartesian_planning>`_

Step 2: Build package
---------------------

Create a catkin workspace and clone the repository

.. code-block:: sh

    $ mkdir pyrobopath_ws/src && cd pyrobopath_ws/src 
    $ git clone git@github.com:alexarbogast/pyrobopath.git

Build the workspace 

.. code-block:: sh

    $ cd ../ 
    $ catkin build

Step 3: Multi-robot system packages
-----------------------------------

If you plan to use pyrobopath with ROS, see the package layout of 
`hydra_ros <https://github.com/alexarbogast/hydra_ros>`_ for a reference
multi-robot system ROS configuration.

.. note:: Pyrobopath is configured to work with system configurations similar to the one pictured below.

.. figure:: https://user-images.githubusercontent.com/46149643/221465891-7995e74a-185d-49c6-80c6-7d63d122b182.png

    hydra_ros

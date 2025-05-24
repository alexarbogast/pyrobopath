.. toctree::
   :maxdepth: 2
   :hidden:


Demos
=====

Two-robot two-material schedule execution in ROS
------------------------------------------------

.. raw:: html

    <iframe width="560" height="315"
    src="https://www.youtube.com/embed/HoNouCSlHS8?si=i_YHTt4498vlDdZ_&amp;start=2"
    title="YouTube video player" frameborder="0" allow="accelerometer; autoplay;
    clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen></iframe>

Multi-robot Tic-Tac-Toe
-----------------------

This demo uses three ``capabilities`` to define the **game board lines, X's,
O's**. Both robots possess the game board capability while each robot has either
the **X** capability or the **O** capability. The
:class:`.MultiAgentToolpathPlanner` schedules the events in the game of
Tic-Tac-Toe. The first move is random, then each player uses the `Minimax
<https://en.wikipedia.org/wiki/Minimax>`_ algorithm to choose their next move.
The code repository for this demo is found at
https://github.com/alexarbogast/hydra_tic_tac_toe.

.. raw:: html

    <iframe width="560" height="315"
    src="https://www.youtube.com/embed/OdXw0MMllgA?si=k46w-m4XwTY7wKNI"
    title="YouTube video player" frameborder="0" allow="accelerometer; autoplay;
    clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen></iframe>

Pyrobopath ROS Schedule execution
---------------------------------

These demonstrations use a two-robot system with markers to demonstrate
heterogeneous task-allocation and schedule execution in ROS. See the
https://github.com/alexarbogast/pyrobopath_ros repository for further details.

.. raw:: html

    <iframe width="560" height="315"
    src="https://www.youtube.com/embed/PtlfFJrBAPE?si=QUwMj8qFvpRa7K3o"
    title="YouTube video player" frameborder="0" allow="accelerometer; autoplay;
    clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

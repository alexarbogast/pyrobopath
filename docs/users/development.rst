.. _development:

Development
===========

Testing
-------

Pyrobopath uses the `pytest <http://doc.pytest.org/en/latest/>`_ framework. The
tests are found in the :file:`test/` folder. Ensure that Pyrobopath has been
:ref:`installed <installation>` with the `[dev]` dependencies enabled.

In the root directory of the repository run one of the following commands.

.. code-block:: sh

   pytest # pytest
   python3 -m unittest # unittest

Building Documentation Locally
------------------------------

To build this documentation locally, first make sure that Pyrobopath has been
:ref:`installed <installation>` with the `[dev]` dependencies enabled. The
documentation is created with `Sphinx
<https://www.sphinx-doc.org/en/master/>`_.

Change directories to the :file:`doc/` folder and run the following commands to
build the documentation.

.. code-block:: sh

   make html

The built docs can be found in the :file:`_build/html` folder. Double-click
:file:`index.html` to show the docs in your default browser.

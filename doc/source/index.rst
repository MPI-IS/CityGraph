.. CityGraph documentation master file, created by
   sphinx-quickstart on Wed Apr 22 10:42:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CityGraph's documentation!
=====================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   pages/demos
   pages/city
   pages/planning
   pages/city_io
   pages/topolgy
   pages/utils

Introduction
------------

``CityGraph`` is a Python framework for representing a city (real or virtual) and moving around it.

Requirements
------------

The application only requires ``Python 3.5`` or higher.

Installation
------------

**CityGraph** releases can be instaled from `PyPI <https://pypi.org/>`_:

.. code::

    $ pip install city-graph

Alternatively, one can also clone the repository and install the package locally:

.. code::

    $ git clone https://github.com/MPI-IS/CityGraph.git
    $ cd CityGraph
    $ pip install .

.. note::

    We strongly advise to install the package in a dedicated virutal environment.

Tests
-----

To run the tests, simply do:

.. code::

    $ python -m unittest

Documentation
-------------

To build the ``Sphinx`` documentation:

.. code::

    $ pip install sphinx sphinx-bootstrap-theme
    $ cd doc
    $ make html

and open the file ``build/html/index.html`` in your web browser.

Authors
-------

`Jean-Claude Passy <mailto:jean-claude.passy@tuebingen.mpg.de>`_,
Software Workshop - Max Planck Institute for Intelligent Systems

`Ivan Oreshnikov <mailto:ivan.oreshnikov@tuebingen.mpg.de>`_,
Software Workshop - Max Planck Institute for Intelligent Systems

`Vincent Berenz <mailto:vberenz@tuebingen.mpg.de>`_,
Max Planck Institute for Intelligent Systems

License
-------

BSD-3-Clause.

The OSM import code contains a modified snippet of code from the project
`osmnx <https://github.com/gboeing/osmnx>`_
that is being distributed under the MIT License.

Copyright
---------
© 2020, Max Planck Society / Software Workshop - Max Planck Institute for Intelligent Systems

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

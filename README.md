[![Build status](https://raw.githubusercontent.com/MPI-IS-BambooAgent/sw_badges/master/badges/plans/citygraph/tag.svg?sanitize=true)](https://atlas.is.localnet/bamboo/browse/BAMEI-CIT7/latest/)

CityGraph
=========

**CityGraph** is a `Python` framework for representing a city (real or virtual) and moving around it.

Requirements
------------

The application only requires `Python 3.5` or higher.

Installation
------------

**CityGraph** releases can be instaled from `PyPI`:

```
$ pip install city-graph
```

Alternatively, one can also clone the repository and install the package locally:

```
$ git clone https://github.com/MPI-IS/CityGraph.git
$ cd CityGraph
$ pip install .
```

*NOTE*: We strongly advise to install the package in a dedicated virtual environment.

Tests
-----

To run the tests, simply do:

```
$ python -m unittest
```

Documentation
-------------

To build the `Sphinx` documentation:

```
$ cd doc
$ make html
```
and open the file `build/html/index.html` in your web browser.

Authors
-------

[Jean-Claude Passy](https://github.com/jcpassy)
(Software Workshop - Max Planck Institute for Intelligent Systems)

[Ivan Oreshnikov](https://github.com/ioreshnikov)
(Software Workshop - Max Planck Institute for Intelligent Systems)

[Vincent Berenz](https://github.com/vincentberenz)
(Max Planck Institute for Intelligent Systems)

License
-------

BSD-3-Clause (see LICENSE.md)

Copyright
---------
© 2020, Max Planck Society / Software Workshop - Max Planck Institute for Intelligent Systems

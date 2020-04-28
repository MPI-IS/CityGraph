Topology
========

.. note::

    This section is for developers mainly.

.. contents::
    :local:
    :depth: 1

Introduction
------------

A ``topology`` is a representation of a virtual or real area.
Its main attribute is a graph that is composed of nodes and edges between them.

Additionally, it must have methods to add nodes and edges, and calculate
the shortest path between two nodes.

The next section provides some examples on how to perform such actions.

Examples
--------

.. warning::

    The user should **NOT** manipulate the topology but instead use the ``City`` interface.

We first define some constants and create a topolgy with some nodes and edges:

.. code:: python

    from city_graph import topology

    # Mobility types
    WALK = 'walk'
    BUS = 'bus'

    # Mobility criteria
    TIME = 'time'
    DISTANCE = 'distance'

    # Create topology
    top = topology.MultiEdgeUndirectedTopology()

    # Add some nodes
    for n in 'city-graph':
        top.add_node(n)

    # Add walk edge through city
    top.add_edge('c', 'i', WALK, **{TIME: 0.1, DISTANCE: 1})
    top.add_edge('i', 't', WALK, **{TIME: 0.2, DISTANCE: 2})
    top.add_edge('t', 'y', WALK, **{TIME: 0.3, DISTANCE: 4})

We can get the shortest path between two nodes according to the ``TIME`` attribute:

.. code:: python

    # Get path by time
    score, path, data = top.get_shortest_path('c', 'y', TIME, [WALK], [TIME, DISTANCE])
    print('Score =', score)
    print('Path =', path)
    print('Data =', data)

.. code::

    # Output
    Score = 0.60
    Path = ['c', 'i', 't', 'y']
    Data = {'type': array(['walk', 'walk', 'walk'], dtype='<U4'), 'time': array([0.1, 0.2, 0.3]), 'distance': array([1, 2, 4])}

We can also get the reversed path by ``DISTANCE`` without extracting the ``TIME`` data:

.. code:: python

    # Get reversed path by distance and we do not want the time
    score, path, data = top.get_shortest_path('y', 'c', DISTANCE, [WALK], [DISTANCE])
    print('Score = {:1.2f}'.format(score))
    print('Path =', path)
    print('Data =', data)

.. code::

    # Output
    Score = 7.00
    Path = ['y', 't', 'i', 'c']
    Data = {'type': array(['walk', 'walk', 'walk'], dtype='<U4'), 'distance': array([4, 2, 1])}

So far, we used only edges of type ``WALK``. Let us add bus edges that make things faster
and calculate the optimal path:

.. code:: python

    # Add bus edge between that makes things much faster
    top.add_edge('t', 'h', BUS, **{TIME: 0.01, DISTANCE: 0.1})
    top.add_edge('h', 'y', BUS, **{TIME: 0.02, DISTANCE: 0.05})

    # Mixed path now
    score, path, data = top.get_shortest_path('c', 'y', TIME, [WALK, BUS], [TIME, DISTANCE])
    print('Score = {:1.2f}'.format(score))
    print('Path =', path)
    print('Data =', data)

.. code::

    # Output
    Score = 0.33
    Path = ['c', 'i', 't', 'h', 'y']
    Data = {'type': array(['walk', 'walk', 'bus', 'bus'], dtype='<U4'), 'time': array([0.1, 0.2 , 0.01, 0.02]), 'distance': array([1., 2., 0.1, 0.05])}

The edge data along the path are stored in ``numpy`` arrays, which makes it very convenient to calculate
relevant quantities:

.. code:: python

    # Calculate relevant information
    print('Previous path had a score = {:1.2f}'.format(score))
    print('Total path time = {:1.2f} ({:1.2f} in {}, {:1.2f} in {})'.format(
        sum(data[TIME]),
        sum(data[TIME][data['type'] == BUS]), BUS,
        sum(data[TIME][data['type'] == WALK]), WALK)
    )
    print('Total path distance = {:1.2f} ({:1.2f} in {}, {:1.2f} in {})'.format(
        sum(data[DISTANCE]),
        sum(data[DISTANCE][data['type'] == BUS]), BUS,
        sum(data[DISTANCE][data['type'] == WALK]), WALK)
    )

.. code::

    # Output
    Previous path had a score = 0.33
    Total path time = 0.33 (0.03 in bus, 0.30 in walk)
    Total path distance = 3.15 (0.15 in bus, 3.00 in walk)

In this case, the score is the criterion value (``TIME``) integrated over the path.
However, one can weight each edge type by a certain value to favor one type over another:

.. code:: python

    # Associate weights to edge types
    types_with_weights = {
        WALK: 0.75,
        BUS: 0.25
    }
    score, path, data = top.get_shortest_path('c', 'y', TIME, types_with_weights, [TIME, DISTANCE])
    print('Score = {:1.2f}'.format(score))
    print('Path =', path)
    print('Total path time = {:1.2f} ({:1.2f} in {}, {:1.2f} in {})'.format(
        sum(data[TIME]),
        sum(data[TIME][data['type'] == BUS]), BUS,
        sum(data[TIME][data['type'] == WALK]), WALK)
    )
    print('Total path distance = {:1.2f} ({:1.2f} in {}, {:1.2f} in {})'.format(
        sum(data[DISTANCE]),
        sum(data[DISTANCE][data['type'] == BUS]), BUS,
        sum(data[DISTANCE][data['type'] == WALK]), WALK)
    )

.. code::

    # Output
    Score = 0.23
    Path = ['c', 'i', 't', 'h', 'y']
    Total path time = 0.33 (0.03 in bus, 0.30 in walk)
    Total path distance = 3.15 (0.15 in bus, 3.00 in walk)

The path or *physical quantities* have not changed, but the score has.

Finally, an exception is raised if not path is found:

.. code:: python

    # Bus only does not work -> exception
    top.get_shortest_path('c', 'y', TIME, [BUS], [TIME, DISTANCE])

.. code::

    # Output
    ...
    ValueError: No path found with type {'bus': None} between c and y.

References
----------

.. automodule:: city_graph.topology
   :members:
   :member-order: bysource
   :exclude-members: __weakref__
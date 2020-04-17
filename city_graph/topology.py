"""
Topology
========

Module for building and operating on graphs.
"""
from collections import namedtuple

import networkx as nx

from .utils import RandomGenerator

# Temporary: default locations and distribution
# JC: This is temporary, I just need it for building the graph
# Location types
LOCATION_TYPE = namedtuple("LocationType", ['name'])
HOUSEHOLD = LOCATION_TYPE("household")
OFFICE = LOCATION_TYPE("office",)
SCHOOL = LOCATION_TYPE("school")
UNIVERSITY = LOCATION_TYPE("university")
SUPERMARKET = LOCATION_TYPE("supermarket")
GROCER = LOCATION_TYPE("grocer")
GYM = LOCATION_TYPE("gym")
PARK = LOCATION_TYPE("park")
DINER = LOCATION_TYPE("diner")
BAR = LOCATION_TYPE("bar")
# Default distribution
DEFAULT_LOCATION_TYPE_DISTRIBUTION = {
    HOUSEHOLD: 300,
    OFFICE: 150,
    SCHOOL: 3,
    UNIVERSITY: 2,
    SUPERMARKET: 5,
    GROCER: 20,
    GYM: 5,
    PARK: 5,
    DINER: 5,
    BAR: 5
}
# Location
LOCATION = namedtuple(
    'Location',
    ['name', 'type', 'x', 'y']
)

# Coordinates
COORDINATES = namedtuple(
    'Coordinates',
    ['x', 'y']
)

# How to calculate distances
# This should probably be in the location module


def compute_distance(_location1, _location2):
    """Computes the Euclidian distance in 2D."""
    return 1.0


class BaseTopology:
    """Abstract class setting up some requirements for a Topology."""

    def __init__(self):
        self.graph = nx.MultiGraph()

    def add_node(self, *args, **kwargs):
        """Method adding a node to the multigraph."""
        raise NotImplementedError

    def add_edge(self, *args, **kwargs):
        """Method adding an edge to the multigraph."""
        raise NotImplementedError


class ScaleFreeTopology(BaseTopology):
    """Class represneting a scale-free topology.

    :param obj rng: Random number generator.
    """

    # Node attribute for storing references to locations
    LOCATION_REFERENCES_NAME = 'locations'

    def __init__(self, rng=None):

        super().__init__()
        self.rng = rng or RandomGenerator()

    @property
    def num_of_nodes(self):
        """The number of nodes in the topology."""
        return self.graph.number_of_nodes()

    @property
    def num_of_edges(self):
        """The number of nodes in the topology."""
        return self.graph.number_of_edges()

    def add_node(self, x_coord, y_coord, location_id):
        """Method adding a node defined by its coordinates and updating the node attributes.

        :param float x_coord: X coordinate
        :param float y_coord: Y coordinate
        :param int location_id: Reference to a location associated to the node
        """

        coordinates = COORDINATES(x_coord, y_coord)
        try:
            # Node already exists
            previous_ids = self.graph.nodes[coordinates][self.LOCATION_REFERENCES_NAME]
            attrs = {
                coordinates: {self.LOCATION_REFERENCES_NAME: previous_ids + (location_id,)}
            }
            nx.set_node_attributes(self.graph, attrs)

        except KeyError:
            # Create node
            attrs = {
                self.LOCATION_REFERENCES_NAME: (location_id,)
            }
            self.graph.add_node(coordinates, **attrs)

    def add_edge(self, *args, **kwargs):
        pass

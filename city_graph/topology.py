"""
Topology
========

Module for building and operating on graphs.
"""
from functools import partial
from networkx import MultiGraph, single_source_dijkstra
from networkx.exception import NetworkXNoPath

from .utils import RandomGenerator

# Some names used for the edges attributes
EDGE_TYPE = 'type'
EDGE_WEIGHT = 'weight'


class BaseTopology:
    """Abstract class setting up some requirements for a Topology."""

    def __init__(self):
        self.graph = MultiGraph()

    def add_node(self, *args, **kwargs):
        """Method adding a node to the multigraph."""
        raise NotImplementedError

    def add_edge(self, *args, **kwargs):
        """Method adding an edge to the multigraph."""
        raise NotImplementedError


class MultiEdgeUndirectedTopology(BaseTopology):
    """Class representing a topology with mutltiple undirected.

    :param obj rng: Random number generator.
    """

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

    def add_node(self, node_label):
        """Add a node defined by its label.

        :param obj node_label: Node label. Must be hashable.
        """
        self.graph.add_node(node_label)

    def add_edge(self, node1, node2, edge_type, edge_weight):
        """Add an edge between two nodes in the graph.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param obj edge_type: Edge type.
        :param float edge_weight: Edge weight.
        """

        if not self.graph.has_node(node1):
            raise ValueError("First node %s does not exist." % node1)
        if not self.graph.has_node(node2):
            raise ValueError("Second node %s does not exist." % node2)
        edge_attrs = {
            EDGE_TYPE: edge_type,
            EDGE_WEIGHT: edge_weight
        }
        self.graph.add_edge(node1, node2, **edge_attrs)

    def get_edges(self, node1, node2):
        """Return all the edges between two nodes.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :returns: A dictionary where the keys are the edge number
            and the values a dictionary of the edges attributes
        :rtype: dict
        :raises: :py:class:`ValueError`: if there is no edge between the nodes
        """
        try:
            return dict(self.graph[node1][node2])
        except KeyError:
            raise ValueError('No edge between nodes %s and %s' % (node1, node2))

    def _get_min_weight(self, edge_types, allowed_edge_types, node1, node2, _edge_attrs):
        """Find the edge with the minimum weight between two nodes."""

        # Get all edges between the two nodes
        edges = self.get_edges(node1, node2)

        # Get all the weights but keep only those that are allowed
        weights = [(attrs[EDGE_TYPE], attrs[EDGE_WEIGHT]) for attrs in edges.values()
                   if attrs[EDGE_TYPE] in allowed_edge_types]

        # Case where there is no valid node
        if not weights:
            return None

        # Otherwise, return minimum weight and save type
        min_type, min_weight = min(weights, key=lambda t: t[1])
        # The algorithm uses the full (symmetric) matrix
        # instead of considering the upper triangular only.
        # This means that for two nodes u ≠ v, the weight is extracted twice:
        # once for (u,v), once for (v,u). We do not need to store both
        if (node2, node1) not in edge_types:
            edge_types[(node1, node2)] = min_type
        return min_weight

    def get_shortest_path(self, node1, node2, allowed_edge_types):
        """Find the shortest path between two nodes using specified edges.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param allowed_edge_types: Edge type(s) allowed to build path
        :type allowed_edge_types: str or itertable
        :returns: A 3-tuple containing in this order

            * the list of edges types along the path
            * the total weight of the path
            * the list of nodes along the path

        :rtype: tuple
        :raises: :py:class:`ValueError`: if no path has been found between the nodes
        """

        # Check that the nodes exist
        for node in node1, node2:
            if not self.graph.has_node(node):
                raise ValueError("Node %s does not exist." % node)

        # Transform allowed_edge_types into a set (hashable and unique elements)
        # Must use the curly brackets in case of a string because it is iterable
        if isinstance(allowed_edge_types, str):
            allowed_edge_types = {allowed_edge_types}
        else:
            allowed_edge_types = set(allowed_edge_types)

        # Using partial function to pass the edge types
        # The weight_func will be called for each edge on the graph
        # Even those which will not been part of the optimial path
        # So edge_types cannot be a simple list to which we append values
        # Instead it will be a dict where the key is a pair of nodes
        # We will extract the relevant values afterwards
        edge_types_dict = {}
        weight_func = partial(self._get_min_weight, edge_types_dict, allowed_edge_types)

        # Calculate path
        try:
            total_weight, path = single_source_dijkstra(
                self.graph, node1, node2, weight=weight_func)
        except NetworkXNoPath:
            raise ValueError("No path found between %s and %s." % (node1, node2))

        # Extract the edge types into a list
        edge_types = []
        for u, v in zip(path, path[1:]):
            # The key could be (u,v) or (v,u)
            try:
                edge_types.append(edge_types_dict[(u, v)])
            except KeyError:
                edge_types.append(edge_types_dict[(v, u)])

        return (edge_types, total_weight, path)

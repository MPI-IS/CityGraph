"""
Topology
========

Module for building and operating on graphs.
"""
import random

from functools import partial
from networkx import MultiGraph, single_source_dijkstra
from networkx.exception import NetworkXNoPath
from numpy import array

from .utils import RandomGenerator

# Name used to hold the edge type
EDGE_TYPE = 'type'


class BaseTopology:
    """Abstract class setting up some requirements for a Topology."""

    def add_node(self, *args, **kwargs):
        """Method adding a node to the multigraph."""
        raise NotImplementedError

    def add_edge(self, *args, **kwargs):
        """Method adding an edge to the multigraph."""
        raise NotImplementedError

    def get_shortest_path(self, node1, node2, *args, **kwargs):
        """Method finding the shortest path between two nodes."""
        raise NotImplementedError


class MultiEdgeUndirectedTopology(BaseTopology):
    """Class representing a topology with mutltiple undirected edges.

    :param rng: Random number generator.
    :type rng: :py:class:`.RandomGenerator`
    """

    def __init__(self, rng=None):

        self.graph = MultiGraph()
        self.rng = rng or RandomGenerator()

    @property
    def num_of_nodes(self):
        """The number of nodes in the topology."""
        return self.graph.number_of_nodes()

    @property
    def num_of_edges(self):
        """The number of nodes in the topology."""
        return self.graph.number_of_edges()

    def add_node(self, node_label, **node_attrs):
        """Add a node defined by its label.

        :param obj node_label: Node label. Must be hashable.
        :param dict node_attrs: Node attributes.
        """
        self.graph.add_node(node_label, **node_attrs)

    def add_edge(self, node1, node2, edge_type, **edge_attrs):
        """Add an edge between two nodes in the graph.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param obj edge_type: Edge type.
        :param dict edge_attrs: Additional attributes of the edge.
        """

        if not self.graph.has_node(node1):
            raise ValueError("First node %s does not exist." % node1)
        if not self.graph.has_node(node2):
            raise ValueError("Second node %s does not exist." % node2)
        # Adding type to attributes
        edge_attrs[EDGE_TYPE] = edge_type
        self.graph.add_edge(node1, node2, **edge_attrs)

    def get_edges(self, node1, node2, edge_types=None):
        """Return all the edges between two nodes.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param iterable edge_types: If provided, return only the edges for these types.
        :returns: A dictionary where the keys are the edge number
            and the values a dictionary of the edges attributes.
        :rtype: dict
        :raises: :py:class:`ValueError`: if there is no edge between the nodes.
        """
        try:
            edges = dict(self.graph[node1][node2])
        except KeyError:
            raise ValueError('No edge between nodes %s and %s' % (node1, node2))

        # Filter if necessary
        if edge_types:
            edges = {e: v for e, v in edges.items() if edges[e][EDGE_TYPE] in edge_types}
        return edges

    def _get_min_weight(self, weight, allowed_types, best_types, node1, node2, _d):
        """
        Find the edge with the minimum weight between two nodes.

        :param str weight: Weight name.
        :param allowed_types: Edge type(s) allowed to build path.
        :type allowed_types: dict-like object with strings as keys.
        :param dict best_types: dictionnary containing the best type for each pair of nodes.

        :note: other arguments are the ones needed by the NetworkX API.
        """

        # The algorithm uses the full (symmetric) matrix
        # instead of considering the upper triangular only.
        # This means that for two nodes u ≠ v, the weight is extracted twice:
        # once for (u,v), once for (v,u). We do not need to do the work for both.
        if (node2, node1) in best_types:
            return None

        # Get all edges between the two nodes
        edges = self.get_edges(node1, node2)

        # Get all the weights but keep only those that are allowed
        # This will throw a KeyError is some edges do not have the weight specified
        try:
            weights = [(attrs[EDGE_TYPE], attrs[weight]) for attrs in edges.values()
                       if attrs[EDGE_TYPE] in allowed_types]
        except KeyError:
            raise ValueError(
                "No edge with attribute %s found between nodes %s and %s" % (
                    weight, node1, node2
                ))

        # If there is no valid edge
        if not weights:
            return None

        # If the weights need to be weighted. Skip if they are all None.
        if list(set(allowed_types.values()))[0]:
            weights = [(t, w * allowed_types[t]) for t, w in weights]

        # Otherwise, save type and return minimum weight
        min_type, min_weight = min(weights, key=lambda t: t[1])
        best_types[(node1, node2)] = min_type
        return min_weight

    def get_shortest_path(self, node1, node2, criterion, allowed_types, edge_data=None):
        """Find the shortest path between two nodes using specified edges.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param str criterion: Criterion used to find shortest path. Must be an edge attribute.
        :param allowed_types: Edge type(s) allowed to build path.
        :type allowed_types: iter(str) or dict-like object with strings as keys.
        :param iter(str) edge_data: Edge attributes for which data along the path is requested.
        :returns: A 3-tuple containing in this order

            * the score of the optimal path
            * the list of nodes along the path
            * a dict containing the edge type and values for the attributes specified in edge_data
              (data are stored in numpy arrays)

        :rtype: tuple
        :raises:
            :py:class:`.ValueError`: if nodes dont exists or no path has been found between them.

        :note: If `allowed_types` is a dict-like object, the weight of an edge will be weighted
            by the value of the given edge type.
        """

        # Check that the nodes exist
        for node in node1, node2:
            if not self.graph.has_node(node):
                raise ValueError("Node %s does not exist." % node)

        # If allowed_types is not a dict-like object,
        # transform the variable into one with None values.
        try:
            _ = allowed_types[random.choice(list(allowed_types))]
        except TypeError:
            allowed_types = {k: None for k in set(allowed_types)}

        # Using partial function to pass the edge types
        # The weight_func will be called for each edge on the graph
        # Even those which will not been part of the optimial path
        # So best_types cannot be a simple list to which we append values
        # Instead it will be a dict where the key is a pair of nodes
        # We will extract the relevant values and attributes afterwards
        best_types = {}
        weight_func = partial(self._get_min_weight, criterion, allowed_types, best_types)

        # Calculate path
        try:
            score, path = single_source_dijkstra(
                self.graph, node1, node2, weight=weight_func)
        except NetworkXNoPath:
            raise ValueError("No path found with type %s between %s and %s." %
                             (allowed_types, node1, node2))

        # Build the dict containing the attributes data
        # Because we do not want to assume anything about the data,
        # we build an intermediate list first.
        data = {}
        # Always extract the edge types
        edge_types = []
        for u, v in zip(path, path[1:]):
            # The key could be (u,v) or (v,u)
            try:
                edge_types.append(best_types[(u, v)])
            except KeyError:
                edge_types.append(best_types[(v, u)])
        data[EDGE_TYPE] = array(edge_types)

        # Additional data if needed
        edge_data = edge_data or []
        for attr in edge_data:
            try:
                temp_gen = [self.get_edges(u, v, str(t))[0][attr]
                            for u, v, t in zip(path, path[1:], data[EDGE_TYPE])]
                data[attr] = array(temp_gen)
            except KeyError:
                raise ValueError("Some nodes do not have the attribute '%s'." % attr)

        return (score, path, data)

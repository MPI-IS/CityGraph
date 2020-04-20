"""
Topology
========

Module for building and operating on graphs.
"""
import networkx as nx

from .utils import RandomGenerator

# Some names used for the edges attributes
EDGE_TYPE = 'type'
EDGE_WEIGHT = 'weight'


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

    def get_edge_attributes(self, node1, node2, edge_attrs=None):
        """Get edges between two nodes.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param iterable edge_attrs: Edge attributes to select.
        """
        try:
            edges = dict(self.graph[node1][node2])
        except KeyError:
            raise ValueError('No edge between nodes %s and %s' % (node1, node2))

        if edge_attrs:
            for e, e_attrs in edges.items():
                edges[e] = {att: e_attrs[att] for att in edge_attrs if att in e_attrs}

            # Remove edges without attributes
            edges = {k: v for k, v in edges.items() if v}
        return edges

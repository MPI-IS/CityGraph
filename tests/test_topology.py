from unittest import TestCase
import random

from city_graph.coordinates import GeoCoordinates
from city_graph.topology import \
    BaseTopology, ScaleFreeTopology, \
    EDGE_TYPE, EDGE_WEIGHT
from city_graph.utils import RandomGenerator

from .fixtures import random_string


class TestBaseTopology(TestCase):
    """Class testing the BaseTopology class."""

    def test_add_methods_required(self):
        """Checks the required method must be implemented."""

        # Methods not implemented in derived class
        class BadTopology(BaseTopology):
            pass

        # Error
        with self.assertRaises(NotImplementedError):
            BadTopology().add_node()
        with self.assertRaises(NotImplementedError):
            BadTopology().add_edge()

        # Methods implemented in derived class
        class GoodTopology(BaseTopology):

            def add_node(self, *args, **kwargs):
                pass

            def add_edge(self, *args, **kwargs):
                pass

        # All good
        GoodTopology().add_node()
        GoodTopology().add_edge()


class TestScaleFreeTopology(TestCase):
    """Class testing the ScaleFreeTopology class."""

    def setUp(self):

        self.rng = RandomGenerator()
        self.sft = ScaleFreeTopology(rng=self.rng)

    def test_init(self):
        """Checks the initialization of an instance."""

        self.assertEqual(self.sft.num_of_nodes, 0)
        self.assertEqual(self.sft.num_of_edges, 0)

    def test_add_node(self):
        """Checks the method adding a node to the graph."""

        # Add first node
        c = GeoCoordinates(self.rng(), self.rng())
        self.sft.add_node(c)

        self.assertEqual(self.sft.num_of_nodes, 1)
        # No attribute stored in the node
        self.assertFalse(self.sft.graph.nodes[c])

        # Try to add identical node
        self.sft.add_node(c)

        self.assertEqual(self.sft.num_of_nodes, 1)
        # Still no attribute
        self.assertFalse(self.sft.graph.nodes[c])

    def test_add_edge(self):
        """Checks the methods adding an edge to the graph."""

        edge_type = random_string()
        edge_weight = self.rng()
        expected_attrs = {
            EDGE_TYPE: edge_type,
            EDGE_WEIGHT: edge_weight
        }

        # Add edge without having created a node
        c1 = GeoCoordinates(self.rng(), self.rng())
        with self.assertRaises(ValueError):
            self.sft.add_edge(c1, c1, edge_type, edge_weight)

        # Create node
        self.sft.add_node(c1)
        self.assertEqual(self.sft.num_of_nodes, 1)

        # Add edge to itself with attributes
        self.sft.add_edge(c1, c1, edge_type, edge_weight)
        self.assertEqual(self.sft.num_of_edges, 1)
        self.assertDictEqual(self.sft.graph.edges[c1, c1, 0], expected_attrs)

        # Add edge to another node but the node is missing
        c2 = GeoCoordinates(self.rng(), self.rng())
        with self.assertRaises(ValueError):
            self.sft.add_edge(c1, c2, edge_type, edge_weight)
        with self.assertRaises(ValueError):
            self.sft.add_edge(c2, c1, edge_type, edge_weight)

        # Create node and add edge
        self.sft.add_node(c2)
        self.assertEqual(self.sft.num_of_nodes, 2)
        self.sft.add_edge(c1, c2, edge_type, edge_weight)
        self.assertEqual(self.sft.num_of_edges, 2)
        self.assertDictEqual(self.sft.graph.edges[c1, c2, 0], expected_attrs)
        self.assertDictEqual(self.sft.graph.edges[c2, c1, 0], expected_attrs)

    def test_get_edge_attributes(self):
        """Checks the methods extracting attributes from edges."""

        # We build two nodes and will create different edges between them
        nodes = [GeoCoordinates(self.rng(), self.rng()) for _ in range(2)]
        for node in nodes:
            self.sft.add_node(node)
        node_start = random.choice(nodes)
        node_end = random.choice(nodes)

        # Type and attributes that will be used
        type1 = random_string()
        type2 = random_string()
        weight1 = self.rng()
        weight2 = self.rng()

        # Build edges - we mix everything
        self.sft.add_edge(node_start, node_end, type1, weight1)
        self.sft.add_edge(node_start, node_end, type1, weight2)
        self.sft.add_edge(node_start, node_end, type2, weight1)
        self.sft.add_edge(node_start, node_end, type2, weight2)

        # No attributes specified - should get all edges and all attributes
        attrs = self.sft.get_edge_attributes(node_start, node_end)
        self.assertSetEqual(set(attrs), {0, 1, 2, 3})
        for key in attrs:
            self.assertIn(EDGE_TYPE, attrs[key])
            self.assertIn(EDGE_WEIGHT, attrs[key])

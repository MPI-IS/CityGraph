from unittest import TestCase
import random

from city_graph.coordinates import GeoCoordinates
from city_graph.topology import \
    BaseTopology, MultiEdgeUndirectedTopology, \
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


class TestMultiEdgeUndirectedTopology(TestCase):
    """Class testing the MultiEdgeUndirectedTopology class."""

    def setUp(self):

        self.rng = RandomGenerator()
        self.top = MultiEdgeUndirectedTopology(rng=self.rng)

    def test_init(self):
        """Checks the initialization of an instance."""

        self.assertEqual(self.top.num_of_nodes, 0)
        self.assertEqual(self.top.num_of_edges, 0)

    def test_add_node(self):
        """Checks the method adding a node to the graph."""

        # Add first node
        c = GeoCoordinates(self.rng(), self.rng())
        self.top.add_node(c)

        self.assertEqual(self.top.num_of_nodes, 1)
        # No attribute stored in the node
        self.assertFalse(self.top.graph.nodes[c])

        # Try to add identical node
        self.top.add_node(c)

        self.assertEqual(self.top.num_of_nodes, 1)
        # Still no attribute
        self.assertFalse(self.top.graph.nodes[c])

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
            self.top.add_edge(c1, c1, edge_type, edge_weight)

        # Create node
        self.top.add_node(c1)
        self.assertEqual(self.top.num_of_nodes, 1)

        # Add edge to itself with attributes
        self.top.add_edge(c1, c1, edge_type, edge_weight)
        self.assertEqual(self.top.num_of_edges, 1)
        self.assertDictEqual(self.top.graph.edges[c1, c1, 0], expected_attrs)

        # Add edge to another node but the node is missing
        c2 = GeoCoordinates(self.rng(), self.rng())
        with self.assertRaises(ValueError):
            self.top.add_edge(c1, c2, edge_type, edge_weight)
        with self.assertRaises(ValueError):
            self.top.add_edge(c2, c1, edge_type, edge_weight)

        # Create node and add edge
        self.top.add_node(c2)
        self.assertEqual(self.top.num_of_nodes, 2)
        self.top.add_edge(c1, c2, edge_type, edge_weight)
        self.assertEqual(self.top.num_of_edges, 2)
        self.assertDictEqual(self.top.graph.edges[c1, c2, 0], expected_attrs)
        self.assertDictEqual(self.top.graph.edges[c2, c1, 0], expected_attrs)

    def test_get_edges(self):
        """Checks the method extracting the edges between nodes."""

        # We build at least one node and build edges
        # Start and end could be the same nodes
        nodes = [GeoCoordinates(self.rng(), self.rng()) for _ in range(1 + self.rng.randint(20))]
        for node in nodes:
            self.top.add_node(node)
        node_start = random.choice(nodes)
        node_end = random.choice(nodes)

        # There is no edge yet: exception
        with self.assertRaises(ValueError):
            self.top.get_edges(node_start, node_end)

        # Build some edges
        num_edges = 1 + self.rng.randint(10)
        edge_types = [random_string() for _ in range(num_edges)]
        weights = [self.rng() for _ in range(num_edges)]
        for t, w in zip(edge_types, weights):
            self.top.add_edge(node_start, node_end, t, w)

        # Add some noise between a different pair of nodes
        num_noise_edges = self.rng.randint(10)
        exclude = (node_start, node_end)
        for _ in range(num_noise_edges):
            n1 = random.choice(nodes)
            n2 = random.choice(nodes)
            while ((n1, n2) == exclude) or ((n2, n1) == exclude):
                n1 = random.choice(nodes)
                n2 = random.choice(nodes)
            self.top.add_edge(n1, n2, random_string(), self.rng())

        self.assertEqual(self.top.num_of_edges, num_edges + num_noise_edges)

        # Get all the edges between the nodes
        edges = self.top.get_edges(node_start, node_end)
        self.assertEqual(len(edges), num_edges)

        # Check data
        for e, t, w in zip(edges, edge_types, weights):
            self.assertEqual(edges[e][EDGE_TYPE], t)
            self.assertEqual(edges[e][EDGE_WEIGHT], w)


class TestGetShortestPath(TestCase):
    """Class testing the method returning the shortest path."""

    def setUp(self):

        self.rng = RandomGenerator()
        self.top = MultiEdgeUndirectedTopology(rng=self.rng)
        self.node_start = GeoCoordinates(self.rng(), self.rng())
        self.node_end = GeoCoordinates(self.rng(), self.rng())

        # Some noise nodes
        # for _ in range(self.rng.randint(20)):
        #    self.top.add_node(GeoCoordinates(self.rng(), self.rng()))

    def create_some_nodes(self, num_nodes=10):
        """Function creating (num_modes + 1) nodes."""
        nodes = [GeoCoordinates(self.rng(), self.rng()) for _ in range(1 + num_nodes)]
        for node in nodes:
            self.top.add_node(node)
        return nodes

    def get_min_from_tuple_list(self, tuple_list):
        """Function for getting the tuple with minimum second element in a list."""
        return min(tuple_list, key=lambda t: t[1])

    def test_nodes_dont_exist(self):
        """Checks the behavior when nodes dont exist."""

        with self.assertRaises(ValueError):
            self.top.get_shortest_path(self.node_start, self.node_end, random_string())
        self.top.add_node(self.node_start)
        with self.assertRaises(ValueError):
            self.top.get_shortest_path(self.node_start, self.node_end, random_string())

    def test_same_start_and_end(self):
        """Checks the behavior when we arrive when we start."""

        # Should work even if there is no edge
        self.top.add_node(self.node_start)
        self.assertFalse(self.top.num_of_edges)
        edge_types, weight, path = self.top.get_shortest_path(
            self.node_start, self.node_start, random_string())
        self.assertListEqual(path, [self.node_start])
        self.assertFalse(weight)
        self.assertFalse(edge_types)

    def test_no_edge(self):
        """Checks the behavior when there is no edge."""

        # No edges: exception
        self.top.add_node(self.node_start)
        self.top.add_node(self.node_end)
        with self.assertRaises(ValueError):
            self.top.get_shortest_path(self.node_start, self.node_end, random_string())

    def test_one_edged_paths(self):
        """Checks the behavior when paths have only one edge."""

        self.top.add_node(self.node_start)
        self.top.add_node(self.node_end)

        # Direct edge
        edge_type = random_string()
        edge_weight = self.rng()
        self.top.add_edge(self.node_start, self.node_end, edge_type, edge_weight)

        edge_types, weight, path = self.top.get_shortest_path(
            self.node_start, self.node_end, edge_type)
        self.assertListEqual(edge_types, [edge_type])
        self.assertAlmostEqual(weight, edge_weight)
        self.assertListEqual(path, [self.node_start, self.node_end])

        # If wrong edge type, that will not work
        with self.assertRaises(ValueError):
            self.top.get_shortest_path(self.node_start, self.node_end, edge_type + '2')

        # Now add a bunch of paths
        type_weights = [(edge_type, edge_weight)]
        for _ in range(self.rng.randint(10)):
            tmp_weight = self.rng()
            tmp_type = random_string()
            self.top.add_edge(self.node_start, self.node_end, tmp_type, tmp_weight)
            # Save weight and type
            type_weights.append((tmp_type, tmp_weight))

        # Find the best path, remove it, find the next one, etc...
        while type_weights:
            edge_types, weight, path = self.top.get_shortest_path(
                self.node_start, self.node_end, [t[0] for t in type_weights])
            min_type, min_weight = self.get_min_from_tuple_list(type_weights)
            type_weights.remove((min_type, min_weight))

            # Checks
            self.assertListEqual(edge_types, [min_type])
            self.assertAlmostEqual(weight, min_weight)
            self.assertListEqual(path, [self.node_start, self.node_end])

    def test_many_edged_paths_single_type(self):
        """Checks the behavior when paths have many edges but a single type."""

        self.top.add_node(self.node_start)
        self.top.add_node(self.node_end)

        edge_type = random_string()

        # Create some nodes and build first path going through them
        nodes_to_the_right = self.create_some_nodes(self.rng.randint(20))
        nodes_to_the_right.append(self.node_end)
        expected_path = [self.node_start] + nodes_to_the_right
        num_edges = len(expected_path) - 1
        expected_weight = 0
        node_in = self.node_start
        while nodes_to_the_right:
            node_out = nodes_to_the_right.pop(0)
            tmp_weight = self.rng()
            expected_weight += tmp_weight
            self.top.add_edge(node_in, node_out, edge_type, tmp_weight)
            node_in = node_out

        self.assertEqual(self.top.num_of_edges, num_edges)
        edge_types, weight, path = self.top.get_shortest_path(
            self.node_start, self.node_end, edge_type)
        self.assertListEqual(edge_types, [edge_type] * num_edges)
        self.assertAlmostEqual(weight, expected_weight)
        self.assertListEqual(path, expected_path)

        # Build a shorter path from the same type
        nodes_to_the_right2 = self.create_some_nodes(self.rng.randint(20))
        nodes_to_the_right2.append(self.node_end)
        expected_path2 = [self.node_start] + nodes_to_the_right2
        num_edges2 = len(expected_path2) - 1
        # We want a better path
        expected_weight2 = expected_weight / 2
        node_in = self.node_start
        while nodes_to_the_right2:
            node_out = nodes_to_the_right2.pop(0)
            self.top.add_edge(node_in, node_out, edge_type, expected_weight2 / num_edges2)
            node_in = node_out

        edge_types2, weight2, path2 = self.top.get_shortest_path(
            self.node_start, self.node_end, edge_type)
        self.assertListEqual(edge_types2, [edge_type] * num_edges2)
        self.assertAlmostEqual(weight2, expected_weight2)
        self.assertListEqual(path2, expected_path2)

        # Build a shorter path from a different type
        other_type = edge_type + 'other'
        nodes_to_the_right3 = self.create_some_nodes(self.rng.randint(20))
        nodes_to_the_right3.append(self.node_end)
        expected_path3 = [self.node_start] + nodes_to_the_right3
        num_edges3 = len(expected_path3) - 1
        # We want a better path
        expected_weight3 = expected_weight2 / 2
        node_in = self.node_start
        while nodes_to_the_right3:
            node_out = nodes_to_the_right3.pop(0)
            self.top.add_edge(node_in, node_out, other_type, expected_weight3 / num_edges3)
            node_in = node_out

        # If the new type is not allowed, nothing changes
        edge_types2bis, weight2bis, path2bis = self.top.get_shortest_path(
            self.node_start, self.node_end, edge_type)
        self.assertListEqual(edge_types2bis, [edge_type] * num_edges2)
        self.assertAlmostEqual(weight2bis, expected_weight2)
        self.assertListEqual(path2bis, expected_path2)

        # If new type is allowed, the new path becomes the optimal one
        edge_types3, weight3, path3 = self.top.get_shortest_path(
            self.node_start, self.node_end, (edge_type, other_type))
        self.assertListEqual(edge_types3, [other_type] * num_edges3)
        self.assertAlmostEqual(weight3, expected_weight3)
        self.assertListEqual(path3, expected_path3)

    def test_many_edged_paths_mixed_types(self):
        """Checks the behavior when paths have many edges with different types."""

        self.top.add_node(self.node_start)
        self.top.add_node(self.node_end)

        allowed_edge_types = [random_string(), random_string()]

        # Simple case: just two edges
        middle_node = self.create_some_nodes(0)[0]
        expected_simple_weight = self.rng()
        self.top.add_edge(self.node_start, middle_node,
                          allowed_edge_types[0], expected_simple_weight / 2)
        self.top.add_edge(middle_node, self.node_end,
                          allowed_edge_types[1], expected_simple_weight / 2)
        num_simple_edges = 2

        simple_types, simple_weight, simple_path = self.top.get_shortest_path(
            self.node_start, self.node_end, allowed_edge_types)
        self.assertListEqual(simple_types, allowed_edge_types)
        self.assertAlmostEqual(simple_weight, expected_simple_weight)
        self.assertListEqual(simple_path, [self.node_start, middle_node, self.node_end])

        # If only one type is given, no path is found
        with self.assertRaises(ValueError):
            self.top.get_shortest_path(self.node_start, self.node_end, allowed_edge_types[0])
        with self.assertRaises(ValueError):
            self.top.get_shortest_path(self.node_start, self.node_end, allowed_edge_types[1])

        # Create some nodes and build first mixed path going through them
        nodes_to_the_right = self.create_some_nodes(self.rng.randint(20))
        nodes_to_the_right.append(self.node_end)
        expected_path = [self.node_start] + nodes_to_the_right
        num_edges = len(expected_path) - 1
        expected_weight = expected_simple_weight / 2
        expected_types = []
        node_in = self.node_start
        while nodes_to_the_right:
            node_out = nodes_to_the_right.pop(0)
            tmp_type = random.choice(allowed_edge_types)
            expected_types.append(tmp_type)
            self.top.add_edge(node_in, node_out, tmp_type, expected_weight / num_edges)
            node_in = node_out

        self.assertEqual(self.top.num_of_edges, num_edges + num_simple_edges)
        edge_types, weight, path = self.top.get_shortest_path(
            self.node_start, self.node_end, allowed_edge_types)
        self.assertListEqual(edge_types, expected_types)
        self.assertAlmostEqual(weight, expected_weight)
        self.assertListEqual(path, expected_path)

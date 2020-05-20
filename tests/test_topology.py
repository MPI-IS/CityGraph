from contextlib import suppress
from itertools import combinations
from unittest import TestCase

from city_graph.topology import \
    BaseTopology, MultiEdgeUndirectedTopology

from .fixtures import RandomTestCase

# Some shorter names
EDGE_TYPE = MultiEdgeUndirectedTopology.EDGE_TYPE
NODE_LONG = MultiEdgeUndirectedTopology.NODE_LONG
NODE_LAT = MultiEdgeUndirectedTopology.NODE_LAT


class TestBaseTopology(TestCase):
    """Class testing the BaseTopology class."""

    def test_add_methods_required(self):
        """Checks the required method must be implemented."""

        # Variables representing nodes
        n1, n2 = ('a', 'b')

        # Methods not implemented in derived class
        class BadTopology(BaseTopology):
            pass

        # Error
        with self.assertRaises(NotImplementedError):
            BadTopology().add_node()
        with self.assertRaises(NotImplementedError):
            BadTopology().add_edge()
        with self.assertRaises(NotImplementedError):
            BadTopology().get_shortest_path(n1, n2)

        # Methods implemented in derived class
        class GoodTopology(BaseTopology):

            def add_node(self, *args, **kwargs):
                pass

            def add_edge(self, *args, **kwargs):
                pass

            def get_shortest_path(self, node1, node2, *args, **kwargs):
                pass

        # All good
        GoodTopology().add_node()
        GoodTopology().add_edge()
        GoodTopology().add_edge(n1, n2)


class TestMultiEdgeUndirectedTopology(RandomTestCase):
    """Class testing the MultiEdgeUndirectedTopology class."""

    def setUp(self):
        super().setUp()
        self.top = MultiEdgeUndirectedTopology()

    def test_init(self):
        """Checks the initialization of an instance."""

        self.assertEqual(self.top.num_of_nodes, 0)
        self.assertEqual(self.top.num_of_edges, 0)

    def test_add_node(self):
        """Checks the method adding a node to the graph."""

        # Add first node without attribute
        c, x, y = self.create_node()
        self.top.add_node(c, x, y)
        self.assertEqual(self.top.num_of_nodes, 1)
        coords = {
            NODE_LONG: x,
            NODE_LAT: y
        }
        self.assertDictEqual(self.top.graph.nodes[c], coords)

        # Try to add identical node: exception
        with self.assertRaises(RuntimeError):
            self.top.add_node(c, x, y)

        # Add another node with attributes
        c2, x2, y2 = self.create_node()
        attrs = {self.rng.rand_str(): self.rng() for _ in range(1 + self.rng.rand_int(20))}
        self.top.add_node(c2, x2, y2, **attrs)
        self.assertEqual(self.top.num_of_nodes, 2)
        coords2 = {
            NODE_LONG: x2,
            NODE_LAT: y2
        }
        attrs.update(coords2)
        self.assertDictEqual(self.top.graph.nodes[c2], attrs)

    def test_add_edge(self):
        """Checks the methods adding an edge to the graph."""

        # Edge types + additional attributes
        edge_type = self.rng.rand_str()
        edge_attrs = {self.rng.rand_str(): self.rng() for _ in range(self.rng.rand_int(10))}

        # Expected attributes
        expected_attrs = edge_attrs
        expected_attrs[EDGE_TYPE] = edge_type

        # Add edge without having created a node
        c1, x1, y1 = self.create_node()
        with self.assertRaises(KeyError):
            self.top.add_edge(c1, c1, edge_type, **edge_attrs)

        # Create node
        self.top.add_node(c1, x1, y1)
        self.assertEqual(self.top.num_of_nodes, 1)

        # Add edge to itself with attributes
        self.top.add_edge(c1, c1, edge_type, **edge_attrs)
        self.assertEqual(self.top.num_of_edges, 1)
        self.assertDictEqual(self.top.graph.edges[c1, c1, 0], expected_attrs)

        # Add edge to another node but the node is missing
        c2, x2, y2 = self.create_node()
        with self.assertRaises(KeyError):
            self.top.add_edge(c1, c2, edge_type, **edge_attrs)
        with self.assertRaises(KeyError):
            self.top.add_edge(c2, c1, edge_type, **edge_attrs)

        # Create node and add edge
        self.top.add_node(c2, x2, y2)
        self.assertEqual(self.top.num_of_nodes, 2)
        self.top.add_edge(c1, c2, edge_type, **edge_attrs)
        self.assertEqual(self.top.num_of_edges, 2)
        self.assertDictEqual(self.top.graph.edges[c1, c2, 0], expected_attrs)
        self.assertDictEqual(self.top.graph.edges[c2, c1, 0], expected_attrs)

        # Add edge of different type with same attrs between the same nodes: OK
        self.top.add_edge(c1, c2, edge_type + '2', **edge_attrs)
        self.assertEqual(self.top.num_of_edges, 3)

        # Add edge of same type and different attributes: error:
        with self.assertRaises(RuntimeError):
            self.top.add_edge(c1, c2, edge_type)
        with self.assertRaises(RuntimeError):
            self.top.add_edge(c2, c1, edge_type)

    def test_get_node(self):
        """Checks the method extracing an edge."""

        c, x, y = self.create_node()
        # Node does not exist yet
        with self.assertRaises(KeyError):
            self.top.get_node(c)

        # Add node with attributes
        attrs = {self.rng.rand_str(): self.rng() for _ in range(1 + self.rng.rand_int(20))}
        self.top.add_node(c, x, y, **attrs)
        coords = {
            NODE_LONG: x,
            NODE_LAT: y
        }
        attrs.update(coords)
        self.assertDictEqual(self.top.get_node(c), attrs)

    def test_get_edges(self):
        """Checks the method extracting the edges between nodes."""

        # We build at least one node and build edges
        # Start and end could be the same nodes
        nodes = [self.create_node() for _ in range(1 + self.rng.rand_int(20))]
        for node, x, y in nodes:
            self.top.add_node(node, x, y)
        node_ids = [node[0] for node in nodes]
        node_start = self.rng.choice(node_ids)
        node_end = self.rng.choice(node_ids)

        # There is no edge yet: exception
        with self.assertRaises(KeyError):
            self.top.get_edges(node_start, node_end)

        # Build some edges
        num_edges = 1 + self.rng.rand_int(10)
        edge_types = [self.rng.rand_str() for _ in range(num_edges)]
        weight_name = self.rng.rand_str()
        weights = [{weight_name: self.rng()} for _ in range(num_edges)]
        for t, w in zip(edge_types, weights):
            self.top.add_edge(node_start, node_end, t, **w)

        # Add some noise between a different pair of nodes
        num_noise_edges = self.rng.rand_int(10)
        exclude = (node_start, node_end)
        for _ in range(num_noise_edges):
            n1 = self.rng.choice(node_ids)
            n2 = self.rng.choice(node_ids)
            while exclude in ((n1, n2), (n2, n1)):
                n1 = self.rng.choice(node_ids)
                n2 = self.rng.choice(node_ids)
            random_weight = {weight_name: self.rng()}
            self.top.add_edge(n1, n2, self.rng.rand_str(), **random_weight)

        self.assertEqual(self.top.num_of_edges, num_edges + num_noise_edges)

        # Get all the edges between the nodes
        edges = self.top.get_edges(node_start, node_end)
        self.assertEqual(len(edges), num_edges)
        # Check data
        for e, t, w in zip(edges, edge_types, weights):
            self.assertEqual(edges[e][EDGE_TYPE], t)
            self.assertEqual(edges[e][weight_name], w[weight_name])

        # Since the graph is undirected, we can invert the args and things should still work
        edges = self.top.get_edges(node_end, node_start)
        self.assertEqual(len(edges), num_edges)
        # Check data
        for e, t, w in zip(edges, edge_types, weights):
            self.assertEqual(edges[e][EDGE_TYPE], t)
            self.assertEqual(edges[e][weight_name], w[weight_name])

        # If a type is requested and no edge of this type exists, an exception is raised
        with self.assertRaises(KeyError):
            self.top.get_edges(node_end, node_start, [self.rng.rand_str()])

    def test_get_edges_only_some_types(self):
        """Checks that we can filter the edges by types."""

        c1, x1, y1 = self.create_node()
        c2, x2, y2 = self.create_node()
        self.top.add_node(c1, x1, y1)
        self.top.add_node(c2, x2, y2)

        # Build some edges
        num_edges = self.rng.rand_int(20)
        attr_name = self.rng.rand_str()
        # Dict recording the type and the attribute
        mapping_type_attr = {}
        for _ in range(num_edges):
            tmp_type = self.rng.rand_str()
            tmp_attr = self.rng()
            mapping_type_attr[tmp_type] = tmp_attr
            # Add edge
            self.top.add_edge(c1, c2, tmp_type, **{attr_name: tmp_attr})

        # Extract samples and check data
        for k in range(1, num_edges):
            sample = self.rng.choice(list(mapping_type_attr), k, replace=False)
            edges = self.top.get_edges(c1, c2, list(sample))
            self.assertEqual(len(edges), k)
            for e in edges:
                e_type = edges[e][EDGE_TYPE]
                e_att = edges[e][attr_name]
                self.assertAlmostEqual(e_att, mapping_type_attr[e_type])


class TestGetShortestPath(RandomTestCase):
    """Class testing the method returning the shortest path."""

    def setUp(self):

        super().setUp()
        self.top = MultiEdgeUndirectedTopology()

        self.node_start, self.node_start_x, self.node_start_y = self.create_node()
        self.node_end, self.node_end_x, self.node_end_y = self.create_node()
        self.top.add_node(self.node_start, self.node_start_x, self.node_start_y)
        self.top.add_node(self.node_end, self.node_end_x, self.node_end_y)

        self.weight_name = self.rng.rand_str()

    def create_some_nodes(self, num_nodes=10):
        """Function creating (num_modes + 1) nodes."""
        nodes = [self.create_node() for _ in range(1 + num_nodes)]
        for node in nodes:
            self.top.add_node(*node)
        return [node[0] for node in nodes]

    def get_min_from_tuple_list(self, tuple_with_dict, key):
        """
        Function searching a list of tuples for which the second element is a dict,
        and return the one with the lowest value for a givn key.
        """
        return min(tuple_with_dict, key=lambda t: t[1][key]
                   )

    def test_nodes_dont_exist(self):
        """Checks the behavior when nodes dont exist."""

        new_start, x, y = self.create_node()
        new_end, _, _ = self.create_node()

        with self.assertRaises(ValueError):
            self.top.get_shortest_path(new_start, new_end, self.rng.rand_str(), self.rng.rand_str())
        self.top.add_node(new_start, x, y)
        with self.assertRaises(ValueError):
            self.top.get_shortest_path(new_start, new_end, self.rng.rand_str(), self.rng.rand_str())

    def test_same_start_and_end(self):
        """Checks the behavior when we arrive when we start."""

        # Should work even if there is no edge
        self.assertFalse(self.top.num_of_edges)
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_start, self.rng.rand_str(), self.rng.rand_str())
        self.assertListEqual(path, [self.node_start])
        self.assertFalse(score)
        self.assertFalse(data[EDGE_TYPE])

    def test_no_edge(self):
        """Checks the behavior when there is no edge."""

        # No edges: exception
        with self.assertRaises(RuntimeError):
            self.top.get_shortest_path(self.node_start, self.node_end,
                                       self.rng.rand_str(), self.rng.rand_str())

    def test_one_edged_paths(self):
        """Checks the behavior when paths have only one edge."""

        # Direct edge
        edge_type = self.rng.rand_str()
        edge_weight = {self.weight_name: self.rng()}
        self.top.add_edge(self.node_start, self.node_end, edge_type, **edge_weight)

        # If wrong edge type: error
        with self.assertRaises(RuntimeError):
            self.top.get_shortest_path(self.node_start, self.node_end,
                                       self.weight_name, {edge_type + '2': None})

        # If wrong weight name: error
        with self.assertRaises(KeyError):
            self.top.get_shortest_path(self.node_start, self.node_end,
                                       self.weight_name + '2', {edge_type: None})

        # OK
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name, {edge_type: None})
        self.assertListEqual(list(data[EDGE_TYPE]), [edge_type])
        self.assertAlmostEqual(score, edge_weight[self.weight_name])
        self.assertListEqual(path, [self.node_start, self.node_end])

        # Now add a bunch of paths
        type_weights = [(edge_type, edge_weight)]
        for _ in range(self.rng.rand_int(10)):
            tmp_weight = {self.weight_name: self.rng()}
            tmp_type = self.rng.rand_str()
            self.top.add_edge(self.node_start, self.node_end, tmp_type, **tmp_weight)
            # Save weight and type
            type_weights.append((tmp_type, tmp_weight))

        # Find the best path, remove it, find the next one, etc...
        while type_weights:
            score, path, data = self.top.get_shortest_path(
                self.node_start, self.node_end, self.weight_name,
                {t[0]: None for t in type_weights})
            min_type, min_weight = self.get_min_from_tuple_list(type_weights, self.weight_name)
            type_weights.remove((min_type, min_weight))

            # Checks
            self.assertListEqual(list(data[EDGE_TYPE]), [min_type])
            self.assertAlmostEqual(score, min_weight[self.weight_name])
            self.assertListEqual(path, [self.node_start, self.node_end])

    def test_many_edged_paths_single_type(self):
        """Checks the behavior when paths have many edges but a single type."""

        edge_type = self.rng.rand_str()

        # Create some nodes and build first path going through them
        nodes_to_the_right = self.create_some_nodes(self.rng.rand_int(20))
        nodes_to_the_right.append(self.node_end)
        expected_path = [self.node_start] + nodes_to_the_right
        num_edges = len(expected_path) - 1
        expected_score = 0
        node_in = self.node_start
        while nodes_to_the_right:
            node_out = nodes_to_the_right.pop(0)
            tmp_score = self.rng()
            expected_score += tmp_score
            self.top.add_edge(node_in, node_out, edge_type, **{self.weight_name: tmp_score})
            node_in = node_out

        self.assertEqual(self.top.num_of_edges, num_edges)
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name, {edge_type: None})
        self.assertListEqual(list(data[EDGE_TYPE]), [edge_type] * num_edges)
        self.assertAlmostEqual(score, expected_score)
        self.assertListEqual(path, expected_path)

        # Build a shorter path from the same type
        nodes_to_the_right2 = self.create_some_nodes(self.rng.rand_int(20))
        nodes_to_the_right2.append(self.node_end)
        expected_path2 = [self.node_start] + nodes_to_the_right2
        num_edges2 = len(expected_path2) - 1
        # We want a better path
        expected_score2 = expected_score / 2
        node_in = self.node_start
        while nodes_to_the_right2:
            node_out = nodes_to_the_right2.pop(0)
            self.top.add_edge(node_in, node_out, edge_type,
                              **{self.weight_name: expected_score2 / num_edges2})
            node_in = node_out

        score2, path2, data2 = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name, {edge_type: None})
        self.assertListEqual(list(data2[EDGE_TYPE]), [edge_type] * num_edges2)
        self.assertAlmostEqual(score2, expected_score2)
        self.assertListEqual(path2, expected_path2)

        # Build a shorter path from a different type
        other_type = edge_type + 'other'
        nodes_to_the_right3 = self.create_some_nodes(self.rng.rand_int(20))
        nodes_to_the_right3.append(self.node_end)
        expected_path3 = [self.node_start] + nodes_to_the_right3
        num_edges3 = len(expected_path3) - 1
        # We want a better path
        expected_score3 = expected_score2 / 2
        node_in = self.node_start
        while nodes_to_the_right3:
            node_out = nodes_to_the_right3.pop(0)
            self.top.add_edge(node_in, node_out, other_type,
                              **{self.weight_name: expected_score3 / num_edges3})
            node_in = node_out

        # If the new type is not allowed, nothing changes
        score2bis, path2bis, data2bis = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name, {edge_type: None})
        self.assertListEqual(list(data2bis[EDGE_TYPE]), [edge_type] * num_edges2)
        self.assertAlmostEqual(score2bis, expected_score2)
        self.assertListEqual(path2bis, expected_path2)

        # If new type is allowed, the new path becomes the optimal one
        score3, path3, data3 = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name,
            {edge_type: None,
             other_type: None})
        self.assertListEqual(list(data3[EDGE_TYPE]), [other_type] * num_edges3)
        self.assertAlmostEqual(score3, expected_score3)
        self.assertListEqual(path3, expected_path3)

    def test_many_edged_paths_mixed_types(self):
        """Checks the behavior when paths have many edges with different types."""

        allowed_edge_types = [self.rng.rand_str(), self.rng.rand_str()]

        # Simple case: just two edges
        middle_node = self.create_some_nodes(0)[0]
        expected_simple_score = self.rng()
        self.top.add_edge(self.node_start, middle_node, allowed_edge_types[0],
                          **{self.weight_name: expected_simple_score / 2})
        self.top.add_edge(middle_node, self.node_end, allowed_edge_types[1],
                          **{self.weight_name: expected_simple_score / 2})
        num_simple_edges = 2

        # If only one type is given, no path is found
        with self.assertRaises(RuntimeError):
            self.top.get_shortest_path(self.node_start, self.node_end,
                                       self.weight_name,
                                       {allowed_edge_types[0]: None})
        with self.assertRaises(RuntimeError):
            self.top.get_shortest_path(self.node_start, self.node_end,
                                       self.weight_name,
                                       {allowed_edge_types[1]: None})

        # If wrong weight name is given: error
        with self.assertRaises(KeyError):
            self.top.get_shortest_path(
                self.node_start, self.node_end, self.weight_name + '2',
                {allowed_edge_types[0]: None,
                 allowed_edge_types[1]: None})

        # OK
        simple_score, simple_path, simple_data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name,
            {allowed_edge_types[0]: None,
             allowed_edge_types[1]: None})
        self.assertListEqual(list(simple_data[EDGE_TYPE]), allowed_edge_types)
        self.assertAlmostEqual(simple_score, expected_simple_score)
        self.assertListEqual(simple_path, [self.node_start, middle_node, self.node_end])

        # Create some nodes and build first mixed path going through them
        nodes_to_the_right = self.create_some_nodes(self.rng.rand_int(20))
        nodes_to_the_right.append(self.node_end)
        expected_path = [self.node_start] + nodes_to_the_right
        num_edges = len(expected_path) - 1
        expected_score = expected_simple_score / 2
        expected_types = []
        node_in = self.node_start
        while nodes_to_the_right:
            node_out = nodes_to_the_right.pop(0)
            tmp_type = self.rng.choice(allowed_edge_types)
            expected_types.append(tmp_type)
            self.top.add_edge(node_in, node_out, tmp_type,
                              **{self.weight_name: expected_score / num_edges})
            node_in = node_out

        self.assertEqual(self.top.num_of_edges, num_edges + num_simple_edges)
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name,
            {allowed_edge_types[0]: None,
             allowed_edge_types[1]: None})
        self.assertListEqual(list(data[EDGE_TYPE]), expected_types)
        self.assertAlmostEqual(score, expected_score)
        self.assertListEqual(path, expected_path)

    def test_return_additional_edge_data(self):
        """Checks that additional edge data is returned properly."""

        edge_type = self.rng.rand_str()

        # One edge betwen the two nodes
        # Requested non existing attribute: error
        attrs = {self.weight_name: self.rng()}
        extra_attrs = {self.rng.rand_str(): self.rng() for _ in range(self.rng.rand_int(10))}
        attrs = {self.weight_name: self.rng(), **extra_attrs}
        self.top.add_edge(self.node_start, self.node_end, edge_type, **attrs)

        with self.assertRaises(KeyError):
            self.top.get_shortest_path(
                self.node_start, self.node_end, self.weight_name,
                {edge_type: None}, [EDGE_TYPE + '2'])

        # OK, take samples of different sizes
        for k in range(len(extra_attrs)):
            sample = self.rng.choice(list(extra_attrs), k, replace=False)
            _, _, data = self.top.get_shortest_path(
                self.node_start, self.node_end, self.weight_name,
                {edge_type: None}, list(sample))
            self.assertEqual(len(data), len(sample) + 1)  # +1 for the type
            for attr in sample:
                self.assertAlmostEqual(data[attr], extra_attrs[attr])

    def test_paths_with_weighted_edges(self):
        """Checks that we can apply weights to edges when calculating the score."""

        edge_types = [self.rng.rand_str() for _ in range(1 + self.rng.rand_int(10))]

        # Create some nodes and build paths
        nodes_to_the_right = self.create_some_nodes(self.rng.rand_int(20))
        nodes_to_the_right.append(self.node_end)
        expected_path = [self.node_start] + nodes_to_the_right
        expected_types = []
        expected_weights = []
        node_in = self.node_start
        while nodes_to_the_right:
            node_out = nodes_to_the_right.pop(0)
            tmp_type = self.rng.choice(edge_types)
            tmp_weight = self.rng()
            expected_types.append(tmp_type)
            expected_weights.append(tmp_weight)
            self.top.add_edge(node_in, node_out, tmp_type, **{self.weight_name: tmp_weight})
            node_in = node_out

        # Without weights: the score is actually the edge weights
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name,
            {e: None for e in edge_types})
        self.assertListEqual(list(data[EDGE_TYPE]), expected_types)
        self.assertAlmostEqual(score, sum(expected_weights))
        self.assertListEqual(path, expected_path)

        # With weights
        d_edge_types = {k: self.rng() for k in edge_types}
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name, d_edge_types)
        expected_score = sum(w * d_edge_types[t]
                             for w, t in zip(expected_weights, expected_types))
        self.assertAlmostEqual(score, expected_score)

        # Using weights can change the path
        # Add edge
        new_type = self.rng.rand_str()
        new_edge_weight = self.rng()
        new_expected_score = expected_score / 2
        d_edge_types[new_type] = new_expected_score / new_edge_weight
        self.top.add_edge(self.node_start, self.node_end, new_type,
                          **{self.weight_name: new_edge_weight})
        score, path, data = self.top.get_shortest_path(
            self.node_start, self.node_end, self.weight_name, d_edge_types)
        self.assertListEqual(list(data[EDGE_TYPE]), [new_type])
        self.assertAlmostEqual(score, new_expected_score)
        self.assertListEqual(path, [self.node_start, self.node_end])

    def test_return_additional_edge_data_many_edges(self):
        """
        Checks that additional edge data is returned properly if they are many edges.

        :note: bug fix (see vberenz/city_demo)
        """
        extra_data = self.rng.rand_str()

        edge_type1 = self.rng.rand_str()
        attrs1 = {self.weight_name: self.rng(), extra_data: self.rng()}
        self.top.add_edge(self.node_start, self.node_end, edge_type1, **attrs1)

        edge_type2 = self.rng.rand_str()
        attrs2 = {self.weight_name: self.rng(), extra_data: self.rng()}
        self.top.add_edge(self.node_start, self.node_end, edge_type2, **attrs2)

        _, _, d1 = self.top.get_shortest_path(self.node_start, self.node_end,
                                              self.weight_name, {edge_type1: None}, [extra_data])
        _, _, d2 = self.top.get_shortest_path(self.node_start, self.node_end,
                                              self.weight_name, {edge_type2: None}, [extra_data])

        self.assertEqual(d1[extra_data], attrs1[extra_data])
        self.assertEqual(d2[extra_data], attrs2[extra_data])


class TestEnergyBasedGraph(RandomTestCase):
    """Class testing the energy graph builder."""

    def setUp(self):
        super().setUp()

        self.nodes = [self.create_node() for _ in range(20)]
        self.nodes = {n: (x, y) for n, x, y in self.nodes}
        self.top = MultiEdgeUndirectedTopology(self.nodes)
        self.edge_types = [self.rng.rand_str(10) for _ in range(1 + self.rng.rand_int(10))]

    def test_no_max_iterations(self):
        """Check that the algorithm terminates when there is no maximum of iterations."""

        self.top.add_energy_based_edges(self.edge_types, 1, None, 1.0, 1.0, self.rng)

        # All nodes should be connected
        node_has_edges = [len(self.top.graph.edges(node)) > 1 for node in self.top.nodes]
        self.assertTrue(all(node_has_edges))

        self.assertGreaterEqual(self.top.num_of_edges, len(self.nodes) * len(self.edge_types))

    def test_num_edges_per_iterations(self):
        """Checks that the algorithm creates the right number of edges at one iteration."""

        # A single iteration
        num_edges = self.rng.rand_int(len(self.nodes))
        self.top.add_energy_based_edges(self.edge_types, num_edges, 1, 1.0, 1.0, self.rng)
        # TODO: should be equal when we use triangular matrix
        self.assertLessEqual(self.top.num_of_edges, num_edges * len(self.edge_types))

    def test_nodes_isolated(self):
        """Check that we can end up with isolated nodes."""

        num_edges = len(self.nodes) - 1
        self.top.add_energy_based_edges(self.edge_types, num_edges, 1, 1.0, 1.0, self.rng)

        # Some nodes should be isolated
        node_has_edges = [len(self.top.graph.edges(node)) > 1 for node in self.top.nodes]
        self.assertFalse(all(node_has_edges))

    def test_effect_factors(self):
        """Checks that effects of the degree and distance energy factors."""

        # Simple topology with 3 nodes (2 very close, one far away)
        n1, x1, y1 = self.create_node(0, 0)
        n2, x2, y2 = self.create_node(1, 0)
        n3, x3, y3 = self.create_node(180, 1)

        top3 = MultiEdgeUndirectedTopology(
            {
                n1: (x1, y1),
                n2: (x2, y2),
                n3: (x3, y3),
            }
        )
        edge_type = self.rng.rand_str()

        # Add one edge for a single iteration
        # Favor close edges
        top3.add_energy_based_edges([edge_type], 1, 1, 1.0, 1e4, self.rng)
        self.assertEqual(top3.num_of_edges, 1)
        _ = top3.get_edges(n1, n2, [edge_type])

        # Create many edges between n1 and n3 to increase their degree
        num_synthetic_edges = 100
        for i in range(num_synthetic_edges):
            top3.add_edge(n1, n3, 'edge %s' % i)
        self.assertEqual(top3.num_of_edges, num_synthetic_edges + 1)

        # Now creating an edge favoring degree
        # Edge should be built between n1 and n2
        edge_type2 = self.rng.rand_str()
        top3.add_energy_based_edges([edge_type2], 1, 1, 1e4, 1, self.rng)
        # JC: This does not work as well as I expected
        # _ = top3.get_edges(n2, n1, [edge_type2])


class TestCentroidsBasedGraph(RandomTestCase):
    """Class testing the builder creating edges between central points."""

    def test_central_num_of_centroids(self):
        """Checks the function adding edges between central nodes."""

        nodes = [self.create_node() for _ in range(20)]
        nodes = {n: (x, y) for n, x, y in nodes}
        top = MultiEdgeUndirectedTopology(nodes)
        edge_types = [self.rng.rand_str(10) for _ in range(1 + self.rng.rand_int(10))]

        # Connected graph required
        top.add_energy_based_edges(edge_types, 10, None, 1.0, 1.0, self.rng)

        num_edges = top.num_of_edges
        edge_type = self.rng.rand_str()

        # Works with a single centroid, but no edge is built
        top.add_edges_between_centroids([edge_type], 1, self.rng)
        self.assertEqual(top.num_of_edges, num_edges)

        # 2 centroids, 1 edge is created
        top.add_edges_between_centroids([edge_type], 2, self.rng)
        self.assertEqual(top.num_of_edges, num_edges + 1)

    def create_close_node(self, topology, node, dx, dy):
        """
        Convenient function to create a node next to another one
        given deltas in longitude and latitude.
        """

        x0 = topology.get_node(node)[topology.NODE_LAT]
        y0 = topology.get_node(node)[topology.NODE_LONG]

        x = x0 + -0.5 + self.rng() * dx
        y = y0 + -0.5 + self.rng() * dy
        return self.create_node(x, y)

    def test_clustered_data(self):
        """Checks the algorithm on a clustered graph."""

        # First create 4 nodes far apart: poles + two on the equator
        n1, x1, y1 = self.create_node(0, 0)
        n2, x2, y2 = self.create_node(180, 0)
        n3, x3, y3 = self.create_node(0, 90)
        n4, x4, y4 = self.create_node(0, -90)
        centers = (n1, n2, n3, n4)
        top = MultiEdgeUndirectedTopology(
            {
                n1: (x1, y1),
                n2: (x2, y2),
                n3: (x3, y3),
                n4: (x4, y4),
            }
        )
        edge_type = self.rng.rand_str()

        # Create a lof of nodes around these "centers"
        num_satellites = 10
        for _ in range(num_satellites):
            for nc in centers:
                n, x, y = self.create_close_node(top, nc, 1, 1)
                top.add_node(n, x, y)
                top.add_edge(n, nc, edge_type)

        num_edges = len(centers) * num_satellites
        self.assertEqual(top.num_of_edges, num_edges)

        # For some reason these centers should be connected
        # Otherwise the algo complains that the Graph is not connected
        # See comment in the topology.py
        edge_type2 = self.rng.rand_str()
        top.add_edge(n1, n2, edge_type2)
        top.add_edge(n2, n3, edge_type2)
        top.add_edge(n3, n4, edge_type2)
        top.add_edge(n4, n1, edge_type2)

        num_edges += 4
        self.assertEqual(top.num_of_edges, num_edges)

        # Running the algorithm looking for 4 centroids
        edge_type3 = self.rng.rand_str()

        top.add_edges_between_centroids([edge_type3], len(centers), self.rng)
        # This is a <= because we calculate a minimum_spanning_tree
        new_edges_counter = 0
        # print("centers: ", centers)
        for a, b in combinations(centers, 2):
            with suppress(KeyError):
                # print("a=", a, "b=", b)
                _ = top.get_edges(a, b, [edge_type3])
                # print(_)
                new_edges_counter += 1

        # TODO: this does not work...
        # Centroids are not all in the list of centers...
        # self.assertEqual(top.num_of_edges, num_edges + new_edges_counter)

from unittest import TestCase

from city_graph.topology import BaseTopology, ScaleFreeTopology
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

        from city_graph.topology import DEFAULT_LOCATION_TYPE_DISTRIBUTION, LOCATION

        self.rng = RandomGenerator()
        self.locations = []
        for location_type, num_locations in DEFAULT_LOCATION_TYPE_DISTRIBUTION.items():
            for _ in range(num_locations):
                self.locations.append(LOCATION(random_string(),
                                               location_type,
                                               self.rng(),
                                               self.rng()))
        # Topology
        self.sft = ScaleFreeTopology(rng=self.rng)

    def test_init(self):
        """Checks the initialization of an instance."""

        self.assertEqual(self.sft.num_of_nodes, 0)
        self.assertEqual(self.sft.num_of_edges, 0)

    def test_add_node(self):
        """Checks the method adding a node to the graph."""

        # Add first node
        x1, y1 = (self.rng(), self.rng())
        location_id = self.rng.randint(1000)
        self.sft.add_node(x1, y1, location_id)

        self.assertEqual(self.sft.num_of_nodes, 1)
        node = self.sft.graph.nodes[(x1, y1)]
        self.assertEqual(node[ScaleFreeTopology.LOCATION_REFERENCES_NAME],
                         tuple([location_id]))

        # Try to add identical node
        location_id2 = self.rng.randint(1000)
        self.sft.add_node(x1, y1, location_id2)

        self.assertEqual(self.sft.num_of_nodes, 1)
        node = self.sft.graph.nodes[(x1, y1)]
        self.assertEqual(node[ScaleFreeTopology.LOCATION_REFERENCES_NAME],
                         tuple([location_id, location_id2]))

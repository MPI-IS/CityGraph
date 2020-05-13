from collections import namedtuple
from unittest import TestCase

from city_graph.utils import RandomGenerator

Point = namedtuple('Point', ['x', 'y'])


class RandomTestCase(TestCase):
    """Fixture TestCase using randomness."""

    def setUp(self):
        self.rng = RandomGenerator()

    def create_node(self, x=None, y=None):
        """Helper for creating a node."""
        x = x if x is not None else self.rng()
        y = y if y is not None else self.rng()
        return self.rng.rand_int(), x, y

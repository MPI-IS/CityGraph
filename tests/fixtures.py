from unittest import TestCase

from city_graph.utils import RandomGenerator


class RandomTestCase(TestCase):
    """Fixture TestCase using randomness."""

    def setUp(self):
        self.rng = RandomGenerator()

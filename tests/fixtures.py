import random
import string
from unittest import TestCase

from city_graph.utils import RandomGenerator


def random_string(length=None):
    """ Generate a random string"""

    length = length or random.randint(5, 64)
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


class RandomTestCase(TestCase):
    """Fixture TestCase using randomness."""

    def setUp(self):
        self.rng = RandomGenerator()

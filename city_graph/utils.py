"""
Utils
=====

Module with different utilities needed for the package.
"""
import collections
import string
import time

from math import sin, cos, sqrt, atan2, radians
from numpy.random import RandomState

# Mean Earth radius in meters.
EARTH_RADIUS_METERS = 6.371 * 1e6


def get_current_time_in_ms():
    """
    Returns the current time in milliseconds.

    :note: Used for seeding the pseudo random number generator.
    """
    return int(time.time() * 1000)


def distance(long1, lat1, long2, lat2):
    """
    Calculate the distance between two points on the Earth.

    :param float long1: longitude of the first point in degrees.
    :param float lat1: latitude of the first point in degrees.
    :param float long2: longitude of the second point in degrees.
    :param float lat2: latitude of the second point in degrees.
    :returns: distance in meters.
    :rtype: float

    :note: We approximate the Earth as a sphere and use the Haversine formula.
    """

    # Convert to radians
    long1 = radians(long1)
    lat1 = radians(lat1)
    long2 = radians(long2)
    lat2 = radians(lat2)

    delta_long = long1 - long2
    delta_lat = lat1 - lat2

    a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_long / 2) ** 2
    d = 2 * atan2(sqrt(a), sqrt(1 - a))
    return d * EARTH_RADIUS_METERS


def group_locations_by(locations, attribute):
    """
    Group locations into a dictionary of lists based on an attribute.

    :param list locations: A list of locations.
    :param str attribute: Attribute name.
    """
    grouped = collections.defaultdict(list)
    for location in locations:
        value = getattr(location, attribute)
        grouped[value].append(location)
    return grouped


def reverse_mapping(mapping):
    """
    Construct a reverse dictionary mapping values to keys.

    :param dict mapping: the original dictionary.
    """
    reverse = {}
    for key, values in mapping.items():
        for value in values:
            reverse[value] = key
    return reverse


class RandomGenerator(RandomState):
    """
    Pseudo-random number generator based on the MT19937.
    Used for the tests and generating random data.

    :param float seed: Seed for the PRNG (default: current time)
    """

    # MT19937: seed should be between 0 and 2**32 - 1
    MAX_SEED = 2**32

    def __init__(self, seed=None):

        seed = seed or get_current_time_in_ms()
        self._seed = seed % self.MAX_SEED
        super().__init__(self._seed)

    @property
    def rng_seed(self):
        """Returns seed."""
        return self._seed

    def seed(self, _seed):
        """Reseeds the generator."""
        self._seed = _seed
        super().seed(self.rng_seed)

    def __call__(self):
        """Returns a random float in [0.0, 1.0)."""
        return self.random_sample()

    def rand_int(self, max_value=MAX_SEED):
        """
        Returns a random integer in [0, max_value).

        :param int max_value: Maximum value.
        """
        if max_value < 1:
            raise ValueError("Maximum value should be at least 1, instead got %s" % max_value)
        return super().randint(0, max_value)

    def rand_str(self, length=None):
        """Returns a random string."""

        length = length or (5 + self.randint(60))
        characters = string.ascii_letters + string.digits
        size = len(characters)
        return ''.join(characters[self.randint(size)] for _ in range(length))

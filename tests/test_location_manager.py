from itertools import combinations
from math import factorial

from city_graph import city

from .fixtures import RandomTestCase

# Note : _LocationManager is a private class of city.
#        It is not part of the user interface.


def _distance(x1, y1, x2, y2):
    return abs(x1 - x2)


class _Location:
    def __init__(self, index, location_type):
        self.index = index
        self.location_type = location_type

    @property
    def x(self):
        return float(self.index)

    @property
    def y(self):
        return 0.0

    def distance(self, other):
        return _distance(self.x, self.y,
                         other.x, other.y)

    def __eq__(self, other):
        if self.index != other.index:
            return False
        if self.location_type != other.location_type:
            return False
        return True

    def __hash__(self):
        return id(self)

    def __str__(self):
        return str(self.index) + " (" + self.location_type + ")"


class TestLocationManager(RandomTestCase):
    """Class testing city._LocationManager."""

    def setUp(self):
        super().setUp()

        self._locations = []
        for index in range(11):
            if index <= 8:
                if index % 2 == 0:
                    self._locations.append(_Location(index, "even"))
                else:
                    self._locations.append(_Location(index, "odd"))
            else:
                self._locations.append(_Location(index, "extra"))

        # Add nodes
        for index, loc in enumerate(self._locations):
            if index % 3 == 0:
                loc.node = 0
            elif index % 3 == 1:
                loc.node = 1
            else:
                loc.node = 2

        # LocationManager
        self.lm = city.LocationManager(self._locations, fdistance=_distance)

    def test_types(self):
        """Check types are extracted properly."""

        self.assertSetEqual(set(self.lm.location_types), set(["even", "odd", "extra"]))

        # List
        evens = self.lm.get_locations("even")
        odds = self.lm.get_locations("odd")
        extras = self.lm.get_locations("extra")
        # Dict
        all_locs = sum(self.lm.get_locations().values(), [])

        self.assertEqual(len(evens), 5)
        self.assertEqual(len(odds), 4)
        self.assertEqual(len(extras), 2)
        self.assertEqual(len(all_locs), 11)

        even_indexes = [e.index for e in evens]
        self.assertListEqual(even_indexes, [0, 2, 4, 6, 8])

    def test_get_closest(self):
        random_loc = self.rng.choice(self._locations)
        self.lm.get_closest(random_loc, "extra")

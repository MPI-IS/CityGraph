from city_graph import city

from .fixtures import RandomTestCase

# Note : _LocationManager is a private class of city.
#        It is not part of the user interface.


class _Location:
    def __init__(self, index, location_type):
        self.index = index
        self.location_type = location_type

    def distance(self, other):
        return abs(self.index - other.index)

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

    def distance(self, l1, l2):
        return abs(l1.index - l2.index)

    def test_types(self):
        """Check types are extracted properly."""
        lm = city.LocationManager(self._locations)
        self.assertSetEqual(set(lm.type_locations), set(["even", "odd", "extra"]))

        evens = lm.get_locations("even")
        odds = lm.get_locations("odd")
        extras = lm.get_locations("extra")

        self.assertTrue(len(evens) == 5)
        self.assertTrue(len(odds) == 4)
        self.assertTrue(len(extras) == 2)

        even_indexes = [e.index for e in evens]
        self.assertTrue(even_indexes == [0, 2, 4, 6, 8])

    def test_compute_all_distances(self):
        lm = city.LocationManager(self._locations)
        lm.compute_all_distances()
        distances = lm.get_all_distances()
        for index, l1 in enumerate(self._locations):
            for l2 in self._locations[index + 1:]:
                self.assertTrue(distances[l1][l2] == self.distance(l1, l2))

    def test_get_closest(self):
        lm = city.LocationManager(self._locations)
        random_loc = self.rng.choice(self._locations)
        with self.assertRaises(NotImplementedError):
            lm.get_closest(random_loc, "extra")

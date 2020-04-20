import unittest
import os
from city_graph import city_io
from city_graph.city import City
from city_graph.types import LocationType
from city_graph.types import Location
from city_graph.types import TransportType
from city_graph.topology import MultiEdgeUndirectedTopology


class _TestTopology(MultiEdgeUndirectedTopology):

    def __init__(self):

        super(MultiEdgeUndirectedTopology, self).__init__()

        # creating a super simple graph for testing purposes
        l0 = Location(LocationType.HOUSEHOLD,
                      (0, 0), name="l0")
        l1 = Location(LocationType.PHARMACY,
                      (50, 0), name="l1")

        # all locations
        self.locations = [l0, l1]
        for location in self.locations:
            self.add_node(location)

        self.add_edge(l0, l1, TransportType.ROAD.value,
                      distance=1, duration=2)

    def get_locations(self):
        return self.locations


class CityIo_TESTCASE(unittest.TestCase):
    """ Class testing module city_io"""

    def setUp(self):

        # fake topology
        self._topology = _TestTopology()

        # name of the city
        self._city_name = "test_city"

        # current directory
        self._path = os.getcwd()

    def tearDown(self):
        files = [f for f in os.listdir(self._path)
                 if os.path.isfile(os.path.join(self._path, f))
                 and f.endswith(city_io.get_extension())]
        for f in files:
            os.remove(f)

    def test_city_name(self):
        """ testing city.get_name"""
        city = City(self._city_name, self._topology)
        name = city.get_name()
        self.assertTrue(city.get_name() == name)

    def test_save_city(self):
        """testing city_io.save"""
        city = City(self._city_name, self._topology)
        path = city_io.save(city, self._path)
        self.assertTrue(os.path.isfile(path))

    def test_load_city(self):
        """testing city_io.load"""
        city = City(self._city_name, self._topology)
        path = city_io.save(city, self._path)
        self.assertTrue(os.path.isfile(path))
        loaded_city = city_io.load(city.get_name(), self._path)
        locations = list(city.get_locations())
        loaded_locations = list(loaded_city.get_locations())
        attributes = ["location_id", "coordinates", "location_type"]
        for attr in attributes:
            original = [getattr(location, attr) for location in locations]
            loaded = [getattr(location, attr) for location in loaded_locations]
            self.assertTrue(original == loaded)

    def test_load_non_existing_city(self):
        """testing city_io.load when city does not exist"""
        thrown = False
        try:
            loaded_city = city_io.load(self._city_name, self._path)
        except FileNotFoundError:
            thrown = True
        self.assertTrue(thrown)

    def test_overwrite_existing_city(self):
        """testing city_io.save when city already exists"""
        city = City(self._city_name, self._topology)
        path = city_io.save(city, self._path)
        thrown = False
        try:
            city_io.save(city, self._path)
        except FileExistsError:
            thrown = True
        self.assertTrue(thrown)

    def test_is_saved(self):
        """testing city_io.is_saved"""
        self.assertFalse(city_io.is_saved(self._city_name, self._path))
        city = City(self._city_name, self._topology)
        city_io.save(city, self._path)
        self.assertTrue(city_io.is_saved(self._city_name, self._path))

import os
import tempfile

from city_graph import city_io
from city_graph.city import City
from city_graph.types import LocationType, Location, TransportType

from .fixtures import RandomTestCase


class TestCityIo(RandomTestCase):
    """Class testing module city_io."""

    def setUp(self):
        super().setUp()

        # name of the city
        self.city_name = "test_city"

        # locations
        l0 = Location(LocationType.HOUSEHOLD, (0, 0), name="l0")
        l1 = Location(LocationType.PHARMACY, (50, 0), name="l1")
        self.locations = [l0, l1]

        # connections between the locations
        # note that we are using the word `connection` here instead of `edge`
        # to differentiate between them
        connections = {
            (l0, l1): (TransportType.ROAD,
                       {'distance': 1, 'duration': 2})
        }
        # city
        self.city = City.build_from_data(self.city_name, self.locations, connections,
                                         create_network=False)

    def test_city_name(self):
        """Testing city.name."""
        self.assertEqual(self.city.name, self.city_name)

    def test_save_city(self):
        """testing city_io.save"""

        with tempfile.TemporaryDirectory() as tf:
            path = city_io.save(self.city, tf)
            self.assertTrue(os.path.isfile(path))

    def test_load_city(self):
        """testing city_io.load"""

        with tempfile.TemporaryDirectory() as tf:
            _ = city_io.save(self.city, tf)

            # Deserialize the city
            loaded_city = city_io.load(self.city.name, tf)
            self.assertEqual(self.city.name, loaded_city.name)
            self.assertEqual(self.city._nb_processes, loaded_city._nb_processes)

            locations = self.city.get_locations()
            loaded_locations = loaded_city.get_locations()
            attributes = ["location_id", "coordinates", "location_type"]
            for attr in attributes:
                original = [getattr(location, attr) for location in locations]
                loaded = [getattr(location, attr) for location in loaded_locations]
                self.assertListEqual(original, loaded)

    def test_load_non_existing_city(self):
        """testing city_io.load when city does not exist"""

        with self.assertRaises(FileNotFoundError):
            city_io.load('toto', 'titi')

    def test_overwrite_existing_city(self):
        """testing city_io.save when city already exists"""

        with tempfile.TemporaryDirectory() as tf:
            _ = city_io.save(self.city, tf)

            # Resave city
            with self.assertRaises(FileExistsError):
                _ = city_io.save(self.city, tf)

    def test_is_saved(self):
        """testing city_io.is_saved"""

        with tempfile.TemporaryDirectory() as tf:
            self.assertFalse(city_io.is_saved(self.city.name, tf))
            city_io.save(self.city, tf)
            self.assertTrue(city_io.is_saved(self.city.name, tf))

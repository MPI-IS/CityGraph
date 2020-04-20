from unittest import TestCase
import math

from city_graph.coordinates import GeoCoordinates
from city_graph.utils import RandomGenerator


class TestGeoCoordinates(TestCase):
    """Class testing the GeoCoordinates."""

    def setUp(self):
        self.rng = RandomGenerator()

        self.x, self.y = (self.rng(), self.rng())
        self.c = GeoCoordinates(self.x, self.y)

    def test_init_and_properties(self):
        """Checks the initializations and the properties."""

        self.assertEqual(self.c.longitude, self.x)
        self.assertEqual(self.c.latitude, self.y)
        self.assertFalse(self.c.locations)

    def test_hash(self):
        """Checks the hash method."""

        # Similar coordinates - same hash
        c1 = GeoCoordinates(self.x, self.y)
        self.assertEqual(hash(self.c), hash(c1))

        # Invert coordinates - different hash
        c2 = GeoCoordinates(self.y, self.x)
        self.assertNotEqual(hash(self.c), hash(c2))

        # List of locations does not influence the hash
        c1.add_location(self.rng())
        self.assertNotEqual(self.c.locations, c1.locations)
        self.assertEqual(hash(self.c), hash(c1))

    def test_add_location(self):
        """Checks the method to add a location."""

        locations = [self.rng.randint(1000) for _ in range(self.rng.randint(100))]
        for location in locations:
            self.c.add_location(location)

        self.assertSetEqual(set(self.c.locations), set(locations))

    def test_compute_distance(self):
        """Checks the method calculating the distance between two coordinates."""

        # Reference point on the equator and opposite point
        p0 = GeoCoordinates(self.rng.randint(360), 0)
        p1 = GeoCoordinates(p0.longitude + 180, 0)

        # North and South poles
        pN = GeoCoordinates(self.rng.randint(360), 90)
        pS = GeoCoordinates(self.rng.randint(360), -90)

        self.assertAlmostEqual(GeoCoordinates.distance(p0, p0), 0)
        # Precision to the cm
        self.assertAlmostEqual(GeoCoordinates.distance(p0, pN),
                               math.pi * GeoCoordinates.EARTH_RADIUS_CM / 2,
                               places=0)
        self.assertAlmostEqual(GeoCoordinates.distance(p0, pS),
                               math.pi * GeoCoordinates.EARTH_RADIUS_CM / 2,
                               places=0)
        self.assertAlmostEqual(GeoCoordinates.distance(pN, pS),
                               math.pi * GeoCoordinates.EARTH_RADIUS_CM,
                               places=0)
        self.assertAlmostEqual(GeoCoordinates.distance(p0, p1),
                               math.pi * GeoCoordinates.EARTH_RADIUS_CM,
                               places=0)

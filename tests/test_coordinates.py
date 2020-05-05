import math

from city_graph.coordinates import GeoCoordinates

from .fixtures import RandomTestCase


class TestGeoCoordinates(RandomTestCase):
    """Class testing the GeoCoordinates."""

    def setUp(self):
        super().setUp()

        self.x, self.y = (self.rng(), self.rng())
        self.c = self.create_node(self.x, self.y)

    def test_init_and_properties(self):
        """Checks the initializations and the properties."""

        self.assertEqual(self.c.longitude, self.x)
        self.assertEqual(self.c.latitude, self.y)

    def test_hash(self):
        """Checks the hash method."""

        # Similar coordinates - same hash
        c1 = self.create_node(self.x, self.y)
        self.assertEqual(hash(self.c), hash(c1))

        # Invert coordinates - different hash
        c2 = self.create_node(self.y, self.x)
        self.assertNotEqual(hash(self.c), hash(c2))

    def test_compute_distance(self):
        """Checks the method calculating the distance between two coordinates."""

        # Reference point on the equator and opposite point
        p0 = self.create_node(self.rng.randint(360), 0)
        p1 = self.create_node(p0.longitude + 180, 0)

        # North and South poles
        pN = self.create_node(self.rng.randint(360), 90)
        pS = self.create_node(self.rng.randint(360), -90)

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

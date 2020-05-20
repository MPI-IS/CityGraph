from math import pi
from time import sleep

from city_graph.utils import RandomGenerator, \
    get_current_time_in_ms, distance, EARTH_RADIUS_METERS

from .fixtures import RandomTestCase


class TestUtilities(RandomTestCase):
    """Class testing some utilities."""

    def test_get_current_time_in_ms(self):
        """Checks the get_current_time_in_ms function."""
        num_times = 1 + self.rng.randint(100)
        times = []
        for _ in range(num_times):
            times.append(get_current_time_in_ms())
            sleep((1 + self.rng.randint(10)) / 1000)  # wait a random number of milliseconds

        # Check that the times are ordered and different
        for i in range(num_times - 1):
            self.assertLess(times[i], times[i + 1])

    def test_distance(self):
        """Checks the distance function."""

        # Reference point on the equator and opposite point
        _, p0x, p0y = self.create_node(self.rng.randint(360), 0)
        _, p1x, p1y = self.create_node(p0x + 180, 0)

        # North and South poles
        _, pNx, pNy = self.create_node(self.rng.randint(360), 90)
        _, pSx, pSy = self.create_node(self.rng.randint(360), -90)

        self.assertAlmostEqual(distance(p0x, p0y, p0x, p0y), 0)
        # Precision to the cm
        self.assertAlmostEqual(distance(p0x, p0y, pNx, pNy),
                               pi * EARTH_RADIUS_METERS / 2,
                               places=0)
        self.assertAlmostEqual(distance(p0x, p0y, pSx, pSy),
                               pi * EARTH_RADIUS_METERS / 2,
                               places=0)
        self.assertAlmostEqual(distance(pNx, pNy, pSx, pSy),
                               pi * EARTH_RADIUS_METERS,
                               places=0)
        self.assertAlmostEqual(distance(p0x, p0y, p1x, p1y),
                               pi * EARTH_RADIUS_METERS,
                               places=0)


class TestRandomGenerator(RandomTestCase):
    """Class testing the RandomGenerator class."""

    def test_init_rng(self):
        """Checks the initialization of a RNG."""

        # Seed is a read-only property
        _ = self.rng.rng_seed
        with self.assertRaises(AttributeError):
            self.rng.rng_seed = 1

        # Seed provided
        seed = self.rng.randint(1e6)
        rng2 = RandomGenerator(seed)
        self.assertEqual(seed, rng2.rng_seed)

    def test_generate_numbers(self):
        """Checks the methods generating random numbers."""

        # Float
        r_float = self.rng()
        self.assertIsInstance(r_float, float)
        self.assertGreaterEqual(r_float, 0)
        self.assertLess(r_float, 1)

        # Integer
        max_value = 1e9
        r_int = self.rng.randint(max_value)
        self.assertIsInstance(r_int, int)
        self.assertGreaterEqual(r_int, 0)
        self.assertLess(r_int, max_value)

        # rngs with different seeds generate different sequences
        rng2 = RandomGenerator()
        rng3 = RandomGenerator()
        while rng2.rng_seed == rng3.rng_seed:
            rng3 = RandomGenerator()

        self.assertNotAlmostEqual(rng2(), rng3())
        self.assertNotEqual(rng2.randint(max_value), rng3.randint(max_value))

        # Check that the integer upper-boundary is consistent with the float case
        with self.assertRaises(ValueError):
            self.rng.randint(0)
        self.assertEqual(self.rng.randint(1), 0)

    def test_generate_strings(self):
        """Checks the methods generating strings."""

        # Random length
        s = self.rng.rand_str()
        self.assertIsInstance(s, str)

        # Specified length
        length = self.rng.randint(64)
        s2 = self.rng.rand_str(length)
        self.assertEqual(len(s2), length)

        # rngs with different seeds generate different sequences
        rng2 = RandomGenerator()
        rng3 = RandomGenerator()
        while rng2.rng_seed == rng3.rng_seed:
            rng3 = RandomGenerator()

        self.assertNotEqual(rng2.rand_str(length), rng3.rand_str(length))

    def test_choice(self):
        """Checks the method for extracting a random element."""

        # If sequence empty: exception
        with self.assertRaises(ValueError):
            self.rng.choice([])

        # Real sequence
        seq = [self.rng() for _ in range(20)]
        c = self.rng.choice(seq)
        self.assertIn(c, seq)

        # rngs with different seeds generate different sequences
        rng2 = RandomGenerator()
        rng3 = RandomGenerator()
        while rng2.rng_seed == rng3.rng_seed:
            rng3 = RandomGenerator()

        length = 10
        self.assertNotEqual([rng2.choice(seq) for _ in range(length)],
                            [rng3.choice(seq) for _ in range(length)])

    def test_sample(self):
        """Checks the method for sampling a sequence."""

        seq = [self.rng() for _ in range(20)]
        length = len(seq)

        # Sample size too large
        with self.assertRaises(ValueError):
            self.rng.choice(seq, length + 1, replace=False)

        # Check sample size and content
        old_seq = seq.copy()
        k = self.rng.randint(length)
        sample = self.rng.choice(seq, k, replace=False)

        self.assertEqual(len(sample), k)
        for e in sample:
            self.assertIn(e, seq)

        # Seq should not be modified
        self.assertListEqual(seq, old_seq)

    def test_uniform(self):
        """Checks the method extracting a value between boundaries."""

        xmin, xmax = tuple(sorted([self.rng.randint(100) for _ in range(2)]))
        self.assertLessEqual(xmin, xmax)

        # Args are in the right order
        val = self.rng.uniform(xmin, xmax)
        self.assertGreaterEqual(val, xmin)
        self.assertLessEqual(val, xmax)

        # Args are flipped
        val = self.rng.uniform(xmax, xmin)
        self.assertGreaterEqual(val, xmin)
        self.assertLessEqual(val, xmax)

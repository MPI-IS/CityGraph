from unittest import TestCase
from random import randint
from time import sleep

from city_graph.utils import RandomGenerator, get_current_time_in_ms


class TestUtilities(TestCase):
    """Class testing some utilities."""

    def test_get_current_time_in_ms(self):
        """Checks the get_current_time_in_ms function."""
        num_times = randint(1, 100)
        times = []
        for _ in range(num_times):
            times.append(get_current_time_in_ms())
            sleep(randint(1, 10) / 1000)  # wait a random number of milliseconds

        # Check that the times are ordered and different
        for i in range(num_times - 1):
            self.assertLess(times[i], times[i + 1])


class TestRandomGenerator(TestCase):
    """Class testing the RandomGenerator class."""

    def setUp(self):
        self.rng = RandomGenerator()

    def test_init_rng(self):
        """Checks the initialization of a RNG."""

        # Seed is a read-only property
        _ = self.rng.seed
        with self.assertRaises(AttributeError):
            self.rng.seed = 1

        # Seed provided
        seed = randint(0, 1e6)
        rng2 = RandomGenerator(seed)
        self.assertEqual(seed, rng2.seed)

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
        while rng2.seed == rng3.seed:
            rng3 = RandomGenerator()

        self.assertNotAlmostEqual(rng2(), rng3())
        self.assertNotEqual(rng2.randint(max_value), rng3.randint(max_value))

        # Check that the integer upper-boundary is consistent with the float case
        with self.assertRaises(ValueError):
            self.rng.randint(0)
        self.assertEqual(self.rng.randint(1), 0)

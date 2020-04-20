"""
Utils
=====

Module with different utilities needed for the package.
"""
import time
from numpy.random import RandomState


def get_current_time_in_ms():
    """Returns the current time in milliseconds.

    :note: Used for seeding the pseudo random number generator.
    """
    return int(time.time() * 1000)


class RandomGenerator:
    """Pseudo-random number generator based on the MT19937.

    :param float seed: Seed for the PRNG (default: current time)
    """

    # MT19937: seed should be between 0 and 2**32 - 1
    MAX_SEED = 2**32

    def __init__(self, seed=None):

        seed = seed or get_current_time_in_ms()
        self._seed = seed % self.MAX_SEED

        self._rng = RandomState(self._seed)

    @property
    def seed(self):
        """Returns the seed (read-only)."""
        return self._seed

    def __call__(self):
        """Returns a random float in [0.0, 1.0)."""
        return self._rng.random_sample()

    def randint(self, max_value):
        """Returns a random integer in [0, max_value).

        :param int max_value: Maximum value.
        """
        if max_value < 1:
            raise ValueError("Maximum value should be at least 1, instead got %s" % max_value)
        return int(self.__call__() * max_value)

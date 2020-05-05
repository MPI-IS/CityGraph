"""
Utils
=====

Module with different utilities needed for the package.
"""
import string
import time

from numpy.random import RandomState


def get_current_time_in_ms():
    """
    Returns the current time in milliseconds.

    :note: Used for seeding the pseudo random number generator.
    """
    return int(time.time() * 1000)


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
        """"Reseed the generator."""
        self._seed = _seed
        super().seed(self.rng_seed)

    def __call__(self):
        """Returns a random float in [0.0, 1.0)."""
        return self.random_sample()

    def rand_int(self, max_value):
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

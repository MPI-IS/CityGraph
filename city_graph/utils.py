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


class RandomGenerator:
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

        self._rng = RandomState(self._seed)

    @property
    def seed(self):
        """Returns the seed (read-only)."""
        return self._seed

    def __call__(self):
        """Returns a random float in [0.0, 1.0)."""
        return self._rng.random_sample()

    def randint(self, max_value):
        """
        Returns a random integer in [0, max_value).

        :param int max_value: Maximum value.
        """
        if max_value < 1:
            raise ValueError("Maximum value should be at least 1, instead got %s" % max_value)
        return int(self.__call__() * max_value)

    def randstr(self, length=None):
        """Returns a random string."""

        length = length or (5 + self.randint(60))
        characters = string.ascii_letters + string.digits
        size = len(characters)
        return ''.join(characters[self.randint(size)] for _ in range(length))

    def choice(self, seq):
        """
        Returns a random element from the non-empty sequence.

        :param iterable seq: Sequence

        raises: IndexError if the sequence is empty.
        """
        length = len(seq)
        if not length:
            raise IndexError("Sequence is empty.")
        return seq[self.randint(length)]

    def sample(self, seq, k):
        """
        Returns a k-lengthed list of unique elements chosen from the population sequence.
        Used for random sampling without replacement.

        :param iterable seq: Population sequence
        :param int k: Sample size

        raises: ValueError if the sample size is larger than the population size.
        """
        length = len(seq)
        if k > length:
            raise ValueError('Sample size larger than population size (%s > %s).'
                             % (k, length))
        # Copy sequence
        data = list(seq)
        results = []
        for _ in range(k):
            results.append(data.pop(self.randint(length)))
            length -= 1
        return results

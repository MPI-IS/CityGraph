import unittest
import math
from city_graph import types


class TypesTestCase(unittest.TestCase):
    """ Class testing types"""

    def test_weight_preferences(self):
        """Checks preference weights are modified for shortest path"""

        weights = {"walk": 0.8,
                   "train": 0.2}

        preferences = types.Preferences(mobility=weights)
        self.assertTrue(math.isclose(preferences.mobility["walk"],
                                     weights["train"],
                                     rel_tol=1e-5))
        self.assertTrue(math.isclose(preferences.mobility["train"],
                                     weights["walk"],
                                     rel_tol=1e-5))

        weights = {"walk": 0.6,
                   "train": 0.4}
        preferences.mobility = weights
        self.assertTrue(math.isclose(preferences.mobility["walk"],
                                     weights["train"],
                                     rel_tol=1e-5))
        self.assertTrue(math.isclose(preferences.mobility["train"],
                                     weights["walk"],
                                     rel_tol=1e-5))

    def test_criterion_preferences(self):
        """Checks on criterion being properly set and checked"""

        preferences = types.Preferences(criterion=types.PathCriterion.DISTANCE)
        self.assertTrue(preferences.criterion == types.PathCriterion.DISTANCE)

        error_thrown = False
        try:
            preferences = types.Preferences(criterion="not a criterion")
        except ValueError:
            error_thrown = True

        self.assertTrue(error_thrown)

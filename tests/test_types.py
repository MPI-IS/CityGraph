from unittest import skip, TestCase

from city_graph.types import (
    Preferences, PathCriterion,
    LocationType, LocationDistribution, MobilityType)

from .fixtures import RandomTestCase


class TypesTestCase(TestCase):
    """Class testing """

    @skip("fails on python3.8 since `walk` and `train` are not enum members")
    def test_weight_preferences(self):
        """Checks preference weights are modified for shortest path"""

        weights = {"walk": 0.8,
                   "train": 0.2}

        preferences = Preferences(mobility=weights)
        self.assertAlmostEqual(preferences.mobility["walk"],
                               weights["train"],
                               5)
        self.assertAlmostEqual(preferences.mobility["train"],
                               weights["walk"],
                               5)

        weights = {"walk": 0.6,
                   "train": 0.4}
        preferences.mobility = weights
        self.assertAlmostEqual(preferences.mobility["walk"],
                               weights["train"],
                               5)
        self.assertAlmostEqual(preferences.mobility["train"],
                               weights["walk"],
                               5)

    def test_criterion_preferences(self):
        """Checks on criterion being properly set and checked"""

        preferences = Preferences(criterion=PathCriterion.DISTANCE)
        self.assertTrue(preferences.criterion == PathCriterion.DISTANCE)

        error_thrown = False
        try:
            preferences = Preferences(criterion="not a criterion")
        except ValueError:
            error_thrown = True

        self.assertTrue(error_thrown)


class TestLocationDistribution(RandomTestCase):
    """Class for testing the LocationDistribution mapping class."""

    def setUp(self):
        super().setUp()

        self.dist = LocationDistribution()

    def test_init(self):
        """Checks the correct initialization of a mapping instance."""

        # Default
        self.assertSetEqual(set(self.dist.keys()), set(LocationType))
        self.assertSetEqual(set(self.dist.values()), {0})

        # With specific values
        num_values_to_set = self.rng.rand_int(len(LocationType))
        choices = self.rng.choice([e.name.lower() for e in LocationType], num_values_to_set)
        kwargs = {m: self.rng.rand_int(10) for m in self.rng.choice(choices, num_values_to_set)}

        dist2 = LocationDistribution(**kwargs)
        for m in LocationType:
            lower_name = m.name.lower()
            if lower_name in kwargs:
                self.assertEqual(dist2[m], kwargs[lower_name])
            else:
                self.assertEqual(dist2[m], 0)

    def test_add_update_members(self):
        """Checks that we can update members but not add new ones."""

        loc_type = list(LocationType)[self.rng.rand_int(len(LocationType))]
        self.assertEqual(self.dist[loc_type], 0)

        # Update is OK
        new_val = self.rng.rand_int(100)
        self.dist[loc_type] = new_val
        self.assertEqual(self.dist[loc_type], new_val)

        # Adding new value is not OK
        with self.assertRaises(RuntimeError):
            self.dist[self.rng.rand_str()] = 1

    def test_delete_member(self):
        """Checks that we cannot delete members."""

        loc_type = list(LocationType)[self.rng.rand_int(len(LocationType))]
        with self.assertRaises(RuntimeError):
            self.dist.pop(loc_type)

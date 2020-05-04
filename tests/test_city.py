import time

from city_graph import planning
from city_graph.city import City
from city_graph.types import LocationType, Location, \
    Preferences, TransportType, PathCriterion
from city_graph.topology import MultiEdgeUndirectedTopology

from .fixtures import RandomTestCase


class TestCity(RandomTestCase):
    """Class testing city.City."""

    def setUp(self):
        super().setUp()

        # name of the city
        self.city_name = "test_city"

        # topology
        self.topology = MultiEdgeUndirectedTopology(self.rng)

        # creating a super simple graph for testing purposes
        # here we use locations as nodes
        # l0 to l6 will be linked through a line
        # l0 to l1, l1 to l2, etc
        l0 = Location(LocationType.HOUSEHOLD,
                      (0, 0), name="l0")
        l1 = Location(LocationType.PHARMACY,
                      (50, 0), name="l1")
        l2 = Location(LocationType.SUPERMARKET,
                      (100, 0), name="l2")
        l3 = Location(LocationType.PUBLIC_TRANSPORT_STATION,
                      (500, 20), name="l3")
        l4 = Location(LocationType.PUBLIC_TRANSPORT_STATION,
                      (10000, -40), name="l4")
        l5 = Location(LocationType.UNIVERSITY,
                      (3000, 10), name="l5")
        l6 = Location(LocationType.PARK,
                      (3000, 40), name="l6")

        # l7 is standalone without edges (cannot be reached)
        l7 = Location(LocationType.GAMBLING,
                      (30000, 40), name="l7")

        # all locations
        self.locations = [l0, l1, l2, l3, l4, l5, l6, l7]
        for location in self.locations:
            self.topology.add_node(location)

        # all location types
        self.locations_types = set([l.location_type for l in self.locations])

        # modes of transportation between l0 and l6,
        # in order
        modes = [TransportType.ROAD,
                 TransportType.WALK,
                 TransportType.ROAD,
                 TransportType.TRAIN,
                 TransportType.BUS,
                 TransportType.WALK]

        # to compute duration
        speeds = {TransportType.ROAD: 60,
                  TransportType.WALK: 1.5,
                  TransportType.TRAIN: 100,
                  TransportType.BUS: 50}

        # adding the edges in the topology. Adding distance and duration
        # as extra attributes.
        # computing the ground true plan "manually".
        plan_steps = []
        distances = []
        durations = []
        for loc1, loc2, mode in zip(
                self.locations, self.locations[1:-1], modes):
            # creating the edges
            distance = loc1.distance(loc2)
            duration = distance / speeds[mode]
            distances.append(distance)
            durations.append(duration)
            self.topology.add_edge(loc1, loc2,
                                   mode.value,
                                   distance=distance,
                                   duration=duration)
            # creating the expected (ground truth) plan
            plan_step = planning.PlanStep(loc1, loc2, mode)
            plan_step.duration = duration
            plan_step.distance = distance
            plan_steps.append(plan_step)

        # "manually" computed plan (i.e. ground truth plan)
        self.expected_plan = planning.Plan(steps=plan_steps)

        # will be set to a value (via set_plan_computation_time)
        # to test parallization of computation of plans taking longer
        # time
        self._plan_computation_time = None

        # all locations in the topology
        # start and target location for test of successful plan,
        # as well as "ground truth" plan
        self._start = self.locations[0]
        self._target = self.locations[-2]

        # a location that is unreachable from the start location
        self._unreachable = self.locations[-1]

        self._preferences = Preferences(criterion=PathCriterion.DURATION,
                                        mobility={TransportType.ROAD: 0.1,
                                                  TransportType.WALK: 0.3,
                                                  TransportType.BUS: 0.4,
                                                  TransportType.TRAIN: 0.1},
                                        data=("duration", "distance"))

        # city
        self.city = City(self.city_name, self.locations, self.topology)

    def test_get_locations(self):
        """checking the city returns all locations"""

        # getting the id of the locations from the city
        locations = self.city.get_locations()
        locations_id = [l.location_id for l in locations]

        # getting the ground truth
        expected_locations_id = [l.location_id for l in self.locations]

        # comparing
        self.assertSetEqual(set(locations_id), set(expected_locations_id))

        # getting the names of the locations from the city
        locations = self.city.get_locations()
        locations_name = [l.name for l in locations]

        # getting the ground truth
        expected_names = [l.name for l in self.locations]

        # comparing
        self.assertSetEqual(set(locations_name), set(expected_names))

    def test_get_locations_specifying_types(self):
        """ checking the city returns locations of specified types """

        # we know the dummy topology has 2 locations for public transport
        # station
        locations = self.city.get_locations(
            location_types=LocationType.PUBLIC_TRANSPORT_STATION)
        self.assertEqual(len(list(locations)), 2)

        # we know the dummy topology has 1 location for supermarket
        locations = self.city.get_locations(location_types=LocationType.SUPERMARKET)
        self.assertEqual(len(list(locations)), 1)

        # we know nb supermarket + public transport station is 3
        locations = self.city.get_locations(
            location_types=[
                LocationType.SUPERMARKET,
                LocationType.PUBLIC_TRANSPORT_STATION])
        self.assertEqual(len(list(locations)), 3)

    def test_get_locations_types(self):
        """ checking the city returns dictionary of locations of specified types """

        # getting the dictionary of locations {type:[locations]}
        locations = self.city.get_locations_by_types(
            [LocationType.SUPERMARKET, LocationType.PUBLIC_TRANSPORT_STATION])

        # comparing with ground truth
        self.assertTrue(len(locations.keys()) == 2)
        self.assertTrue(LocationType.SUPERMARKET in locations)
        self.assertTrue(LocationType.PUBLIC_TRANSPORT_STATION in locations)

    def test_get_closest(self):

        # Right now this is not implemented but we keep the test for later
        for _ in range(2):

            target = self.locations[0]

            with self.assertRaises(NotImplementedError):
                closest = self.city.get_closest(target)
                self.assertTrue(closest == self.locations[1])

            with self.assertRaises(NotImplementedError):
                closest = self.city.get_closest(target, LocationType.SUPERMARKET)
                self.assertTrue(closest == self.locations[2])

            with self.assertRaises(NotImplementedError):
                closest = self.city.get_closest(
                    target, [
                        LocationType.PUBLIC_TRANSPORT_STATION, LocationType.UNIVERSITY])
                self.assertTrue(closest == self.locations[3])

            with self.assertRaises(NotImplementedError):
                closest = self.city.get_closest(target, LocationType.SCHOOL)
                self.assertTrue(closest is None)

            # second run will use pre-computed distances
            self.city.compute_distances()

    def test_request_blocking_plan(self):
        """ check the plan returned by request_plan is as expected"""

        # requesting a plan
        plan = self.city.request_plan(self._start,
                                      self._target,
                                      self._preferences,
                                      blocking=True)

        # checking a instance of Plan is indeed returned
        self.assertTrue(isinstance(plan, planning.Plan))

        # checking the plan is tagged valid
        self.assertTrue(plan.is_valid())

        # checking all steps has the expected attributes
        for step in plan.steps():
            self.assertTrue(hasattr(step, "distance"))
            self.assertTrue(hasattr(step, "duration"))

        # checking the plan is the same as our (manually created)
        # ground truth
        self.assertTrue(plan == self.expected_plan)

    def test_request_blocking_plan_not_found(self):

        # requesting an impossible plan
        plan = self.city.request_plan(self._start,
                                      self._unreachable,  # !
                                      self._preferences,
                                      blocking=True)

        # checking a instance of Plan is indeed returned
        self.assertTrue(isinstance(plan, planning.Plan))

        # checking the plan is tagged as invalid
        self.assertFalse(plan.is_valid())

    def test_request_non_blocking_plan(self):

        # asking the topology to pretend it takes time to
        # compute
        # TODO: JC: dont know what is happening here
        # but sounds like we should use mock.
        #
        # self.topology.set_computation_time(0.2)

        # requesting non blocking computation, the id is returned
        # and computation continues in background
        plan_id = self.city.request_plan(self._start,
                                         self._target,
                                         self._preferences,
                                         blocking=False)

        # checking the id is an int
        self.assertTrue(isinstance(plan_id, int))

        # the plan is not supposed to be ready right
        # away (we did ask the topology to take some time)
        ready = self.city.is_plan_ready(plan_id)
        self.assertFalse(ready)

        # waiting long enough to ensure computation is done
        time.sleep(0.3)

        # checking plan is indeed finished
        ready = self.city.is_plan_ready(plan_id)
        self.assertTrue(ready)

        # getting the plan
        plan = self.city.retrieve_plan(plan_id)

        # checking plan is as expected
        self.assertTrue(isinstance(plan, planning.Plan))
        self.assertTrue(plan.is_valid())
        for step in plan.steps():
            self.assertTrue(hasattr(step, "distance"))
            self.assertTrue(hasattr(step, "duration"))
        self.assertTrue(plan == self.expected_plan)

    def test_request_failing_non_blocking_plan(self):

        # asking the topology to pretend it takes time to
        # compute
        # TODO: JC: dont know what is happening here
        # but sounds like we should use mock.
        #
        # self.topology.set_computation_time(0.2)

        # requesting non blocking computation, the id is returned
        # and computation continues in background
        plan_id = self.city.request_plan(self._start,
                                         self._unreachable,  # !
                                         self._preferences,
                                         blocking=False)

        # waiting long enough to ensure computation is done
        time.sleep(0.3)

        # checking plan is indeed finished
        ready = self.city.is_plan_ready(plan_id)
        self.assertTrue(ready)

        # getting the plan
        plan = self.city.retrieve_plan(plan_id)

        # checking plan is as expected (i.e. not valid)
        self.assertFalse(plan.is_valid())

    def test_request_non_existing_plan(self):

        # asking the topology to pretend it takes time to
        # compute
        # TODO: JC: dont know what is happening here
        # but sounds like we should use mock.
        #
        # self.topology.set_computation_time(0.2)

        # non existing plan
        plan_id = -1

        # checking status of non existing plan throws
        with self.assertRaises(ValueError):
            self.city.is_plan_ready(plan_id)

        # retrieving non existing plan throws
        with self.assertRaises(ValueError):
            self.city.retrieve_plan(plan_id)

        # requesting non blocking computation, the id is returned
        # and computation continues in background
        plan_id = self.city.request_plan(self._start,
                                         self._unreachable,  # !
                                         self._preferences,
                                         blocking=False)

        # waiting long enough to ensure computation is done
        time.sleep(0.3)

        # checking plan is indeed finished
        ready = self.city.is_plan_ready(plan_id)
        self.assertTrue(ready)

        # getting the plan
        _ = self.city.retrieve_plan(plan_id)

        # getting the plan again:
        # this should fail, city deletes plan
        # once returned (avoiding infinite growth of
        # memory)
        with self.assertRaises(ValueError):
            self.city.retrieve_plan(plan_id)

    def test_compute_plans_single_process(self):

        # requesting 3 plans
        requests = []
        requests.append((self._start, self._target, self._preferences))
        requests.append((self._start, self._target, self._preferences))
        requests.append(
            (self._start,
             self._unreachable,
             self._preferences))  # !

        # single process because nb_processes=1
        plans = self.city.compute_plans(requests)

        # 3 requests -> 3 plans
        self.assertTrue(len(plans), len(requests))

        # checking ground truth of first two plans
        for plan in plans[:2]:
            self.assertTrue(plan.is_valid())
            self.assertTrue(plan == self.expected_plan)

        # checking last plan is invalid (because to unreachable node)
        self.assertFalse(plans[-1].is_valid())

    def test_compute_plans_multiple_process(self):

        # creating the city # ! 2 processes
        city_2p = City(self.city_name, self.locations, self.topology, nb_processes=2)

        # requesting 3 plans
        requests = []
        requests.append((self._start, self._target, self._preferences))
        requests.append((self._start, self._target, self._preferences))
        requests.append(
            (self._start,
             self._unreachable,
             self._preferences))  # !

        # multi-processes because nb_processes=2
        plans = city_2p.compute_plans(requests)

        # 3 requests -> 3 plans
        self.assertTrue(len(plans), len(requests))

        # checking ground truth of first two plans
        for plan in plans[:2]:
            self.assertTrue(plan.is_valid())
            self.assertTrue(plan == self.expected_plan)

        # checking last plan is invalid (because to unreachable node)
        self.assertFalse(plans[-1].is_valid())

    def test_parallelism(self):

        # asking the topology to pretend it takes time to
        # compute
        # TODO: JC: dont know what is happening here
        # but sounds like we should use mock.
        #
        # self.topology.set_computation_time(0.2)

        # creating the city # ! nb_processes is 2
        city_2p = City(self.city_name, self.locations, self.topology, nb_processes=2)

        # starting time
        start_time = time.time()

        # requesting 2 plans, in parallel.
        plan_ids = [None] * 2
        for index in range(2):
            plan_ids[index] = city_2p.request_plan(self._start,
                                                   self._target,
                                                   self._preferences,
                                                   blocking=False)

        # waiting for both plans to finish
        while not city_2p.are_plans_ready(plan_ids):
            time.sleep(0.001)

        # end time
        end_time = time.time()

        # if things have been running in parallel, total time
        # should be less than 2 seconds (1 job is 1 second)
        self.assertTrue((end_time - start_time) < 2)
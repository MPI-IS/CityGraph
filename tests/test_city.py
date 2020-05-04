import unittest
import time
from addict import Dict
from city_graph import topology
from city_graph import planning
from city_graph.city import City
from city_graph.types import LocationType
from city_graph.types import Location
from city_graph.types import Preferences
from city_graph.types import TransportType
from city_graph.types import PathCriterion
from city_graph.topology import MultiEdgeUndirectedTopology


class _TestTopology(MultiEdgeUndirectedTopology):

    def __init__(self):

        super().__init__()

        # creating a super simple graph for testing purposes

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
            self.add_node(location)

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
            self.add_edge(loc1, loc2,
                          mode.value,
                          distance=distance,
                          duration=duration)
            # creating the expected (ground truth) plan
            plan_step = planning.PlanStep(loc1, loc2, mode)
            plan_step.duration = duration
            plan_step.distance = distance
            plan_steps.append(plan_step)

            edge = self.get_edges(loc1, loc2, mode)

        # "manually" computed plan (i.e. ground truth plan)
        self.expected_plan = planning.Plan(steps=plan_steps)

        # will be set to a value (via set_plan_computation_time)
        # to test parallization of computation of plans taking longer
        # time
        self._plan_computation_time = None

    def get_locations(self):
        return self.locations

    def get_expected_plan(self):
        return self.expected_plan

    def set_computation_time(self, t):
        self._plan_computation_time = t


class City_TESTCASE(unittest.TestCase):
    """ Class testing city.City"""

    def setUp(self):

        # fake topology
        self._topology = _TestTopology()

        # all locations in the topology
        self._locations = self._topology.get_locations()

        # start and target location for test of successful plan,
        # as well as "ground truth" plan
        self._start = self._locations[0]
        self._target = self._locations[-2]
        self._expected_plan = self._topology.get_expected_plan()

        # a location that is unreachable from the start location
        self._unreachable = self._locations[-1]

        self._preferences = Preferences(criterion=PathCriterion.DURATION,
                                        mobility={TransportType.ROAD: 0.1,
                                                  TransportType.WALK: 0.3,
                                                  TransportType.BUS: 0.4,
                                                  TransportType.TRAIN: 0.1},
                                        data=("duration", "distance"))

    def test_get_locations(self):
        """checking the city returns all locations"""

        # creating a city based on the dummy topology
        city = City("TestCity", self._topology)

        # getting the id of the locations from the city
        locations = city.get_locations()
        locations_id = [l.location_id for l in locations]

        # getting the ground truth
        expected_locations_id = [l.location_id for l in self._locations]

        # comparing
        self.assertTrue(locations_id == expected_locations_id)

        # getting the names of the locations from the city
        locations = city.get_locations()
        locations_name = [l.name for l in locations]

        # getting the ground truth
        expected_names = [l.name for l in self._locations]

        # comparing
        self.assertTrue(locations_name == expected_names)

    def test_get_locations_specifying_types(self):
        """ checking the city returns locations of specified types """

        # creating city from dummy topology
        city = City("TestCity", self._topology)

        # we know the dummy topology has 2 locations for public transport
        # station
        locations = city.get_locations(
            location_types=LocationType.PUBLIC_TRANSPORT_STATION)
        self.assertTrue(len(list(locations)) == 2)

        # we know the dummy topology has 1 location for supermarket
        locations = city.get_locations(location_types=LocationType.SUPERMARKET)
        self.assertTrue(len(list(locations)) == 1)

        # we know nb supermarket + public transport station is 3
        locations = city.get_locations(
            location_types=[
                LocationType.SUPERMARKET,
                LocationType.PUBLIC_TRANSPORT_STATION])
        self.assertTrue(len(list(locations)) == 3)

    def test_get_locations_types(self):
        """ checking the city returns types of locations """

        # creating city from dummy topology
        city = City("TestCity", self._topology)

        # listing the types of location found in the city
        loc_types = set(city.get_location_types())

        # comparing with ground truth
        # TODO: What do do there?
        #self.assertTrue(self._locations_types, self._topology.locations_types)

    def test_get_locations_types(self):
        """ checking the city returns dictionary of locations of specified types """

        # creating city from dummy topology
        city = City("TestCity", self._topology)

        # getting the dictionary of locations {type:[locations]}
        locations = city.get_locations_by_types(
            [LocationType.SUPERMARKET, LocationType.PUBLIC_TRANSPORT_STATION])

        # comparing with ground truth
        self.assertTrue(len(locations.keys()) == 2)
        self.assertTrue(LocationType.SUPERMARKET in locations)
        self.assertTrue(LocationType.PUBLIC_TRANSPORT_STATION in locations)

    def test_get_closest(self):

        # creating city from dummy topology
        city = City("TestCity", self._topology)

        for _ in range(2):

            locations = self._topology.locations
            target = locations[0]

            closest = city.get_closest(target)
            self.assertTrue(closest == locations[1])

            closest = city.get_closest(target, LocationType.SUPERMARKET)
            self.assertTrue(closest == locations[2])

            closest = city.get_closest(
                target, [
                    LocationType.PUBLIC_TRANSPORT_STATION, LocationType.UNIVERSITY])
            self.assertTrue(closest == locations[3])

            closest = city.get_closest(target, LocationType.SCHOOL)
            self.assertTrue(closest is None)

            # second run will use pre-computed distances
            city.compute_distances()

    def test_request_blocking_plan(self):
        """ check the plan returned by request_plan is as expected"""

        # creating city from dummy topology
        city = City("TestCity", self._topology, nb_processes=1)

        # requesting a plan
        plan = city.request_plan(self._start,
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
        self.assertTrue(plan == self._topology.expected_plan)

    def test_request_blocking_plan_not_found(self):

        # requesting an impossible plan
        city = City("TestCity", self._topology, nb_processes=1)
        plan = city.request_plan(self._start,
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
        self._topology.set_computation_time(0.2)

        # creating the city
        city = City("TestCity", self._topology, nb_processes=1)

        # requesting non blocking computation, the id is returned
        # and computation continues in background
        plan_id = city.request_plan(self._start,
                                    self._target,
                                    self._preferences,
                                    blocking=False)

        # checking the id is an int
        self.assertTrue(isinstance(plan_id, int))

        # the plan is not supposed to be ready right
        # away (we did ask the topology to take some time)
        ready = city.is_plan_ready(plan_id)
        self.assertFalse(ready)

        # waiting long enough to ensure computation is done
        time.sleep(0.3)

        # checking plan is indeed finished
        ready = city.is_plan_ready(plan_id)
        self.assertTrue(ready)

        # getting the plan
        plan = city.retrieve_plan(plan_id)

        # checking plan is as expected
        self.assertTrue(isinstance(plan, planning.Plan))
        self.assertTrue(plan.is_valid())
        for step in plan.steps():
            self.assertTrue(hasattr(step, "distance"))
            self.assertTrue(hasattr(step, "duration"))
        self.assertTrue(plan == self._topology.expected_plan)

    def test_request_failing_non_blocking_plan(self):

        # asking the topology to pretend it takes time to
        # compute
        self._topology.set_computation_time(0.2)

        # creating the city
        city = City("TestCity", self._topology, nb_processes=1)

        # requesting non blocking computation, the id is returned
        # and computation continues in background
        plan_id = city.request_plan(self._start,
                                    self._unreachable,  # !
                                    self._preferences,
                                    blocking=False)

        # waiting long enough to ensure computation is done
        time.sleep(0.3)

        # checking plan is indeed finished
        ready = city.is_plan_ready(plan_id)
        self.assertTrue(ready)

        # getting the plan
        plan = city.retrieve_plan(plan_id)

        # checking plan is as expected (i.e. not valid)
        self.assertFalse(plan.is_valid())

    def test_request_non_existing_plan(self):

        # asking the topology to pretend it takes time to
        # compute
        self._topology.set_computation_time(0.2)

        # creating the city
        city = City("TestCity", self._topology, nb_processes=1)

        # non existing plan
        plan_id = -1

        # checking status of non existing plan throws
        thrown = False
        try:
            city.is_plan_ready(plan_id)
        except ValueError:
            thrown = True
        self.assertTrue(thrown)

        # retrieving non existing plan throws
        thrown = False
        try:
            city.retrieve_plan(plan_id)
        except ValueError:
            thrown = True
        self.assertTrue(thrown)

        # requesting non blocking computation, the id is returned
        # and computation continues in background
        plan_id = city.request_plan(self._start,
                                    self._unreachable,  # !
                                    self._preferences,
                                    blocking=False)

        # waiting long enough to ensure computation is done
        time.sleep(0.3)

        # checking plan is indeed finished
        ready = city.is_plan_ready(plan_id)
        self.assertTrue(ready)

        # getting the plan
        plan = city.retrieve_plan(plan_id)

        # getting the plan again:
        # this should fail, city deletes plan
        # once returned (avoiding infinite growth of
        # memory)
        thrown = False
        try:
            city.retrieve_plan(plan_id)
        except ValueError:
            thrown = True
        self.assertTrue(thrown)

    def test_compute_plans_single_process(self):

        # creating the city
        city = City("TestCity", self._topology, nb_processes=1)

        # requesting 3 plans
        requests = []
        requests.append((self._start, self._target, self._preferences))
        requests.append((self._start, self._target, self._preferences))
        requests.append(
            (self._start,
             self._unreachable,
             self._preferences))  # !

        # single process because nb_processes=1
        plans = city.compute_plans(requests)

        # 3 requests -> 3 plans
        self.assertTrue(len(plans), len(requests))

        # checking ground truth of first two plans
        for plan in plans[:2]:
            self.assertTrue(plan.is_valid())
            self.assertTrue(plan == self._expected_plan)

        # checking last plan is invalid (because to unreachable node)
        self.assertFalse(plans[-1].is_valid())

    def test_compute_plans_multiple_process(self):

        # creating the city # ! 2 processes
        city = City("TestCity", self._topology, nb_processes=2)

        # requesting 3 plans
        requests = []
        requests.append((self._start, self._target, self._preferences))
        requests.append((self._start, self._target, self._preferences))
        requests.append(
            (self._start,
             self._unreachable,
             self._preferences))  # !

        # multi-processes because nb_processes=2
        plans = city.compute_plans(requests)

        # 3 requests -> 3 plans
        self.assertTrue(len(plans), len(requests))

        # checking ground truth of first two plans
        for plan in plans[:2]:
            self.assertTrue(plan.is_valid())
            self.assertTrue(plan == self._expected_plan)

        # checking last plan is invalid (because to unreachable node)
        self.assertFalse(plans[-1].is_valid())

    def test_parallelism(self):

        # asking the topology to pretend it takes time to
        # compute
        self._topology.set_computation_time(1.0)

        # creating the city # ! nb_processes is 2
        city = City("TestCity", self._topology, nb_processes=2)

        # starting time
        start_time = time.time()

        # requesting 2 plans, in parallel.
        plan_ids = [None] * 2
        for index in range(2):
            plan_ids[index] = city.request_plan(self._start,
                                                self._target,
                                                self._preferences,
                                                blocking=False)

        # waiting for both plans to finish
        while not city.are_plans_ready(plan_ids):
            time.sleep(0.001)

        # end time
        end_time = time.time()

        # if things have been running in parallel, total time
        # should be less than 2 seconds (1 job is 1 second)
        self.assertTrue((end_time - start_time) < 2)

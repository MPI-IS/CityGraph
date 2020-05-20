import time
from unittest.mock import patch

from city_graph.city import City
from city_graph.planning import Plan, PlanStep
from city_graph.topology import MultiEdgeUndirectedTopology
from city_graph.types import LocationType, Location, \
    Preferences, TransportType, PathCriterion

from .fixtures import RandomTestCase


# see method : test_parallelism
SLEEP_TIME = 0.001
TOTAL_TIME_CHECK = 2.0

# Some shorter names
EDGE_TYPE = MultiEdgeUndirectedTopology.EDGE_TYPE


class TestCity(RandomTestCase):
    """Class testing city.City."""

    def setUp(self):
        super().setUp()

        # name of the city
        self.city_name = "test_city"

        # creating a super simple city for testing purposes
        # l0 to l6 will be linked through a line
        # l0 to l1, l1 to l2, etc
        l0 = Location(LocationType.HOUSEHOLD,
                      (0, 0), name="l0")
        l1 = Location(LocationType.PHARMACY,
                      (1, 0), name="l1")
        l2 = Location(LocationType.GROCERY,
                      (2, 0), name="l2")
        l3 = Location(LocationType.PUBLIC_TRANSPORT_STATION,
                      (3, 0), name="l3")
        l4 = Location(LocationType.PUBLIC_TRANSPORT_STATION,
                      (4, 0), name="l4")
        l5 = Location(LocationType.UNIVERSITY,
                      (5, 0), name="l5")
        l6 = Location(LocationType.PARK,
                      (6, 0), name="l6")

        # l7 is standalone without edges (cannot be reached)
        l7 = Location(LocationType.GAMBLING,
                      (30000, 40), name="l7")

        # all locations
        self.locations = [l0, l1, l2, l3, l4, l5, l6, l7]

        # all location types
        self.locations_types = {l.location_type for l in self.locations}

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

        # connections between the locations
        # note that we are using the word `connection` here instead of `edge`
        # to differentiate between them.
        # planning: computing the ground true plan "manually".
        plan_steps = []
        distances = []
        durations = []
        self.connections = {}
        for loc1, loc2, mode in zip(
                self.locations, self.locations[1:-1], modes):
            # creating the edges
            distance = loc1.distance(loc2)
            duration = distance / speeds[mode]
            distances.append(distance)
            durations.append(duration)
            self.connections[(loc1, loc2)] = (
                mode,
                {'distance': distance, 'duration': duration}
            )
            # creating the expected (ground truth) plan
            plan_step = PlanStep(loc1, loc2, mode, duration)
            plan_step.duration = duration
            plan_step.distance = distance
            plan_steps.append(plan_step)

        # city
        self.city = City.build_from_data(
            self.city_name, self.locations, self.connections, create_network=False)
        # city with 2 processes
        self.city_2p = City.build_from_data(
            self.city_name, self.locations, self.connections, create_network=False)
        self.city_2p._nb_processes = 2

        # "manually" computed plan (i.e. ground truth plan)
        self.expected_plan = Plan(steps=plan_steps)

        # will be set to a value (via set_plan_computation_time)
        # to test parallization of computation of plans taking longer
        # time
        self._plan_computation_time = None

        # all locations
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

    def test_default_constructor(self):
        """Check the default constructor."""

        city = City(self.city_name, self.locations, None)
        self.assertEqual(city.name, self.city_name)

        for att in ['_pool', '_plans', '_plan_id', '_topology', '_locations_manager']:
            self.assertTrue(hasattr(city, att))
            self.assertTrue(hasattr(self.city, att))

    @patch.object(City, 'create_connections_by_energy')
    def test_create_city_from_data(self, create_mocked):
        """Checks that we can create a city from pre-computed locations and connetions."""

        # We use the city built in the setup
        # Check locations
        top = self.city._topology
        locs_from_lm = sum(self.city._locations_manager.get_locations().values(), [])
        self.assertSetEqual(set(locs_from_lm), set(self.locations))

        # Check nodes in the topology
        self.assertEqual(top.num_of_nodes, len(self.locations))
        # There should be exactly one node per location
        for loc in locs_from_lm:
            _ = top.get_node(loc.node)

        # Check edges
        self.assertEqual(top.num_of_edges, len(self.connections))
        for (l1, l2), (mode, attrs) in self.connections.items():
            edge_data = top.get_edges(l1.node, l2.node)
            self.assertEqual(len(edge_data), 1)  # only one edge
            edge_data = edge_data[0]
            self.assertEqual(edge_data.pop(EDGE_TYPE), mode)  # transportation mode
            self.assertDictEqual(edge_data, attrs)

        # Energy algorithm should not have been called
        create_mocked.assert_not_called()

        # New city, this time network created
        _ = City.build_from_data(self.city_name, self.locations, self.connections)
        self.assertTrue(create_mocked.called)

    @patch.object(City, 'create_connections_by_energy')
    def test_create_city_from_distribution(self, create_mocked):
        """Checks that we can create a city from the location type distribution."""

        distribution = {t: self.rng.randint(10) for t in LocationType}
        city = City.build_random(self.city_name, distribution, rng=self.rng, create_network=False)
        lm = city._locations_manager

        # Check the number of locations
        total_num_loc = 0
        for loct, num_loc in distribution.items():
            self.assertEqual(len(lm.get_locations(loct)), num_loc)
            total_num_loc += num_loc

        # Check the nodes in the topology
        self.assertEqual(city._topology.num_of_nodes, total_num_loc)
        # There should be exactly one node per location
        for loct in distribution:
            for loc in lm.get_locations(loct):
                _ = city._topology.get_node(loc.node)

        # Energy algorithm should not have been called
        create_mocked.assert_not_called()

        # New city, this time network created
        _ = City.build_random(self.city_name, distribution, rng=self.rng)
        self.assertTrue(create_mocked.called)

    def test_get_locations(self):
        """Checking the city returns all locations"""

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
        locations = self.city.get_locations(location_types=LocationType.GROCERY)
        self.assertEqual(len(list(locations)), 1)

        # we know nb supermarket + public transport station is 3
        locations = self.city.get_locations(
            location_types=[
                LocationType.GROCERY,
                LocationType.PUBLIC_TRANSPORT_STATION])
        self.assertEqual(len(list(locations)), 3)

    def test_get_locations_types(self):
        """ checking the city returns dictionary of locations of specified types """

        # getting the dictionary of locations {type:[locations]}
        locations = self.city.get_locations_by_types(
            [LocationType.GROCERY, LocationType.PUBLIC_TRANSPORT_STATION])

        # comparing with ground truth
        self.assertEqual(len(locations.keys()), 2)
        self.assertIn(LocationType.GROCERY, locations)
        self.assertIn(LocationType.PUBLIC_TRANSPORT_STATION, locations)

    def test_get_closest(self):
        """Test that city succeed in returning closest locations"""

        # running twice, because distances are buffered, so
        # results could vary between two iterations
        for _ in range(2):

            target = self.locations[0]

            closest = self.city.get_closest(target)
            self.assertTrue(closest == self.locations[1])

            closest = self.city.get_closest(target, LocationType.GROCERY)
            self.assertTrue(closest == self.locations[2])

            closest = self.city.get_closest(
                target, [
                    LocationType.PUBLIC_TRANSPORT_STATION, LocationType.UNIVERSITY])
            self.assertTrue(closest == self.locations[3])

            closest = self.city.get_closest(target, LocationType.SCHOOL)
            self.assertTrue(closest is None)

    def test_get_closest_with_exclusion(self):
        """
        Test that city succeed in returning closest locations
        even when some locations are excluded
        """
        # running twice, because distances are buffered, so
        # results could vary between two iterations
        for _ in range(2):

            target = self.locations[0]

            closest = self.city.get_closest(target)
            self.assertTrue(closest == self.locations[1])

            closest = self.city.get_closest(target,excluded_locations=[self.locations[1]])
            self.assertTrue(closest== self.locations[2])


    def test_request_blocking_plan(self):
        """ check the plan returned by request_plan is as expected"""

        # requesting a plan
        plan = self.city.request_plan(self._start,
                                      self._target,
                                      self._preferences,
                                      blocking=True)

        # checking a instance of Plan is indeed returned
        self.assertTrue(isinstance(plan, Plan))

        # checking the plan is tagged valid
        self.assertTrue(plan.is_valid())

        # checking all steps has the expected attributes
        for step in plan.steps():
            self.assertTrue(hasattr(step, "distance"))
            self.assertTrue(hasattr(step, "duration"))

        # checking the plan is the same as our (manually created)
        # ground truth
        self.assertEqual(plan, self.expected_plan)

    def test_request_blocking_plan_not_found(self):

        # requesting an impossible plan
        plan = self.city.request_plan(self._start,
                                      self._unreachable,  # !
                                      self._preferences,
                                      blocking=True)

        # checking a instance of Plan is indeed returned
        self.assertTrue(isinstance(plan, Plan))

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
        self.assertTrue(isinstance(plan, Plan))
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
        with self.assertRaises(KeyError):
            self.city.is_plan_ready(plan_id)

        # retrieving non existing plan throws
        with self.assertRaises(KeyError):
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
        with self.assertRaises(KeyError):
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

        # requesting 3 plans
        requests = []
        requests.append((self._start, self._target, self._preferences))
        requests.append((self._start, self._target, self._preferences))
        requests.append(
            (self._start,
             self._unreachable,
             self._preferences))  # !

        # multi-processes because nb_processes=2
        plans = self.city_2p.compute_plans(requests)

        # 3 requests -> 3 plans
        self.assertTrue(len(plans), len(requests))

        # checking ground truth of first two plans
        for plan in plans[:2]:
            self.assertTrue(plan.is_valid())
            self.assertEqual(plan, self.expected_plan)

        # checking last plan is invalid (because to unreachable node)
        self.assertFalse(plans[-1].is_valid())

    def test_parallelism(self):

        # asking the topology to pretend it takes time to
        # compute
        # TODO: JC: dont know what is happening here
        # but sounds like we should use mock.
        #
        # self.topology.set_computation_time(0.2)

        # starting time
        start_time = time.time()

        # requesting 2 plans, in parallel.
        plan_ids = [None] * 2
        for index in range(2):
            plan_ids[index] = self.city_2p.request_plan(self._start,
                                                        self._target,
                                                        self._preferences,
                                                        blocking=False)

        # waiting for both plans to finish
        while not self.city_2p.are_plans_ready(plan_ids):
            time.sleep(SLEEP_TIME)

        # end time
        end_time = time.time()

        # if things have been running in parallel, total time
        # should be less than 2 seconds (1 job is 1 second)
        self.assertTrue((end_time - start_time) < TOTAL_TIME_CHECK)


    def test_plan_durations(self):
        """Check the where method of Plan"""
        l0,l1,l2,l3 = self.locations[:4]
        step1 = PlanStep(l0,l1,TransportType.ROAD,60)
        step2 = PlanStep(l1,l2,TransportType.BUS,30)
        step3 = PlanStep(l2,l3,TransportType.TRAIN,20)
        plan = Plan([step1,step2,step3])
        starting_time = 100
        # checking unfinished plans
        # on first segment
        finished,(r1,r2,mode) = plan.where(100)
        self.assertEqual(r1,l0)
        self.assertEqual(r2,l1)
        self.assertFalse(finished)
        self.assertEqual(mode,TransportType.ROAD)
        # still on first segment
        finished,(r1,r2,mode) = plan.where(starting_time+20)
        self.assertEqual(r1,l0)
        self.assertEqual(r2,l1)
        self.assertFalse(finished)
        self.assertEqual(mode,TransportType.ROAD)
        # on second segment
        finished,(r1,r2,mode) = plan.where(starting_time+70)
        self.assertEqual(r1,l1)
        self.assertEqual(r2,l2)
        self.assertFalse(finished)
        self.assertEqual(mode,TransportType.BUS)
        # on third segment
        finished,(r1,r2,mode) = plan.where(starting_time+110)
        self.assertEqual(r1,l2)
        self.assertEqual(r2,l3)
        self.assertFalse(finished)
        self.assertEqual(mode,TransportType.TRAIN)
        # check finished plan
        finished,(r1,r2,mode) = plan.where(starting_time+130)
        self.assertEqual(r1,l2)
        self.assertEqual(r2,l3)
        self.assertEqual(finished,20)
        self.assertEqual(mode,TransportType.TRAIN)
        # check invalid query time
        with self.assertRaises(ValueError):
            plan.where(starting_time-1)

    @patch.object(MultiEdgeUndirectedTopology, 'add_energy_based_edges')
    def test_create_connections_by_energy(self, add_mocked):
        """Checks the algo creating connections by energy."""

        self.city.create_connections_by_energy()
        self.assertTrue(add_mocked.called)

    @patch.object(MultiEdgeUndirectedTopology, 'add_edges_between_centroids')
    def test_create_central_connections(self, add_mocked):
        """Checks the algo creating connections to central locations."""

        self.city.create_central_connections(EDGE_TYPE)
        self.assertTrue(add_mocked.called)

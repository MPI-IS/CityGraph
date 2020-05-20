from itertools import chain
import multiprocessing

from .planning import get_plan
from .topology import MultiEdgeUndirectedTopology
from .types import (
    LocationType, Location, Point, TransportType, LocationDistribution)
from .utils import distance, group_locations_by, RandomGenerator


DEFAULT_LOCATION_DISTRIBUTION = LocationDistribution(
    household=295, office=150, school=3, university=2,
    grocery=25, retail=20, sports_centre=5, park=5,
    restaurant=5, bar=5)


class LocationManager:
    """
    Helper class that manages locations.

    :param iter locations: Locations
    :param func fdistance: Function used to calculate distances
    """

    __slots__ = (
        "_distances", "_locations", "_func_distance"
    )

    def __init__(self, locations, fdistance=distance):

        # Function used to calculate distances
        self._func_distance = fdistance

        # saving all distances computation, so they not done twice
        self._distances = {}

        # Save locations by type
        self._locations = group_locations_by(locations, "location_type")

    @property
    def location_types(self):
        """Returns all location types."""
        return list(self._locations)

    # return {location_type:[locations]}
    def get_locations(self, location_types=None):
        """
        Returns the locations.

        :param location_types: location types requested
        :type location_types: str or iter(str)
        :returns: Locations matching the types
        :rtype: list or dict
        """

        location_types = location_types or self.location_types
        if isinstance(location_types, str):
            return self._get_locations(location_types)
        # Otherwise return dict
        return {type_: self._get_locations(type_)
                for type_ in location_types}

    # return list of locations of specified type
    def _get_locations(self, location_type):
        try:
            return self._locations[location_type]
        except KeyError:
            # TODO Replace by logger
            print("No location of type %s" % location_type)
            return []

    # returns {(l1,l2):location:distance}}
    def get_all_distances(self):
        return self._distances

    def get_distance(self, l1, l2):
        """
        Returns the distance between l1 and l2 by either
          * retrieving it from the _distances attributes (if previously computed)
          * by computing it (and then saving it in _distances)

        :todo: lru_cache here?
        """
        try:
            return self._get_distance(l1, l2)
        except KeyError:
            d = self._func_distance(l1.x, l1.y, l2.x, l2.y)
            self._distances[(l1, l2)] = d
            return d

    def _get_distance(self, l1, l2):
        """Try to get the distance between two locations by flipping them."""
        try:
            return self._distances[(l1, l2)]
        except KeyError:
            return self._distances[(l2, l1)]

    def get_closest(self, location,
                    location_types, excluded_locations=[]):
        """
        Returns the location the closest to location, which is of one
        of the specified authorized type, and not excluded.  Note:
        this does not return the closest location in terms of shortest
        path.  Uses distances as the crows fly.

        :param location: target location
        :param location_types: list of authorized type
        :param excluded_location: list of excluded locations (default:
            empty list)

        :returns: the location that is the closest to location (which
                  is of the authorized type and not excluded)
        """

        # creating the list of candidate locations
        if location_types is None:
            location_types = self.location_types
        elif isinstance(location_types, str):
            location_types = [location_types]
        candidates = list(
            chain.from_iterable(
                self._get_locations(location_type)
                for location_type in location_types))

        # to avoid returning location (which is indeed the closest location
        # to itself)
        excluded_locations.append(location)

        # lower indexes will be closest location
        sorted_candidates = sorted(
            candidates, key=lambda l: self.get_distance(l, location))

        # the closest is the candidate with the lowest
        # index which is not in excluded
        for candidate in sorted_candidates:
            if candidate not in excluded_locations:
                return candidate

        # not a single location that is not excluded
        return None


class City:
    """
    Class for representing a city.

    :param str name: name of the city
    :param rng: Random number generator.
    :type rng: :py:class: `.RandomGenerator`
    :param int nb_processes: number of processes to use (when computing shortest paths)
    :param location_cls: A callable used to construct the location.

    :note: The `location_cls` default constructor must accept as input arguments:
        * a location type
        * a 2-tuple of random values between ``x_lim`` and ``y_lim``
    :note: The location coordinates (longitude and latitude)
        should be accessible through its attributes ``x`` and ``y``.
    """

    __slots__ = (
        "name", "_topology", "_pool", "_nb_processes",
        "_plans", "_plan_id", "_locations_manager", "_location_cls",
        "_locations_by_node"
    )

    def __init__(
        self, name, locations, topology,
        nb_processes=1,
        location_cls=Location
    ):
        self.name = name

        # Create the LocationManager
        self._locations_manager = LocationManager(locations)

        # Topology
        self._topology = topology

        # pool of processes in charge of computing plans
        self._nb_processes = nb_processes
        self._pool = multiprocessing.Pool(processes=nb_processes)

        # key: plan_id, value: result(future) of the plan,
        # as provided by the process in charge of computing the plan
        self._plans = {}

        # used for attributing a unique new id to each plan
        self._plan_id = 0

        # remember the location class. This will come handy.
        self._location_cls = location_cls

        # Build the lookup dictionary. XXX: It used to inside the
        # LocationManager, but it becomes too aware about the location
        # class and the topology. Probably it should be merged with
        # the city.
        self._locations_by_node = group_locations_by(locations, "node")

    def __del__(self):
        self._pool.terminate()
        self._pool.join()

    @classmethod
    def build_from_data(
        cls, name, locations, connections=None, rng=None,
        location_cls=Location, create_network=True, nb_processes=1,
        **kwargs
    ):
        """
        Creates a city with locations and connections provided as
        input.

        :param str name: name of the city
        :param iter locations: locations
        :param dict connections: connections between a pair of
            locations (key).  The value is a 2-tuple containing: * the
            connection type * a dictionary for the extra attributes of
            the connection
        :param rng: Random number generator.
        :type rng: :py:class:`RandomGenerator<city_graph.utils.RandomGenerator>`

        :param location_cls: A callable used to construct the
            location.
        :param int create_network: Whether to run create connections
            using the energy algorithm
        :param int nb_processes: number of processes to use (when
            computing shortest paths)
        :param dict kwargs: Additional kwargs to pass to the energy
            algorithm

        :note: The `location_cls` default constructor must accept
            as input arguments: * a location type * a 2-tuple of
            random values between ``x_lim`` and ``y_lim`` :note: The
            location coordinates (longitude and latitude) should be
            accessible through its attributes ``x`` and ``y``.
        """
        rng = rng or RandomGenerator()

        # Assign node IDs if necessary
        locations = list(locations)
        for i, loc in enumerate(locations):
            if not loc.node:
                loc.node = i

        # All locations should have a node now
        assert all([hasattr(loc, 'node') for loc in locations])

        # Create the Topology
        nodes = {loc.node: (loc.x, loc.y) for loc in locations}
        edges = None
        # if connections are provided, we need to specify the edges
        if connections:
            edges = {
                (l1.node, l2.node): d
                for (l1, l2), d in
                connections.items()
            }
        topology = MultiEdgeUndirectedTopology(nodes, edges)

        # Default constructor
        city = cls(
            name, locations, topology,
            nb_processes=nb_processes, location_cls=location_cls)

        # Call energy algorithm if necessary
        if create_network:
            city.create_connections_by_energy(rng=rng, **kwargs)
        return city

    @classmethod
    def build_random(
        cls, name, distribution, x_lim=(0, 360), y_lim=(-90, 90),
        rng=None, location_cls=Location, create_network=True,
        **kwargs
    ):
        """
        Creates a city with random locations and connections.

        :param str name: name of the city.
        :param distribution: the wanted number of locations (values)
            of a given type (key).
        :type distribution: dict-like object (e.g. :py:class:
            `city_graph.types.LocationDistribution`)

        :param tuple x_lim: Coordinates range on the x-axis.
        :param tuple y_lim: Coordinates range on the y-axis.
        :param rng: Random number generator.
        :type rng: :py:class:`RandomGenerator<city_graph.utils.RandomGenerator>`

        :param cls location_cls: class used for representing a
            location.
        :param int create_network: whether to run create connections
            using the energy algorithm.
        :param dict kwargs: additional kwargs to pass to the energy
            algorithm.

        :note: The `location_cls` default constructor must accept
            as input arguments: * a location type * a 2-tuple of
            random values between ``x_lim`` and ``y_lim`` :note: The
            location coordinates (longitude and latitude) should be
            accessible through its attributes ``x`` and ``y``.
        """
        rng = rng or RandomGenerator()

        # One element (the type) for each location in a flattened list
        types_list = ([k] * v for k, v in distribution.items() if v > 0)
        types_list = (t for sublist in types_list for t in sublist)

        # Create random locations
        locations = (
            location_cls(t, (rng.uniform(*x_lim), rng.uniform(*y_lim)))
            for t in types_list
        )

        # Call builder
        return cls.build_from_data(
            name, locations, rng=rng,
            location_cls=location_cls,
            create_network=create_network, **kwargs)

    def create_connections_by_energy(
            self, connection_types=(TransportType.ROAD, TransportType.WALK),
            connections_per_step=10, max_iterations=None,
            degree_factor=None, distance_factor=None, rng=None):
        """
        Create random connections of given types using an energy
        sampling mechanism.

        At each iteration, a number of new random connections are
        created based on the degree and the distance between
        locations.  Locations are more likely to be connected if *
        they are close * they are already connected to other locations

        :param connection_types: Types of the connections to add.
        :type connection_types: str or iterable

        :param int connections_per_step: Number of new connections
            created at each step.
        :param int max_iterations: Maximum number of iterations.  If
            None, the algorithm stops when each location has been
            connected at least once.
        :param float degree_factor: Multiplier applied to the degree
            enery component during the search for new connections
            (higher means more prominent).
        :param float distance_factor: Multiplier applied to the
            distance energy component during the search for new
            connections (higher means more prominent).
        :param rng: Random number generator.
        :type rng: :py:class:`RandomGenerator<city_graph.utils.RandomGenerator>`
        """
        rng = rng or RandomGenerator()

        # Normalize the factors if necessary
        if degree_factor and distance_factor:
            sum_factors = degree_factor + distance_factor
            degree_factor /= sum_factors
            distance_factor /= sum_factors

        # Make edge_types iterable if it is not
        if not isinstance(connection_types, list):
            connection_types = list(connection_types)

        # Check configuration for running the algorithm
        if connections_per_step < 1:
            raise RuntimeError("Samples must have a minimum size of 1.")

        # Run algorithm
        self._topology.add_energy_based_edges(
            connection_types, connections_per_step, max_iterations,
            degree_factor, distance_factor, rng)

    def create_central_connections(
        self, connection_types,
        num_central_locations=10, rng=None
    ):
        """
        Create random connections of given types between central
        locations.  These central locations are determined by a
        clustering algorithm.

        :param connection_types: Types of the connections to add.
        :type connection_types: str or iterable

        :param int num_central_locations: Number of requested central
            locations.
        :param rng: Random number generator.
        :type rng: :py:class:`RandomGenerator<city_graph.utils.RandomGenerator>`
        """
        rng = rng or RandomGenerator()

        # Make edge_types iterable if it is not
        if not isinstance(connection_types, list):
            connection_types = list(connection_types)

        # Run algorithm
        self._topology.add_edges_between_centroids(
            connection_types, num_central_locations, rng)

    def __getstate__(self):
        # this is used to inform pickle which attributes should
        # be saved. These are all the attributes, except _pool,
        # which can not be pickled (see city_io module)
        state = {attr: getattr(self, attr)
                 for attr in self.__slots__}
        del state["_pool"]
        return state

    def __setstate__(self, state):
        # this will be used by pickle to recreate the instance.
        # according to the __getstate__ method, all attributes
        # are saved, except for _pool, that we reinstantiate here
        # (see city_io module)
        for attr, value in state.items():
            setattr(self, attr, value)
        self._pool = multiprocessing.Pool(processes=self._nb_processes)

    def _next_plan_id(self):
        # attributing an id to each new plan
        self._plan_id += 1
        return self._plan_id

    def get_distance(self, location1, location2):
        """
        Returns the "as the crows fly" distance between
        location1 and location2, given that location1 and location2
        are part of the city. Distance in meters.
        """
        return self._locations_manager.get_distance(location1, location2)

    def get_locations(self, location_types=LocationType):
        """
        Returns an iterator over the locations hosted by the city,
        possibly filtering by location type

        :param location_types: either a:
            :py:class:`LocationType<city_graph.types.LocationType>`, or an interable
            of(default: all types available)

        :returns: All the locations of the given type(s)
        :rtype: iter(:py:class:`LocationType<city_graph.types.LocationType>`)
        """
        if isinstance(location_types, str):
            location_types = [location_types]
        for location_type in location_types:
            for location in self._locations_manager.get_locations(location_type):
                yield location

    def get_locations_by_types(self, location_types):
        """
        Return dictionary, which keys are a location type, and the
        values lists of the location of that type.

        :param location_types: locations types
        :type location_types: iter(:py:class:`LocationType<city_graph.types.LocationType>`)

        :returns: A dictionary {type: list of Location instances}
        """
        return self._locations_manager.get_locations(location_types)

    def get_location_types(self):
        """
        Returns the set of all location types having at least one
        representative location in the city
        """
        return self._locations_manager.location_types

    def compute_distances(self):
        """
        Compute and store the pairwise distances between all known
        locations.  This will make calls to :py:meth:`.get_closest`
        faster.  This method can also be called before calls to
        :py:meth:`save<city_graph.city_io.save>` so that cities may be
        reloaed with pre - computed distances.
        """
        self._locations_manager.compute_all_distances()

    def get_closest(self, location, location_types=None,
                    excluded_locations=[]):
        """
        Returns the location the closest to location, which is of one
        of the specified authorized type, and not excluded.  Note:
        this does not return the closest location in terms of shortest
        path.  Uses distances as the crows fly.

        :param location: target location
        :param location_types: list of authorized type
        :param excluded_locations: list of excluded locations
            (default: empty list)

        :returns: the location that is the closest to location (which
                  is of the authorized type and not excluded)
        """
        return self._locations_manager.get_closest(location, location_types,
                                                   excluded_locations)

    def _blocking_plan(self,
                       start,
                       target,
                       preferences):
        # blocking function that compute a plan
        # (in the main process)
        plan = get_plan(self._topology,
                        start,
                        target,
                        preferences)
        plan = self._from_nodes_to_locations(plan)
        return plan

    def _non_blocking_plan(self,
                           start,
                           target,
                           preferences):
        # non blocking function that compute a plan in a separate process,
        # and returns a corresponding plan_id.
        # dev notes : assumes self._graph is threadsafe as far the planning
        # algorithm goes. networkx does not guaranty threadsafe operations,
        # but seems to be fine in practice. But has to be tested on bigger graphs.
        # Some (reasonable) refactor would be required to work on hard copies
        # of graph
        result = self._pool.apply_async(
            get_plan, args=(self._topology, start, target, preferences))
        plan_id = self._next_plan_id()
        self._plans[plan_id] = result
        return plan_id

    def request_plan(self, start, target, preferences, blocking=True):
        """
        Requests a plan, i.e. the preferred path between start and
        target, taking the transportation preferences modes into
        consideration.  If blocking is True (the default), this
        function blocks and returns plan once computed.  If blocking
        is False, this function requests a separate process to compute
        the plan, and returns immediately a plan_id.  This plan_id can
        be used to query the state of the
        computation(:py:meth:`is_plan_ready`) and retrieve the plan
        once the computation is terminated(:py:meth:`get_plan`).

        :param start: starting location
        :type start: :py:class:`Location<city_graph.types.Location>`

        :param target: target location
        :type target: :py:class:`Location<city_graph.types.Location>`

        :param preferences: preferrence of each transportation mode
        :type preferences: :py:class:`Preferences<city_graph.types.Preferences>`

        :param bool blocking: True if a plan should be computed then
            returned, False if the function should return immediately
            a job id

        :returns: A plan (blocking is True) or a job id (blocking is False)
        :rtype: :py:class:`Plan<city_graph.planning.Plan>` or int
        """

        # run shortest path and return the plan
        if blocking:
            return self._blocking_plan(start, target, preferences)

        # send the request to a pool of processes,
        # which returns immediately a plan_id allowing to get the
        # results later on once computation done
        plan_id = self._non_blocking_plan(start, target, preferences)
        return plan_id

    def _serial_compute_plans(self, requests):
        # compute all the plans corresponding to the requests,
        # one by one, on the main process
        plans = [self.request_plan(*request) for request in requests]
        return plans

    def _multiprocesses_compute_plans(self, requests):
        # compute all the plans corresponding to the requests,
        # using the pool of processes, and wait for the completion
        # of all computation.
        results = [
            self._pool.apply_async(get_plan, args=(self._topology, *request))
            for request in requests
        ]
        plans = [self._from_nodes_to_locations(r.get()) for r in results]
        return plans

    def compute_plans(self, requests):
        """
        Compute a plan for each request.  A request is a tuple (start
        location, target_location, preferences).  See
        :py:class:`Location<city_graph.types.Location>`,
        :py:class:`Preferences<city_graph.types.Preferences>`.

        :param requests: An iterable of requests, i.e tuple(start
            location, target_location, preferences)

        :returns: A list of plans with the same ordering as the requests
        :rtype: list(:py:class:`Plan<city_graph.planning.Plan>`)
        """
        # non multiprocess computation
        if self._nb_processes <= 1 or len(requests) == 1:
            return self._serial_compute_plans(requests)
        # multiprocess computation
        return self._multiprocesses_compute_plans(requests)

    def is_plan_ready(self, plan_id):
        """
        Returns True if the plan corresponding to the passed id is finished,
        False otherwise.
        See: py: meth: `.request_plan`.

        :raises: :py:class:`KeyError` if plan_id is unknown
        """
        result = self._plans[plan_id]
        return result.ready()

    def are_plans_ready(self, plan_ids):
        """
        Returns True if all the plans ready.
        False otherwise.
        See :py:meth:`request_plan`.

        :param: plan_ids, an iterable of plan ids
        :returns: True if all plans ready, False otherwise
        :raises: :py:class:`KeyError` if any invalid plan_id
        """
        return all([self.is_plan_ready(plan_id) for plan_id in plan_ids])

    def retrieve_plan(self, plan_id):
        """
        Returns the corresponding plan if its computation is finished,
        None otherwise.  If the passed id does not correspond to a
        planning job, a KeyError exception is raised.  See
        :py:meth:`request_plan`.
        """
        result = self._plans[plan_id]
        if not result.ready():
            return None
        plan = result.get()
        self._from_nodes_to_locations(plan)
        del self._plans[plan_id]
        return plan

    def retrieve_plans(self, plan_ids):
        """
        Waits until completion of all related computation and then
        returns for each id the corresponding plan.

        :param requests_id: an iterable of ids as returned by
            :py:meth:`request_plan`.

        :returns: a dictionary {plan_id: :py:class:`Plan<city_graph.planning.Plan>`}
        """
        plans = {}
        for plan_id in plan_ids:
            try:
                result = self._plans[plan_id]
            except KeyError:
                result = None
            if result:
                plan = result.get()
                self._from_nodes_to_locations(plan)
                plans[plan_id] = plan
        return plans

    def _from_nodes_to_locations(self, plan):
        """
        Reframe the plan in terms of locations.

        The plan as returned by `retrieve_plans` is constructed in
        terms of topological nodes.  The user is cearly more
        interested in the locations that they might see along the way.
        This function looks up the locations corresponding to the
        nodes and puts them into the plan.
        """
        if not plan.is_valid():
            return plan

        for step in plan.steps():
            start_locations = self._locations_by_node.get(step.start)
            target_locations = self._locations_by_node.get(step.target)

            if not start_locations:
                node_data = self._topology.nodes[step.start]
                lon = node_data[self._topology.NODE_LONG]
                lat = node_data[self._topology.NODE_LAT]
                start_locations = [
                    self._location_cls(
                        LocationType.NONE,
                        node=step.start,
                        coordinates=Point(lon, lat))
                ]
            if not target_locations:
                node_data = self._topology.nodes[step.start]
                lon = node_data[self._topology.NODE_LONG]
                lat = node_data[self._topology.NODE_LAT]
                target_locations = [
                    self._location_cls(
                        LocationType.NONE,
                        node=step.target,
                        coordinates=Point(lon, lat))
                ]

            step.start = tuple(start_locations)
            step.target = tuple(target_locations)

        return plan

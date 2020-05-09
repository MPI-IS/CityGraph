from itertools import combinations_with_replacement
import multiprocessing

from .coordinates import GeoCoordinates
from .planning import get_plan, Plan
from .topology import MultiEdgeUndirectedTopology
from .types import LocationType, Location
from .utils import RandomGenerator


class LocationManager:
    """
    Helper class that manages locations.

    :param iter locations: Locations
    :param func fdistance: Function used to calculate distances
    """

    __slots__ = ("_distances", "_locations", "_func_distance", "_locations_by_node")

    def __init__(self, locations, fdistance=GeoCoordinates.distance):

        # Function used to calculate distances
        self._func_distance = fdistance

        # saving all distances computation, so they not done twice
        self._distances = {}

        # mapping between nodes and locations
        self._locations_by_node = {}

        # Save locations by type
        self._locations = {}
        for loc in locations:

            # Save location by node
            try:
                self._locations_by_node[loc.node].append(loc)
            except KeyError:
                self._locations_by_node[loc.node] = [loc]

            # Save location by type
            try:
                self._locations[loc.location_type].append(loc)
            except KeyError:
                self._locations[loc.location_type] = [loc]

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
            * retrieving it from the _distances attributes(if previously computed)
            * by computing it (and then saving it in _distances)
        : todo: lru_cache here?
        """
        try:
            return self._get_distance(l1, l2)
        except KeyError:
            d = self._func_distance(l1, l2)
            self._distances[(l1, l2)] = d
            return d

    def _get_distance(self, l1, l2):
        """Try to get the distance between two locations by flipping them."""
        try:
            return self._distances[(l1, l2)]
        except KeyError:
            return self._distances[(l2, l1)]

    # fills _distances will all distances values, so no distance
    # will ever need to be computed again.
    # TODO: JC I do not think this is needed
    def compute_all_distances(self):
        """Calculate and save all distances."""

        # Itertools is so cool!
        for l1, l2 in combinations_with_replacement(sum(self._locations.values(), []), 2):
            self.get_distance(l1, l2)

    # returns the location of one of the specified type
    # closest to location
    # TODO: fix implementation and test
    def get_closest(self, location, location_types):
        # Right now we raise an error
        raise NotImplementedError

        if location_types is None:
            locations = self._locations
            index = 1
            # if there is one entry, then it is location,
            # so the method would return itself
            if len(locations) <= 1:
                return None
        else:
            if isinstance(location_types, str):
                location_types = [location_types]
            locations = []
            for location_type in location_types:
                locations += self.get_locations(location_type)
            # index of the closest location in the sorted array
            index = 0
            # location is part of the candidate location ...
            if location.location_type in location_types:
                # so index 0 will be location, so returning the next one
                index = 1
                if len(locations) < 2:
                    return None
            elif len(locations) < 1:
                # there is only one entry in the list, so location itself
                return None
        return sorted(locations,
                      key=lambda l: self.get_distance(l, location))[index]


class City:
    """
    Class for representing a city.

    :param str name: name of the city
    :param rng: Random number generator.
    :type rng: :py:class: `.RandomGenerator`
    :param int nb_processes: number of processes to use(when computing shortest paths)
    """

    __slots__ = ("name", "rng", "_topology", "_pool", "_nb_processes",
                 "_plans", "_plan_id", "_locations_manager")

    def __init__(self, name, rng=None, nb_processes=1):

        self.name = name
        self._nb_processes = nb_processes
        self.rng = rng or RandomGenerator()

        # pool of processes in charge of computing plans
        self._pool = multiprocessing.Pool(processes=nb_processes)

        # key: plan_id, value: result(future) of the plan,
        # as provided by the process in charge of computing the plan
        self._plans = {}

        # used for attributing a unique new id to each plan
        self._plan_id = 0

    @classmethod
    def build_from_data(cls, name, locations, connections=None, rng=None,
                        node_cls=GeoCoordinates):
        """
        Creates a city with locations and connections provided as input.

        :param str name: name of the city
        :param iter locations: locations
        :param dict connections: connections between a pair of locations (key) with data (value)
        :param rng: Random number generator.
        :type rng: :py:class: `.RandomGenerator`
        :param cls node_cls: Class used for representing a node

        :note: The `node_cls` default constructor must accept a location as input argument.
        """
        rng = rng or RandomGenerator()

        # Default constructor
        city = cls(name, rng=rng)

        # Build nodes if necessary
        for loc in locations:
            if not loc.node:
                loc.node = node_cls(loc)

        # All locations should have a node now
        assert all([hasattr(loc, 'node') for loc in locations])

        # Create the LocationManager
        city._locations_manager = LocationManager(locations)

        # Create the Topology
        nodes = (l.node for l in locations)
        edges = None
        # if connections are provided, we need to specify the edges
        if connections:
            edges = {(l1.node, l2.node): _ for (l1, l2), _ in connections.items()}
        city._topology = MultiEdgeUndirectedTopology(nodes, edges)
        return city

    @classmethod
    def build_random(cls, name, distribution, rng=None, x_lim=(0, 360), y_lim=(-90, 90),
                     location_cls=Location, node_cls=GeoCoordinates):
        """
        Creates a city with random locations and connections.

        :param str name: name of the city
        :param dict distribution: distribution of locations by type
        :param rng: Random number generator.
        :param tuple x_lim: Coordinates range on the x-axis
        :param tuple y_lim: Coordinates range on the y-axis
        :param cls location_cls: Class used for representing a location
        :param cls node_cls: Class used for representing a node

        :note: The `location_cls` default constructor must accept a location type
            and a 2-tuple as input arguments.
        :note: The `node_cls` default constructor must accept an intance
            of type `location_cls` as input argument.
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

        # Call builder - edges are still missing
        city = cls.build_from_data(name, locations, rng=rng, node_cls=node_cls)
        return city

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

    def get_locations(self, location_types=LocationType):
        """
        Returns an iterator over the locations hosted by the city, possibly filtering by
        location type

        : param location_types: either a: py: class: `city_graph.types.LocationType`,
             or an interable of(default: all types available)

        : returns: An iterator of: py: class: `city_graph.types.Location` instances.
        """
        if isinstance(location_types, str):
            location_types = [location_types]
        for location_type in location_types:
            for location in self._locations_manager.get_locations(location_type):
                yield location

    def get_locations_by_types(self, location_types):
        """
        Return dictionary, which keys are a location type, and the values
        lists of the location of that type.

        : param location_types: an iterator of: py: class: `types.LocationType`

        : returns: A dictionary {type: list of Location instances}
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
        Compute and store the pairwise distances between all known locations.
        This will make calls to: py: meth: `.get_closest` faster.
        This method can also be called before calls to: py: meth: `city_graph.city_io.save`
        so that cities may be reloaed with pre - computed distances.
        """
        self._locations_manager.compute_all_distances()

    def get_closest(self, location, location_types=None):
        """
        Returns the closest location. Note: this method will reuse pre - computed
        distances if: py: meth: `.compute_distances` has been previously called.
        Limitation: this method uses the coordinates of the position, and is
        not based on shortest path.

        : param obj location: the location
        : param location_types: the location_type or a list of location_types
               used to generate the list of candidate locations(default: None, i.e.
               any type)
        : returns: the closest location of one of the specified type, or an
               None of no such location exists.
        """
        return self._locations_manager.get_closest(location, location_types)

    def _blocking_plan(self,
                       start,
                       target,
                       preferences):
        # blocking function that compute a plan
        # (in the main process)
        return get_plan(self._topology,
                        start,
                        target,
                        preferences)

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
            get_plan, args=(
                self._topology, start, target, preferences,))
        plan_id = self._next_plan_id()
        self._plans[plan_id] = result
        return plan_id

    def request_plan(self,
                     start,
                     target,
                     preferences,
                     blocking=True):
        """
        Requests a plan, i.e. the preferred path between start and target,
        taking the transportation preferences modes into consideration.
        If blocking is True (the default), this function blocks and returns
        plan once computed. If blocking is False, this function requests a separate
        process to compute the plan, and returns immediately a plan_id. This plan_id can be
        used to query the state of the computation(: py: meth: `.is_plan_ready`) and retrieve
        the plan once the computation is terminated(: py: meth: `.get_plan`).

        : param start: starting location
        : type start:: py: class: `city_graph.types.Location`

        : param target: target location
        : type target:: py: class: `city_graph.types.Location`

        : param preferences: preferrence of each transportation mode
        : type preferences:: py: class: `city_graph.types.Preferences`

        : param bool blocking: True if a plan should be computed then returned,
            False if the function should return immediately a job id

        : returns: A plan(blocking is True) or a job id(blocking is False)
        : rtype:: py: class: `city_graph.planning.Plan` or int
        """

        # run shortest path and return the plan
        if blocking:
            return self._blocking_plan(start,
                                       target,
                                       preferences)

        # send the request to a pool of processes,
        # which returns immediately a plan_id allowing to get the
        # results later on once computation done
        plan_id = self._non_blocking_plan(start, target,
                                          preferences)
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
            self._pool.apply_async(
                get_plan,
                args=(
                    self._topology,
                    *request,)) for request in requests]
        plans = [r.get() for r in results]
        return plans

    def compute_plans(self, requests):
        """
        Compute a plan for each request. A request is a tuple
        (start location, target_location, preferences).
        See: py: class: `city_graph.types.Location`, : py: class: `city_graph.types.Preferences`.

        : param requests: An iterable of requests,
            i.e tuple(start location, target_location, preferences)
        : returns: A list of plans(: py: class: `.planning.Plan`),
            with the same ordering as the requests

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

        : raises:: py: class: `KeyError`: if an unknown plan_id
        """
        result = self._plans[plan_id]
        return result.ready()

    def are_plans_ready(self, plan_ids):
        """
        Returns True if all the plans ready.
        False otherwise.
        See: py: meth: `.request_plan`.

        : param: plan_ids, an iterable of plan ids
        : returns: True if all plans ready, False otherwise
        : raises:: py: class: `KeyError`: if any invalid plan_id
        """
        return all([self.is_plan_ready(plan_id)
                    for plan_id in plan_ids])

    def retrieve_plan(self, plan_id):
        """
        Returns the corresponding plan if its computation is finished, None otherwise.
        If the passed id does not correspond to a planning job,
        a KeyError exception is raised.
        See: py: meth: `.request_plan`.
        """
        result = self._plans[plan_id]
        if not result.ready():
            return None
        plan = result.get()
        del self._plans[plan_id]
        return plan

    def retrieve_plans(self, plan_ids):
        """
        Waits until completion of all related computation and
        then returns for each id the corresponding plan.

        : param requests_id: an iterable of ids as returned by: py: meth: `.request_plan`.
        : returns: a dictionary {plan_id:: py: class: `.planning.Plan`}
        """
        plans = {}
        for plan_id in plan_ids:
            try:
                result = self._plans[plan_id]
            except KeyError:
                result = None
            if result:
                steps = result.get()
                plan = Plan()
                if steps is not None:
                    plan.set_steps(steps)
                plans[plan_id] = plan
        return plans

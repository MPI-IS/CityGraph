import multiprocessing
import logging
import time
import atexit
import copy
import math
import queue
from addict import Dict
from functools import partial
from . import planning
from . import types

# helper class that manages distances between locations.
# will be used as attribute of class City.


class _LocationManager:

    __slots__ = ("_distances", "_locations", "_type_locations")

    def __init__(self, locations):
        # saving all distances computation, so they not done twice
        self._distances = Dict()
        self._locations = locations
        types = set([l.location_type
                     for l in locations])
        # also saving dict {location types: locations}, also
        # for saving computation later on
        self._type_locations = {t: [] for t in types}
        for location in locations:
            self._type_locations[location.location_type].append(location)

    # return a set of all known location types
    def get_location_types(self):
        return set(self._type_locations.keys())

    # return {location_type:[locations]}
    def get_locations_by_types(self, location_types):
        return {type_: self._type_locations[type_]
                for type_ in location_types
                if type_ in self._type_locations}

    # return list of locations of specified type
    def get_locations(self, location_type):
        try:
            return self._type_locations[location_type]
        except KeyError:
            return []

    # returns {location:{location:distance}}
    def get_all_distances(self):
        return self._distances

    # returns distance between l1 and l2,
    # by either retrieving it from _distances
    # (if previously computed) or by computing
    # it (and then saving it in _distances)
    def get_distances(self, l1, l2):
        d = self._distances[l1][l2]
        if isinstance(d, float) or len(d) > 0:
            return d
        d = l1.distance(l2)
        self._distances[l1][l2] = d
        self._distances[l2][l1] = d
        return d

    # returns the location of one of the specified type
    # closest to location
    def get_closest(self, location, location_types):
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
                      key=lambda l: self.get_distances(l, location))[index]

    # fills _distances will all distances values, so no distance
    # will ever need to be computed again.
    def compute_all_distances(self):
        for index, l1 in enumerate(self._locations):
            for l2 in self._locations[index + 1:]:
                d = l1.distance(l2)
                self._distances[l1][l2] = d
                self._distances[l2][l1] = d


class City:

    """
    User-end interface for querying a city

    :param str name: name of the city
    :param topology: topology of the city
    :param int nb_processes: number of processes to use (when computing shortest paths)
    """

    __slots__ = ("_name", "_topology", "_pool", "_nb_processes",
                 "_plans", "_plan_id", "_locations_manager")

    def __init__(self, name, topology, nb_processes=1):
        # arbitrary name provided by the user
        self._name = name
        # topology object, allows to generate plan to go from
        # one location to the other
        self._topology = topology
        # number of processes spawn for plan generation
        self._nb_processes = nb_processes
        # pool of processes in charge of computing plans
        self._pool = multiprocessing.Pool(processes=nb_processes)
        # key: plan_id, value: result(future) of the plan,
        # as provided by the process in charge of computing the plan
        self._plans = {}
        # used for attributing a unique new id to each plan
        self._plan_id = 0
        # used to order the locations
        self._locations_manager = _LocationManager(list(self.get_locations()))

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

    def get_name(self):
        """Returns the name of the city"""
        return self._name

    def _next_plan_id(self):
        # attributing an id to each new plan
        self._plan_id += 1
        return self._plan_id

    def get_locations(self, location_types=None):
        """
        Returns an iterator over the locations hosted by the city, possibly filtering by
        location type

        :param location_types: either a :py:class:`city_graph.types.LocationType`,
             or an interable of (default:None)

        :returns: An iterator of :py:class:`city_graph.types.Location` instances.
        """
        if location_types is None:
            for n in self._topology.get_nodes():
                yield n
            return
        if isinstance(location_types, str):
            location_types = [location_types]
        for location_type in location_types:
            for location in self._locations_manager.get_locations(
                    location_type):
                yield(location)

    def get_locations_by_types(self, location_types):
        """
        Return dictionary, which keys are a location type, and the values
        lists of the location of that type.

        :param location_types: an iterator of :py:class:`types.LocationType`

        :returns: A dictionary {type:list of Location instances}
        """
        return self._locations_manager.get_locations_by_types(location_types)

    def get_location_types(self):
        """
        Returns the set of all location types having at least one
        representative location in the city
        """
        return self._locations_manager.get_location_types()

    def compute_distances(self):
        """
        Compute and store the pairwise distances between all known locations.
        This will make calls to :py:meth:`.get_closest` faster.
        This method can also be called before calls to :py:meth:`city_graph.city_io.save`
        so that cities may be reloaed with pre-computed distances.
        """
        self._locations_manager.compute_all_distances()

    def get_closest(self, location, location_types=None):
        """
        Returns the closest location. Note: this method will reuse pre-computed
        distances if :py:meth:`.compute_distances` has been previously called.
        Limitation: this method uses the coordinates of the position, and is
        not based on shortest path.

        :param obj location: the location
        :param location_types: the location_type or a list of location_types
               used to generate the list of candidate locations (default: None, i.e.
               any type)
        :returns: the closest location of one of the specified type, or an
               None of no such location exists.
        """
        return self._locations_manager.get_closest(location, location_types)

    def _blocking_plan(self,
                       start,
                       target,
                       preferences):
        # blocking function that compute a plan
        # (in the main process)
        return planning.get_plan(self._topology,
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
            planning.get_plan, args=(
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
        used to query the state of the computation (:py:meth:`.is_plan_ready`) and retrieve
        the plan once the computation is terminated (:py:meth:`.get_plan`).

        :param start: starting location
        :type start: :py:class:`city_graph.types.Location`

        :param target: target location
        :type target: :py:class:`city_graph.types.Location`

        :param preferences: preferrence of each transportation mode
        :type preferences: :py:class:`city_graph.types.Preferences`

        :param bool blocking: True if a plan should be computed then returned,
            False if the function should return immediately a job id

        :returns: A plan (blocking is True) or a job id (blocking is False)
        :rtype: :py:class:`city_graph.planning.Plan` or int
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
                planning.get_plan,
                args=(
                    self._topology,
                    *request,)) for request in requests]
        plans = [r.get() for r in results]
        return plans

    def compute_plans(self, requests):
        """
        Compute a plan for each request. A request is a tuple
        (start location, target_location, preferences).
        See :py:class:`city_graph.types.Location`, :py:class:`city_graph.types.Preferences`.

        :param requests: An iterable of requests,
            i.e tuple (start location, target_location, preferences)
        :returns: A list of plans (:py:class:`.planning.Plan`),
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
        See :py:meth:`.request_plan`.

        :raises: :py:class:`ValueError`: if an unknown plan_id
        """
        try:
            result = self._plans[plan_id]
        except BaseException:
            raise ValueError()
        return result.ready()

    def are_plans_ready(self, plan_ids):
        """
        Returns True if all the plans ready.
        False otherwise.
        See :py:meth:`.request_plan`.

        :param: plan_ids, an iterable of plan ids
        :returns: True if all plans ready, False otherwise
        :raises: :py:class:`ValueError`: if any invalid plan_id
        """
        return all([self.is_plan_ready(plan_id)
                    for plan_id in plan_ids])

    def retrieve_plan(self, plan_id):
        """
        Returns the corresponding plan if its computation is finished, None otherwise.
        If the passed id does not correspond to a planning job,
        a ValueError exception is raised.
        See :py:meth:`.request_plan`.
        """
        try:
            result = self._plans[plan_id]
        except BaseException:
            raise ValueError()
        if not result.ready():
            return None
        plan = result.get()
        del self._plans[plan_id]
        return plan

    def retrieve_plans(self, plan_ids):
        """
        Waits until completion of all related computation and
        then returns for each id the corresponding plan.

        :param requests_id: an iterable of ids as returned by :py:meth:`.request_plan`.
        :returns: a dictionary {plan_id: :py:class:`.planning.Plan`}
        """
        plans = {}
        for plan_id in plan_ids:
            try:
                result = self._plans[plan_id]
            except BaseException:
                result = None
            if result:
                steps = result.get()
                plan = planning.Plan()
                if steps is not None:
                    plan.set_steps(steps)
                plans[plan_id] = plan
        return plans

import time
from .types import TransportType, AVERAGE_SPEEDS


class PlanStep:
    """
    A plan step represents a segment in a bigger plan for going from a
    starting location to a target location (a plan is a list of
    successive plan steps, in which the target location of a step is
    the starting location of the next step).

    :param start: starting location of the step
    :type start: :py:class:`Location<city_graph.types.Location>`
    :param target: target location of the step
    :type target: :py:class:`Location<city_graph.types.Location>`
    :param mode: transportation mode used
    :type mode: :py:class:`TransportType<city_graph.types.TransportType>`
    :param float duration: expected duration for going from start to
        target uing the selected transport type
    """

    def __init__(self, start, target, mode, duration):
        self.start = start
        self.target = target
        self.mode = mode
        self.duration = duration

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        attrs = [
            attr
            for attr in vars(self)
            if attr != "start" and attr != "target"
        ]
        if any([getattr(self, attr) != getattr(other, attr) for attr in attrs]):
            return False
        return True

    def __str__(self):
        output = "PlanStep:\n"
        output += "\tmode: {}\n".format(self.mode)
        start = (
            set(str(loc) for loc in self.start)
            if isinstance(self.start, list)
            else str(self.start))
        target = (
            set(str(loc) for loc in self.target)
            if isinstance(self.target, list)
            else str(self.target))
        output += "\tstart: {}\n".format(start)
        output += "\ttarget: {}\n".format(target)
        return output


class Plan:
    """
    A valid plan is a path for going from a start to a target location. A plan
    consists of an ordered list of :py:class:`.PlanStep` , each plan step
    expliciting how sub-targets are reached.
    An invalid plan is the result of the planner failing to find a suitable
    path between the start and target locations.
    Instances of Plan are returned by planning functions,
    see :py:class:`City<city_graph.city.City>`.

    :param steps: list of plan steps
    :type steps: list(:py:class:`.PlanStep`)
    :param float score: score of the plan, as computed by the shortest path algorithm
    """

    __slots__ = (
        "_steps", "_valid", "_error", "_score",
        "_start_time", "_average_speeds")

    def __init__(self, steps=None, score=None):
        self.set_steps(steps)
        self._error = None
        self._score = score
        self._start_time = None
        self._average_speeds = AVERAGE_SPEEDS

    def set_average_speed(self, speeds):
        """
        Set the average speed for each transportation type.
        If this function is not called, then the default average
        speed (see :py:const:`AVERAGE_SPEEDS<city_graph.types.AVERAGE_SPEEDS>`).
        Average speed are used to compute the current position
        as returned by :py:meth:`where<.Plan.where>`.
        See :py:class:`TransportType<city_graph.types.TransportType>`.

        :param speeds: dictionary {TransportType:speed in meters per seconds}
        """
        self._average_speeds = speeds

    def _start(self, current_time):
        # Set a time stamp (in seconds) at which this plan starts to be executed.
        # This starting time will be used as reference for computing the position
        # of the person executed then plan (see :py:meth:`.Plan.where`)
        # current_time: current time in seconds
        if current_time is None:
            self._start_time = time.time()
        else:
            self._start_time = current_time

    def where(self, current_time):
        """
        Returns the current location of a person executing the plan,
        i.e. the location of a person going through all the steps of the plan
        moving at the average speed of the transport type used.
        The plan is initialized at the first call to this function.
        See :py:meth:`set_average_speed<.Plan.set_average_speed>`.

        :param int current_time: current time, in seconds. If None,
            the current machine time is used.

        :returns: tuple (finished,(location1,location2,TransportType)), finished being
            None if the plan is not finished yet or a duration (in second) indicating for how
            long the plan has been finished. (Location1, location2, TransportType) indicates the
            current road (or the last road taken in case of finished plan)

        :raises:
            :py:class:`ValueError` if attempting to query a status
            prior to the plan starting time

        """
        if self._start_time is None:
            self._start(current_time)
        if current_time is None:
            current_time = time.time()

        relative_time = current_time - self._start_time
        if relative_time < 0:
            raise ValueError(
                "PlanStep: trying to get position at a time "
                "prior to the plan starting time")

        d = 0
        for step in self._steps:
            d += step.duration
            if relative_time < d:
                return False, (step.start, step.target, step.mode)

        step = self._steps[-1]
        return relative_time - d, (step.start, step.target, step.mode)

    @property
    def score(self):
        return self._score

    def is_valid(self):
        """
        :returns: True if this plan is valid (does provide a suitable plan to go
            from the start to the target location), False otherwise (i.e. the planner
            failed to find a suitable plan).
        """
        return self._valid

    def get_error(self):
        """
        If the plan is invalid, attemps to return the corresponding error (as str).

        :returns: error (str)
        :raises: :py:class:`ValueError` if there is no error message
        """
        try:
            return self._error
        except BaseException:
            raise ValueError()

    def set_error(self, error):
        """
        Set an error message
        """
        self._error = str(error)

    def set_steps(self, steps):
        """
        set the lists of plan steps defining this plan

        :param steps: list of steps
        :type steps: list(:py:class:`.PlanStep`)
        """
        self._steps = steps
        if steps is None:
            self._valid = False
        else:
            self._valid = True

    def steps(self):
        """
        Iterator over the :py:class:`.PlanStep` defining the plan.

        :raises: :py:class:`ValueError` if the plan is invalid
        """
        if not self._valid:
            raise ValueError
        for step in self._steps:
            yield step

    def __eq__(self, other):
        if self._valid != other._valid:
            return False
        if len(self._steps) != len(other._steps):
            return False
        return all([sa == sb for sa, sb
                    in zip(self.steps(), other.steps())])

    def __str__(self):
        if self._valid:
            return "Plan\n----\n" + "\n".join([str(plan_step)
                                               for plan_step in self._steps])
        return "Plan\n---\ninvalid plan"


def get_plan(topology,
             start_location,
             target_location,
             preferences):
    """
    Compute ap plan that describes how to go from the start to the target location
    under the provided user preferences.

    :param start_location: start location
    :type start_location: :py:class:`Location<city_graph.types.Location>`
    :param target_location: target location
    :type target_location: :py:class:`Location<city_graph.types.Location>`
    :param preferences: user preferences
    :type preferences: :py:class:`Preferences<city_graph.types.Preferences>`

    :returns: the optimal plan if any found, else None
    :rtype: :py:class:`.Plan`
    """

    assert all([mode in TransportType for mode in preferences.mobility])

    try:
        # TODO: get_shortest_path API expected to take the mobility dict instead of
        #       its keys
        # We need to take the nodes here
        score, path, data = topology.get_shortest_path(
            start_location.node, target_location.node, preferences.criterion.value,
            preferences.mobility, preferences.data)
    except RuntimeError as error:
        # No Plan found, returning an invalid Plan
        # (an empty Plan is invalid, i.e. plan.is_valid())
        # dev note : returning an invalid plan, as opposed to raising
        # an expection, makes the multiprocessing computation of plans
        # easier.
        p = Plan()
        p.set_error(str(error))
        return p

    def _get_duration(distance, mode, preferences):
        average_speed = preferences.get_average_speed(mode)
        duration = distance / average_speed
        return duration

    plan_steps = [
        PlanStep(start, target, mode, _get_duration(distance, mode, preferences))
        for (start, target, mode, distance)
        in zip(path, path[1:], data["type"], data["distance"])
    ]

    # adding to the steps the extra attributes (e.g. distance, duration)
    # also returned by the topology
    for attr, values in data.items():
        if attr != "type":
            for plan_step, value in zip(plan_steps, values):
                setattr(plan_step, attr, value)

    # packaging the plan steps into a plan

    plan = Plan(steps=plan_steps, score=score)

    return plan

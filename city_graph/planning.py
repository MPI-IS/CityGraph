from .types import TransportType


class PlanStep:
    """
    A plan step represents a segment in a bigger plan for going from a starting
    location to a target location (a plan is a list of successive plan steps,
    in which the target location of a step is the starting location of the next step).

    :param obj start: starting location of the step (:py:class:`city_graph.types.Location`)
    :param obj target: target location of the step (:py:class:`city_graph.types.Location`)
    :param obj mode: transportation mode used (:py:class:`city_graph.types.TransportType`)
        to go from the start to the target location.
    """

    def __init__(self, start, target, mode):
        self.start = start
        self.target = target
        self.mode = mode

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        attrs = [attr for attr in vars(self) if attr != "start"
                 and attr != "target"]
        if any([getattr(self, attr) != getattr(other, attr)
                for attr in attrs]):
            return False
        return True

    def __str__(self):
        attrs = vars(self)
        s = "PlanStep:\n\t"
        return s + "\n\t".join([attr + ": " + str(getattr(self, attr))
                                for attr in attrs])


class Plan:
    """
    A valid plan is a path for going from a start to a target location. A plan
    consists of an ordered list of :py:class:`.PlanStep` , each plan step
    expliciting how sub-targets are reached.
    An invalid plan is the result of the planner failing to find a suitable
    path between the start and target locations.
    Instances of Plan are returned by planning functions,
    see :py:class:`city_graph.city.City`.

    :param steps: list of :py:class:`.PlanStep`
    :param score: score of the plan, as computed by the shortest path algorithm
    """

    __slots__ = ("_steps", "_valid", "_error", "_score")

    def __init__(self, steps=None, score=None):
        self.set_steps(steps)
        self._error = None
        self._score = score

    @property
    def score(self):
        return self._score

    def set_steps(self, steps):
        """
        set the lists of plan steps defining this plan

        :param steps: list of :py:class:`.PlanStep`
        """
        self._steps = steps
        if steps is None:
            self._valid = False
        else:
            self._valid = True

    def is_valid(self):
        """
        :returns: True if this plan is valid (does provide a suitable plan to go
            from the start to the target location), False otherwise (i.e. the planner
            failed to find a suitable plan).
        """
        return self._valid

    @property
    def steps(self):
        """
        :returns: the ordered list of :py:class:`.PlanStep` (or None for an invalid plan)
        """
        return self._steps

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

    :param obj start_location: start location (:py:class:`city_graph.types.Location`)
    :param obj target_location: target location (:py:class:`city_graph.types.Location`)
    :param obj preferences: user preferences (:py:class:`city_graph.types.Preferences`)

    :returns: :py:class:`.Plan` if a plan is found, None otherwise
    """

    assert all([mode in TransportType for mode in preferences.mobility])

    try:
        # TODO: get_shortest_path API expected to take the mobility dict instead of
        #       its keys
        # We need to take the nodes here
        score, path, data = topology.get_shortest_path(
            start_location.node, target_location.node,
            preferences.criterion.value,
            list(preferences.mobility.keys()),
            preferences.data)
    except RuntimeError as error:
        # No Plan found, returning an invalid Plan
        # (an empty Plan is invalid, i.e. plan.is_valid())
        # dev note : returning an invalid plan, as opposed to raising
        # an expection, makes the multiprocessing computation of plans
        # easier.
        p = Plan()
        p.set_error(str(error))
        return p

    # for each segment in the path, creating a plan step, i.e.
    # start location, end location, transportation mode
    plan_steps = [PlanStep(start, target, mode)
                  for start, target, mode in zip(path, path[1:], data["type"])]

    # adding to the steps the extra attributes (e.g. distance, duration)
    # also returned by the topology
    for attr, values in data.items():
        if attr != "type":
            for plan_step, value in zip(plan_steps, values):
                setattr(plan_step, attr, value)

    # packaging the plan steps into a plan

    plan = Plan(steps=plan_steps, score=score)

    return plan

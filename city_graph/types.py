"""
Types
=====

Module containing custom types and constants needed for the package.
"""
from collections import UserDict
from enum import Enum, unique
from shapely.geometry import Point


@unique
class TransportType(Enum):
    """Type of transport."""

    ROAD = 0
    BIKE = 1
    WALK = 2

    BUS = 3
    FERRY = 4
    SUBWAY = 5
    TRAIN = 6
    TRAM = 7
    TROLLEYBUS = 8


# TODO: move to mapping
#: Average speed (in meters/seconds) of transport types
#: (dictionary, key: TransportType, value: speed in m/s)
AVERAGE_SPEEDS = {
    TransportType.WALK: 1.39,
    TransportType.BIKE: 4.3,
    TransportType.ROAD: 13.8,
    TransportType.BUS: 11.1,
    TransportType.TROLLEYBUS: 11.1,
    TransportType.FERRY: 12.78,
    TransportType.TRAM: 11.1,
    TransportType.TRAIN: 16.66
}


@unique
class MobilityType(tuple, Enum):
    """Type of human mobility."""

    PUBLIC_TRANSPORT = (
        TransportType.WALK,
        TransportType.BUS,
        TransportType.TROLLEYBUS,
        TransportType.FERRY,
        TransportType.TRAIN,
        TransportType.TRAM
    )

    CAR = (TransportType.ROAD, TransportType.WALK)
    BIKE = (TransportType.BIKE, ) + PUBLIC_TRANSPORT
    WALK = (TransportType.WALK, )


def _mobility_to_transport(d):
    """
    This function takes a dictionary in which the keys are either
    members of TransportTyep or MobilityType and converts it into a
    dictionary for in which the keys are all members of TransportType,
    e.g. it deploys PUBLIC_TRANSPORT into each corresponding
    TransportType.

    :param dict d: input dictionary.
    """
    r = {}
    for k, v in d.items():
        if k in MobilityType:
            for tt in k:
                r[tt] = v
        else:
            r[k] = v
    return r


class LocationType(str, Enum):
    """
    Type of location.

    :note: Compiled from amenity, leisure and building type values in
        OpenStreetMap. For the original lists, please refer to:

        * https://wiki.openstreetmap.org/wiki/Key:amenity
        * https://wiki.openstreetmap.org/wiki/Key:leisure
        * https://wiki.openstreetmap.org/wiki/Key:building
    """

    # This is a special case: sometimes we want to convey to the user
    # that there is an important geographical point that does not
    # correspond to a venue. It can be a cross-road or a waypoint
    # along the path. In this case we choose constructing a special
    # non-location with the type below.
    # NOTE: By convention, those are allowed to be only constructed in
    #       runtime and we NEVER put those inside a city.
    NONE = -1

    # Sustenance section: reduced down to two types depending on
    # whether you want to eat or to drink.
    BAR = 0
    RESTAURANT = 1

    # Education section: reduced down to three types based on the age
    # bracket of the student. Colleges are treated as schools.
    KINDERGARDEN = 2
    SCHOOL = 3
    UNIVERSITY = 4

    # Transportation section: only interested in stations and parking
    # lots.
    PUBLIC_TRANSPORT_STATION = 5
    PARKING = 6

    # Financial section: ignored. Banks will be classified as offices.

    # Healthcare: hospitals, doctors and pharmacies. The first two is
    # where the sick people might go (and stay in case of hospital).
    HOSPITAL = 7
    DOCTOR = 8
    PHARMACY = 9

    # Entertainment, Arts & Culture: reducing to gambling, theater
    # (including cinema) and social centre (including community
    # center). Sorry to all the stripclubs, brothels and swinger clubs
    # out there.
    GAMBLING = 10
    NIGHTCLUB = 11
    THEATER = 12
    SOCIAL_CENTER = 13

    # Leisure:
    BEACH = 14
    SPORTS_CENTER = 15
    PARK = 16
    STADIUM = 17

    # Building: everything livable is (including a cabin and a static
    # caravan) is a household, everything religious is a church.
    HOUSEHOLD = 18

    OFFICE = 19
    RETAIL = 20  # for non-essentials
    GROCERY = 21

    CHURCH = 22


class BaseEnumMapping(UserDict):
    """
    Abstract class mapping members of an enumeration to values.
    Values can be set using the enumeration members or their lower - cased names.
    """

    def __init__(self, enum_cls, **kwargs):
        super().__init__()

        # Initialize everything
        self.data = {
            member: kwargs.pop(member.name.lower(), 0)
            for member in enum_cls
        }

    # Cannot delete item
    def __delitem__(self, *args, **kwargs):
        raise RuntimeError('Deleting an item is not allowed.')

    # Cannot add new item
    def __setitem__(self, key, value):

        for m in self.keys():
            if key in (m, m.name.lower()):
                super().__setitem__(m, value)
                return
        # If non-exisiting item, raise exception
        message = 'Adding new item is not allowed, you can only reset one of these:\n - '
        message += '\n - '.join(m.__str__() for m in self.keys())
        raise RuntimeError(message)

    def __repr__(self):
        return "{}:\n - {}".format(
            type(self).__name__,
            '\n - '.join('{} = {}'.format(m.__str__(), self.data[m])
                         for m in self.keys()))


class LocationDistribution(BaseEnumMapping):
    """
    Class representing a distribution of location types.
    The values represent the number of locations for a given type.
    """

    def __init__(self, **kwargs):
        super().__init__(LocationType, **kwargs)


@unique
class PathCriterion(str, Enum):
    """
    Shortest path between two locations can be computed based on either
    duration or distance.
    """
    DURATION = "duration"
    DISTANCE = "distance"


class Preferences:
    """
    Encapsulate the user's preferences for path planning,
    e.g.  his / her relative preferrances for the various transporation modes.

    :param criterion: graph edge attribute used as weight
    :type criterion: :py:class:`.PathCriterion`
    :param dict mobility: preferred transportation modes.
        The keys are members of :py:class:`.MobilityType` and the values are floats
        (the highest the value, the preferred the transportation mode)
    :param dict average_speeds: dictionary relating: :py:class:`.TransportType` to
        related average speed in m/s.
    :param list data: types of data to be extracted during plan computation
        (see: :py:func:`request_plan<city_graph.City.request_plan>`).
    """

    __slots__ = ("_mobility", "_criterion", "_data", "_average_speeds")

    def __init__(self,
                 criterion=PathCriterion.DISTANCE,
                 mobility=None,
                 average_speeds=AVERAGE_SPEEDS,
                 data=["distance"]):

        if criterion not in list(PathCriterion):
            message = str(criterion) + ": "
            message += "Unknown criterion, use a member of the PathCriterion enum:\n\t- "
            message += '\n\t- '.join(t.__str__() for t in PathCriterion)
            raise ValueError(message)

        self._set_mobility(mobility)
        self._criterion = criterion
        self._data = data or []
        self._average_speeds = average_speeds

    def get_average_speed(self, transport_type):
        """
        Returns the average speed for the transportation type

        :param transport_mode: transport mode
        :type transport_mode: :py:class:`.TransportType`

        :raises:
            :py:class:`.KeyError`: if the average speed for the transport type
            has not been set
        """
        return self._average_speeds[transport_type]

    @property
    def average_speeds(self):
        return self._average_speeds

    @property
    def data(self):
        return self._data

    @property
    def criterion(self):
        return self._criterion

    @property
    def mobility(self):
        return self._mobility

    def _set_mobility(self, values):
        if values is not None:
            self._mobility = values
            self._mobility = _mobility_to_transport(self._mobility)
            self._normalize_mobility()

    @mobility.setter
    def mobility(self, values):
        self._set_mobility(values)

    # because it is more intuitive, weights is provided by the user
    # as (for example):
    # {"walk":0.9,
    #  "train":0.1}
    # to indicated walk is (much) preferred.
    # The below transforms this to
    # {"walk":0.1,
    #  "train":0.9}
    # as when computing shortest path,
    # paths for walking will have lower weights

    def _normalize_mobility(self):
        sum_mobility = sum(self._mobility.values())
        if sum_mobility != 0:
            normalized_mobility = {mode: value / sum_mobility
                                   for mode, value in self._mobility.items()}
            self._mobility = {mode: (1.0 - nw)
                              for mode, nw in normalized_mobility.items()}


class Location:
    """
    Class representing a location.

    :param int location_id: unique id of the location
    :param location_type: the type of location
    :param tuple coordinates: location coordinates
    :param str name: the location name
    :param int node: id of a node in a graph
    """

    __id_count = 0

    # TODO: class not tested
    def __init__(self, location_type, coordinates, name=None, node=None):

        if location_type not in LocationType:
            message = str(location_type) + ": "
            message += "Unknown location type, use a member of the LocationType enum:\n\t- "
            message += '\n\t- '.join(t.__str__() for t in LocationType)
            raise ValueError(message)

        self._location_id = self._get_id()

        self._location_type = location_type

        if isinstance(coordinates, Point):
            self._coordinates = coordinates
        else:
            self._coordinates = Point(coordinates)

        self.name = name
        self.node = node

    def __str__(self):
        return (
            str(self.location_id) + " | " +
            (self.name if self.name else "<nameless>") +
            " (" + self._location_type.name + ")")

    @classmethod
    def _get_id(cls):
        cls.__id_count += 1
        return cls.__id_count

    @property
    def location_id(self):
        return self._location_id

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def x(self):
        return self._coordinates.x

    @property
    def y(self):
        return self._coordinates.y

    @property
    def location_type(self):
        return self._location_type

    def distance(self, other):
        """
        Returns the distance with other
        """
        return self._coordinates.distance(other._coordinates)

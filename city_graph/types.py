from enum import Enum
import shapely.geometry


class TransportType(str, Enum):
    """Type of transport."""

    ROAD = "road"
    BIKE = "bike"
    WALK = "walk"

    BUS = "bus"
    TROLLEYBUS = "trolleybus"
    FERRY = "ferry"
    TRAIN = "train"
    TRAM = "tram"


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


class LocationType(str, Enum):
    """
    Type of location.

    :note: Compiled from amenity, leisure and building type values in
        OpenStreetMap. For the original lists, please refer to:
            * https://wiki.openstreetmap.org/wiki/Key:amenity
            * https://wiki.openstreetmap.org/wiki/Key:leisure
            * https://wiki.openstreetmap.org/wiki/Key:building
    """

    # Sustenance section: reduced down to two types depending on
    # whether you want to eat or to drink.
    BAR = "bar"
    RESTAURANT = "restaurant"

    # Education section: reduced down to three types based on the age
    # bracket of the student. Colleges are treated as schools.
    KINDERGARDEN = "kindergarden"
    SCHOOL = "school"
    UNIVERSITY = "university"

    # Transportation section: only interested in stations and parking
    # lots.
    PUBLIC_TRANSPORT_STATION = "public_transport_station"
    PARKING = "parking"

    # Financial section: ignored. Banks will be classified as offices.

    # Healthcare: hospitals, doctors and pharmacies. The first two is
    # where the sick people might go (and stay in case of hospital).
    HOSPITAL = "hospital"
    DOCTOR = "doctor"
    PHARMACY = "pharmacy"

    # Entertainment, Arts & Culture: reducing to gambling, theater
    # (including cinema) and social centre (including community
    # center). Sorry to all the stripclubs, brothels and swinger clubs
    # out there.
    GAMBLING = "gambling"
    NIGHTCLUB = "nightclub"
    THEATER = "theater"
    SOCIAL_CENTRE = "social_centre"  # spelling from OSM

    # Leisure:
    BEACH = "beach"
    SPORTS_CENTRE = "sports_centre"
    PARK = "park"
    STADIUM = "stadium"

    # Building: everything livable is (including a cabin and a static
    # caravan) is a household, everything religious is a church.
    HOUSEHOLD = "household"

    OFFICE = "office"
    RETAIL = "retail"  # for non-essentials
    SUPERMARKET = "supermarket"

    CHURCH = "church"


LOCATION_TYPES = set(item.value for item in LocationType)


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
    e.g.  his/her relative preferrances for the various transporation modes.

    :param criterion: graph edge attribute used as weight (default:duration)
    :param weights: dictionary of keys of type :py:class:`.MobilityType` related
        to a preference weight (the highest the value, the preferred the transportation mode)
    """

    __slots__ = ("_mobility", "_criterion", "_data")

    def __init__(self,
                 criterion=PathCriterion.DURATION,
                 mobility=None,
                 data=None):

        if criterion not in PathCriterion:
            message = str(criterion) + ": "
            message += "Unknown criterion, use a member of the PathCriterion enum:\n\t- "
            message += '\n\t- '.join(t.__str__() for t in PathCriterion)
            raise ValueError(message)

        self._mobility = mobility or {}
        self._normalize_mobility()
        self._criterion = criterion
        self._data = data or []

    @property
    def data(self):
        return self._data

    @property
    def criterion(self):
        return self._criterion

    @property
    def mobility(self):
        return self._mobility

    @mobility.setter
    def mobility(self, values):
        self._mobility = values
        self._normalize_mobility()

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
        normalized_mobility = {mode: value / sum_mobility
                               for mode, value in self._mobility.items()}
        self._mobility = {mode: (1.0 - nw)
                          for mode, nw in normalized_mobility.items()}


class Location:

    """
    :param int location_id: unique id of the location
    :param location_type: the type of location
    :param coordinates: where the location is (instance of shapely.geometry.Point)
    :param str name: the location name (optional)
    :param node: id of a node in a graph (optional)
    """

    _id_count = 0

    def __init__(self, location_type, coordinates, name=None, node=None):

        if location_type not in LOCATION_TYPES:
            message = str(location_type) + ": "
            message += "Unknown location type, use a member of the LocationType enum:\n\t- "
            message += '\n\t- '.join(t.__str__() for t in LocationType)
            raise ValueError(message)

        self._location_id = self._get_id()

        self._location_type = location_type

        if isinstance(coordinates, shapely.geometry.Point):
            self._coordinates = coordinates
        else:
            self._coordinates = shapely.geometry.Point(coordinates)

        self.name = name
        self.node = node

    def __str__(self):
        return str(self.location_id) + " | " + self.name + " (" + self._location_type + ")"

    @classmethod
    def _get_id(cls):
        cls._id_count += 1
        return cls._id_count

    @property
    def location_id(self):
        return self._location_id

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def location_type(self):
        return self._location_type

    @classmethod
    def _get_id(cls):
        cls._id_count += 1
        return cls._id_count

    def distance(self, other):
        """
        Returns the distance with other
        """
        return self._coordinates.distance(other._coordinates)

from enum import Enum
import shapely


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
    PUBLIC_TRANPORT_STATION = "public_transport_station"
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


class Preferences:
    """
    Encapsulate the user's preferences for path planning,
    e.g.  his/her relative preferrances for the various transporation modes.

    :param weights: dictionary of keys of type :py:class:`.MobilityType` related
    to a preference weight (the highest the value, the preferred the transportation mode)
    """

    __slots__ = ("_weights",)

    def __init__(self, weights=None):
        self._weights = weights or {}
        self._normalize_weights()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, values):
        self._weights = values
        self._normalize_weights()

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

    def _normalize_weights(self):
        sum_weights = sum(self._weights.values())
        normalized_weights = {mode: value / sum_weights
                              for mode, value in self._weights.items()}
        self._weights = {mode: (1.0 - nw)
                         for mode, nw in normalized_weights.items()}


class Location:
    """
    A location in the city.

    :param :py:class:`LocationType` location_type: the type of location
    :param coordinates: where the location is (instance of shapely.geometry.Point)
    :param str name: the location name (optional)
    :param int location_id: unique id of the location
    """

    __id_count = 0

    def __init__(self, location_type, coordinates, name=None):

        if location_type not in LocationType:
            message = "Unknown location type, use a member of the LocationType enum:\n\t- "
            message += '\n\t- '.join(t.__str__() for t in LocationType)
            raise ValueError(message)

        self.name = name
        self._location_type = location_type
        self._coordinates = coordinates
        # generating a unique id
        self._location_id = self._get_id()

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
        cls.__id_count += 1
        return cls.__id_count

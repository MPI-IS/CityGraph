from enum import Enum


class EdgeType(Enum):
    """
    Type of Topology edge.
    """

    ROAD = 0
    BIKE = 1
    WALK = 2

    BUS = 3
    TROLLEYBUS = 4
    FERRY = 5
    TRAIN = 6
    TRAM = 7


class MobilityType(tuple, Enum):
    """
    Type of human mobility.
    """

    PUBLIC_TRANSPORT = (
        EdgeType.WALK,
        EdgeType.BUS,
        EdgeType.TROLLEYBUS,
        EdgeType.FERRY,
        EdgeType.TRAIN,
        EdgeType.TRAM
    )

    CAR = (EdgeType.ROAD, EdgeType.WALK)
    BIKE = (EdgeType.BIKE, ) + PUBLIC_TRANSPORT
    WALK = (EdgeType.WALK, )


class LocationType(Enum):
    # Compiled from amenity, leisure and building type values in
    # OpenStreetMap. Please refer to:
    #   https://wiki.openstreetmap.org/wiki/Key:amenity
    #   https://wiki.openstreetmap.org/wiki/Key:leisure
    #   https://wiki.openstreetmap.org/wiki/Key:building
    # for the original lists.

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
    PUBLIC_TRANPORT_STATION = 5
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
    SOCIAL_CENTRE = 13  # spelling from OSM

    # Leisure:
    BEACH = 14
    SPORTS_CENTRE = 15
    PARK = 16
    STADIUM = 17

    # Building: everything livable is (including a cabin and a static
    # caravan) is a household, everything religious is a church.
    HOUSEHOLD = 18

    OFFICE = 19
    RETAIL = 20  # for non-essentials
    SUPERMARKET = 21

    CHURCH = 22

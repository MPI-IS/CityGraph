from enum import Enum


class EdgeType(str, Enum):
    """
    Type of Topology edge.
    """

    ROAD = "road"
    BIKE = "bike"
    WALK = "walk"

    BUS = "bus"
    TROLLEYBUS = "trolleybus"
    FERRY = "ferry"
    TRAIN = "train"
    TRAM = "tram"


class MobilityType(Enum):
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


class LocationType(str, Enum):
    # Compiled from amenity, leisure and building type values in
    # OpenStreetMap. Please refer to:
    #   https://wiki.openstreetmap.org/wiki/Key:amenity
    #   https://wiki.openstreetmap.org/wiki/Key:leisure
    #   https://wiki.openstreetmap.org/wiki/Key:building
    # for the original lists.

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

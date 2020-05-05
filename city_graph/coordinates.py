"""
Coordinates
===========

Module for using coordinates and calculating distances.
"""
from math import sin, cos, sqrt, atan2, radians


class GeoCoordinates:
    """Class representing a geolocation.

    :param obj point: An objet with coordinates that are reachable
        through the attributes ``x`` and ``y`` (e.g. a ``shapely Point``).
    """

    # Mean Earth radius in cm.
    EARTH_RADIUS_CM = 6371. * 1e5

    def __init__(self, point):
        self.longitude = float(point.x)
        self.latitude = float(point.y)

    def __hash__(self):
        # An instance is defined by its coordinates only
        return tuple([self.longitude, self.latitude]).__hash__()

    @classmethod
    def distance(cls, c1, c2):
        """Calculate the distance between two coordinates in cm.

        :param c1: First coordinates
        :type c1: :py:class:`.GeoCoordinates`
        :param c2: Second coordinates
        :type c2: :py:class:`.GeoCoordinates`

        :returns: Distance in cm
        :rtype: float

        :note: We approximate the Earth as a sphere and use the Haversine formula.
        """

        long1 = radians(c1.longitude)
        long2 = radians(c2.longitude)
        lat1 = radians(c1.latitude)
        lat2 = radians(c2.latitude)

        delta_long = long1 - long2
        delta_lat = lat1 - lat2

        a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_long / 2) ** 2
        d = 2 * atan2(sqrt(a), sqrt(1 - a))
        return d * GeoCoordinates.EARTH_RADIUS_CM

    def distance_to(self, other):
        """Same as the class method but for the instance."""
        return GeoCoordinates.distance(self, other)

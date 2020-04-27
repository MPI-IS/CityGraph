import contextlib
import math
import logging

import geopandas
import networkx
import osmnx
from shapely.geometry import Point, Polygon, MultiPolygon

from .city import City
from .types import TransportType, LocationType, Location
from .topology import MultiEdgeUndirectedTopology
from .utils import (
    EARTH_RADIUS_METERS, distance, group_locations_by, reverse_mapping)


TRANSPORT_TO_NETWORK = {
    TransportType.ROAD: "drive",
    TransportType.BIKE: "bike",
    TransportType.WALK: "walk",
}

TRANSPORT_TO_ROUTE = {
    TransportType.BUS: "bus",
    TransportType.FERRY: "ferry",
    TransportType.SUBWAY: "subway",
    TransportType.TRAIN: "train",
    TransportType.TRAM: "tram",
    TransportType.TROLLEYBUS: "trolleybus",
}

LOCATION_TO_AMENITIES = {
    LocationType.BAR: ["bar", "biergarten", "pub"],
    LocationType.RESTAURANT: ["cafe", "fast_food", "restaurant"],
    LocationType.KINDERGARDEN: ["kindergarden"],
    LocationType.SCHOOL: ["school", "college"],
    LocationType.UNIVERSITY: ["university"],
    LocationType.PARKING: ["parking", "parking_space"],
    LocationType.HOSPITAL: ["clinic", "hospital"],
    LocationType.DOCTOR: ["dentist", "doctors"],
    LocationType.PHARMACY: ["pharmacy"],
    LocationType.GAMBLING: ["casino", "gambling"],
    LocationType.NIGHTCLUB: ["nightclub"],
    LocationType.THEATER: ["cinema", "theatre"],
    LocationType.SOCIAL_CENTER: [
        "arts_centre", "community_centre", "social_centre"],
    # ^^^ please DON'T rename those. OSM uses british spelling.
}

LOCATION_TO_LEISURE = {
    LocationType.BEACH: ["beach_resort"],
    LocationType.SPORTS_CENTER: [
        "bowling_alley", "fitness_centre", "sports_centre", "track"],
    # ^^^ please DON'T rename those. OSM uses british spelling.
    LocationType.PARK: [
        "garden", "park", "trampoline_park", "water_park"],
    LocationType.STADIUM: ["ice_rink", "pitch", "stadium"],
}

LOCATION_TO_BUILDING = {
    LocationType.HOUSEHOLD: [
        "apartments", "bungalow", "cabin", "detached", "dormitory",
        "ger", "house", "residential", "semidetached_house", "static_caravan",
        "yes"
    ],
    LocationType.OFFICE: ["commercial", "industrial", "office"],
    LocationType.RETAIL: ["retail"],
    LocationType.GROCERY: ["supermarket"],
    LocationType.CHURCH: [
        "catherdral", "chapel", "church", "mosque",
        "religious", "shrine", "synagogue", "temple"]
}

GROCERIES = (
    "alcohol", "bakery", "beverages", "brewing_supplies", "butcher",
    "cheese", "chocolate", "coffee", "confectionery", "convenience",
    "deli", "dairy", "farm", "frozen_food", "greengrocer", "health_food",
    "pasta", "pastry", "seafood", "spices", "tea", "water"
)


AMENITY_TO_LOCATION = reverse_mapping(LOCATION_TO_AMENITIES)
LEISURE_TO_LOCATION = reverse_mapping(LOCATION_TO_LEISURE)
BUILDING_TO_LOCATION = reverse_mapping(LOCATION_TO_BUILDING)


class Tesselation:
    """
    Base class for plane tesselation strategy.
    """

    @property
    def ncells(self):
        """
        Get total number of cells.
        """
        raise NotImplementedError()

    def index(self, x, y):
        """
        Get cell index by coordinates.

        :param float x: x coordinate
        :param float y: y coordinate
        """
        raise NotImplementedError()

    def center(self, index):
        """
        Get coordinates of the cell center by index.

        :param int index: cell index.
        """
        raise NotImplementedError()


class GridTesselation(Tesselation):
    """
    Plane tesselation with regular rectangular grid with fixed step
    size.

    :param polygon: polygon enclosing the place.
    :param resolution: grid resolution in degrees.

    :type resolution: a tuple of longitudinal and latitudinal
        resolution or an int if you want a square grid.
    """

    def __init__(self, polygon, resolution=(100, 100)):
        self.polygon = polygon

        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.dx, self.dy = resolution

        self.xmin, self.ymin, self.xmax, self.ymax = polygon.bounds

        self.nx = int((self.xmax - self.xmin) / self.dx)
        self.ny = int((self.ymax - self.ymin) / self.dy)

    @classmethod
    def from_resolution_in_meters(cls, polygon, resolution=100):
        """
        Construct a plane tesselation using a square grid with fixed
        resolution given in meters.

        :param polygon: a :py:class:`shapely.geometry.Polygon`.
        :param int resolution: resolution in meters.
        """
        latmin, _, latmax, _ = polygon.bounds
        latmid = (latmin + latmax) / 2 / 180 * math.pi

        dtheta = resolution / EARTH_RADIUS_METERS
        dlat = (180 / math.pi) * dtheta
        dlon = (180 / math.pi) * 2 * math.asin(
            1 / math.cos(latmid) * math.sin(dtheta/2))

        return cls(polygon, resolution=(dlon, dlat))

    def _ixiy(self, x, y):
        ix = int((x - self.xmin) / self.dx)
        iy = int((y - self.ymin) / self.dy)

        # Clamp everything into the region borders. Since we round
        # down the indices and then offset the coordinates by
        # resolution/2 up, the points that lie on the edges end up in
        # the cell to the top (or to the right). This is not a big
        # deal for the points that are initially inside the original
        # polygon, but that can mess up the points on the
        # boundary. Those we have to put back inside the polygon
        # manually and clamping the indicies does exactly that.

        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)

        return ix, iy

    @property
    def ncells(self):
        return self.nx * self.ny

    def index(self, x, y):
        ix, iy = self._ixiy(x, y)
        return iy * self.nx + ix

    def center(self, index):
        iy = int(index / self.nx)
        ix = index % self.nx

        x = self.xmin + ix * self.dx + self.dx/2
        y = self.ymin + iy * self.dy + self.dy/2

        return x, y


def route_from_relation(relation, nodes, paths):
    """
    Construct a route given the OSM relation.

    :param dict relation: OSM relation corresponding to the route.
    :param dict nodes: a lookup dictionary of the graph nodes.
    :param dict paths: a lookup dictionary of the graph edges.

    :returns: a tuple of line name and a multigraph.
    """

    route_nodes = {}
    route_paths = {}

    rel_tags = relation["tags"]
    line = rel_tags.get("line") or rel_tags.get("ref")
    tags = {
        "line": line,
        "name": rel_tags.get("name"),
        "type": rel_tags.get("route")
    }

    route = networkx.MultiDiGraph(name=line, crs=osmnx.settings.default_crs)

    for member in relation["members"]:
        key = member["ref"]

        if member["type"] == "node":
            route_nodes[key] = nodes[key]

        if member["type"] == "way":
            path = paths[key].copy()
            path.update(**tags)

            if "nodes" not in path:
                continue

            route_paths[key] = path
            for node_key in path["nodes"]:
                route_nodes[node_key] = nodes[node_key]

    if not route_paths:
        return

    for key, data in route_nodes.items():
        route.add_node(key, **data)

    route = osmnx.add_paths(route, route_paths)
    route = osmnx.add_edge_lengths(route)

    return line, route


def routes_from_polygon(polygon, route_type):
    """
    Create a network graph corresponding to a selected route type from
    OSM data within the spatial boundaries of the passed-in shapely
    polygon.

    This function mostly reproduces that is already done in
    :py:func:`osmnx.graph_from_polygon`, but it preserves the route
    structure imposed by relations.  The graph is always truncated to
    the polygon and simplified.

    :param shapely.Polygon polygon: the shpae to get network data
        within.  Coordinates should be in units of latitude-longiute.
    :param str route_type: what type of route to get.

    :returns: a dictionary mapping route name to the corresponding
              multigraph.
    """

    for tag in ("name", "public_transport"):  # might be expanded later
        if tag not in osmnx.settings.useful_tags_node:
            osmnx.settings.useful_tags_node.append(tag)

    response_jsons = osmnx.osm_net_download(
        polygon=polygon,
        infrastructure='rel["route"]',
        custom_filter='["route"="{}"]'.format(route_type))

    # Collect all the elements from the response.
    elements = []
    for osm_data in response_jsons:
        elements.extend(osm_data["elements"])

    # Sort them into paths, nodes and relations building a global
    # lookup dictionary.
    nodes = {}
    paths = {}
    relations = {}
    for element in elements:
        key = element["id"]
        if element["type"] == "node":
            nodes[key] = osmnx.get_node(element)
        if element["type"] == "way":
            paths[key] = osmnx.get_path(element)
        if element["type"] == "relation":
            relations[key] = element

    # Build a graph for every relation.
    routes = {}

    for _, relation in relations.items():
        try:
            line, route = route_from_relation(relation, nodes, paths)
            route = osmnx.truncate_graph_polygon(
                route, polygon, retain_all=True)
        except Exception:
            # route might be empty or the line might be outside of
            # region
            continue

        if route.edges:
            routes[line] = route

    return routes


def pois_from_polygon(polygon, timeout=180, maxsize=''):
    """
    Get OSM points of interest with a polygon.

    This is a recreation of :py:func:`osmnx.pois_from_polygon` with
    one crucial difference.  `osmnx` targets the nodes, the ways and
    the relations based on "amenity" tag.  This is useful but it
    misses pretty much all the grocery stores (don't have an amenity,
    but have a "shop" tag) and all the fitness centers (no amenity,
    but have a "leisure" tag).  Here we use a slightly different
    approach: we find the ways and the relations with an "amenity"
    tag, but also fetch all the named nodes (i.e. those with the tag
    "name" defined).  This covers pretty much everything except for
    occasional nameless locations.

    :param shapely.geometry.Polygon: the polygon that will be used to
        limit the POI search.
    :param int timeout: timeout for the API request.
    """

    west, south, east, north = polygon.bounds
    query_template = (
        '{settings};('
            '(node["name"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});(._;>;););'
            '(node["amenity"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});(._;>;););'
            '(way["amenity"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});(._;>;););'
            '(relation["amenity"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});(._;>;););'
        ');out;')
    query = query_template.format(
        north=north, south=south, east=east, west=west,
        settings=osmnx.settings.default_overpass_query_settings.format(
            timeout=timeout, maxsize=maxsize))

    responses = osmnx.overpass_request(data={"data": query}, timeout=timeout)

    # NOTE: Starting from here and until the end of the function we
    # copy the code from osmnx.pois.create_poi_gdf() with very
    # superficial changes. Please refer to LICENSE_OSMNX.md for the
    # original licence and copyright attribution of the copied code.
    # --- BORROWED FROM osmnx.pois.create_poi_gdf() -----------------------

    # Parse coordinates from all the nodes in the response
    coords = osmnx.parse_nodes_coords(responses)

    # POI nodes
    poi_nodes = {}

    # POI ways
    poi_ways = {}

    # A list of POI relations
    relations = []

    for result in responses['elements']:
        if result['type'] == 'node' and 'tags' in result:
            poi = osmnx.parse_osm_node(response=result)
            # Add element_type
            poi['element_type'] = 'node'
            # Add to 'pois'
            poi_nodes[result['id']] = poi
        elif result['type'] == 'way':
            # Parse POI area Polygon
            poi_area = osmnx.parse_polygonal_poi(coords=coords, response=result)
            if poi_area:
                # Add element_type
                poi_area['element_type'] = 'way'
                # Add to 'poi_ways'
                poi_ways[result['id']] = poi_area

        elif result['type'] == 'relation':
            # Add relation to a relation list (needs to be parsed
            # after all nodes and ways have been parsed)
            relations.append(result)

    # Create GeoDataFrames
    gdf_nodes = geopandas.GeoDataFrame(poi_nodes).T
    gdf_nodes.crs = osmnx.settings.default_crs

    gdf_ways = geopandas.GeoDataFrame(poi_ways).T
    gdf_ways.crs = osmnx.settings.default_crs

    # Parse relations (MultiPolygons) from 'ways'
    gdf_ways = osmnx.parse_osm_relations(relations=relations, osm_way_df=gdf_ways)

    # Combine GeoDataFrames
    gdf = gdf_nodes.append(gdf_ways, sort=False)

    if polygon:
        gdf = gdf.loc[gdf['geometry'].centroid.within(polygon)==True]

    return gdf

    # --- END OF BORROWED CODE -------------------------------------------------


def fetch_osm_data(name):
    """
    Fetch OSM data for a given place.

    :param str name: name of the place.
    """

    logging.info("Fetching city graph for {!r}".format(name))

    # Find the place by name and convert it to a polygon
    gdf = osmnx.gdf_from_place(name)
    logging.info("Found geodata frame")
    polygon = gdf["geometry"].unary_union

    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise ValueError(
            "invalid geometry. "
            "Expected Polygon or MultiPolygon, but got {} instead. "
            "Perhabs the place does not exist.".format(type(polygon)))

    # Fetch all the networks and public transportation routes
    networks = {}
    for path_type in list(TransportType):
        if path_type in TRANSPORT_TO_NETWORK:
            logging.info("Fetching {} network".format(path_type))
            network_type = TRANSPORT_TO_NETWORK[path_type]
            networks[path_type] = osmnx.graph_from_polygon(
                polygon, network_type=network_type, simplify=False)
        elif path_type in TRANSPORT_TO_ROUTE:
            logging.info("Fetching {} routes".format(path_type))
            route_type = TRANSPORT_TO_ROUTE[path_type]
            networks[path_type] = routes_from_polygon(
                polygon, route_type=route_type)
        else:
            logging.warning(
                "TransportType {} is not recognized as either network type "
                "or route type. Ignoring it".format(path_type))

    # Fetch all the building footprints
    logging.info("Importing footprints")
    footprints = osmnx.footprints_from_polygon(polygon)
    footprints["name"] = footprints["name"].fillna(value="")

    # Fetch all the points of interest
    logging.info("Importing points of interest")

    pois = pois_from_polygon(polygon)
    pois["name"] = pois["name"].fillna(value="")

    return gdf, networks, footprints, pois


def combine_network_lines(routes):
    """
    Combine all the public transportation lines into a single graph.

    :param dict route: a dictionary of bust lines.  The keys are line
        names and the values are graphs.
    """
    result = networkx.MultiDiGraph(crs=osmnx.settings.default_crs)
    for route in routes.values():
        result = networkx.compose(result, route)
    return result


def downsample_network(tesselation, in_network, offset=0):
    """
    Construct a downsampled network using a given plane tesselation.

    :param tesselation: a plane tesselation instance.
    :param in_network: a single graph representing a transport network.
    :param int offset: optional offset for node id.
    """

    out_network = networkx.MultiDiGraph(**in_network.graph)

    for src_node, dst_node in in_network.edges():
        src_data = in_network.nodes[src_node]
        dst_data = in_network.nodes[dst_node]

        src_index = tesselation.index(src_data["x"], src_data["y"]) + offset
        dst_index = tesselation.index(dst_data["x"], dst_data["y"]) + offset

        src_x, src_y = tesselation.center(src_index - offset)
        dst_x, dst_y = tesselation.center(dst_index - offset)

        if src_index == dst_index:
            continue

        if (src_index, dst_index) in out_network.edges:
            continue

        if src_index not in out_network.nodes:
            out_network.add_node(src_index, x=src_x, y=src_y)

        if dst_index not in out_network.nodes:
            out_network.add_node(dst_index, x=dst_x, y=dst_y)

        out_network.add_edge(
            src_index, dst_index,
            distance=distance(src_x, src_y, dst_x, dst_y))

    return out_network


def copy_network_to_topology(transport_type, network, topology):
    """
    Copy the network into topology.

    :param tranport_type: transport that the network is representing.
    :param network: the graph to be copied.
    :param topology: the target topology.
    """

    for node, data in network.nodes(data=True):
        with contextlib.suppress(RuntimeError):
            topology.add_node(node, data["x"], data["y"], **data)

    for src, dst, data in network.edges(data=True):
        with contextlib.suppress(RuntimeError):
            topology.add_edge(src, dst, transport_type, **data)


def is_platform(node_data):
    """
    Checks OSM node attributes and determines whether the node is a
    public transport platform (e.g bus station).

    :param dict node_data: a dictionary of the node attributes.
    """
    return (
        node_data.get("highway") == "bus_stop" or
        node_data.get("public_transport") == "platform")


def extract_platforms(network):
    """
    Extract the platforms from the networks.

    :param network: a networkx graph of a transportation network.
    """
    platforms = []

    for node, data in list(network.nodes(data=True)):
        if not is_platform(data):
            continue

        platforms.append(data)
        if not network.degree(node):
            network.remove_node(node)

    return platforms


def import_networks(networks, topology, tesselation):
    """
    Import the transportation networks into an empty topology.

    :param networks: a dictionary of transport networks as returned by
        :py:func:`fetch_osm_data`.
    :param topology: a topology instance.
    :param tesselation: plane tesselation.
    """

    # We process the networks one by one. Pedestrian paths, bike paths
    # and roads need only a simple downsampling.
    walk_network = None
    for transport_type in TRANSPORT_TO_NETWORK:
        network = downsample_network(
            tesselation, networks[transport_type])
        copy_network_to_topology(transport_type, network, topology)

        if transport_type == TransportType.WALK:
            walk_network = network

    # Public transport routes are a bit different. While they go
    # through the same points on the map as the previously processed
    # networks, they should not go through the same graph nodes. They
    # exist in a parallel plane in some sense, and you can enter that
    # plain only at specific points -- the public transportation
    # platforms. In those points we will create a zero-length
    # connection between the node on the route and the node on the
    # path.
    locations = []

    for layer, transport_type in enumerate(TRANSPORT_TO_ROUTE, start=1):
        if transport_type not in networks:
            continue

        transport_network = combine_network_lines(networks[transport_type])
        offset = layer * tesselation.ncells

        platforms = extract_platforms(transport_network)
        transport_network = downsample_network(
            tesselation, transport_network, offset=offset)

        copy_network_to_topology(transport_type, transport_network, topology)

        px = [p["x"] for p in platforms]
        py = [p["y"] for p in platforms]

        transport_nodes = osmnx.get_nearest_nodes(
            transport_network, px, py)
        walk_nodes = osmnx.get_nearest_nodes(
            walk_network, px, py)

        for tn, wn, platform in zip(transport_nodes, walk_nodes, platforms):
            with contextlib.suppress(RuntimeError):
                topology.add_edge(tn, wn, TransportType.WALK, distance=0)
            location = extract_route_location(platform)
            location["node"] = tn
            locations.append(location)

    return locations


def extract_route_location(node_data):
    """
    Constructalocation from the route node.

    :param dict node_data: node attributes.
    """
    return {
        "location_type": LocationType.PUBLIC_TRANSPORT_STATION,
        "coordinates": Point(node_data["x"], node_data["y"]),
        "name": node_data.get("name")
    }


def extract_location_type(attrs):
    """
    Figure out the location type given the location attributes.

    :param dict attrs: a dictionary of the location attributes.
    """

    # Stores are a special case -- we have only two types of them. The
    # grocery shops are determined by constant GROCERIES. Anything
    # else with "shop" tag is a retail.
    shop_type = attrs.get("shop")
    if isinstance(shop_type, str):
        if shop_type in GROCERIES:
            return LocationType.GROCERY
        else:
            return LocationType.RETAIL

    # Location type can be also encoded in one of the following
    # fields.
    amenity_type = attrs.get("amenity")
    leisure_type = attrs.get("leisure")
    building_type = attrs.get("building")

    # Most of the times the attribute is either a string or a
    # NaN. Comparing to NaN in python is really awkward, so instead we
    # check the instance type. If it's a string we try to resolve it
    # according to a dictionary. If it's something else we set it to
    # None for consistency. XXX: bool(nan) is True.
    if isinstance(amenity_type, str):
        amenity_type = AMENITY_TO_LOCATION.get(amenity_type)
    else:
        amenity_type = None

    if isinstance(leisure_type, str):
        leisure_type = LEISURE_TO_LOCATION.get(leisure_type)
    else:
        leisure_type = None

    if isinstance(building_type, str):
        building_type = BUILDING_TO_LOCATION.get(building_type)
    else:
        building_type = None

    return amenity_type or leisure_type or building_type


def extract_footprint_location(footprint):
    """
    Constructs a location from a building footprint.

    :param footprint: pandas row representing a building footprint.
    """

    type_ = extract_location_type(footprint)
    if not type_:
        return

    polygon = footprint["geometry"]
    point = polygon.centroid

    if point.is_empty:
        return

    name = footprint.get("name")
    if not name or isinstance(name, float):  # a silly way to compare to nan :)
        name = footprint.get("addr:housename")
    if not name or isinstance(name, float):
        street = footprint.get("addr:street")
        housenumber = footprint.get("addr:housenumber")
        if isinstance(street, float) or isinstance(street, float):
            name = None
        else:
            name = "{} {}".format(street, housenumber)

    return {
        "location_type": type_,
        "coordinates": point,
        "name": name
    }


def extract_poi_location(poi):
    """
    Constructs a location from a point of interest.

    :param footprint: pandas row representing a POI.
    """

    type_ = extract_location_type(poi)
    if not type_:
        return

    geometry = poi["geometry"]

    if isinstance(geometry, Point):
        point = geometry
    elif isinstance(geometry, Polygon):
        point = geometry.centroid
    elif isinstance(geometry, MultiPolygon):
        point = max(list(geometry), key=lambda p: p.area).centroid
    else:
        logging.error("geometry is {}".format(geometry))

    if point.is_empty:
        return

    return {
        "location_type": type_,
        "coordinates": point,
        "name": poi["name"]
    }


def import_osm_data(
    place,
    resolution=100,
    location_class=Location
):
    """
    Import the city from OSM given the place name.

    :param str place: a string uniquely defining a geographical
        region.  That could be, for example, a name of the city or a
        city district.
    :param resolution: the resolution of the grid upon which we clip
        the data.
    :param location_class: a callable location constructor.

    :type resolution: a tuple of longitudinal and latitudinal
        resolution or an int if you want a square grid.
    """

    # XXX: This kind of caching is not very elegant (for example, the
    # location is hardcoded), but it works great for development. It's
    # dead code, but please don't remove it :)

    # import os
    # import pickle

    # filename = place + ".place"

    # if not os.path.exists(filename):
    #     gdf, networks, footprints, pois =  fetch_osm_data(place)
    #     with open(filename, "wb") as fd:
    #         pickle.dump((gdf, networks, footprints, pois), fd)
    # else:
    #     with open(filename, "rb") as fd:
    #         gdf, networks, footprints, pois = pickle.load(fd)

    # Fetch raw OSM data.
    gdf, networks, footprints, pois =  fetch_osm_data(place)

    topology = MultiEdgeUndirectedTopology()
    locations = []
    tesselation = GridTesselation.from_resolution_in_meters(
        gdf["geometry"].unary_union,
        resolution=resolution)

    # Import the transportation networks into topology. As a side
    # effect this function extract all the transportation platforms.
    platforms = import_networks(networks, topology, tesselation)

    logging.info(
        "Successfully imported the transportation network:"
        " {} nodes and {} edges in total.".format(
            topology.num_of_nodes,
            topology.num_of_edges))

    # Convert all the points of interest into locations.
    for _, poi in pois.iterrows():
        location = extract_poi_location(poi)
        if location:
            locations.append(location)

    # Convert all the footprints into locations.
    for _, footprint in footprints.iterrows():
        location = extract_footprint_location(footprint)
        if location:
            locations.append(location)

    # Bind all the locations to the topology nodes.
    for location in locations:
        node = tesselation.index(
            location["coordinates"].x,
            location["coordinates"].y)
        if node in topology.nodes:
            location["node"] = node

    # Wrap all the location data into an appropriate class.
    locations += platforms
    locations = [
        location_class(**location)
        for location in locations
    ]

    # Some of the locations will not be bound to a topology node. We
    # will have drop those.
    bound_locations = []
    unbound_locations = []
    for location in locations:
        if location.node is None:
            logging.warn(
                "Location {} is not bound to a node. Skipping it."
                .format(location))
            unbound_locations.append(location)
        else:
            bound_locations.append(location)

    if unbound_locations:
        logging.info(
            "Of {} locations {} ({:.1f}%) are not connected to "
            "a graph node and will be dropped."
            .format(
                len(locations),
                len(unbound_locations),
                100 * len(unbound_locations) / len(locations)))
    locations = bound_locations

    # Do some reporting.
    locations_by_type = group_locations_by(locations, "location_type")
    for location_type, locs in locations_by_type.items():
        logging.info(
            "Imported {} {} locations".format(len(locs), location_type.name))

    for location in locations:
        if not location.node:
            logging.info("{} is not bound to a node".format(location))

    # Wrap it all into a city and return.
    return City(place, locations, topology)

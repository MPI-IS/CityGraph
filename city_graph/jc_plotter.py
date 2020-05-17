import itertools
import collections

from descartes import PolygonPatch

from matplotlib import pyplot
from matplotlib.collections import LineCollection, PatchCollection

from .jc_importer import TRANSPORT_TO_ROUTE
from .types import LocationType, TransportType


def group_locations_by_type(locations):
    """
    Group a list of locations in a dictionary based their type.

    :note: Used mostly for reporting and plotting.
    """
    locations_by_type = collections.defaultdict(list)
    for location in locations:
        locations_by_type[location.location_type].append(location)
    return locations_by_type


COLOR_BACKGROUND = "#d8f2c2"
COLOR_GROUND = "#fefdfb"

COLOR_BUILDING = "#f2efeb"

COLOR_ROAD_FACE = "#ffffff"
COLOR_ROAD_EDGE = "#a8a8a8"

COLOR_PALETTE = (
    "#18bd9d",
    "#1c9b8a",
    "#2dcc70",
    "#29ad61",
    "#3398dc",
    "#2981b1",
    "#9a58b9",
    "#8d45ab",
    "#f1c413",
    "#fc9616",
    "#e77f1c",
    "#d45400",
    "#fc4713",
    "#c73b2c"
)


def _alpha(rgb, a):
    """
    Add alpha channel to a hex color.

    :param str rgb: hex color.
    :param float a: alpha channel (float between 0 and 1)
    """
    a = int(a * 255)
    return "{}{:02x}".format(rgb, a)


def plot_gdf(gdf, margin=0.1):
    """
    Plot GeoDataFrame and set the correst aspect ratio.

    :param gdf: A :py:class:`Polygon` or :py:class:`MultiPolygon`
    """
    ax = pyplot.gca()
    ax.set_facecolor(COLOR_BACKGROUND)

    polygon = gdf["geometry"].unary_union

    patch = PolygonPatch(
        polygon,
        fc=COLOR_GROUND,
        ec=COLOR_GROUND,
        zorder=-1)
    ax.add_patch(patch)

    west, south, east, north = polygon.bounds

    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin

    ax.set_xlim((west - margin_ew, east + margin_ew))
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_aspect("equal")


def plot_network(
    network,
    facecolor=None,
    edgecolor=COLOR_ROAD_EDGE,
    linewidth=1,
    linestyle="solid",
    zorder=1,
    use_geometry=True,
    label=None
):
    """
    Plot a single network.

    :param netowrk: a :py:class:`networkx.Graph` representing the
        network.
    :param str facecolor: optional color for the path filling.  If
        empty the edge color will be used.
    :param str edgecolor: optional color for the path outline
    :param int linewidth: line width.
    :param str linestyle: line style used for the paths.
    :param zorder: z-order of the plot.
    :param use_geometry: use geometric data of the edges if provided.
        Handful when drawing the graphs simplified by osmnx.
    :param label: plot label.
    """
    ax = pyplot.gca()

    lines = []
    for u, v, data in network.edges(keys=False, data=True):
        if use_geometry and "geometry" in data:
            xs, ys = data["geometry"].xy
            lines.append(list(zip(xs, ys)))
        else:
            x1 = network.nodes[u]["long"]
            x2 = network.nodes[v]["long"]
            y1 = network.nodes[u]["lat"]
            y2 = network.nodes[v]["lat"]
            lines.append([(x1, y1), (x2, y2)])

    if not facecolor:
        lc = LineCollection(
            lines,
            colors=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            label=label)

        ax.add_collection(lc)
    else:
        lc_edge = LineCollection(
            lines,
            colors=edgecolor,
            linewidth=(linewidth + 2),
            linestyle=linestyle,
            zorder=zorder,
            label=label)

        lc_face = LineCollection(
            lines,
            colors=facecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder + 1)

        ax.add_collection(lc_edge)
        ax.add_collection(lc_face)


def plot_footprints(
    footprints,
    facecolor=COLOR_BUILDING,
    edgecolor=COLOR_BUILDING,
    linewidth=0.5,
):
    """
    Plot building footprints.

    :param str facecolor: color used to fill in the building footprint.
    :param str edgecolor: color used for the building outline.
    :param int linewidth: thickness of the outline.
    """
    ax = pyplot.gca()

    patches = []
    for _, row in footprints.iterrows():
        polygon = row["geometry"]
        patch = PolygonPatch(polygon)
        patches.append(patch)

    pc = PatchCollection(
        patches,
        fc=facecolor,
        ec=edgecolor,
        linewidth=linewidth)

    ax.add_collection(pc)


def plot_locations(locations):
    """
    Plot a list of locations.

    :param list locations: a list of locations.
    """

    plots_by_type = {}
    locations_by_type = group_locations_by_type(locations)

    colors = itertools.cycle(COLOR_PALETTE)
    for color, location_type in zip(colors, list(LocationType)):
        if location_type == LocationType.HOUSEHOLD:
            zorder = 50
            color = COLOR_ROAD_EDGE
            linewidth = 1
        else:
            zorder = 51
            linewidth = 1.5

        locations = locations_by_type[location_type]
        if not locations:
            continue

        x = [loc.coordinates.x for loc in locations]
        y = [loc.coordinates.y for loc in locations]

        plot = pyplot.scatter(
            x, y, s=50,
            color="#ffffff", edgecolor=color,
            label=location_type.name,
            zorder=zorder,
            linewidth=linewidth)
        plots_by_type[location_type] = plot

    fig, ax = pyplot.gcf(), pyplot.gca()
    annotation = ax.annotate(
        "", xy=(0, 0),
        xytext=(5, 5),
        textcoords="offset points",
        zorder=100,
        color="white",
        backgroundcolor=_alpha("#000000", 0.75))
    annotation.set_visible(False)

    def highlight_location(plot, location_type, index):
        index = index["ind"][0]
        position = plot.get_offsets()[index]
        location = locations_by_type[location_type][index]

        text = location_type.name
        text = location.node
        if location.name:
            text += " «" + location.name + "»"

        annotation.set_text(text)
        annotation.xy = position

    def on_hover(event):
        if not event.inaxes == ax:
            return

        for location_type, plot in plots_by_type.items():
            contains, index = plot.contains(event)
            if contains:
                highlight_location(plot, location_type, index)
                annotation.set_visible(True)
                fig.canvas.draw_idle()
                break
        else:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    pyplot.legend(bbox_to_anchor=(1, 1), loc="upper left")


def plot_topology(topology):
    """
    Plot a topology.

    :param topology: a topology instance.
    """

    def extract_network(topology, transport_type):
        network = topology.graph.__class__(**topology.graph.graph)

        for src_node, dst_node, edge_data in topology.graph.edges(data=True):
            if transport_type == edge_data[topology.EDGE_TYPE]:
                network.add_node(src_node, **topology.graph.nodes[src_node])
                network.add_node(dst_node, **topology.graph.nodes[dst_node])
                network.add_edge(src_node, dst_node, **edge_data)

        return network

    road_network = extract_network(topology, TransportType.ROAD)
    walk_network = extract_network(topology, TransportType.WALK)
    bike_network = extract_network(topology, TransportType.BIKE)

    plot_network(
        road_network,
        linewidth=4,
        facecolor=COLOR_ROAD_FACE)
    plot_network(bike_network)
    plot_network(walk_network, linestyle="dashed")

    for level, (color, transport_type) in enumerate(zip(COLOR_PALETTE[::3], TRANSPORT_TO_ROUTE)):
        network = extract_network(topology, transport_type)
        plot_network(
            network,
            edgecolor=color,
            zorder=(10 + level),
            label=transport_type.name)

    pyplot.legend()


def plot_city(city, margin=0.1):
    """
    Plot a city.

    :param city: an instance of a city to plot.
    """

    # We first have to find the extent of the city.
    x = [data["long"] for _, data in city._topology.graph.nodes(data=True)]
    y = [data["lat"] for _, data in city._topology.graph.nodes(data=True)]

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    margin_ew = (xmax - xmin) * margin
    margin_ns = (ymax - ymin) * margin

    ax = pyplot.gca()

    ax.set_xlim((xmin - margin_ew, xmax + margin_ew))
    ax.set_ylim((ymin - margin_ns, ymax + margin_ns))
    ax.set_aspect("equal")

    plot_topology(city._topology)
    plot_locations(city.get_locations())

    pyplot.show()

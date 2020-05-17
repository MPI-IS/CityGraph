from city_graph.city import City, DEFAULT_LOCATION_DISTRIBUTION
from city_graph.city import TransportType
from city_graph.utils import RandomGenerator
from city_graph.jc_plotter import plot_city

connection_types = (
    TransportType.ROAD, TransportType.WALK
)
allowed_types = {k: None for k in connection_types}
seed = 1234
# print(DEFAULT_LOCATION_DISTRIBUTION)
print("Creating city with", sum(DEFAULT_LOCATION_DISTRIBUTION.values()), "locations")
rng = RandomGenerator(seed)
print(rng.rng_seed)
my_city = City.build_random(
    'Awesome City', DEFAULT_LOCATION_DISTRIBUTION, rng=rng,
    create_network=True, max_iterations=11, connections_per_step=5,
    connection_types=connection_types)
print('city built')

# Plot city
plot_city(my_city)

# direct path: 485 -> 216: OK
# Command:
# my_city._topology.get_shortest_path(485, 216, 'distance', allowed_types)
# Output:
# (69673672.55039248, [485, 216], {'type': array([<TransportTyp...pe=object)})

# long path: 199 -> 215
# Command:
# my_city._topology.get_shortest_path(215, 199, 'distance', allowed_types)
# Ouput:
# (2096789602.6957245, [215, 31, 39, 280, 199], {'type': array([<TransportTyp...pe=object)})

# no path: 243 -> 415
# Command:
# my_city._topology.get_shortest_path(243,415, 'distance', allowed_types)
# Output:
# RuntimeError: No path found with type {<TransportType.ROAD: 0>: None, <TransportType.WALK: 2>: None} between 243 and 415.

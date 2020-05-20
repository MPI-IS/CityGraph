"""
Topology
========

Module for building and operating on graphs.
"""
from contextlib import suppress
from enum import Enum
from functools import partial
from itertools import product, combinations
from networkx import Graph, MultiGraph, \
    minimum_spanning_tree, single_source_dijkstra
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.exception import NetworkXNoPath
import numpy as np

from .utils import distance as utility_distance


# Precision for float comparisons
EPS_PRECISION = 1e-6


class TerminationReason(Enum):
    """Enumeration of the reasons for termination of an algorithm."""

    NO_TERMINATION = 0
    MAX_ITER = 1
    ALL_NODES_CONNECTED = 2


class BaseTopology:
    """Abstract class setting up some requirements for a Topology."""

    def add_node(self, *args, **kwargs):
        """Method adding a node to the multigraph."""
        raise NotImplementedError

    def add_edge(self, *args, **kwargs):
        """Method adding an edge to the multigraph."""
        raise NotImplementedError

    def get_shortest_path(self, node1, node2, *args, **kwargs):
        """Method finding the shortest path between two nodes."""
        raise NotImplementedError


class MultiEdgeUndirectedTopology(BaseTopology):
    """Class representing a topology with mutltiple undirected edges.

    :param iter nodes: Nodes in the graph
    :param iter edges: Edges in the graph
    """

    # Names used to hold the node longitude and latitude
    NODE_LONG = 'long'
    NODE_LAT = 'lat'

    # Name used to hold the edge type
    EDGE_TYPE = 'type'

    def __init__(self, nodes=None, edges=None):

        self.graph = MultiGraph()

        # Nodes
        nodes = nodes or {}
        for node_id, (lon, lat) in nodes.items():
            self.add_node(node_id, lon, lat)

        # Edges
        edges = edges or {}
        for (n1, n2), (edge_type, dict_attrs) in edges.items():
            self.add_edge(n1, n2, edge_type, **dict_attrs)

    @property
    def num_of_nodes(self):
        """The number of nodes in the topology."""
        return self.graph.number_of_nodes()

    @property
    def num_of_edges(self):
        """The number of nodes in the topology."""
        return self.graph.number_of_edges()

    @property
    def nodes(self):
        """The nodes in the graph."""
        return self.graph.nodes

    def add_node(self, node_id, longitude, latitude, **node_attrs):
        """Add a node defined by its label.

        :param int node_id: Node id.
        :param dict node_attrs: Node attributes.

        :raises:
            :py:class:`RuntimeError`: if a node with the same ID already exists.
        """

        # If there is no node with the same ID already, raise exception
        if self.graph.has_node(node_id):
            raise RuntimeError("Node %s already exists." % node_id)

        # Adding longitude and latitude to attributes
        node_attrs[self.NODE_LONG] = longitude
        node_attrs[self.NODE_LAT] = latitude
        # Create node
        self.graph.add_node(node_id, **node_attrs)

    def get_node(self, node_id):
        """Return node from its ID."""

        try:
            return self.graph.nodes[node_id]
        except KeyError:
            raise KeyError("Node %s does not exist." % node_id)

    def distance(self, n1, n2):
        """
        Calculate the distance between two nodes on the Earth.

        :param int n1: ID of the first node.
        :param int n2: ID of the second node.
        :returns: Distance in cm
        :rtype: float
        """
        n1 = self.get_node(n1)
        n2 = self.get_node(n2)

        return utility_distance(
            n1[self.NODE_LONG],
            n1[self.NODE_LAT],
            n2[self.NODE_LONG],
            n2[self.NODE_LAT]
        )

    def add_edge(self, node1, node2, edge_type, **edge_attrs):
        """Add an edge between two nodes in the graph.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param obj edge_type: Edge type.
        :param dict edge_attrs: Additional attributes of the edge.

        :raises:
            :py:class:`RuntimeError`: if an edge with the same type already exists.
        """

        _ = self.get_node(node1)
        _ = self.get_node(node2)

        # If an edge of the given type between the two nodes already exits: error
        # We take into account that the graph in undirected
        with suppress(KeyError):
            types_existing_edges = [e[self.EDGE_TYPE] == edge_type
                                    for e in self.get_edges(node1, node2).values()]

            if any(types_existing_edges):
                raise RuntimeError(
                    'Already existing edge %s between nodes %s and %s' % (edge_type, node1, node2)
                )

        # Adding type to attributes
        edge_attrs[self.EDGE_TYPE] = edge_type
        # Create edge
        self.graph.add_edge(node1, node2, **edge_attrs)

    def get_edges(self, node1, node2, edge_types=None):
        """Return all the edges between two nodes.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param iterable edge_types: If provided, return only the edges for these types.
        :returns: A dictionary where the keys are the edge number
            and the values a dictionary of the edges attributes.
        :rtype: dict
        :raises: :py:class:`ValueError`: if there is no edge between the nodes.
        """
        try:
            edges = dict(self.graph[node1][node2])
        except KeyError:
            # undirected graph
            try:
                edges = dict(self.graph[node2][node1])
            except KeyError:
                raise KeyError('No edge between nodes %s and %s' % (node1, node2))

        # Filter if necessary
        if edge_types:
            edges = {e: v for e, v in edges.items()
                     if edges[e][self.EDGE_TYPE] in edge_types}

            # If dict is empty, there is no edge
            if not edges:
                raise KeyError('No edge between nodes %s and %s' % (node1, node2))

        return edges

    def _get_min_weight(self, weight, allowed_types, best_types, node1, node2, _d):
        """
        Find the edge with the minimum weight between two nodes.

        :param str weight: Weight name.
        :param allowed_types: Edge type(s) allowed to build path.
        :type allowed_types: dict-like object with TransportType as keys.
        :param dict best_types: dictionnary containing the best type for each pair of nodes.

        :note: other arguments are the ones needed by the NetworkX API.
        """

        # The algorithm uses the full (symmetric) matrix
        # instead of considering the upper triangular only.
        # This means that for two nodes u ≠ v, the weight is extracted twice:
        # once for (u,v), once for (v,u). We do not need to do the work for both.
        if (node2, node1) in best_types:
            return None

        # Get all edges between the two nodes
        edges = self.get_edges(node1, node2)

        # Get all the weights but keep only those that are allowed
        # This will throw a KeyError is some edges do not have the weight specified
        try:
            weights = [(attrs[self.EDGE_TYPE], attrs[weight]) for attrs in edges.values()
                       if attrs[self.EDGE_TYPE] in allowed_types]
        except KeyError:
            raise KeyError(
                "No edge with attribute %s found between nodes %s and %s" % (
                    weight, node1, node2
                ))

        # If there is no valid edge
        if not weights:
            return None

        # If the weights need to be weighted. Skip if they are all None.
        if list(set(allowed_types.values()))[0]:
            weights = [(t, w * allowed_types[t]) for t, w in weights]

        # Otherwise, save type and return minimum weight
        min_type, min_weight = min(weights, key=lambda t: t[1])
        best_types[(node1, node2)] = min_type
        return min_weight

    def get_shortest_path(self, node1, node2, criterion, allowed_types, edge_data=None):
        """Find the shortest path between two nodes using specified edges.

        :param obj node1: Label of the first node.
        :param obj node2: Label of the second node.
        :param str criterion: Criterion used to find shortest path. Must be an edge attribute.
        :param dict allowed_types: Edge type(s) allowed to build path.
            The keys are the edge types, the values (if any) indicate the preference for each type.
        :param iter(str) edge_data: Edge attributes for which data along the path is requested.
        :returns: A 3-tuple containing in this order

            * the score of the optimal path
            * the list of nodes along the path
            * a dict containing the edge type and values for the attributes specified in edge_data
              (data are stored in numpy arrays)

        :rtype: tuple
        :raises:
            :py:class:`.ValueError`: if nodes dont exists or no path has been found between them.

        :note: If `allowed_types` is a dict-like object, the weight of an edge will be weighted
            by the value of the given edge type.
        """

        # Check that the nodes exist
        for node in node1, node2:
            if not self.graph.has_node(node):
                raise ValueError("Node %s does not exist." % node)

        # Using partial function to pass the edge types
        # The weight_func will be called for each edge on the graph
        # Even those which will not been part of the optimial path
        # So best_types cannot be a simple list to which we append values
        # Instead it will be a dict where the key is a pair of nodes
        # We will extract the relevant values and attributes afterwards
        best_types = {}
        weight_func = partial(self._get_min_weight, criterion, allowed_types, best_types)

        # Calculate path
        try:
            score, path = single_source_dijkstra(
                self.graph, node1, node2, weight=weight_func)
        except NetworkXNoPath:
            raise RuntimeError("No path found with type %s between %s and %s." %
                               (allowed_types, node1, node2))

        # Build the dict containing the attributes data
        # Because we do not want to assume anything about the data,
        # we build an intermediate list first.
        data = {}
        # Always extract the edge types
        edge_types = []
        for u, v in zip(path, path[1:]):
            # The key could be (u,v) or (v,u)
            try:
                edge_types.append(best_types[(u, v)])
            except KeyError:
                edge_types.append(best_types[(v, u)])
        data[self.EDGE_TYPE] = np.array(edge_types)

        # Additional data if needed
        edge_data = edge_data or []
        for attr in edge_data:
            try:
                temp_list = [
                    list(self.get_edges(u, v, [t]).values())[0][attr]
                    for u, v, t in zip(
                        path, path[1:], data[self.EDGE_TYPE])]
                data[attr] = np.array(temp_list)
            except KeyError:
                raise KeyError("Some edges do not have the attribute '%s'." % attr)

        return (score, path, data)

    def add_energy_based_edges(
            self, edge_types, num_edges_per_step, max_iterations,
            degree_energy_factor, distance_energy_factor, rng,
            attribute_name='distance'):
        """
        This algorithm creates edges based on an energy sampling mechanism.
        At each iteration, the total energy (degree energy + potential energy)
        is calculated between all nodes and transformed into a Boltzmann distribution.
        Edges are more likely to be created between nodes which are:

        * close
        * already connected to other nodes

        A fixed number of random connections are then created based on this distribution,
        masking previous connections and self-edges.
        The process is repeated until either the maximum number of iterations is reached,
        or all nodes have been connected at least once to another node.

        :param iter edge_types: Types of the edges to add.
        :param int num_edges_per_step: Number of edges to create at each step
        :param int max_iterations: Maximum number of iterations.
            If None, the algorithm stops when each node has been connected at least once.
        :param float degree_energy_factor: Multiplier applied to the degree enery component
            during the search for new edges (higher means more prominent).
        :param float distance_energy_factor: Multiplier applied to the distance energy component
            during the search for new edges (higher means more prominent).
        :param rng: Random number generator.
        :type rng: :py:class:`RandomGenerator<city_graph.utils.RandomGenerator>`

        :note: The algorithm does take into account already exisiting edges
            to compute the probabilities.
        """

        print("[Topology] Starting energy sampling algorithm to build edges.")

        # Step 1: prepare
        # Compute all distances: graph is undirected so we want all combinations with replacements
        # so we should have in total C(2,n) + n distances
        # TODO: We compute things twice for now, so things might be improved
        array_shape = (self.num_of_nodes,) * 2
        distances = np.zeros(shape=array_shape)
        array_size = distances.size
        for (i, n1), (j, n2) in product(enumerate(self.nodes), repeat=2):
            distances[i, j] = self.distance(n1, n2)

        # Normalize by maximum distance - save scaling factor as it will be needed
        max_distance = distances.max()
        distances /= max_distance

        # Adjacency matrix (not taking into account previously existing edges)
        adjacency_matrix = np.zeros(shape=array_shape)

        # Step 2: sampling
        step = 0
        termination = TerminationReason.NO_TERMINATION
        while termination == TerminationReason.NO_TERMINATION:

            # This loop could be put in a separate function.
            # We will see if it is necessary

            # Sampling step
            # a) compute the degree energies
            degree_energy = -1 / 2 * (
                adjacency_matrix.sum(0, keepdims=True) +
                adjacency_matrix.sum(1, keepdims=True)
            )
            if degree_energy_factor:
                degree_energy = degree_energy_factor * degree_energy

            # b) compute the distance energies
            distance_energy = distances
            if distance_energy_factor:
                distance_energy = distance_energy_factor * distances

            # c) compute edge sampling probability distribution
            total_energy = degree_energy + distance_energy

            # mask existing edges
            total_energy[adjacency_matrix > EPS_PRECISION] = np.inf
            # mask self-edges
            np.fill_diagonal(total_energy, np.inf)

            # compute Boltzmann distribution and normalize it
            bolz_dist = np.exp(-total_energy)
            bolz_dist /= bolz_dist.sum()

            # d) sample step
            # sample indices to activate
            sample_indices = rng.choice(
                np.arange(array_size), size=(num_edges_per_step,),
                p=bolz_dist.ravel(), replace=True)

            # unravel indices - get a tuple
            unravel_indices = np.unravel_index(sample_indices, array_shape)

            # add sample edges to the adjacency matrix
            # and make matric symmetric
            adjacency_matrix[unravel_indices] = 1
            adjacency_matrix[unravel_indices[::-1]] = 1

            # Increase counter
            step += 1

            # Check termination
            # Maximum step number reached
            if max_iterations:
                if step >= max_iterations:
                    termination = TerminationReason.MAX_ITER
            # All nodes are connected
            if adjacency_matrix.sum(0).min() > EPS_PRECISION:
                termination = TerminationReason.ALL_NODES_CONNECTED

        # Inform the user why the algorithm has stopped
        print("[Topology] Sampling finished after", step, "steps. Reason:", termination)

        # Build edges from the adjacency matrix
        # Rescale the distances
        distances *= max_distance

        # Get pair of nodes to connect
        # Here we use indices because we will need them later on for the distances
        pairs = [(i, j) for i, j in zip(*adjacency_matrix.nonzero())]

        # Create edges
        old_num_edges = self.num_of_edges
        node_ids = list(self.nodes)
        for i, j in pairs:
            for edge_type in edge_types:

                # TODO: exception might be raised here because we now check
                # That there is no outgoing/incoming edges.
                # Should we fix when we use an actual triangular matrix
                with suppress(RuntimeError):
                    self.add_edge(node_ids[i], node_ids[j], edge_type,
                                  **{attribute_name: distances[i, j]})

        # Inform that edges have been built
        print("[Topology] %i edges have been created" % (self.num_of_edges - old_num_edges))

    def add_edges_between_centroids(self, edge_types, num_centroids, rng, attribute_name='distance'):
        """
        This algorithm creates edges between central nodes. These central nodes
        are determined by a clustering algorithm (here the Fluid Communities algorithm).

        :param iter edge_types: Types of the edges to add.
        :param int num_centroids: Number of centroids.
        :param rng: Random number generator.
        :type rng: :py:class:`RandomGenerator<city_graph.utils.RandomGenerator>`
        """

        print("[Topology] Starting building edges between %s central nodes." % num_centroids)

        # Calculate clusters
        # TODO: I think it would make sense to instead use e.g. a k-means for two reasons:
        #  * this algo assumes that the clusters have the same density, which is not necessarily
        # application to cities (some areas are more crowded)
        #  *  the implementation needs the graph to be fully connected to begin with. It seems
        # to be a limitation
        clusters = (list(c) for c in asyn_fluidc(
            self.graph, k=num_centroids, seed=rng.rand_int()))

        # Extract centroids: we take the node with the highest degree
        centroids = [c[int(np.argmax([self.graph.degree[n] for n in c]))] for c in clusters]

        # Create temporary graph for the centroids
        tmp_graph = Graph()
        tmp_graph.add_nodes_from(centroids)
        # We need the combinations because the graph is undirected and we dont want self-edges

        for n1, n2 in combinations(centroids, 2):
            tmp_graph.add_edge(n1, n2, **{attribute_name: self.distance(n1, n2)})

        # Calculate subgraph with the minimum sum of edge weights
        # TODO: to investigate why we do this here...
        subgraph = minimum_spanning_tree(tmp_graph, weight=attribute_name)

        # Build edges
        # Here we can reuse the previously calculated distances
        old_num_edges = self.num_of_edges
        for (n1, n2) in subgraph.edges:
            for edge_type in edge_types:

                # TODO: exception might be raised here because we now check
                # That there is no outgoing/incoming edges.
                # Should we fix when we use an actual triangular matrix
                with suppress(RuntimeError):
                    self.add_edge(n1, n2, edge_type, **subgraph[n1][n2])

        # Inform that edges have been built
        print("[Topology] %i edges have been created" % (self.num_of_edges - old_num_edges))

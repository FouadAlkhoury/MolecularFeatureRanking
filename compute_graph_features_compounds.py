# Compute graph-level features for compound datasets.
import datetime
import math
import os
import pickle
import subprocess
import networkx as nx
import numpy as np
from scipy.stats import kurtosis, skew
import graphlets
from util import writeToReport

def list_to_str(list):
    str_list = ''
    for l in list:
        str_list += str(l) + ','
    return str_list

def load_compounds_dataset(filename):
    with open(filename, 'rb') as f:
        compounds_dataset = pickle.load(f)
    print(len(compounds_dataset))
    return compounds_dataset

def compute_feature_distribution(feature_list):
    if (isinstance(feature_list, dict)):
        feature_list = list(feature_list.values())
    mean = np.mean(feature_list)
    std = np.std(feature_list)
    skewness = skew(feature_list)
    if (math.isnan(skewness)):
        skewness = 0
    kurtosis_value = kurtosis(feature_list)
    if (math.isnan(kurtosis_value)):
        kurtosis_value = 0

    return str(mean) + ',' + str(std) + ',' + str(skewness) + ',' + str(kurtosis_value) + ','


def features_header(feature):
    return 'avg. ' + feature + ', std. ' + feature + ', skewness ' + feature + ', kurtosis ' + feature + ','

def write_csv(graph, path):
    """
    Export a NetworkX graph to CSV edge-list format.
    Nodes are 0-indexed.
    """
    mapping = {n: i for i, n in enumerate(graph.nodes())}

    with open(path, "w") as f:
        for u, v in graph.edges():
            f.write(f"{mapping[u]},{mapping[v]}\n")


def run_glasgow(pattern_file, target_file, induced=False, count=False):
    cmd = ["glascow/glasgow-subgraph-solver/build/glasgow_subgraph_solver"]

    if induced:
        cmd.append("--induced")
    if count:
        cmd.append("--count-solutions")

    cmd.append("--pattern-format=csv")
    cmd.append("--target-format=csv")
    cmd.append("--format=csv")
    cmd.append("--count-solutions")
    cmd.extend([pattern_file, target_file])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def parse_solution_count(stdout):
    """Parse Glasgow output and return the integer solution count."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("solution_count"):
            return int(line.split("=")[1].strip())

    return 0


def compute_features(graph_id, graph):
    start = datetime.datetime.now()

    G = nx.Graph(graph.graph)
    directory_graphs = 'Real/' + dataset + '/'
    os.makedirs(directory_graphs, exist_ok=True)
    pickle.dump(G, open(directory_graphs + dataset + '_' + str(graph_id) + '.pickle', 'wb'))
    y = graph.graph['class_label']
    nodes_count = G.number_of_nodes()
    edges_count = G.number_of_edges()

    report_file_features = directory_features + str(graph_id) + '.pickle.csv'

    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random, ' + ' Target ')

    if (nx.is_connected(G)):

        diameter = nx.diameter(G)

        lengths = dict(nx.all_pairs_shortest_path_length(G))
        all_lengths = [l for target_dict in lengths.values() for l in target_dict.values()]

        mean_path_length = np.mean(all_lengths)
        std_path_length = np.std(all_lengths)
    else:
        diameter = -1
        mean_path_length = -1
        std_path_length = -1

    metrics_count = 26

    subgraph_nodes = list(G.nodes)

    graphlets_arr_counting = np.zeros(len(graphs))

    total_time_graphlets_list = []

    target_file = os.path.abspath(dataset + '/' + 'target.csv')
    write_csv(G, target_file)

    for graph_index, g in enumerate(graphs):
        start_time_graphlet = datetime.datetime.now()

        pattern_file = os.path.abspath(dataset + '/' + 'pattern_' + str(graph_index) + '.csv')
        write_csv(g, pattern_file)

        output = run_glasgow(pattern_file, target_file)
        count = parse_solution_count(output)

        end_time_graphlet = datetime.datetime.now()
        total_time_graphlet = (end_time_graphlet - start_time_graphlet)
        total_time_graphlets_list.append(total_time_graphlet)

        graphlets_arr_counting[graph_index] = count

    graphlets_arr_binary = [1 if gr > 0 else 0 for gr in graphlets_arr_counting]

    start_time_degree_centrality = datetime.datetime.now()
    degree_centrality = nx.degree_centrality(G)
    end_time_degree_centrality = datetime.datetime.now()
    total_time_degree_centrality = (end_time_degree_centrality - start_time_degree_centrality)

    start_time_eigenvector_centrality = datetime.datetime.now()
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100, tol=1e-03)
    end_time_eigenvector_centrality = datetime.datetime.now()
    total_time_eigenvector_centrality = (end_time_eigenvector_centrality - start_time_eigenvector_centrality)

    start_time_closeness_centrality = datetime.datetime.now()
    closeness_centrality = nx.closeness_centrality(G)
    end_time_closeness_centrality = datetime.datetime.now()
    total_time_closeness_centrality = (end_time_closeness_centrality - start_time_closeness_centrality)

    start_time_harmonic_centrality = datetime.datetime.now()
    harmonic_centrality = nx.harmonic_centrality(G)
    end_time_harmonic_centrality = datetime.datetime.now()
    total_time_harmonic_centrality = (end_time_harmonic_centrality - start_time_harmonic_centrality)

    start_time_betweenness_centrality = datetime.datetime.now()
    betweenness_centrality = nx.betweenness_centrality(G)
    end_time_betweenness_centrality = datetime.datetime.now()
    total_time_betweenness_centrality = (end_time_betweenness_centrality - start_time_betweenness_centrality)

    start_time_coloring_lf = datetime.datetime.now()
    coloring_largest_first = nx.coloring.greedy_color(G, strategy='largest_first')
    end_time_coloring_lf = datetime.datetime.now()
    total_time_coloring_lf = (end_time_coloring_lf - start_time_coloring_lf)

    start_time_coloring_sl = datetime.datetime.now()
    coloring_smallest_last = nx.coloring.greedy_color(G, strategy='largest_first')
    end_time_coloring_sl = datetime.datetime.now()
    total_time_coloring_sl = (end_time_coloring_sl - start_time_coloring_sl)

    start_time_coloring_is = datetime.datetime.now()
    coloring_independent_set = nx.coloring.greedy_color(G, strategy='independent_set')
    end_time_coloring_is = datetime.datetime.now()
    total_time_coloring_is = (end_time_coloring_is - start_time_coloring_is)

    start_time_coloring_rs = datetime.datetime.now()
    coloring_random_sequential = nx.coloring.greedy_color(G, strategy='random_sequential')
    end_time_coloring_rs = datetime.datetime.now()
    total_time_coloring_rs = (end_time_coloring_rs - start_time_coloring_rs)

    start_time_coloring_dfs = datetime.datetime.now()
    coloring_connected_sequential_dfs = nx.coloring.greedy_color(G, strategy='connected_sequential_dfs')
    end_time_coloring_dfs = datetime.datetime.now()
    total_time_coloring_dfs = (end_time_coloring_dfs - start_time_coloring_dfs)

    start_time_coloring_bfs = datetime.datetime.now()
    coloring_connected_sequential_bfs = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs')
    end_time_coloring_bfs = datetime.datetime.now()
    total_time_coloring_bfs = (end_time_coloring_bfs - start_time_coloring_bfs)

    start_time_node_clique_number = datetime.datetime.now()
    node_clique_number = nx.node_clique_number(G)
    end_time_node_clique_number = datetime.datetime.now()
    total_time_node_clique_number = (end_time_node_clique_number - start_time_node_clique_number)

    start_time_number_of_cliques = datetime.datetime.now()
    number_of_cliques = {n: sum(1 for c in nx.find_cliques(G) if n in c) for n in G}
    end_time_number_of_cliques = datetime.datetime.now()
    total_time_number_of_cliques = (end_time_number_of_cliques - start_time_number_of_cliques)
    start_time_clustering_coefficient = datetime.datetime.now()
    clustering_coefficient = nx.clustering(G)
    end_time_clustering_coefficient = datetime.datetime.now()
    total_time_clustering_coefficient = (end_time_clustering_coefficient - start_time_clustering_coefficient)

    start_time_square_clustering = datetime.datetime.now()
    square_clustering = nx.square_clustering(G)
    end_time_square_clustering = datetime.datetime.now()
    total_time_square_clustering = (end_time_square_clustering - start_time_square_clustering)

    start_time_average_neighbor_degree = datetime.datetime.now()
    average_neighbor_degree = nx.average_neighbor_degree(G)
    end_time_average_neighbor_degree = datetime.datetime.now()
    total_time_average_neighbor_degree = (end_time_average_neighbor_degree - start_time_average_neighbor_degree)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    start_time_hubs = datetime.datetime.now()
    hubs, _ = nx.hits(G)
    end_time_hubs = datetime.datetime.now()
    total_time_hubs = (end_time_hubs - start_time_hubs)

    start_time_page_rank = datetime.datetime.now()
    page_rank = nx.pagerank(G)
    end_time_page_rank = datetime.datetime.now()
    total_time_page_rank = (end_time_page_rank - start_time_page_rank)

    start_time_core_number = datetime.datetime.now()
    G1 = G
    G1.remove_edges_from(nx.selfloop_edges(G1))
    core_number = nx.core_number(G1)
    end_time_core_number = datetime.datetime.now()
    total_time_core_number = (end_time_core_number - start_time_core_number)

    total_time_egonet = datetime.timedelta()
    total_time_triangles = datetime.timedelta()
    total_time_random = datetime.timedelta()

    atomic_num = nx.get_node_attributes(graph, 'atomic_num')
    degree_feature = nx.get_node_attributes(graph, 'degree')
    formal_charge = nx.get_node_attributes(graph, 'formal_charge')
    hybridization = nx.get_node_attributes(graph, 'hybridization')
    is_aromatic = nx.get_node_attributes(graph, 'is_aromatic')
    mass = nx.get_node_attributes(graph, 'mass')
    num_hs = nx.get_node_attributes(graph, 'num_hs')

    mol_wt = graph.graph['graph_features']['mol_wt']
    tpsa = graph.graph['graph_features']['tpsa']
    logp = graph.graph['graph_features']['logp']
    num_chiral_centers = graph.graph['graph_features']['num_chiral_centers']
    num_rotatable_bonds = graph.graph['graph_features']['num_rotatable_bonds']
    fingerprints = graph.graph['maccs_fingerprint']

    edges_within_egonet_list = []

    X = np.zeros([nx.number_of_nodes(G), metrics_count])
    for i, v in enumerate(G):

        start_time_degree = datetime.datetime.now()
        X[i][0] = G.degree(v)
        end_time_degree = datetime.datetime.now()
        total_time_degree = (end_time_degree - start_time_degree)
        X[i][1] = degree_centrality[subgraph_nodes[i]]
        neighborhood_degrees = [G.degree(n) for n in nx.neighbors(G, v)]
        if (len(neighborhood_degrees) == 0):
            max_neighbor_degree = 0
            min_neighbor_degree = 0
            std_neighbor_degree = 0
        else:
            max_neighbor_degree = np.max(neighborhood_degrees)
            min_neighbor_degree = np.min(neighborhood_degrees)
            std_neighbor_degree = np.std(neighborhood_degrees)
        X[i][2] = max_neighbor_degree
        X[i][3] = min_neighbor_degree
        X[i][4] = average_neighbor_degree[subgraph_nodes[i]]
        X[i][5] = std_neighbor_degree
        X[i][6] = eigenvector_centrality[subgraph_nodes[i]]
        X[i][7] = closeness_centrality[subgraph_nodes[i]]
        X[i][8] = harmonic_centrality[subgraph_nodes[i]]
        X[i][9] = betweenness_centrality[subgraph_nodes[i]]
        X[i][10] = coloring_largest_first[subgraph_nodes[i]]
        X[i][11] = coloring_smallest_last[subgraph_nodes[i]]
        X[i][12] = coloring_independent_set[subgraph_nodes[i]]
        X[i][13] = coloring_random_sequential[subgraph_nodes[i]]
        X[i][14] = coloring_connected_sequential_dfs[subgraph_nodes[i]]
        X[i][15] = coloring_connected_sequential_bfs[subgraph_nodes[i]]

        start_time_egonet = datetime.datetime.now()
        egonet = nx.ego_graph(G, v, radius=1)
        edges_within_egonet = nx.number_of_edges(egonet)
        edges_within_egonet_list.append(edges_within_egonet)
        end_time_egonet = datetime.datetime.now()
        total_time_egonet += (end_time_egonet - start_time_egonet)

        X[i][16] = edges_within_egonet
        X[i][17] = node_clique_number[subgraph_nodes[i]]
        X[i][18] = number_of_cliques[subgraph_nodes[i]]
        X[i][19] = clustering_coefficient[subgraph_nodes[i]]
        X[i][20] = square_clustering[subgraph_nodes[i]]
        X[i][21] = 0
        X[i][22] = 0
        start_time_triangles = datetime.datetime.now()
        X[i][23] = 1
        end_time_triangles = datetime.datetime.now()
        total_time_triangles += (end_time_triangles - start_time_triangles)

        X[i][24] = 0
        start_time_random = datetime.datetime.now()
        X[i][25] = np.random.normal(0, 1, 1)[0]
        end_time_random = datetime.datetime.now()
        total_time_random += (end_time_random - start_time_random)

    degrees = [d for _, d in G.degree()]
    degrees_str = compute_feature_distribution(degrees)
    eigenvector_centrality_str = compute_feature_distribution(eigenvector_centrality)
    closeness_centrality_str = compute_feature_distribution(closeness_centrality)
    harmonic_centrality_str = compute_feature_distribution(harmonic_centrality)
    betweenness_centrality_str = compute_feature_distribution(betweenness_centrality)
    coloring_largest_first_str = compute_feature_distribution(coloring_largest_first)
    edges_within_egonet_str = compute_feature_distribution(edges_within_egonet_list)
    node_clique_number_str = compute_feature_distribution(node_clique_number)
    number_of_cliques_str = compute_feature_distribution(number_of_cliques)
    clustering_coefficient_str = compute_feature_distribution(clustering_coefficient)
    square_clustering_str = compute_feature_distribution(square_clustering)
    page_rank_str = compute_feature_distribution(page_rank)
    hubs_str = compute_feature_distribution(hubs)
    core_number_str = compute_feature_distribution(core_number)
    random_per_graph = np.random.normal(0, 1, 1)[0]
    atomic_num_str = compute_feature_distribution(atomic_num)
    degree_feature_str = compute_feature_distribution(degree_feature)
    formal_charge_str = compute_feature_distribution(formal_charge)
    hybridization_str = compute_feature_distribution(hybridization)
    is_aromatic_str = compute_feature_distribution(is_aromatic)
    mass_str = compute_feature_distribution(mass)
    num_hs_str = compute_feature_distribution(num_hs)

    print(dataset)
    writeToReport(report_graph_level_features,
                  dataset + '_' + str(graph_id) + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + str(
                      diameter) + ','
                  + str(mean_path_length) + ',' + str(
                      std_path_length) + ',' + degrees_str + eigenvector_centrality_str + closeness_centrality_str + harmonic_centrality_str +
                  betweenness_centrality_str + coloring_largest_first_str + edges_within_egonet_str + node_clique_number_str + number_of_cliques_str +
                  clustering_coefficient_str + square_clustering_str + page_rank_str + hubs_str + core_number_str + str(
                      random_per_graph) + ',' + list_to_str(graphlets_arr_counting) + list_to_str(
                      graphlets_arr_binary) + atomic_num_str + degree_feature_str + formal_charge_str + hybridization_str + is_aromatic_str + mass_str + num_hs_str + str(
                      mol_wt) + ',' + str(tpsa) + ',' + str(logp) + ',' + str(num_chiral_centers) + ',' + str(
                      num_rotatable_bonds) + ',' + list_to_str(fingerprints) + str(y))  ## real: str(data.y[0].item())

    for x in X:
        writeToReport(report_file_features, list_to_str(x))
    writeToReport(report_file_features, '\n')

    writeToReport(test_file, str(graph_id) + '.pickle,' + str(y))

    end = datetime.datetime.now()
    total_time = (end - start)
    print('Computing all features + graphlets time: ' + str(total_time))

    report_file = 'reports/features_data/computing_time_' + dataset + '_compounds_100_v3_datasets.csv'
    writeToReport(report_file,
                  'graph_name , nodes_count , edges_count, degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random,' + list_to_str([ind for ind in range(0, 31)]) + ' Total Time ')
    writeToReport(report_file, str(graph_id) + ',' + str(nodes_count) + ',' + str(
        edges_count) + ',' + str(total_time_degree) + ',' + str(total_time_degree_centrality) + ',' + str(
        total_time_average_neighbor_degree) + ',' + str(total_time_average_neighbor_degree)
                  + ',' + str(total_time_average_neighbor_degree) + ',' + str(
        total_time_average_neighbor_degree) + ',' +
                  str(total_time_eigenvector_centrality) + ',' + str(total_time_closeness_centrality) + ',' + str(
        total_time_harmonic_centrality) + ',' + str(total_time_betweenness_centrality)
                  + ',' + str(total_time_coloring_lf) + ',' + str(total_time_coloring_sl) + ',' + str(
        total_time_coloring_is) + ',' + str(total_time_coloring_rs) + ',' + str(total_time_coloring_dfs)
                  + ',' + str(total_time_coloring_bfs) + ',' + str(total_time_egonet) + ',' + str(
        total_time_node_clique_number) + ',' + str(total_time_number_of_cliques) + ',' + str(
        total_time_clustering_coefficient)
                  + ',' + str(total_time_square_clustering) + ',' + str(total_time_page_rank) + ',' + str(
        total_time_hubs) + ',' + str(total_time_triangles) + ',' + str(total_time_core_number) + ',' + str(
        total_time_random) + ',' + list_to_str(total_time_graphlets_list) + ',' + str(total_time))

graphs = graphlets.graphs
compounds_datasets = [
    'CHEMBL1163125',
    'CHEMBL1914',
    'CHEMBL204',
    'CHEMBL214',
    'CHEMBL218',
    'CHEMBL220',
    'CHEMBL230',
    'CHEMBL261',
    'CHEMBL2973',
    'CHEMBL4822',
]


for dataset in compounds_datasets:
    test_file = 'data/real/' + dataset + '.train'

    if not os.path.exists(dataset):
        os.mkdir(dataset)

    print(dataset)
    graphs_dataset = load_compounds_dataset('./pickles_100_v3/' + dataset + '_graphs.pkl')

    is_compute_features = True

    directory_features = 'data/real/' + dataset + '_fingerprints_v3/'
    os.makedirs(directory_features, exist_ok=True)
    report_graph_level_features = directory_features + dataset + '_compounds.csv'

    writeToReport(
        report_graph_level_features,
        'graph_id, nodes count, edges count, diameter, avg. path length, std path length, '
        + features_header('degree')
        + features_header('eigenvector_cent.')
        + features_header('closeness cent.')
        + features_header('harmonic cent.')
        + features_header('betweenness cent. ')
        + features_header('coloring LF')
        + features_header('edges within egonet')
        + features_header('node clique number')
        + features_header('number of cliques')
        + features_header('clustering coefficient')
        + features_header('square clustering coefficient')
        + features_header('pagerank')
        + features_header('hubs')
        + features_header('core number')
        + 'random,'
        + list_to_str(['c. g_' + str(_) for _ in range(0, 31)])
        + list_to_str(['b. g_' + str(_) for _ in range(0, 31)])
        + features_header('atomic_num')
        + features_header('degree_feature')
        + features_header('formal_charge')
        + features_header('hybridization')
        + features_header('is_aromatic')
        + features_header('mass')
        + features_header('num_hs')
        + 'mol_wt, tpsa, logp, num_chiral_centers, num_rotatable_bonds,'
        + list_to_str(['f_' + str(f_i) for f_i in range(0, 167)])
        + '  Target',
    )

    if (is_compute_features):
        for graph_id, graph in enumerate(graphs_dataset):
            if (graph_id < 10000):
                print(graph_id)
                compute_features(graph_id, graph)


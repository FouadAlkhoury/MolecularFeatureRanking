# The script computes graph-level features of the three types: Graphlet Features, Aggregated Node Features, and Global Graph Properties.
# The computed features are saved under data/real/dataset.train

import math
import os
import collections
import numpy as np
import torch
from networkx.algorithms.centrality import eigenvector_centrality
from sympy.codegen.cnodes import sizeof
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import ppi, Reddit
from torch_geometric.datasets import wikics, AttributedGraphDataset, Actor, GitHub, HeterophilousGraphDataset, Twitch, \
    Airports, TUDataset, BA2MotifDataset
from torch_geometric.datasets import Amazon, Coauthor, GNNBenchmarkDataset, NELL, CitationFull, Reddit, Reddit2, Flickr, \
    Yelp, AmazonProducts, Entities
import networkx as nx
from networkx.algorithms import community, bipartite
import csv
import datetime
import random
from sklearn.feature_selection import mutual_info_classif
import torch_geometric.transforms as T
from sklearn.ensemble import RandomForestClassifier
import util
from util import writeToReport
from sklearn.model_selection import cross_val_score
import pickle
from torch_geometric.nn import global_mean_pool
import ReadData
from torch_geometric.loader import DataLoader
from collections import OrderedDict
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import export_text
import graphlets
from scipy.stats import skew, kurtosis, entropy

def list_to_str(list):
    str_list = ''
    for l in list:
        # print(l)
        str_list += str(l) + ','
    return str_list

graphs = graphlets.graphs
print(len(graphs))


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


def one_hot(y):
    labels = [0, 0, 0, 0, 0]
    labels[y] = 1
    return labels


def compute_features(graph_id, graph):
    start = datetime.datetime.now()
    graph_name = graph
    # data = torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + graph, 'rb')))#real
    data = pickle.load(open(dataset + '/' + graph, 'rb'))  # synthetic
    G = to_networkx(data, to_undirected=True)

    # data = graph   #real
    print(graph_id)
    # G = to_networkx(graph, to_undirected=True)

    directory_graphs = 'Real/' + dataset + '/'
    os.makedirs(directory_graphs, exist_ok=True)
    pickle.dump(G, open(directory_graphs + dataset + '_' + str(graph_id) + '.pickle', 'wb'))
    # data = torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + dataset, 'rb')))
    print(graph)
    print('data:')
    print(data)
    print(data.y.item())
    y = data.y.item()
    # y = data.y[0].item()   ## for real datasets
    nodes_count = data.num_nodes
    edges_count = data.num_edges / 2

    report_file_features = directory_features + str(graph_id) + '.pickle.csv'

    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random, ' + ' Target ')

    G = nx.Graph()
    G = to_networkx(data, to_undirected=True)
    print('to network x')

    start_total = datetime.datetime.now()
    nodes_count = nx.number_of_nodes(G)
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

    print(nx.number_of_nodes(G))
    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    subgraph_nodes = list(G.nodes)

    print('Number of nodes: ')
    print(nx.number_of_nodes(G))
    print(nx.number_of_edges(G))

    features_count = 0

    graphlets_arr_counting = np.zeros(len(graphs))
    feature_index = 0

    total_time_graphlets_list = []
    total_time_graphlets = datetime.timedelta()

    for graph_index, g in enumerate(graphs):
        print('Graphlet: ' + str(graph_index))

        start_time_graphlet = datetime.datetime.now()

        GM = nx.isomorphism.GraphMatcher(G, g)
        g_iter = GM.subgraph_isomorphisms_iter()
        counter_graphlets = 0
        unique_matches = set()
        for match in g_iter:
            match_nodes = tuple(sorted(match.keys()))
            unique_matches.add(match_nodes)

        counter_graphlets = len(unique_matches)
        end_time_graphlet = datetime.datetime.now()
        total_time_graphlet = datetime.timedelta()
        total_time_graphlet = (end_time_graphlet - start_time_graphlet)
        total_time_graphlets_list.append(total_time_graphlet)
        total_time_graphlets += total_time_graphlet
        graphlets_arr_counting[graph_index] = counter_graphlets
    graphlets_arr_binary = [1 if gr > 0 else 0 for gr in graphlets_arr_counting]

    print('Graphlets time: ' + str(total_time_graphlets))

    start_time_features = datetime.datetime.now()

    start_time_degree_centrality = datetime.datetime.now()
    degree_centrality = nx.degree_centrality(G)
    end_time_degree_centrality = datetime.datetime.now()
    total_time_degree_centrality = datetime.timedelta()
    total_time_degree_centrality = (end_time_degree_centrality - start_time_degree_centrality)

    start_time_eigenvector_centrality = datetime.datetime.now()
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100, tol=1e-03)
    end_time_eigenvector_centrality = datetime.datetime.now()
    total_time_eigenvector_centrality = datetime.timedelta()
    total_time_eigenvector_centrality = (end_time_eigenvector_centrality - start_time_eigenvector_centrality)

    start_time_closeness_centrality = datetime.datetime.now()
    closeness_centrality = nx.closeness_centrality(G)
    end_time_closeness_centrality = datetime.datetime.now()
    total_time_closeness_centrality = datetime.timedelta()
    total_time_closeness_centrality = (end_time_closeness_centrality - start_time_closeness_centrality)

    start_time_harmonic_centrality = datetime.datetime.now()
    harmonic_centrality = nx.harmonic_centrality(G)
    end_time_harmonic_centrality = datetime.datetime.now()
    total_time_harmonic_centrality = datetime.timedelta()
    total_time_harmonic_centrality = (end_time_harmonic_centrality - start_time_harmonic_centrality)

    start_time_betweenness_centrality = datetime.datetime.now()
    betweenness_centrality = nx.betweenness_centrality(G)
    end_time_betweenness_centrality = datetime.datetime.now()
    total_time_betweenness_centrality = datetime.timedelta()
    total_time_betweenness_centrality = (end_time_betweenness_centrality - start_time_betweenness_centrality)

    start_time_coloring_lf = datetime.datetime.now()
    coloring_largest_first = nx.coloring.greedy_color(G, strategy='largest_first')
    end_time_coloring_lf = datetime.datetime.now()
    total_time_coloring_lf = datetime.timedelta()
    total_time_coloring_lf = (end_time_coloring_lf - start_time_coloring_lf)

    start_time_coloring_sl = datetime.datetime.now()
    # coloring_smallest_last = nx.coloring.greedy_color(G, strategy='smallest_last')
    coloring_smallest_last = nx.coloring.greedy_color(G, strategy='largest_first')
    end_time_coloring_sl = datetime.datetime.now()
    total_time_coloring_sl = datetime.timedelta()
    total_time_coloring_sl = (end_time_coloring_sl - start_time_coloring_sl)

    start_time_coloring_is = datetime.datetime.now()
    coloring_independent_set = nx.coloring.greedy_color(G, strategy='independent_set')
    end_time_coloring_is = datetime.datetime.now()
    total_time_coloring_is = datetime.timedelta()
    total_time_coloring_is = (end_time_coloring_is - start_time_coloring_is)

    start_time_coloring_rs = datetime.datetime.now()
    coloring_random_sequential = nx.coloring.greedy_color(G, strategy='random_sequential')
    end_time_coloring_rs = datetime.datetime.now()
    total_time_coloring_rs = datetime.timedelta()
    total_time_coloring_rs = (end_time_coloring_rs - start_time_coloring_rs)

    start_time_coloring_dfs = datetime.datetime.now()
    coloring_connected_sequential_dfs = nx.coloring.greedy_color(G, strategy='connected_sequential_dfs')
    end_time_coloring_dfs = datetime.datetime.now()
    total_time_coloring_dfs = datetime.timedelta()
    total_time_coloring_dfs = (end_time_coloring_dfs - start_time_coloring_dfs)

    start_time_coloring_bfs = datetime.datetime.now()
    coloring_connected_sequential_bfs = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs')
    end_time_coloring_bfs = datetime.datetime.now()
    total_time_coloring_bfs = datetime.timedelta()
    total_time_coloring_bfs = (end_time_coloring_bfs - start_time_coloring_bfs)

    start_time_node_clique_number = datetime.datetime.now()
    node_clique_number = nx.node_clique_number(G)
    end_time_node_clique_number = datetime.datetime.now()
    total_time_node_clique_number = datetime.timedelta()
    total_time_node_clique_number = (end_time_node_clique_number - start_time_node_clique_number)

    # number_of_cliques = nx.number_of_cliques(G)
    start_time_number_of_cliques = datetime.datetime.now()
    number_of_cliques = {n: sum(1 for c in nx.find_cliques(G) if n in c) for n in G}
    end_time_number_of_cliques = datetime.datetime.now()
    total_time_number_of_cliques = datetime.timedelta()
    total_time_number_of_cliques = (end_time_number_of_cliques - start_time_number_of_cliques)
    start_time_clustering_coefficient = datetime.datetime.now()
    clustering_coefficient = nx.clustering(G)
    end_time_clustering_coefficient = datetime.datetime.now()
    total_time_clustering_coefficient = datetime.timedelta()
    total_time_clustering_coefficient = (end_time_clustering_coefficient - start_time_clustering_coefficient)
    print('clustering coefficient: ')
    print(clustering_coefficient)

    start_time_square_clustering = datetime.datetime.now()
    square_clustering = nx.square_clustering(G)
    end_time_square_clustering = datetime.datetime.now()
    total_time_square_clustering = datetime.timedelta()
    total_time_square_clustering = (end_time_square_clustering - start_time_square_clustering)

    start_time_average_neighbor_degree = datetime.datetime.now()
    average_neighbor_degree = nx.average_neighbor_degree(G)
    end_time_average_neighbor_degree = datetime.datetime.now()
    total_time_average_neighbor_degree = datetime.timedelta()
    total_time_average_neighbor_degree = (end_time_average_neighbor_degree - start_time_average_neighbor_degree)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    start_time_hubs = datetime.datetime.now()
    hubs, authorities = nx.hits(G)
    end_time_hubs = datetime.datetime.now()
    total_time_hubs = datetime.timedelta()
    total_time_hubs = (end_time_hubs - start_time_hubs)

    start_time_page_rank = datetime.datetime.now()
    page_rank = nx.pagerank(G)
    end_time_page_rank = datetime.datetime.now()
    total_time_page_rank = datetime.timedelta()
    total_time_page_rank = (end_time_page_rank - start_time_page_rank)
    print('page rank:')
    print(page_rank)
    print(type(page_rank))

    start_time_core_number = datetime.datetime.now()
    G1 = G
    G1.remove_edges_from(nx.selfloop_edges(G1))
    core_number = nx.core_number(G1)
    end_time_core_number = datetime.datetime.now()
    total_time_core_number = datetime.timedelta()
    total_time_core_number = (end_time_core_number - start_time_core_number)

    end_time_features = datetime.datetime.now()
    total_time_features = datetime.timedelta()
    total_time_features = (end_time_features - start_time_features)
    print('total time features: ')
    print(total_time_features)

    total_time_egonet = datetime.timedelta()
    total_time_triangles = datetime.timedelta()
    total_time_random = datetime.timedelta()

    edges_within_egonet_list = []

    X = np.zeros([nx.number_of_nodes(G), metrics_count])
    Y = np.zeros([nx.number_of_nodes(G)])
    for i, v in enumerate(G):
        print(i)

        start_time_degree = datetime.datetime.now()
        X[i][0] = G.degree(v)
        end_time_degree = datetime.datetime.now()
        total_time_degree = datetime.timedelta()
        total_time_degree = (end_time_degree - start_time_degree)
        # print(degree_centrality[i])
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
        X[i][21] = page_rank[subgraph_nodes[i]]
        X[i][22] = hubs[subgraph_nodes[i]]

        start_time_triangles = datetime.datetime.now()
        X[i][23] = nx.triangles(G, subgraph_nodes[i])
        end_time_triangles = datetime.datetime.now()
        total_time_triangles += (end_time_triangles - start_time_triangles)

        X[i][24] = core_number[subgraph_nodes[i]]

        start_time_random = datetime.datetime.now()
        X[i][25] = np.random.normal(0, 1, 1)[0]
        end_time_random = datetime.datetime.now()
        total_time_random += (end_time_random - start_time_random)

        # Y[i] = data.y[subgraph_nodes[i]]

    # X = np.concatenate((X, res), axis=1)

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

    writeToReport(report_graph_level_features,
                  dataset + '_' + str(graph_id) + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + str(
                      diameter) + ','
                  + str(mean_path_length) + ',' + str(
                      std_path_length) + ',' + degrees_str + eigenvector_centrality_str + closeness_centrality_str + harmonic_centrality_str +
                  betweenness_centrality_str + coloring_largest_first_str + edges_within_egonet_str + node_clique_number_str + number_of_cliques_str +
                  clustering_coefficient_str + square_clustering_str + page_rank_str + hubs_str + core_number_str + str(
                      random_per_graph) + ',' + list_to_str(graphlets_arr_counting) + list_to_str(
                      graphlets_arr_binary) + str(y))  ## real: str(data.y[0].item())

    for i, x in enumerate(X):
        writeToReport(report_file_features, list_to_str(x))
    writeToReport(report_file_features, '\n')

    writeToReport(test_file, str(graph_id) + '.pickle,' + str(y))

    end = datetime.datetime.now()
    total_time = datetime.timedelta()
    total_time = (end - start)
    print('Computing all features + graphlets time: ' + str(total_time))

    report_file = 'reports/features_data/computing_time_' + dataset + '.csv'
    writeToReport(report_file,
                  'graph_name , nodes_count , edges_count, degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random,' + list_to_str([ind for ind in range(0, 30)]) + ' Total Time ')
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


data_dir = "./graph_classification_datasets"
os.makedirs(data_dir, exist_ok=True)

## main
# dataset = 'PROTEINS'
# dataset = 'ENZYMES'
# dataset = 'IMDB-BINARY'
# dataset = 'MUTAG'
# dataset = 'CSL'
# dataset = 'BA2MotifDataset'
# dataset = 'NCI1','BZR', 'DHFR', 'PTC_FR', 'PTC_MM', 'COX2'
# dataset = 'MUTAG'
dataset = 'ws_s5'
test_file = 'data/real/' + dataset + '.train'
# test_file = open('data/real/' + dataset + '.train', 'w+')

if (
        dataset == 'MUTAG' or dataset == 'PROTEINS' or dataset == 'IMDB-MULTI' or dataset == 'ENZYMES' or dataset == 'NCI1' or dataset == 'BZR' or dataset == 'DHFR' or dataset == 'PTC_FR' or dataset == 'PTC_MM' or dataset == 'COX2' or dataset == 'Tox21_ATAD5_evaluation' or dataset == 'KKI' or dataset == 'MSRC_9' or dataset == 'Frankenstein' or dataset == 'facebook_ct1' or dataset == 'IMDB-BINARY'):
    graphs_dataset = TUDataset(root=data_dir, name=dataset)
    dataset_length = len(graphs_dataset)
if (dataset == 'CSL'):
    graphs_dataset = GNNBenchmarkDataset(root=data_dir, name=dataset)
    dataset_length = len(graphs_dataset)
if (dataset == 'BA2MotifDataset'):
    graphs_dataset = BA2MotifDataset(root=data_dir)
    dataset_length = len(graphs_dataset)
if (
        dataset == 'er_single' or dataset == 'er_and2or' or dataset == 'er_or' or dataset == 'er_imbalanced' or dataset == 'er_counting' or dataset == 'er_s1' or dataset == 'er_s2' or dataset == 'er_s3' or dataset == 'er_s4' or dataset == 'er_s5'):
    graphs_dataset = [f for f in os.listdir(dataset + '/') if 'Erdos_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)
if (
        dataset == 'ba_single' or dataset == 'ba_and2or' or dataset == 'ba_or' or dataset == 'ba_imbalanced' or dataset == 'ba_counting' or dataset == 'ba_s1' or dataset == 'ba_s2' or dataset == 'ba_s3' or dataset == 'ba_s4' or dataset == 'ba_s5'):
    graphs_dataset = [f for f in os.listdir(dataset + '/') if 'Barabasi_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)
if (
        dataset == 'ws_single' or dataset == 'ws_and2or' or dataset == 'ws_or' or dataset == 'ws_imbalanced' or dataset == 'ws_counting' or dataset == 'ws_s1' or dataset == 'ws_s2' or dataset == 'ws_s3' or dataset == 'ws_s4' or dataset == 'ws_s5'):
    graphs_dataset = [f for f in os.listdir(dataset + '/') if 'Watts_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)

if (
        dataset == 'pl_single' or dataset == 'pl_and2or' or dataset == 'pl_or' or dataset == 'pl_imbalanced' or dataset == 'pl_counting' or dataset == 'pl_s1' or dataset == 'pl_s2' or dataset == 'pl_s3' or dataset == 'pl_s4' or dataset == 'pl_s5'):
    graphs_dataset = [f for f in os.listdir(dataset + '/') if 'PowerLaw_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)

if (dataset == 'Barabasi_'):
    graphs_dataset = [f for f in os.listdir('Synthetic/') if
                      'Barabasi_10_' in f or 'Barabasi_15_' in f or 'Barabasi_20_' in f or 'Barabasi_25_' in f or 'Barabasi_30_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)
if (dataset == 'Watts'):
    graphs_dataset = [f for f in os.listdir('Synthetic/') if
                      'Watts_20_' in f or 'Watts_25_' in f or 'Watts_30_' in f or 'Watts_35_' in f or 'Watts_40_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)
if (dataset == 'Power-Law'):
    graphs_dataset = [f for f in os.listdir('Synthetic/') if
                      'Power-Law_20_' in f or 'Power-Law_25_' in f or 'Power-Law_30_' in f or 'Power-Law_35_' in f or 'Power-Law_40_' in f]
    # graphs_dataset = [f for f in os.listdir('Synthetic/') if 'Erdos_6_' in f]
    dataset_length = len(graphs_dataset)

K = 6
train_length_gc = int(dataset_length * (480) / (dataset_length))
batch = 10
num_epochs = 10
train_iterations = 5

modes = ['vanilla', 'C', 'top k', 'random k']

directory_features = 'data/real/' + dataset + '/'
os.makedirs(directory_features, exist_ok=True)
report_graph_level_features = directory_features + dataset + '.csv'

writeToReport(report_graph_level_features,
              'graph_id, nodes count, edges count, diameter, avg. path length, std path length, ' + features_header(
                  'degree') + features_header('eigenvector_cent.') + features_header('closeness cent.')
              + features_header('harmonic cent.') + features_header('betweenness cent. ') + features_header(
                  'coloring LF') + features_header('edges within egonet')
              + features_header('node clique number') + features_header('number of cliques') + features_header(
                  'clustering coefficient') +
              features_header('square clustering coefficient') + features_header('pagerank') + features_header(
                  'hubs') + features_header('core number') + 'random,'
              + list_to_str(['c. g_' + str(_) for _ in range(0, 30)]) + list_to_str(
                  ['b. g_' + str(_) for _ in range(0, 30)]) + '  Target')



for graph_id, graph in enumerate(graphs_dataset):
    compute_features(graph_id, graph)


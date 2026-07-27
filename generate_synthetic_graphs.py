# generate synthetic graphs for the six defined learning problems explained in Section 4.1 of the paper.
# Each dataset consists of 600 graphs generated using four graph models:
# Erdos (ER), Barabasi (BA), Watts-Strogatz (WS), Power-Law (PL).
# Each dataset in this script has the name: X_Y where X denotes the base graph model and Y refers to the different learnign problems (s1...s6)
# The generated graphs are saved under Synthetic/dataset_name/

import random
import networkx as nx
import numpy as np
import os
import random
import pickle
import datetime
import util
from util import writeToReport, list_to_str
import torch_geometric
from torch_geometric.utils import to_networkx
import torch
import graphlets

def create_motif(shape):
    G = nx.Graph()
    if (shape == 'house'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 3, 'x': 1}), (counter + 4, {'y': 3, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter, counter + 2))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 1, counter + 3))
        G.add_edge(*(counter + 2, counter + 4))
        G.add_edge(*(counter + 3, counter + 4))
        motif_size = 5

    if (shape == 'star'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 2, 'x': 1}), (counter + 4, {'y': 2, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter, counter + 2))
        G.add_edge(*(counter, counter + 3))
        G.add_edge(*(counter, counter + 4))
        motif_size = 5

    if (shape == 'path'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 2, 'x': 1}), (counter + 4, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter + 4))
        motif_size = 5

    if (shape == 'cycle'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 1, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 1, 'x': 1}), (counter + 4, {'y': 1, 'x': 1}),
                             (counter + 5, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter + 4))
        G.add_edge(*(counter + 4, counter + 5))
        G.add_edge(*(counter + 5, counter))
        motif_size = 6

    if (shape == 'grid'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 2, 'x': 1}), (counter + 4, {'y': 3, 'x': 1}),
                             (counter + 5, {'y': 2, 'x': 1}),
                             (counter + 6, {'y': 1, 'x': 1}), (counter + 7, {'y': 2, 'x': 1}),
                             (counter + 8, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 3, counter + 4))
        G.add_edge(*(counter + 4, counter + 5))
        G.add_edge(*(counter + 6, counter + 7))
        G.add_edge(*(counter + 7, counter + 8))
        G.add_edge(*(counter, counter + 3))
        G.add_edge(*(counter + 3, counter + 6))
        G.add_edge(*(counter + 1, counter + 4))
        G.add_edge(*(counter + 4, counter + 7))
        G.add_edge(*(counter + 2, counter + 5))
        G.add_edge(*(counter + 5, counter + 8))
        motif_size = 9

    if (shape == 'G8'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 1, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter, counter + 2))
        G.add_edge(*(counter, counter + 3))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 1, counter + 3))
        G.add_edge(*(counter + 2, counter + 3))

        motif_size = 4

    if (shape == 'G15'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 1, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 1, 'x': 1}), (counter + 4, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter + 4))
        G.add_edge(*(counter, counter + 4))

        motif_size = 5

    if (shape == 'G20'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 1, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 1, 'x': 1}), (counter + 4, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter))
        G.add_edge(*(counter + 1, counter + 4))
        G.add_edge(*(counter + 3, counter + 4))

        motif_size = 5

    if (shape == 'G21'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 3, 'x': 1}), (counter + 4, {'y': 3, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 4))
        G.add_edge(*(counter + 1, counter + 4))
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter))
        motif_size = 5

    if (shape == 'G24'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 3, 'x': 1}), (counter + 4, {'y': 3, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter, counter + 2))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 1, counter + 3))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 2, counter + 4))
        G.add_edge(*(counter + 3, counter + 4))
        motif_size = 5

    if (shape == 'G30'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 1, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}), (counter + 3, {'y': 1, 'x': 1}),
                             (counter + 4, {'y': 1, 'x': 1}), (counter + 5, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter + 4))
        G.add_edge(*(counter + 4, counter + 5))
        G.add_edge(*(counter + 5, counter))

        motif_size = 6

    return G, motif_size


def create_base_graph(nodes_count, edges_count):
    if (base_graph_type == 'Barabasi'):
        base_graph = nx.barabasi_albert_graph(nodes_count, edges_count)
    if (base_graph_type == 'Erdos'):
        base_graph = nx.gnp_random_graph(nodes_count, edges_count)
    if (base_graph_type == 'Watts'):
        base_graph = nx.connected_watts_strogatz_graph(nodes_count, edges_count, 0.1, tries=20)
    if (base_graph_type == 'PowerLaw'):
        base_graph = nx.powerlaw_cluster_graph(nodes_count, edges_count, 0.3)
    if (base_graph_type == 'Geometric'):
        base_graph = nx.random_geometric_graph(nodes_count, 0.2)
    if (base_graph_type == 'Grid'):
        base_graph = nx.grid_graph(dim=(40, 40))
        delete_prob = 0.1
        edges_to_remove = [edge for edge in base_graph.edges if random.random() < delete_prob]
        base_graph.remove_edges_from(edges_to_remove)

    nx.set_node_attributes(base_graph, 1, 'x')
    nx.set_node_attributes(base_graph, 0, 'y')

    return base_graph


def add_attributes(graphs, names):
    new_graphs_list = []
    for i, graph in enumerate(graphs):
        if ('house' in names[i]):
            graph.g_label = 0
            print('house')
        if ('grid' in names[i]):
            graph.g_label = 1
            print('grid')
        if ('path' in names[i]):
            graph.g_label = 2
            print('path')
        if ('star' in names[i]):
            graph.g_label = 3
            print('star')
        if ('cycle' in names[i]):
            graph.g_label = 4
            print('cycle')
        if ('G8' in names[i]):
            graph.g_label = 0
            print('G8')
        if ('G20' in names[i]):
            graph.g_label = 1
            print('G20')
        if ('G21' in names[i]):
            graph.g_label = 2
            print('G21')
        if ('G24' in names[i]):
            graph.g_label = 3
            print('G24')
        if ('no_label' in names[i]):
            graph.g_label = 0
            print('no_label')
        if ('G8-G20' in names[i]):
            graph.g_label = 0
            print('G8-G20')
        if ('G8-G21' in names[i]):
            graph.g_label = 1
            print('G8-G21')
        if ('G20-G21' in names[i]):
            graph.g_label = 2
            print('G20-G21')
        if ('G8-G20-G21' in names[i]):
            graph.g_label = 1
            print('G8-G20-G21')

        new_graphs_list.append(graph)
    return new_graphs_list


# datasets = ['ba_single','er_single','ws_single','pl_single']
# datasets = ['ba_and']
# datasets = ['ba_and2or','er_and2or','ws_and2or','pl_and2or']
datasets = ['ba_s5', 'er_s5', 'ws_s5', 'pl_s5']
# datasets = ['ba_imbalanced','er_imbalanced','ws_imbalanced','pl_imbalanced']
# datasets = ['ba_counting','er_counting','ws_counting','pl_counting']
# datasets = ['pl_s1']

for dataset in datasets:
    str_arr = dataset.split('_')
    if (str_arr[0] == 'ba'):
        base_graph_type = 'Barabasi'
        base_graph_nodes_count_list = [j for j in range(8, 14)]
        base_graph_edges_count_list = [2, 2, 2, 2, 2]
    if (str_arr[0] == 'er'):
        base_graph_type = 'Erdos'
        base_graph_nodes_count_list = [j for j in range(12, 18)]
        base_graph_edges_count_list = [0.10, 0.12, 0.14, 0.16, 0.18]
    if (str_arr[0] == 'ws'):
        base_graph_type = 'Watts'
        base_graph_nodes_count_list = [j for j in range(8, 14)]
        base_graph_edges_count_list = [2, 3, 3, 3, 3]
    if (str_arr[0] == 'pl'):
        base_graph_type = 'PowerLaw'
        base_graph_nodes_count_list = [j for j in range(8, 14)]
        base_graph_edges_count_list = [2, 2, 2, 2, 2]
    if (str_arr[1] == 'single'):
        motifs_count_list = [0]
        # motifs_shapes_list = ['G8','G20','G21','G24']
        motifs_shapes_list = ['single']
    if (str_arr[1] == 'and'):
        motifs_count_list = [1, 1, 2, 2]
        motifs_shapes_list = ['and']
    if (str_arr[1] == 'or'):
        motifs_count_list = [1, 2]
        motifs_shapes_list = ['or']
    if (str_arr[1] == 'imbalanced'):
        motifs_count_list = [1, 2]
        motifs_shapes_list = ['imbalanced']
    if (str_arr[1] == 'counting'):
        motifs_count_list = [1, 1, 1, 1, 1, 1, 1, 1]
        motifs_shapes_list = ['counting']
    if (str_arr[1] == 's1'):
        motifs_count_list = [0]
        motifs_shapes_list = ['s1']
    if (str_arr[1] == 's2'):
        motifs_count_list = [1, 1, 1, 1, 2, 2, 2, 2]
        motifs_shapes_list = ['s2']
    if (str_arr[1] == 's3'):
        motifs_count_list = [0]
        motifs_shapes_list = ['s3']
    if (str_arr[1] == 's4'):
        motifs_count_list = [1, 1, 1, 2, 2, 2]
        motifs_shapes_list = ['s4']
    if (str_arr[1] == 's5'):
        motifs_count_list = [1]
        motifs_shapes_list = ['s5']


    noisy_edges_count_list = [2]
    generated_graphs = []
    generated_graphs_names = []
    graphs_ind = 271
    copies = 1
    label = 1

    total_nodes = 0
    total_edges = 0
    for c in range(copies):
        for n in base_graph_nodes_count_list:
            for m in base_graph_edges_count_list:
                for motifs_count in motifs_count_list:
                    for motifs_shape in motifs_shapes_list:
                        for noisy_edges_count in noisy_edges_count_list:

                            start = datetime.datetime.now()
                            print('graph: ' + str(graphs_ind))
                            graphs = [graphlets.graphs[index] for index in [8, 20, 21, 30]]
                            counter_graphlets = 1

                            # Check whether the base graph contains one of the used graphlets.

                            while counter_graphlets != 0:

                                base_graph = create_base_graph(n, m)

                                data = torch_geometric.utils.from_networkx(base_graph)
                                counter_graphlets = 0
                                for graph_index, g in enumerate(graphs):
                                    print('Graphlet: ' + str(graph_index))

                                    start_time_graphlet = datetime.datetime.now()
                                    GM = nx.isomorphism.GraphMatcher(base_graph, g)
                                    g_iter = GM.subgraph_isomorphisms_iter()
                                    unique_matches = set()
                                    for match in g_iter:
                                        match_nodes = tuple(sorted(match.keys()))
                                        unique_matches.add(match_nodes)
                                    counter_graphlets += len(unique_matches)
                                    print('Counter: ')
                                    print(counter_graphlets)

                            base_graph = base_graph.subgraph(max(nx.connected_components(base_graph), key=len)).copy()

                            n = nx.number_of_nodes(base_graph)

                            if (n < 4):
                                motif_graph, motif_size = create_motif('cycle')
                                base_graph = nx.disjoint_union(base_graph, motif_graph)
                                base_graph.add_edge(*(0, 3))
                                n = base_graph.number_of_nodes()
                            counter = n
                            print('nodes: ' + str(n))
                            # attach motifs
                            if (motifs_shape == 'and'):
                                motif_shape_arr = ['G8', 'G20', 'G21', 'G24']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 'or'):
                                motif_shape_arr = ['G8']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 'imbalanced'):
                                motif_shape_arr = ['G20', 'G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size
                            if (motifs_shape == 'counting'):
                                motif_shape_arr = ['G8']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 's1'):
                                motif_shape_arr = ['G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size
                            if (motifs_shape == 's2'):
                                motif_shape_arr = ['G20', 'G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 's3'):
                                motif_shape_arr = ['G20', 'G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 's4'):
                                motif_shape_arr = ['G8', 'G20', 'G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 's5'):
                                motif_shape_arr = ['G8', 'G20', 'G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            if (motifs_shape == 'and2or'):
                                motif_shape_arr = ['G20', 'G21']
                                for motif_s in motif_shape_arr:
                                    for motif in range(0, motifs_count):

                                        motif_graph, motif_size = create_motif(motif_s)
                                        base_graph = nx.disjoint_union(base_graph, motif_graph)
                                        print(base_graph.number_of_nodes())
                                        random_base_node = random.randint(0, n - 1)
                                        random_motif_node = random.randint(counter, counter + motif_size - 1)
                                        base_graph.add_edge(*(random_base_node, random_motif_node))
                                        counter += motif_size

                            noisy_edges_count = int(nx.number_of_edges(base_graph) * 0.01)
                            for e in range(noisy_edges_count):
                                noisy_edge_node_1 = random.randint(0, counter - 1)
                                noisy_edge_node_2 = random.randint(0, counter - 1)
                                base_graph.add_edge(*(noisy_edge_node_1, noisy_edge_node_2))

                            base_graph.remove_edges_from(nx.selfloop_edges(base_graph))
                            total_nodes += base_graph.number_of_nodes()
                            total_edges += base_graph.number_of_edges()
                            print(base_graph.number_of_nodes())

                            graph_name = base_graph_type + '_' + str(graphs_ind) + '_' + str(n) + '_' + str(
                                m) + '_' + motifs_shape + '_' + str(
                                motifs_count) + '_' + str(noisy_edges_count) + '.pickle'
                            # G = to_networkx(data, to_undirected=True)
                            for node in base_graph.nodes:
                                base_graph.nodes[node]['x'] = 0
                                base_graph.nodes[node]['y'] = 0
                            data = torch_geometric.utils.from_networkx(base_graph)

                            if (motifs_shape == 'house'):
                                data.y = torch.tensor([0])
                            if (motifs_shape == 'grid'):
                                data.y = torch.tensor([1])
                            if (motifs_shape == 'path'):
                                data.y = torch.tensor([2])
                            if (motifs_shape == 'star'):
                                data.y = torch.tensor([3])
                            if (motifs_shape == 'cycle'):
                                data.y = torch.tensor([4])
                            if (motifs_shape == 'G8'):
                                data.y = torch.tensor([0])
                            if (motifs_shape == 'G20'):
                                data.y = torch.tensor([1])
                            if (motifs_shape == 'G21'):
                                data.y = torch.tensor([2])
                            if (motifs_shape == 'G24'):
                                data.y = torch.tensor([3])
                            if (motifs_shape == 'G8-G20'):
                                data.y = torch.tensor([0])
                            if (motifs_shape == 'G8-G21'):
                                data.y = torch.tensor([1])
                            if (motifs_shape == 'G20-G21'):
                                data.y = torch.tensor([2])
                            if (motifs_shape == 'no_label'):
                                data.y = torch.tensor([0])
                            if (motifs_shape == 'G8-G20-G21'):
                                data.y = torch.tensor([1])
                            data.y = torch.tensor([label])
                            pickle.dump(data, open(dataset + '/' + graph_name, 'wb'))

                            end = datetime.datetime.now()
                            total_time = datetime.timedelta()
                            total_time = (end - start)
                            print('Generating time: ' + str(total_time))
                            graphs_ind += 1
                            print(base_graph.number_of_nodes())

    print('Nodes: ' + str(total_nodes / 1))
    print('Edges: ' + str(total_edges / 1))

    print(counter_graphlets)

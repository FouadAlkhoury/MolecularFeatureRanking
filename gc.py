# This script runs the graph classification model on three GNN architectures: GCN, GAT, and GraphSAGE.

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import ReadData
import os
import pickle
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import ReadData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import random
import util
from util import writeToReport
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import export_text
from torch_geometric.datasets import Amazon, CitationFull, Entities, Twitch, Airports, Actor, GitHub
from torch_geometric.datasets import GNNBenchmarkDataset, AttributedGraphDataset, HeterophilousGraphDataset
import datetime
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, Coauthor, BA2MotifDataset
from collections import OrderedDict
import torch.cuda


def list_to_str(list):
    str_list = ''
    for l in list:
        str_list += str(l) + ','
    return str_list


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels + num_graph_features, num_classes)

    def forward(self, data):
        x, edge_index, batch, graph_features = data.x, data.edge_index, data.batch, data.graph_features
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        if (num_graph_features > 0):

            graph_features = graph_features.view(batch_size, num_graph_features)
            # graph_features = graph_features.view(-1, num_graph_features)
            x = torch.cat([x, graph_features], dim=1)
        # x = torch.cat([x, graph_features], dim=1)
        out = self.lin1(x)

        return out


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels + num_graph_features, num_classes)

    def forward(self, data):
        x, edge_index, batch, graph_features = data.x, data.edge_index, data.batch, data.graph_features
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        if (num_graph_features > 0):
            graph_features = graph_features.view(batch_size, num_graph_features)
            x = torch.cat([x, graph_features], dim=1)
        # x = torch.cat([x, graph_features], dim=1)
        out = self.lin1(x)

        return out


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels + num_graph_features, num_classes)

    def forward(self, data):
        x, edge_index, batch, graph_features = data.x, data.edge_index, data.batch, data.graph_features
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        if (num_graph_features > 0):
            graph_features = graph_features.view(batch_size, num_graph_features)
            x = torch.cat([x, graph_features], dim=1)
        # x = torch.cat([x, graph_features], dim=1)
        out = self.lin1(x)

        return out


def train():
    model.train()
    for i, data in enumerate(train_loader):
        # print(data)
        optimizer.zero_grad()
        data = data.to(device)
        out1 = model(data)
        loss = criterion(out1, data.y)
        if (epoch % 250 == 0):
            print('Epoch: ' + str(epoch) + '   train loss: ' + str(loss.item()))
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()
    correct = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        out1 = model(data)
        pred1 = out1.argmax(dim=1)
        correct += int((pred1 == data.y).sum())
    acc = int(correct) / len(loader.dataset)
    if (epoch == num_epochs - 1):
        test_results.append(pred1)
    return acc


def add_attributes(graphs):
    new_graphs_list = []
    for i, graph in enumerate(graphs):
        x_train = np.ones((graph.num_nodes, 10),
                          dtype=np.float32)  # graph.num_nodes, graph.num_node_features for real datasets
        x_train = torch.from_numpy(x_train)
        graph.x = x_train

        graph_features = torch.from_numpy(X[i])
        graph.graph_features = graph_features
        new_graphs_list.append(graph)
    return new_graphs_list


# datasets = ['MUTAG', 'BZR', 'DHFR',  'BA2MotifDataset', 'PROTEINS', 'NCI1' ]
# datasets = ['er_s2','ba_s2','ws_s2','pl_s2']
datasets = ['ba_s1', 'ba_s2', 'ba_s3', 'ba_s4', 'er_s1', 'er_s2', 'er_s3', 'er_s4', 'ws_s1', 'ws_s2', 'ws_s3', 'pl_s1',
            'pl_s2', 'pl_s3']
# datasets = ['er_s5','ba_s5','ws_s5','pl_s5']
# datasets = ['ws_s2','ws_s1','ws_s3']

for dataset in datasets:
    print(dataset)
    X_train = []

    Y_train = []
    X_test = []
    Y_test = []
    X_val = []
    Y_val = []

    num_epochs = 1000
    data_dir = "./datasets"
    os.makedirs(data_dir, exist_ok=True)
    dataPath = 'data/real/split'

    train_length = 0
    num_graph_features = 0

    if (dataset == 'MUTAG'):
        dataset_length = 147
        train_length = 126
        batch_size = 21
        num_node_features = 7
        num_classes = 2
        # top_k variable is set to the global ranking aggregated for all training graphs in each dataset.
        top_k = [65, 62, 58, 0, 50, 49, 28, 17, 20, 3, 64, 1, 9, 7, 52]  # MUTAG
    if (dataset == 'ENZYMES'):
        dataset_length = 540
        train_length = 480
        batch_size = 20
        num_node_features = 3
        num_classes = 6
        top_k = [87, 7, 10, 71, 75, 32, 28, 50]
    if (dataset == 'PROTEINS'):
        dataset_length = 1000
        train_length = 900
        batch_size = 20
        num_node_features = 3
        num_classes = 2
        top_k = [7, 15, 4, 9, 50, 13, 31, 36]
    if (dataset == 'BA2MotifDataset'):
        dataset_length = 900
        train_length = 800
        batch_size = 20
        num_node_features = 10
        num_classes = 2
        top_k = [35, 42, 36, 47, 34, 45, 29, 43]
    if (dataset == 'CSL'):
        dataset_length = 135
        train_length = 120
        batch_size = 15
        num_node_features = 0
        num_classes = 10
        # top_k = [35,42,36,47,34,45,29,43]
    if (dataset == 'NCI1'):
        dataset_length = 3700
        train_length = 3300
        batch_size = 50
        num_node_features = 37
        num_classes = 2
        top_k = [15, 11, 19, 25, 21, 18, 7, 23]
    if (dataset == 'BZR'):
        dataset_length = 360
        train_length = 320
        batch_size = 20
        num_node_features = 53
        num_classes = 2
        top_k = [15, 55, 30, 4, 10, 65]
    if (dataset == 'DHFR'):
        dataset_length = 680
        train_length = 600
        batch_size = 40
        num_node_features = 53
        num_classes = 2
        top_k = [24, 20, 19, 10, 52, 28]
    if (dataset == 'PTC_FR'):
        dataset_length = 320
        train_length = 280
        batch_size = 20
        num_node_features = 19
        num_classes = 2
        top_k = [52, 12, 10, 20, 54, 26, 40]
    if (dataset == 'PTC_MM'):
        dataset_length = 300
        train_length = 270
        batch_size = 10
        num_node_features = 20
        num_classes = 2
        top_k = [56, 61, 15, 11, 52, 24, 8, 12]

    # ,'er_s2','ba_s2','ws_s2','pl_s2','er_s3','ba_s3','ws_s3','pl_s3','er_s4','ba_s4','ws_s4','pl_s4']):
    if (dataset in ['er_s1', 'ba_s1', 'ws_s1', 'pl_s1', 'er_s4', 'ba_s4', 'ws_s4', 'pl_s4']):
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 4
        top_k = [100, 112, 113, 115, 33, 46]

    if (dataset in ['er_s2', 'ba_s2', 'ws_s2', 'pl_s2']):
        print('er_s2')
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 3
        top_k = [100, 112, 113, 115, 33, 46]
        # top_k = [73,74,75,76,77,78]
    if (dataset in ['er_s3', 'ba_s3', 'ws_s3', 'pl_s3']):
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 2
        top_k = [100, 112, 113, 115, 33, 46]

    if (dataset == 'ba_s5'):
        print('ba_s5')
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 4
        # top_k = [45,55,11,59,15,54,10,60,29,56,12,4,5,17]
        top_k = [45, 55, 11, 59, 15, 54, 10, 60, 29, 56, 12, 4, 5, 17, 57, 28, 24, 46, 35, 75, 9, 50, 23, 48, 44, 16,
                 26, 30, 92, 22, 41]

    if (dataset == 'er_s5'):
        print('er_s5')
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 4
        # top_k = [47,48,45,46,44,29,36,5,30,32,43,35,10]
        top_k = [47, 48, 45, 46, 44, 29, 36, 5, 30, 32, 43, 35, 10, 39, 8, 38, 31, 4, 59, 7, 6, 17, 22, 26, 15, 9, 63,
                 18, 20, 1, 52, 34, 61, 41, 62, 11, 33]
    if (dataset == 'ws_s5'):
        print('ws_s5')
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 4
        # top_k = [47,48,45,71,46,29,63,17,30,65,92,5,36,123]
        top_k = [47, 48, 45, 71, 46, 29, 63, 17, 30, 65, 92, 5, 36, 123, 10, 9, 35, 0, 53, 66, 21, 72, 18, 49, 54, 32,
                 1, 50, 57, 62, 15]
    if (dataset == 'pl_s5'):
        print('pl_s5')
        dataset_length = 500
        train_length = 400
        batch_size = 25
        num_node_features = 10
        num_classes = 4
        # top_k = [45,15,5,29,50,55,28,35,11,92,61,13,123,10]
        top_k = [45, 15, 5, 29, 50, 55, 28, 35, 11, 92, 61, 13, 123, 10, 25, 3, 54, 36, 41, 9, 67, 4, 98, 48, 46, 26,
                 43, 24, 14, 44, 33, 47]

    results_per_k_file = 'reports/results_all_synthetic' + '_all.csv'

    writeToReport(results_per_k_file, 'k, features, test_acc')

    gnn_classes = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': GraphSAGE}
    #gnn_classes = {'GCN': GCN}
    features_variants = ['F_0', 'F_C', 'R_6', 'T_6']
    #features_variants = ['T_6']

    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name, model_class in gnn_classes.items():
        for feature_variant in features_variants:

            for k in range(6, 7):

                file_A = "data/real/split/gc_training_" + dataset + ".csv"
                df_A = pd.read_csv(file_A, header=None)

                if (feature_variant == 'T_6'):
                    print('T_6')
                    top_k_indices = top_k[:k]
                    num_graph_features = k
                    # print(top_k_indices)
                if (feature_variant == 'F_C'):
                    print('F_C')
                    top_k_indices = [j for j in range(122)]
                    num_graph_features = 122
                if (feature_variant == 'F_0'):
                    print('F_0')
                    top_k_indices = [j for j in range(122)]
                    num_graph_features = 0
                if (feature_variant == 'R_6'):
                    print('R_6')
                    top_k_indices = random.sample([j for j in range(0, 122)], 6)
                    num_graph_features = 6

                for iter_ in range(1):
                    num_graphs, num_features = df_A.shape[0], df_A.shape[1] - 2
                    X = np.zeros((num_graphs, num_features)).astype(np.float32)

                    for i in range(num_graphs):
                        graph_name = df_A.iloc[i, 0]
                        feature_values = df_A.iloc[i, 1:].values
                        # Keep only top k features in their positions
                        for idx in top_k_indices:
                            X[i, idx] = feature_values[idx]

                    if (feature_variant == 'R_6' or feature_variant == 'T_6'):

                        X = X[:, top_k_indices]

                    indices_list = []
                    graph_names = df_A.iloc[:, 0].astype(str)
                    indices_list = [int(name.split(dataset + '_')[1]) for name in graph_names]

                    if (
                            dataset == 'MUTAG' or dataset == 'PROTEINS' or dataset == 'IMDB-BINARY' or dataset == 'ENZYMES' or dataset == 'NCI1' or dataset == 'BZR' or dataset == 'DHFR' or dataset == 'PTC_FR' or dataset == 'PTC_MM'):
                        graphs = TUDataset(root='data/TUDataset', name=dataset)
                    # if (dataset == 'CSL'):
                    # graphs_dataset = GNNBenchmarkDataset(root=data_dir, name = dataset)

                    if (dataset == 'BA2MotifDataset'):
                        graphs = BA2MotifDataset(root='data/BA2MotifDataset')

                    if (dataset in ['er_s1', 'ba_s1', 'ws_s1', 'pl_s1', 'er_s2', 'ba_s2', 'ws_s2', 'pl_s2', 'er_s3',
                                    'ba_s3', 'ws_s3', 'pl_s3', 'er_s4', 'ba_s4', 'ws_s4', 'pl_s4', 'er_s5', 'ba_s5',
                                    'ws_s5', 'pl_s5']):
                        graphs_names = [f for f in os.listdir(dataset)]
                        graphs = []
                        # print(graphs_names)
                        for graph in graphs_names:
                            data = pickle.load(open(dataset + '/' + graph, 'rb'))
                            # G = to_networkx(data, to_undirected=True)

                            graphs.append(data)
                        # dataset_length = len(graphs_dataset)

                    # graphs = graphs[indices_list]
                    graphs = [graphs[i] for i in indices_list]
                    # print(graphs)
                    graphs = add_attributes(graphs)
                    train_indices = random.sample(range(0, dataset_length), train_length)
                    train_dataset = [item for i, item in enumerate(graphs) if i in train_indices]
                    test_dataset = [item for i, item in enumerate(graphs) if i not in train_indices]
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    model = model_class(128).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                    criterion = torch.nn.CrossEntropyLoss()
                    # criterion = torch.nn.BCELoss()
                    test_results = []
                    for epoch in range(1, num_epochs):
                        train()
                    test_acc = test(test_loader)
                    print(test_acc)
                    writeToReport(results_per_k_file,
                                  dataset + ',' + model_name + ',' + feature_variant + ',' + str(test_acc))

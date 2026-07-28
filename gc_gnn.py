# names of training graphs are stored in the file synthetic.train, the learned model is saved under model/

import datetime
import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import from_networkx

from util import writeToReport


def list_to_str(values):
    str_list = ''
    for value in values:
        str_list += str(value) + ','
    return str_list


def load_compounds_dataset(filename):
    with open(filename, 'rb') as f:
        compounds_dataset = pickle.load(f)
    return compounds_dataset


def chembl_to_pyg(dataset):
    """
    Convert CHEMBL pickled dataset
    to a list of torch_geometric.data.Data objects.
    """
    pyg_graphs = []

    for entry in dataset:
        G = nx.Graph(entry.graph)

        data = from_networkx(G)
        data.y = torch.tensor(
            [int(entry.graph['class_label'])],
            dtype=torch.long
        )

        graph_features = entry.graph.get('graph_features', {})
        if graph_features:
            data.graph_features = torch.tensor(
                list(graph_features.values()),
                dtype=torch.float
            )

        data.name = entry.graph.get('name', None)
        data.cid = entry.graph.get('cid', None)
        data.core_id = entry.graph.get('core_id', None)
        data.sets = entry.graph.get('set', None)

        pyg_graphs.append(data)

    return pyg_graphs


def get_train_indices_for_run(pyg_graphs, run_index):
    if not hasattr(pyg_graphs[0], "sets"):
        raise ValueError(
            "Graphs do not have the 'sets' attribute. "
            "Did you include it during conversion?"
        )

    num_runs = len(pyg_graphs[0].sets)
    if run_index < 1 or run_index > num_runs:
        raise ValueError(
            f"Invalid run_index {run_index}. "
            f"Must be between 1 and {num_runs}."
        )

    run_idx = run_index - 1
    train_indices = [
        i for i, g in enumerate(pyg_graphs)
        if g.sets[run_idx] == 'train'
    ]
    return train_indices


def get_test_indices_for_run(pyg_graphs, run_index):
    if not hasattr(pyg_graphs[0], "sets"):
        raise ValueError(
            "Graphs do not have the 'sets' attribute. "
            "Did you include it during conversion?"
        )

    num_runs = len(pyg_graphs[0].sets)
    if run_index < 1 or run_index > num_runs:
        raise ValueError(
            f"Invalid run_index {run_index}. "
            f"Must be between 1 and {num_runs}."
        )

    run_idx = run_index - 1
    test_indices = [
        i for i, g in enumerate(pyg_graphs)
        if g.sets[run_idx] == 'test'
    ]
    return test_indices


# Feature-ranking GNN.
# The name GCN is kept because models saved with torch.save(model, ...)
# from fr_gnn.py may refer to __main__.GCN when they are loaded.
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(5, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, 167)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin1(x)
        return out


# Graph-classification GCN.
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN_Graph, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(
            hidden_channels + num_graph_features,
            num_classes
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        graph_features = data.graph_features

        if x.dim() == 1:
            x = x.unsqueeze(1)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)

        graph_features = graph_features.view(x.shape[0], -1)
        if num_graph_features != num_real_graph_features:
            x = torch.cat([x, graph_features], dim=1)

        out = self.lin1(x)
        return out


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(
            hidden_channels + num_graph_features,
            num_classes
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        graph_features = data.graph_features

        if x.dim() == 1:
            x = x.unsqueeze(1)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)

        graph_features = graph_features.view(x.shape[0], -1)
        if num_graph_features != num_real_graph_features:
            x = torch.cat([x, graph_features], dim=1)

        out = self.lin1(x)
        return out


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(
            hidden_channels + num_graph_features,
            num_classes
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        graph_features = data.graph_features

        if x.dim() == 1:
            x = x.unsqueeze(1)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)

        graph_features = graph_features.view(x.shape[0], -1)
        if num_graph_features != num_real_graph_features:
            x = torch.cat([x, graph_features], dim=1)

        out = self.lin1(x)
        return out


def train():
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out1 = model(data)
        loss = criterion(out1, data.y)

        if epoch % 500 == 0:
            print(
                'Epoch: '
                + str(epoch)
                + '   train loss: '
                + str(loss.item())
            )

        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out1 = model(data)
            pred1 = out1.argmax(dim=1)
            correct += int((pred1 == data.y).sum())

    acc = int(correct) / len(loader.dataset)
    return acc


def add_attributes(graphs):
    new_graphs_list = []

    for i, graph in enumerate(graphs):
        x_train = np.ones(
            (graph.num_nodes, 10),
            dtype=np.float32
        )
        graph.x = torch.from_numpy(x_train)

        added_graph_features = torch.from_numpy(X[i])
        graph.graph_features = added_graph_features
        new_graphs_list.append(graph)

    return new_graphs_list


def load_feature_ranking_model(dataset, run):
    model_paths = [
        'model/' + dataset + '_' + str(run) + '.pth',
        'model/model_' + dataset + '_' + str(run) + '.pth'
    ]

    model_path = None
    for candidate in model_paths:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            'No feature-ranking model found for '
            + dataset
            + ', run '
            + str(run)
            + '. Checked: '
            + str(model_paths)
        )

    try:
        saved_model = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        saved_model = torch.load(
            model_path,
            map_location=device
        )

    if isinstance(saved_model, torch.nn.Module):
        feature_ranking_model = saved_model.to(device)
    elif isinstance(saved_model, dict):
        feature_ranking_model = GCN(128).to(device)

        if 'model_state_dict' in saved_model:
            feature_ranking_model.load_state_dict(
                saved_model['model_state_dict']
            )
        else:
            feature_ranking_model.load_state_dict(saved_model)
    else:
        raise TypeError(
            'Unsupported saved model type: '
            + type(saved_model).__name__
        )

    feature_ranking_model.eval()
    return feature_ranking_model


def predict_top_features(dataset, run):
    graphs_dataset = load_compounds_dataset(
        './pickles_100_v3/'
        + dataset
        + '_graphs.pkl'
    )
    graphs = chembl_to_pyg(graphs_dataset)

    test_indices = get_test_indices_for_run(graphs, run)
    test_graphs = [graphs[i] for i in test_indices]

    if len(test_graphs) == 0:
        raise ValueError(
            'No test graphs found for '
            + dataset
            + ', run '
            + str(run)
        )

    for graph in test_graphs:
        graph.x = torch.ones(
            (graph.num_nodes, 5),
            dtype=torch.float32
        )

    feature_ranking_loader = DataLoader(
        test_graphs,
        batch_size=16,
        shuffle=False
    )

    feature_ranking_model = load_feature_ranking_model(
        dataset,
        run
    )

    importance_vectors = []

    with torch.no_grad():
        for data in feature_ranking_loader:
            data = data.to(device)
            importance = feature_ranking_model(
                data.x,
                data.edge_index,
                data.batch
            )

            if importance.shape[1] != 167:
                raise ValueError(
                    'Expected 167 importance values, but received '
                    + str(importance.shape[1])
                )

            importance_vectors.append(
                importance.detach().cpu().numpy()
            )

    importance_vectors = np.vstack(importance_vectors)
    importance_aggregated = np.mean(
        importance_vectors,
        axis=0
    )

    top_k_temp = np.argsort(
        importance_aggregated
    )[::-1].tolist()

    return top_k_temp


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
    'CHEMBL4822'
]

#compounds_datasets = [
#    'CHEMBL230',
#    ]

torch.cuda.set_device(0)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


for d, dataset in enumerate(compounds_datasets):
    print(dataset)

    for run in range(1, 11):
        print('Run: ' + str(run))

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        X_val = []
        Y_val = []

        num_epochs = 10
        data_dir = './datasets'
        os.makedirs(data_dir, exist_ok=True)
        dataPath = 'data/real/split'

        train_length = 0
        num_graph_features = 0

        top_k_temp = predict_top_features(dataset, run)

        # The FR-GNN returns fingerprint indices from 0 to 166.
        # Fingerprints occupy columns 157 to 323 in the combined CSV.
        top_k = [i + 157 for i in top_k_temp]

        results_per_k_file = 'reports/results_gat_combined_top_k.txt'

        writeToReport(
            results_per_k_file,
            (
                'dataset, size, run, model, variant, k, '
                'test_acc, train_acc, gc_training_time'
            )
        )

        gnn_classes = {
            'GAT': GAT,
            'GraphSAGE': GraphSAGE
        }
        features_variants = ['T_K']

        for model_name, model_class in gnn_classes.items():
            for feature_variant in features_variants:
                start_time_gc_training = datetime.datetime.now()

                for k in range(1, 16):
                    graphs_dataset = load_compounds_dataset(
                        './pickles_100_v3/'
                        + dataset
                        + '_graphs.pkl'
                    )

                    graphs = chembl_to_pyg(graphs_dataset)
                    num_real_graph_features = 0

                    file_A = (
                        'data/real/split/gc_training_'
                        + dataset
                        + '_run'
                        + str(run)
                        + '_repeat2_compounds_v3.csv'
                    )
                    df_A = pd.read_csv(file_A, header=None)

                    if feature_variant == 'T_K':
                        print('T_K')
                        top_k_indices = top_k[:k]
                        num_graph_features = (
                            num_real_graph_features + k
                        )
                        print(top_k_indices)

                    for iter_ in range(1):
                        num_graphs = df_A.shape[0]
                        num_features = df_A.shape[1] - 2
                        X = np.zeros(
                            (num_graphs, num_features)
                        ).astype(np.float32)

                        for i in range(num_graphs):
                            feature_values = (
                                df_A.iloc[i, 1:].values
                            )

                            for idx in top_k_indices:
                                X[i, idx] = feature_values[idx]

                        X = X[:, top_k_indices]

                        graph_names = df_A.iloc[:, 0].astype(str)
                        indices_list = [
                            int(name.split(dataset + '_')[1])
                            for name in graph_names
                        ]
                        graphs = [graphs[i] for i in indices_list]

                        num_classes = 2
                        graphs = add_attributes(graphs)
                        dataset_length = len(graphs)
                        num_node_features = 10

                        train_indices = get_train_indices_for_run(
                            graphs,
                            run
                        )
                        train_dataset = [
                            item
                            for i, item in enumerate(graphs)
                            if i in train_indices
                        ]
                        test_dataset = [
                            item
                            for i, item in enumerate(graphs)
                            if i not in train_indices
                        ]

                        batch_size = 16
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=False
                        )
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            shuffle=False
                        )

                        train_length = len(train_dataset)

                        model = model_class(128).to(device)
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=0.005
                        )
                        criterion = torch.nn.CrossEntropyLoss()

                        for epoch in range(1, num_epochs):
                            train()

                        test_acc = test(test_loader)
                        train_acc = test(train_loader)
                        print(test_acc)

                        end_time_gc_training = datetime.datetime.now()
                        total_time_gc_training = (
                            end_time_gc_training
                            - start_time_gc_training
                        )

                        writeToReport(
                            results_per_k_file,
                            (
                                dataset
                                + ','
                                + str(dataset_length)
                                + ','
                                + str(run)
                                + ','
                                + model_name
                                + ','
                                + feature_variant
                                + ','
                                + str(k)
                                + ','
                                + str(round(test_acc * 100.0, 2))
                                + ','
                                + str(round(train_acc * 100.0, 2))
                                + ','
                                + str(total_time_gc_training)
                            )
                        )

# the feature learning graph classification model
# names of training graphs are stored in the file synthetic.train, the learned model is saved under model/

import datetime
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import ReadData


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, k_features)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin1(x)
        return out


def train():
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        out1 = model(data.x, data.edge_index, data.batch)
        #data.y = torch.reshape(data.y, (batch, k_features))
        target = data.y.reshape(-1, k_features)
        loss = criterion(out1, target)
        if epoch % 50 == 0:
            print('Epoch: ' + str(epoch) + '   train loss: ' + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = 'CHEMBL230'
data_dir = "./datasets"
os.makedirs(data_dir, exist_ok=True)
dataPath = 'data'
run = 1
X, Y = ReadData.readDataMolecules('train', dataset, run, dataPath)
dataset_length = len(X)
train_length = len(X)
k_features = 167

offset = 0
Y = Y[offset:offset + dataset_length]
X = X[offset:offset + dataset_length]
Y = np.reshape(Y, (len(Y), k_features, 1))

batch = 10
y_all = []

num_node_features = 5

for y in Y:
    tmp = []
    for f in y:
        tmp.append(f)
    y_all.append(tmp)


def add_attributes(dataset):
    data_list = []
    for i, data in enumerate(dataset):
        data.y = y_all[i]
        x_train = np.ones((data.num_nodes, num_node_features), dtype=np.float32)
        x_train = np.array(x_train)
        x_train = torch.from_numpy(x_train)
        data.x = x_train
        data_list.append(data)
    return data_list


y_all = np.array(y_all, dtype=np.float32)
y_all = torch.Tensor(y_all)
y_all = y_all.type(torch.FloatTensor)

if '_s' in dataset:  # synthetic dataset
    graphs_list = []
    graphs_files = [file for file in os.listdir(dataset + '/')]
    for x in X:
        data = pickle.load(open(dataset + '/' + graphs_files[int(x.split(dataset + '_')[1])], 'rb'))
        graphs_list.append(data)
else:  # real dataset
    graphs_list = [
        torch_geometric.utils.from_networkx(
            pickle.load(open('Real/' + x.split('_')[0] + '/' + x + '.pickle', 'rb'))
        )
        for x in X
    ]

graphs_list = graphs_list[:dataset_length]
dataset_list = add_attributes(graphs_list)
dataset_list_train = dataset_list[:train_length]

train_loader = DataLoader(dataset_list_train, batch_size=batch, shuffle=False)
for i in train_loader:
    print(i)

model = GCN(128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

start_time = datetime.datetime.now()
num_epochs = 10
for epoch in range(1, num_epochs):
    train()

end_time = datetime.datetime.now()
diff_time = end_time - start_time
print(diff_time)

torch.save(model, 'model/model_' + dataset + '_' + str(run) + '.pth')

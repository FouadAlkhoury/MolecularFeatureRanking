This repository contains the source code and datasets associated with the manuscript:
Feature Selection-Guided Graph Neural Networks for Compound Activity Prediction. Authors: Fouad Alkhoury, Tamás Horváth, Tiago Janela, Andrea Mastropietro, and Jürgen Bajorath.
In the following we describe the scripts used to run the experimental pipeline.

### compute_graph_features_compounds.py
This script loads the molecular graphs of each ChEMBL activity dataset and computes graph-level feature representation for every compound. The extracted representation comprises three main feature groups: graphlet features, including occurrence counts and binary presence indicators for predefined graphlets, aggregated node features, obtained by summarizing node structural features including centrality scores and degree-based measures, and global graph properties, such as the number of nodes and edges, graph diameter, and shortest path length statistics.
The script additionally includes molecular descriptors such as molecular weight, TPSA, logP, the numbers of chiral centers and rotatable bonds, as well as the MACCS fingerprints.
The resulting graph-level feature matrix is stored in: data/real/dataset_fingerprints_v3/dataset_compounds.csv. The script also records feature-computation run times for each molecular graph under reports/feature_data/.

### graphlets.py
This module defines the graphlet templates used during structural feature extraction. It contains the considered graphlet patterns with two to five nodes. These predefined subgraphs are used by compute_graph_features_compounds.py to identify graphlet occurrences in each molecular graph and to construct the corresponding count and presence-based graphlet features.

### split_graphs_compounds_repeats.py
This script partitions the original training set D_train into two disjoint subsets. The first subset, D_FR, is used to train the feature ranking GNN to predict informative molecular features.
The second subset, D_GC, is used for training and evaluating the downstream graph classification model using the selected features.

### train_rf_compounds.py
This script generates the feature importance targets used to train the feature ranking model. For each molecular graph in D_FR, the graph is treated as the positive instance and is contrasted with a sampled set of graphs from the opposite activity class. The script stores both the raw importance vectors and the resulting rankings.

### fr_gnn.py
This script trains the feature ranking graph neural network (FR-GNN) on the graphs in D_FR. The model uses graph convolutional layers followed by global mean pooling to produce one graph level feature importance prediction per molecule. The learned models are stored under model/model_dataset_run.pth. These saved models are subsequently applied to unseen graphs to predict feature importance vectors and derive the feature rankings used by the downstream graph-classification model.

### gc_gnn.py
This script trains and evaluates the downstream graph classification GNN using he feature rankings predicted by the previously trained FR-GNN model. The graph-classification stage supports three GNN architectures: Graph Convolutional Network (GCN), Graph Attention Network (GAT), and GraphSAGE. The script records the selected feature count, training and test accuracy, model architecture, and experimental run in the output reports.




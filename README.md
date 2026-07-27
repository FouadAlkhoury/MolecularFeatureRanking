Code and datasets associated with the manuscript:
Feature Selection-Guided Graph Neural Networks for Compound Activity Prediction. Authors: Fouad Alkhoury, Tamás Horváth, Tiago Janela, Andrea Mastropietro, and Jürgen Bajorath.

## Repository Structure

This folder contains the scripts used to run the experiments, we explain in turn each file: 

### split_graphs.py
The script splits the set of training graphs D_train into two subsets:
D_FR to learn to predict feature ranking,
and D_GC which will be used to for the graph classification task.

### compute_graph_features.py
The script computes graph-level features of the three types: Graphlet Features, Aggregated Node Features, and Global Graph Properties.
The computed features are saved under data/real/dataset.train

### graphlets.py
It contains the definition of graphlet features of 2-5 nodes plus the hexagon.

### train_fr_c_gnn.py
The script computes the feature ranking vectors for the training graphs in the set D_FR based on the random forest algorithm.

### train_svm_importance.py
The Support Vector Machine version of computing the feature ranking vectors for the training graphs in the set D_FR.

### train_nn_importance.py
Using local explainer, this script computes the feature ranking vectors for the training graphs in the set D_FR.

### feature_learning_gc_gnn
The scripts trains the feature ranking (FR-GNN) model. The learned model is saved under model/model_dataset.pth

### gc.py
This script runs the graph classification model on three GNN architectures: GCN, GAT, and GraphSAGE.

The folders ba_s1, er_s2, ws_s3, and pl_s4 contain the generated synthetic graphs for four graph models and learning problems.
The results, reports, times are exported to 'reports' folder. 




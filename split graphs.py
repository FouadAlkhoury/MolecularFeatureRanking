# The script splits the set of training graphs D_train into two subsets:
# D_FR to learn to predict feature ranking,
# and D_GC which will be used to for the graph classification task.
import numpy as np
import os
import random

#datasets = ['MUTAG','ENZYMES','PROTEINS','IMDB-BINARY','CSL','BA2MotifDataset','NCI1', 'BZR','DHFR', 'PTC_FR','PTC_MM','COX2','MSRC_9','IMDB-MULTI']
#datasets = ['or_graphs']
#datasets = ['ba_single','er_single','ws_single','pl_single']
#datasets = ['ba_and']
datasets = ['pl_s1']
#datasets = ['ba_or','er_or','ws_or','pl_or']
#datasets = ['ba_imbalanced','er_imbalanced','ws_imbalanced','pl_imbalanced']
#datasets = ['ba_counting','er_counting','ws_counting','pl_counting']

#dataset_length = [188,600,1113,1000,150,1000, 4110, ]
dataset_length = [600]
training_graphs_count = 60
#fr_training_list = []


for i,dataset in enumerate(datasets):
    fr_training_list = []
    gc_training_list = []
    fr_training_indexes = random.sample(range(1,dataset_length[i]+1),training_graphs_count)
    with open('data/real/' + dataset + '/' + dataset + '.csv','r') as file:
        for line_index,line in enumerate(file):
            #if (line_index == 0):
            #    fr_training_list.append(line)
            #    gc_training_list.append(line)
            if (line_index in fr_training_indexes):
                fr_training_list.append(line)
            else:
                if (line_index != 0):
                    gc_training_list.append(line)

    with open('data/real/split/gc_training_'+dataset+'.csv', 'w') as file_write_gc:
        for graph_line in gc_training_list:
            file_write_gc.write(str(graph_line))

    with open('data/real/split/fr_training_' + dataset + '.csv','w') as file_write_fr:
        for graph_line in fr_training_list:
            file_write_fr.write(str(graph_line))

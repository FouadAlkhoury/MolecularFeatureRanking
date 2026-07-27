# The script computes the feature ranking vectors for the training graphs in the set D_FR based on the random forest algorithm.

from posixpath import split
from util import writeToReport, list_to_str
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import shap
import random
import pandas as pd
import datetime

def get_negative_samples(i, total_graphs, group_size, intra_count, inter_count):
    group_index = i // group_size
    same_group_indices = list(range(group_index * group_size, (group_index + 1) * group_size))
    same_group_indices.remove(i)
    # intra_negatives = random.sample(same_group_indices, intra_count)
    other_group_indices = [j for j in range(total_graphs) if j // group_size != group_index]
    inter_negatives = random.sample(other_group_indices, inter_count)
    # print(intra_negatives)
    return inter_negatives


dataset = 'ws_s5'
training_data_file = 'data/real/split/fr_training_' + dataset + '.csv'
graphs = np.loadtxt(training_data_file, delimiter=',', dtype=str)
X = graphs[:, 1:-1]
Y = graphs[:, -1]
# X = np.delete(X,D,axis =1)
X = X.astype(np.float32)
Y_clean = np.array([
    int(y[7:-1]) if isinstance(y, str) and y.startswith('tensor(') else int(y)
    for y in Y
])
Y = Y_clean.astype(int)

def features_header(feature):
    return 'avg. ' + feature + ', std. ' + feature + ', skewness ' + feature + ', kurtosis ' + feature + ','


report_graph_level_importance_raw = 'data/real/split/importance_' + dataset + '.csv'
report_graph_level_ranking_raw = 'data/real/split/ranking_' + dataset + '.csv'
report_graph_level_importance_agg = 'data/real/split/importance_agg_' + dataset + '.csv'
report_graph_level_ranking_agg = 'data/real/split/ranking_agg_' + dataset + '.csv'

'''
writeToReport(report_graph_level_importance,'graph_id, nodes count, edges count, diameter, avg. path length, std path length, ' + features_header('degree') + features_header('eigenvector_cent.') + features_header('closeness cent.')
              + features_header('harmonic cent.') + features_header('betweenness cent. ') + features_header('coloring LF') + features_header('edges within egonet')
              + features_header('node clique number') + features_header('number of cliques') + features_header('clustering coefficient') +
              features_header('square clustering coefficient')+ features_header('pagerank') + features_header('hubs') + features_header('core number') + 'random,'
              + list_to_str(['c. g_'+str(_) for _ in range(0,30)]) + list_to_str(['b. g_'+str(_) for _ in range(0,30)]))
writeToReport(report_graph_level_ranking,'graph_id, nodes count, edges count, diameter, avg. path length, std path length, ' + features_header('degree') + features_header('eigenvector_cent.') + features_header('closeness cent.')
              + features_header('harmonic cent.') + features_header('betweenness cent. ') + features_header('coloring LF') + features_header('edges within egonet')
              + features_header('node clique number') + features_header('number of cliques') + features_header('clustering coefficient') +
              features_header('square clustering coefficient')+ features_header('pagerank') + features_header('hubs') + features_header('core number') + 'random,'
              + list_to_str(['c. g_'+str(_) for _ in range(0,30)]) + list_to_str(['b. g_'+str(_) for _ in range(0,30)]))
'''

def adaptive_threshold_and_normalize(arr, threshold=0.02):
    """Apply thresholding (keep values >= threshold) and normalize using Min-Max scaling."""

    # Apply thresholding: Set values below threshold to zero
    arr_thresholded = np.where(arr >= threshold, arr, 0)
    min_vals = np.min(arr_thresholded)  # .min(axis=1, keepdims=True)
    max_vals = np.max(arr_thresholded)  # .max(axis=1, keepdims=True)
    arr_normalized = (arr_thresholded - min_vals) / (max_vals - min_vals + 1e-8)
    arr_normalized = np.round(arr_normalized, 3)
    return arr_normalized


start_time = datetime.datetime.now()
arr_agg = np.zeros(124)
all_feature_orderings = []

for i in range(len(X)):

    mask = np.where(Y != Y[i])[0]
    arr_total = np.zeros(124)
    # arr_agg = np.zeros(122)
    for j in range(10):
        random_graphs = random.sample(list(mask), 5)
        X_negative = X[random_graphs]

        X_train = np.vstack([X[i], X_negative])
        Y_train = np.array([1] + [0] * len(X_negative))

        model = RandomForestClassifier(n_estimators=128, max_depth=15, max_features=12, random_state=j)
        model.fit(X_train, Y_train)
        arr = model.feature_importances_
        arr_total += arr
        arr_agg += arr
        arr_ordered_pro_iter = np.argsort(arr)[::-1]
        all_feature_orderings.append(arr_ordered_pro_iter)

    arr_ordered = np.argsort(arr_total)[::-1]
    print('Ranking: ' + str(arr_ordered))
    print(arr_ordered)
    arr_total = arr_total / 10

    writeToReport(report_graph_level_importance_raw, graphs[i][0] + ',' + list_to_str(list(arr_total)))
    writeToReport(report_graph_level_ranking_raw, graphs[i][0] + ',' + list_to_str(arr_ordered))

from collections import Counter

feature_counter = Counter()
for feature_ordering in all_feature_orderings:
    top_20_features = feature_ordering[:25]
    feature_counter.update(top_20_features)

sorted_features = feature_counter.most_common()
for feature, count in sorted_features:
    print(f"Feature {feature}: {count} times")

print(sorted_features)

arr_ordered_agg = np.argsort(arr_agg)[::-1]

writeToReport(report_graph_level_importance_agg, graphs[i][0] + ',' + list_to_str(list(arr_agg)))
writeToReport(report_graph_level_ranking_agg, graphs[i][0] + ',' + list_to_str(list(arr_ordered_agg)))

end_time = datetime.datetime.now()
diff_time = datetime.timedelta()
diff_time = (end_time - start_time)

print(diff_time)
time_file = 'data/real/split/computing_importance_time.txt'

writeToReport(time_file, 'RF,' + dataset + ', ' + str(diff_time))

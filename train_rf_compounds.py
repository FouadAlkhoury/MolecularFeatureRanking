from util import writeToReport, list_to_str
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import datetime


compounds_datasets = ['CHEMBL1163125','CHEMBL1914','CHEMBL204','CHEMBL214', 'CHEMBL218','CHEMBL220', 'CHEMBL230','CHEMBL261','CHEMBL2973','CHEMBL4822']
negative_samples = [10, 10, 9, 6, 5, 10, 8, 8, 3, 6 ]


for d, dataset in enumerate(compounds_datasets):
    print(dataset)
    for run in range(1, 11):
        print('run: ' + str(run))
        training_data_file = (
            'data/real/split/fr_training_'
            + dataset
            + '_run'
            + str(run)
            + '_repeat1_compounds_v3.csv'
        )

        with open(training_data_file, 'r') as f:
            first_line = f.readline().strip()

        skiprows = 1 if 'graph_id' in first_line else 0
        graphs = np.loadtxt(
            training_data_file,
            delimiter=',',
            dtype=str,
            skiprows=skiprows
        )

        X = graphs[:, 158:-1]
        Y = graphs[:, -1]
        X = X.astype(np.float32)

        Y_clean = np.array([
            int(y[7:-1]) if isinstance(y, str) and y.startswith('tensor(') else int(y)
            for y in Y
        ])
        Y = Y_clean.astype(int)

        report_graph_level_importance_raw = (
            'data/real/split/importance_'
            + dataset
            + '_run'
            + str(run)
            + '_compounds_v3.csv'
        )
        report_graph_level_ranking_raw = (
            'data/real/split/ranking_'
            + dataset
            + '_run'
            + str(run)
            + '_compounds_v3.csv'
        )
        report_graph_level_importance_agg = (
            'data/real/split/importance_agg_'
            + dataset
            + '_run'
            + str(run)
            + '_compounds_v3.csv'
        )
        report_graph_level_ranking_agg = (
            'data/real/split/ranking_agg_'
            + dataset
            + '_run'
            + str(run)
            + '_compounds_v3.csv'
        )

        start_time = datetime.datetime.now()
        arr_agg = np.zeros(167)
        all_feature_orderings = []

        for i in range(len(X)):
            mask = np.where(Y != Y[i])[0]
            arr_total = np.zeros(167)

            for j in range(2):
                random_graphs = random.sample(list(mask), negative_samples[d])
                X_negative = X[random_graphs]

                X_train = np.vstack([X[i], X_negative])
                Y_train = np.array([1] + [0] * len(X_negative))

                model = RandomForestClassifier(
                    n_estimators=16,
                    max_depth=5,
                    max_features=20,
                    random_state=j
                )
                model.fit(X_train, Y_train)

                arr = model.feature_importances_
                arr_total += arr
                arr_agg += arr

                arr_ordered_pro_iter = np.argsort(arr)[::-1]
                all_feature_orderings.append(arr_ordered_pro_iter)

            arr_ordered = np.argsort(arr_total)[::-1]
            arr_total = arr_total / 3

            writeToReport(
                report_graph_level_importance_raw,
                graphs[i][0] + ',' + list_to_str(list(arr_total))
            )
            writeToReport(
                report_graph_level_ranking_raw,
                graphs[i][0] + ',' + list_to_str(arr_ordered)
            )

        from collections import Counter

        feature_counter = Counter()

        for feature_ordering in all_feature_orderings:
            top_20_features = feature_ordering[:15]
            feature_counter.update(top_20_features)

        sorted_features = feature_counter.most_common()

        for feature, count in sorted_features:
            print(f"Feature {feature}: {count} times")

        print(sorted_features)

        arr_ordered_agg = np.argsort(arr_agg)[::-1]

        writeToReport(
            report_graph_level_importance_agg,
            graphs[i][0] + ',' + list_to_str(list(arr_agg))
        )
        writeToReport(
            report_graph_level_ranking_agg,
            graphs[i][0] + ',' + list_to_str(list(arr_ordered_agg))
        )
        writeToReport(
            report_graph_level_ranking_agg,
            graphs[i][0] + ',' + list_to_str(sorted_features)
        )

        end_time = datetime.datetime.now()
        diff_time = end_time - start_time

        print(diff_time)
        time_file = 'data/real/split/computing_importance_time_compounds_v3.txt'

        writeToReport(
            time_file,
            'RF,' + dataset + '_run' + str(run) + ', ' + str(diff_time)
        )

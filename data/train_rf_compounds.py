import numpy as np
import random
import datetime
from sklearn.ensemble import RandomForestClassifier
from util import writeToReport, list_to_str
from collections import Counter

# === CONFIG ===

DATASETS = ['CHEMBL1163125','CHEMBL1914','CHEMBL204','CHEMBL214', 'CHEMBL218','CHEMBL220', 'CHEMBL230','CHEMBL261','CHEMBL2973','CHEMBL4822']
NEGATIVE_SAMPLES = [10, 10, 9, 6, 5, 10, 8, 8, 3, 6 ]
N_SPLITS = 1                    
N_NEG_ITERS = 3                # number of negative sampling loops
N_TREES = 64
MAX_DEPTH = 15
MAX_FEATURES = 15

# Only fingerprints features
FEATURE_COUNT = 167

def run_rf_on_split(X, Y, neg_count):
    """
    Runs the RF importance extraction on one split.
    Returns aggregated importance array and the list of orderings.
    """
    arr_agg = np.zeros(FEATURE_COUNT)
    all_orderings = []

    for i in range(len(X)):
        mask = np.where(Y != Y[i])[0]

        for j in range(N_NEG_ITERS):
            negatives = random.sample(list(mask), neg_count)
            X_negative = X[negatives]

            X_train = np.vstack([X[i], X_negative])
            Y_train = np.array([1] + [0] * len(negatives))

            model = RandomForestClassifier(
                n_estimators=N_TREES,
                max_depth=MAX_DEPTH,
                max_features=MAX_FEATURES,
                random_state=j
            )
            model.fit(X_train, Y_train)

            arr = model.feature_importances_
            arr_agg += arr
            all_orderings.append(np.argsort(arr)[::-1])


    return arr_agg, all_orderings


# ============================================================
#                       MAIN
# ============================================================

for d, dataset in enumerate(DATASETS):
    print("\n====================================================")
    print("Dataset:", dataset)
    print("====================================================")

    for run in range(1, 11):
        print(f"\n▶ RUN {run}")

        # ==== load all 7 DFR splits ====
        split_files = []
        for s in range(1, N_SPLITS + 1):
            fname = f"data/real/split/fr_training_{dataset}_run{run}_repeat{s}_compounds_v3.csv"
            split_files.append(fname)

        # Overall aggregation across 7 splits
        overall_importance = np.zeros(FEATURE_COUNT)

        # For collecting orderings of all splits
        overall_counter = Counter()

        start_time = datetime.datetime.now()

        for s, fname in enumerate(split_files, start=1):
            print(f"  → Processing split {s}")

            graphs = np.loadtxt(fname, delimiter=",", dtype=str, skiprows=1)

            X = graphs[:,158:158+167].astype(np.float32)


            Y = np.array([int(y) for y in graphs[:,-1]])

            # === Train RF on this split ===
            arr_agg, orderings = run_rf_on_split(X, Y, NEGATIVE_SAMPLES[d])

            # update global
            overall_importance += arr_agg

            # Count top features for ranking
            for ordering in orderings:
                top15 = ordering[:2048]
                overall_counter.update(top15)

            # SAVE per-split outputs
            imp_file  = f"data/real/split/importance_{dataset}_run{run}_split{s}_v3_all_fingerprints_one_split.csv"
            rank_file = f"data/real/split/ranking_{dataset}_run{run}_split{s}_v3_all_fingerprints_one_split.csv"

            ordering_split = np.argsort(arr_agg)[::-1]

            writeToReport(imp_file,  "importance," + list_to_str(list(arr_agg)))
            writeToReport(rank_file, "ranking,"    + list_to_str(list(ordering_split)))

        # ======================================================
        #  Save overall results from 7 splits
        # ======================================================
        print("  → Saving overall results...")

        overall_ordering = np.argsort(overall_importance)[::-1]

        over_imp_file  = f"data/real/split/importance_overall_{dataset}_run{run}_v3_all_fingerprints_one_split.csv"
        over_rank_file = f"data/real/split/ranking_overall_{dataset}_run{run}_v3_all_fingerprints_one_split.csv"

        writeToReport(over_imp_file, "importance," + list_to_str(list(overall_importance)))
        writeToReport(over_rank_file, "ranking,"    + list_to_str(list(overall_ordering)))

        # Write frequency-based ranking
        freq_sorted = overall_counter.most_common()
        writeToReport(over_rank_file, "frequency," + list_to_str(freq_sorted))

        end_time = datetime.datetime.now()
        diff = end_time - start_time
        print("  Time:", diff)


print("\nALL DONE.")

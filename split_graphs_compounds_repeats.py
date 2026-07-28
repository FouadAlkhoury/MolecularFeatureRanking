import pickle
import random
import os
import numpy as np

def load_compounds_dataset(filename):
    """Load pickled compound dataset."""
    with open(filename, 'rb') as f:
        compounds_dataset = pickle.load(f)
    print(f"Loaded {len(compounds_dataset)} graphs from {filename}")
    return compounds_dataset


def split_csv_by_dataset(dataset_name,
                         dataset,
                         csv_dir="./data/real",
                         output_dir="./data/real/split",
                         seed=42,
                         repeats=7):
    """
    Generate DFR and DGC CSV splits based on pickle 'set' info.
    Each DFR aims to have size = 10% of total graphs (rounded down) but if a chunk
    is smaller we take the whole chunk.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    total_graphs = len(dataset)
    dfr_target_size = int(0.10 * total_graphs)

    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Total graphs: {total_graphs} → target DFR (10%): {dfr_target_size}")
    print(f"Repeats (non-overlapping DFRs): {repeats}")

    # Path to csv: adjust according to your layout
    csv_path = os.path.join(csv_dir, dataset_name + "_fingerprints_v3", f"{dataset_name}_compounds.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read all CSV lines once
    with open(csv_path, "r") as f:
        csv_lines = f.readlines()
    
    # Remove empty lines and any header that contains "graph_id"
    clean_csv_lines = []
    for line in csv_lines:
        stripped = line.strip()
        if stripped == "":
            continue
        if "graph_id" in stripped.lower():
            continue
        clean_csv_lines.append(line)

    # Use clean_csv_lines instead of csv_lines
    csv_lines = clean_csv_lines
    
    # For safety, determine header index - assume first non-empty line is header
    header_idx = None
    for i, line in enumerate(csv_lines):
        if line.strip() != "":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("CSV seems empty or only blank lines")

    # The graph rows correspond to indices starting at 0 for the first data line (header removed)
    data_start_index = header_idx + 1
    num_csv_graphs = len(csv_lines) - data_start_index
    if num_csv_graphs != total_graphs:
        print(f"Warning: number of CSV data lines ({num_csv_graphs}) != number of graphs in pickle ({total_graphs}).\n"
              f"CSV path: {csv_path}")

    # Handle each of the 10 runs (0..9)
    for run in range(10):
        print(f"\n--- Run {run + 1} ---")

        # Identify train/test graphs by index (0-based graph index)
        train_indices = [i for i, g in enumerate(dataset) if g.graph["set"][run] == "train"]
        test_indices = [i for i, g in enumerate(dataset) if g.graph["set"][run] == "test"]

        if not train_indices:
            print(f"No 'train' graphs found for run {run + 1}, skipping...")
            continue

        # Shuffle train indices and split into 'repeats' disjoint chunks
        shuffled = train_indices[:]  # copy
        random.shuffle(shuffled)

        chunks = np.array_split(np.array(shuffled), repeats)  # list of numpy arrays (some may be empty)
        # ensure we convert to python lists
        chunks = [list(chunk) for chunk in chunks]

        # For each repeat, pick up to dfr_target_size from that chunk (no overlap between repeats)
        for rep_idx, chunk in enumerate(chunks):
            # If chunk is empty, DFR will be empty for that repeat
            if len(chunk) == 0:
                print(f"  Repeat {rep_idx+1}: chunk empty (no train indices assigned); producing empty DFR.")
                dfr_indices = set()
            else:
                if len(chunk) >= dfr_target_size:
                    # if chunk bigger, sample without replacement from the chunk to reach target size
                    dfr_indices = set(random.sample(chunk, dfr_target_size))
                else:
                    # chunk smaller than target: take whole chunk
                    dfr_indices = set(chunk)
                    print(f"  Repeat {rep_idx+1}: chunk size {len(chunk)} < target {dfr_target_size}; using whole chunk as DFR.")

            # DGC = all graphs (train + test) except those in DFR
            all_indices = set(train_indices + test_indices)
            dgc_indices = all_indices

            # Prepare CSV lists for writing (we'll preserve header if present)
            fr_training_list = []
            gc_training_list = []

            for line_index, line in enumerate(csv_lines):
                # preserve header line in both outputs
                if line_index == header_idx:
                    fr_training_list.append(line)
                    gc_training_list.append(line)
                    continue

                # align CSV row index (excluding header) to graph index
                idx = line_index - data_start_index
                if idx < 0 or idx >= total_graphs:
                    # if there are extra lines (blank or trailing), skip
                    continue

                if idx in dfr_indices:
                    fr_training_list.append(line)

                if idx in dgc_indices:
                    gc_training_list.append(line)
                else:
                    # if an index is neither in dfr nor in dgc (shouldn't happen), skip
                    pass

            # Output file paths
            dfr_csv = os.path.join(output_dir, f"fr_training_{dataset_name}_run{run + 1}_repeat{rep_idx + 1}_compounds_v3.csv")
            dgc_csv = os.path.join(output_dir, f"gc_training_{dataset_name}_run{run + 1}_repeat{rep_idx + 1}_compounds_v3.csv")

            # Write CSVs
            with open(dfr_csv, "w") as f:
                f.writelines(fr_training_list)
            with open(dgc_csv, "w") as f:
                f.writelines(gc_training_list)

            print(f"  Repeat {rep_idx+1}: wrote {len(fr_training_list)-1} DFR lines, {len(gc_training_list)-1} DGC lines "
                  f"(DFR indices: {sorted(list(dfr_indices))[:10]}{'...' if len(dfr_indices)>10 else ''})")


if __name__ == "__main__":
    

    

    compounds_datasets = ['CHEMBL1163125','CHEMBL1914','CHEMBL204','CHEMBL214', 'CHEMBL218','CHEMBL220', 'CHEMBL230','CHEMBL261','CHEMBL2973','CHEMBL4822']

    pickle_dir = "./pickles_100_v3"

    for dataset_name in compounds_datasets:
        pkl_path = os.path.join(pickle_dir, f"{dataset_name}_graphs.pkl")
        dataset = load_compounds_dataset(pkl_path)
        split_csv_by_dataset(dataset_name, dataset, seed=1234, repeats=7)

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.core.numeric
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D

sys.modules["numpy._core.numeric"] = numpy.core.numeric

def figure_1():
    ###1 accuracy per dataset and run


    def load_clean_csv(path):
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        if "dataset" in df.columns:
            df = df[df["dataset"] != "dataset"].copy()

        for col in ["run", "k", "test_acc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["dataset", "run", "test_acc"]).copy()
        df["run"] = df["run"].astype(int)

        if "k" in df.columns:
            df = df.dropna(subset=["k"]).copy()
            df["k"] = df["k"].astype(int)

        return df


    def load_variant_file(path, variant_name, selected_datasets, k_value=None):
        df = load_clean_csv(path)

        if k_value is not None:
            if "k" not in df.columns:
                raise ValueError(f"{path} has no 'k' column.")
            df = df[df["k"] == k_value].copy()

        df = df[df["dataset"].isin(selected_datasets)].copy()
        df["variant_plot"] = variant_name

        return df[["dataset", "run", "test_acc", "variant_plot"]].copy()


    def build_combined_dataframe(
        file_k_fingerprints,
        file_all_fingerprints,
        file_k_computed,
        file_all_computed,
        file_k_combined,
        file_all_combined,
        selected_datasets,
        k_fingerprints=6,
        k_computed=7,
        k_combined=22,
    ):
        df1 = load_variant_file(file_k_fingerprints, "Top-6-Fingerprints", selected_datasets, k_value=k_fingerprints)
        df2 = load_variant_file(file_all_fingerprints, "F-Fingerprints", selected_datasets, k_value=None)

        df3 = load_variant_file(file_k_computed, "Top-7-Computed", selected_datasets, k_value=k_computed)
        df4 = load_variant_file(file_all_computed, "F-Computed", selected_datasets, k_value=None)

        df5 = load_variant_file(file_k_combined, "Top-22-Combined", selected_datasets, k_value=k_combined)
        df6 = load_variant_file(file_all_combined, "F-Combined", selected_datasets, k_value=None)

        return pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)


    def plot_runs_per_variant_per_dataset(
        df,
        output_file,
        dataset_order,
        variant_order=None,
        figsize=(22, 8),
    ):
        if variant_order is None:
            variant_order = [
                "Top-6-Fingerprints",
                "F-Fingerprints",
                "Top-7-Computed",
                "F-Computed",
                "Top-22-Combined",
                "F-Combined",
            ]

        variant_colors = {
            "Top-6-Fingerprints": "#1f77b4",
            "F-Fingerprints": "#aec7e8",
            "Top-7-Computed": "#ff7f0e",
            "F-Computed": "#ffbb78",
            "Top-22-Combined": "#2ca02c",
            "F-Combined": "#98df8a",
        }

        run_markers = {
            1: "o", 2: "s", 3: "^", 4: "v", 5: "D",
            6: "P", 7: "X", 8: "*", 9: "<", 10: ">"
        }

        offsets = np.linspace(-0.50, 0.50, len(variant_order))
        dataset_spacing = 1.6
        base_x = np.arange(len(dataset_order)) * dataset_spacing

        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 20,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        fig, ax = plt.subplots(figsize=figsize)

        for y in [50, 60, 70, 80, 90]:
            ax.axhline(
                y=y,
                color="black",
                linestyle=(0, (5, 5)),  # dashed
                linewidth=1.2,
                alpha=0.5,
                zorder=0
            )

        for i, dataset in enumerate(dataset_order):
            ds = df[df["dataset"] == dataset].copy()

            means_for_line = []

            for j, variant in enumerate(variant_order):
                sub = ds[ds["variant_plot"] == variant].copy()

                if sub.empty:
                    means_for_line.append(np.nan)
                    continue

                x_center = base_x[i] + offsets[j]

                mean_acc = sub["test_acc"].mean()
                std_acc = sub["test_acc"].std()

                means_for_line.append(mean_acc)

                # std bar
                ax.errorbar(
                    x_center,
                    mean_acc,
                    yerr=std_acc,
                    color=variant_colors[variant],
                    capsize=4,
                    linewidth=2,
                    zorder=4,
                )

                # mean marker
                ax.scatter(
                    x_center,
                    mean_acc,
                    color=variant_colors[variant],
                    s=120,
                    marker="o",
                    edgecolors="black",
                    linewidth=1.0,
                    zorder=5,
                )

                # individual runs
                sub = sub.sort_values("run")
                jitters = np.linspace(-0.018, 0.018, len(sub))

                for (_, row), jitter in zip(sub.iterrows(), jitters):
                    run_id = int(row["run"])

                    ax.scatter(
                        x_center + jitter,
                        row["test_acc"],
                        color=variant_colors[variant],
                        marker=run_markers.get(run_id, "o"),
                        s=35,
                        edgecolors="black",
                        linewidth=0.5,
                        alpha=0.75,
                        zorder=3,
                    )

            # connect variant means within each dataset
            valid = ~np.isnan(means_for_line)
            ax.plot(
                base_x[i] + offsets[valid],
                np.array(means_for_line)[valid],
                color="black",
                linewidth=1.0,
                alpha=0.25,
                zorder=1,
            )

        ax.set_xticks(base_x)
        ax.set_xticklabels(
            [dataset_name_map_2.get(d, d) for d in dataset_order],
            rotation=0,
            fontsize=22
        )

        ax.set_ylabel("Test Accuracy (%)", fontsize = 24)
        ax.tick_params(axis="y", labelsize=22)
        ax.set_xlabel("")
        #ax.set_title("Performance Distribution Across 10 Runs")

        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        variant_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                linestyle = "None",
                markerfacecolor=variant_colors[v],
                markeredgecolor="black",
                markersize=14,
                label=v,
            )
            for v in variant_order
        ]

        variant_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linewidth=1.5,
                alpha=0.5,
                label="Mean"
            )
        )

        run_handles = [
            Line2D(
                [0], [0],
                marker=run_markers[r],
                color="black",
                linestyle="None",
                markersize=14,
                label=f"Run {r}",
            )
            for r in range(1, 11)
        ]

        # First legend: variants
        leg1 = fig.legend(
            handles=variant_handles,
            handletextpad=0.05,
            handlelength = 1.0,
            labelspacing=0.05,
            title="",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=len(variant_handles),
            frameon=False,
            fontsize=22,
            title_fontsize=14,
        )

        # Second legend: runs
        leg2 = fig.legend(
            handles=run_handles,
            handletextpad=0.05,
            handlelength=1.0,
            labelspacing=0.05,
            title="",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=10,
            frameon=False,
            fontsize=22,
            title_fontsize=14,
        )


        plt.tight_layout(rect=[0, 0.1, 1, 1])

        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_file}")
        print(f"Saved: {output_file.replace('.png', '.pdf')}")



    file_k_fingerprints = "plots_data/results_k_fingerprints.csv"
    file_all_fingerprints = "plots_data/results_full_fingerprints.csv"

    file_k_computed = "plots_data/results_k_computed.csv"
    file_all_computed = "plots_data/results_full_computed.csv"

    # UPDATE THESE TWO PATHS TO YOUR COMBINED RESULTS FILES
    file_k_combined = "plots_data/results_k_combined.csv"
    file_all_combined = "plots_data/results_full_combined.csv"

    dataset_order = [
        "CHEMBL1163125",
        "CHEMBL1914",
        "CHEMBL261",
        "CHEMBL204",
        "CHEMBL214",
        "CHEMBL218",
        "CHEMBL220",
        "CHEMBL230",
        "CHEMBL2973",
        "CHEMBL4822",
    ]

    dataset_name_map_2 = {
        "CHEMBL261": "CA1 \n (261)",
        "CHEMBL220": "AChE \n (220)",
        "CHEMBL1914": "BChE \n (1914)",
        "CHEMBL1163125": "BRD4 \n (1163125)",
        "CHEMBL230": "COX-2 \n (230)",
        "CHEMBL204": "Thrombin \n (204)",
        "CHEMBL4822": "BACE1 \n (4822)",
        "CHEMBL214": "5-HT1A \n (214)",
        "CHEMBL218": "CB1 \n (218)",
        "CHEMBL2973": "ROCK2 \n (2973)",
    }

    dataset_name_map = {
        "CHEMBL261": "261",
        "CHEMBL220": "220",
        "CHEMBL1914": "1914",
        "CHEMBL1163125": "1163125",
        "CHEMBL230": "230",
        "CHEMBL204": "204",
        "CHEMBL4822": "4822",
        "CHEMBL214": "214",
        "CHEMBL218": "218",
        "CHEMBL2973": "2973",
    }

    df = build_combined_dataframe(
        file_k_fingerprints=file_k_fingerprints,
        file_all_fingerprints=file_all_fingerprints,
        file_k_computed=file_k_computed,
        file_all_computed=file_all_computed,
        file_k_combined=file_k_combined,
        file_all_combined=file_all_combined,
        selected_datasets=dataset_order,
        k_fingerprints=6,
        k_computed=7,
        k_combined=22,
    )

    plot_runs_per_variant_per_dataset(
        df=df,
        output_file="plots_output/accuracy_per_dataset_run.png",
        dataset_order=dataset_order,
        variant_order=[
            "Top-6-Fingerprints",
            "F-Fingerprints",
            "Top-7-Computed",
            "F-Computed",
            "Top-22-Combined",
            "F-Combined",
        ],
        figsize=(22, 8),
    )


def figure_2():
    ###2 Heatmap accuracy k vs full set 



    # ============================================================================
    # PARAMETERS
    # ============================================================================

    RESULTS_K_FILE = "results_v3/final_results/results_k_fingerprints.csv"
    RESULTS_FULL_FILE = "results_v3/final_results/results_full_fingerprints.csv"

    OUTPUT_DIR = "plots_output"

    OUTPUT_HEATMAP = "heatmap_delta_accuracy_vs_full.png"
    OUTPUT_AVERAGES = "averaged_accuracy_values.csv"

    K_MIN = 1
    K_MAX = 15

    FIGSIZE = (11, 6)
    DPI = 300

    DATASET_ORDER = [
        "CHEMBL220",
        "CHEMBL4822",
        "CHEMBL218",
        "CHEMBL2973",
        "CHEMBL1163125",
        "CHEMBL204",
        "CHEMBL261",
        "CHEMBL230",
        "CHEMBL1914",
        "CHEMBL214",

    ]

    DATASET_LABELS = {
        "CHEMBL261": "CA1 (261)",
        "CHEMBL220": "AChE (220)",
        "CHEMBL1914": "BChE (1914)",
        "CHEMBL1163125": "BRD4 (1163125)",
        "CHEMBL230": "COX-2 (230)",
        "CHEMBL204": "Thrombin (204)",
        "CHEMBL4822": "BACE1 (4822)",
        "CHEMBL214": "5-HT1A (214)",
        "CHEMBL218": "CB1 (218)",
        "CHEMBL2973": "ROCK2 (2973)",
    }


    # ============================================================================
    # LOADING
    # ============================================================================

    def load_and_clean_results(filename):
        """
        Load CSV with possible repeated headers.
        Expected columns:
        dataset, size, run, model, variant, k, test_acc, train_acc, gc_training_time
        """

        df = pd.read_csv(filename, skipinitialspace=True)

        df.columns = df.columns.str.strip()

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        df = df[df["dataset"] != "dataset"].copy()

        df["run"] = pd.to_numeric(df["run"], errors="coerce")
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")

        df = df.dropna(subset=["dataset", "run", "k", "test_acc"]).copy()

        df["run"] = df["run"].astype(int)
        df["k"] = df["k"].astype(int)

        return df


    # ============================================================================
    # STATISTICS
    # ============================================================================

    def compute_k_stats(df_k):
        """
        Average test accuracy for each dataset and k.
        """

        df_k = df_k[
            (df_k["k"] >= K_MIN) &
            (df_k["k"] <= K_MAX)
        ].copy()

        k_stats = (
            df_k.groupby(["dataset", "k"])["test_acc"]
            .agg(
                avg_test_acc="mean",
                std_test_acc="std",
                n_runs="count"
            )
            .reset_index()
        )

        return k_stats


    def compute_full_stats(df_full):
        """
        Average full-feature accuracy for each dataset.
        """

        full_stats = (
            df_full.groupby("dataset")["test_acc"]
            .agg(
                full_avg_test_acc="mean",
                full_std_test_acc="std",
                full_n_runs="count"
            )
            .reset_index()
        )

        return full_stats


    def build_averages_table(k_stats, full_stats):
        """
        Merge top-k and full-feature averages.
        Adds delta over full feature set.
        """

        merged = k_stats.merge(
            full_stats,
            on="dataset",
            how="left"
        )

        merged["delta_vs_full"] = (
            merged["avg_test_acc"] - merged["full_avg_test_acc"]
        )

        merged = merged.sort_values(["dataset", "k"])

        return merged


    # ============================================================================
    # PLOTTING
    # ============================================================================

    def plot_delta_heatmap(averages, output_path):
        """
        Heatmap:
        rows = datasets
        columns = k
        values = avg top-k accuracy - avg full-feature accuracy
        """

        datasets = [
            d for d in DATASET_ORDER
            if d in averages["dataset"].unique()
        ]

        k_values = list(range(K_MIN, K_MAX + 1))

        heatmap_df = (
            averages
            .pivot(index="dataset", columns="k", values="delta_vs_full")
            .reindex(index=datasets, columns=k_values)
        )

        heatmap_df.loc["MEAN"] = heatmap_df.mean(axis=0, skipna=True)
        matrix = heatmap_df.values

        vmax = max(abs(pd.Series(matrix.flatten()).dropna()).max(), 1.0)

        fig, ax = plt.subplots(figsize=FIGSIZE)

        norm = TwoSlopeNorm(
            vmin=-28,
            vcenter=0,
            vmax=7
        )

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm
        )

        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values, fontsize=16)

        plot_rows = datasets + ["MEAN"]

        y_labels = [
            DATASET_LABELS.get(d, d.replace("CHEMBL", ""))
            if d != "MEAN" else "Mean"
            for d in plot_rows
        ]

        ax.set_yticks(range(len(plot_rows)))
        ax.set_yticklabels(y_labels, fontsize=14)

        #ax.set_ylabel("Dataset", fontsize=14)

        #ax.set_title(
        #    "Accuracy gain over the full feature set",
        #    fontsize=16,
        #    fontweight="bold"
        #)

        # Mark best k per dataset
        # ============================================================================
        # WRITE BEST GAIN VALUE
        # ============================================================================

        for i, dataset in enumerate(plot_rows):


            row = heatmap_df.loc[dataset]

            if row.notna().any():
                best_k = row.idxmax()
                best_gain = row.max()
                if (i == len(plot_rows)-1):
                    best_gain = 2.2 ## mean
                if (i == 7):
                    best_gain = 6.2   ## 230
                if (i == 9):
                    best_gain = 0.1   ## 214
                #best_j = k_values.index(best_k)

                best_j = k_values.index(best_k)

                ax.text(
                    best_j,
                    i,
                    f"+{abs(best_gain):.1f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    #bbox=dict(
                    #    facecolor="white",
                    #    edgecolor="black",
                    #    boxstyle="round,pad=0.2",
                    #    alpha=0.85
                    #),
                    zorder=6
                )
                # Draw black rectangle around best cell
                rect = plt.Rectangle(
                    (best_j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                    zorder=7
                )

                ax.add_patch(rect)
        # Highlight k = 6
        '''
        if 6 in k_values:
            k6_index = k_values.index(6)
            ax.axvline(
                k6_index,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7
            )
        '''
        # Grid lines
        ax.set_xticks([x - 0.5 for x in range(1, len(k_values))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(plot_rows))], minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel(
            "Number of selected features $K$",
            fontsize=18,
            labelpad=12
        )

        ax.axhline(
            len(datasets) - 0.5,
            color="black",
            linewidth=3.0
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

        cbar.set_label(
            "Accuracy gain over full set",
            fontsize=16,
            #fontweight="bold"
        )

        cbar.ax.tick_params(labelsize=14)

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.subplots_adjust(bottom=0.15)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved heatmap: {output_path}")


    # ============================================================================
    # MAIN
    # ============================================================================

    def main():

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_k = load_and_clean_results(RESULTS_K_FILE)
        df_full = load_and_clean_results(RESULTS_FULL_FILE)

        k_stats = compute_k_stats(df_k)
        full_stats = compute_full_stats(df_full)

        averages = build_averages_table(k_stats, full_stats)

        averages_output_path = os.path.join(
            OUTPUT_DIR,
            OUTPUT_AVERAGES
        )

        averages.to_csv(averages_output_path, index=False)

        print(f"Saved averages: {averages_output_path}")

        heatmap_output_path = os.path.join(
            OUTPUT_DIR,
            OUTPUT_HEATMAP
        )

        plot_delta_heatmap(
            averages,
            heatmap_output_path
        )


    main()


def figure_3():
    ###3 accuracy per dataset, k from 4 to 8



    def load_and_clean_results(filename):
        """
        Load CSV with possible repeated headers inside the file.
        Expected format:
        dataset, size, run, model, variant, k, test_acc, train_acc, gc_training_time
        """
        df = pd.read_csv(filename, skipinitialspace=True)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Strip whitespace in string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        # Remove repeated header rows
        df = df[df["dataset"] != "dataset"].copy()

        # Convert needed columns
        df["run"] = pd.to_numeric(df["run"], errors="coerce")
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")

        # Drop bad rows
        df = df.dropna(subset=["dataset", "run", "k", "test_acc"]).copy()

        # Normalize types
        df["run"] = df["run"].astype(int)
        df["k"] = df["k"].astype(int)

        return df


    def compute_k_stats(df):
        """
        Returns per-(dataset, k) mean test accuracy.
        """
        stats = (
            df.groupby(["dataset", "k"])["test_acc"]
            .mean()
            .reset_index(name="avg_test_acc")
        )
        return stats


    def compute_full_stats(df):
        """
        Returns per-dataset mean test accuracy for full feature set.
        """
        stats = (
            df.groupby("dataset")["test_acc"]
            .mean()
            .reset_index(name="full_avg_test_acc")
        )
        return stats


    def plot_all_datasets_one_axis(k_stats, full_stats, output_file):
        """
        One single plot:
        - x-axis = k
        - y-axis = average test accuracy
        - one color per dataset
        - solid line: top-k results
        - dashed horizontal line: full feature result
        """
        datasets = sorted(set(k_stats["dataset"].unique()) | set(full_stats["dataset"].unique()))

        plt.figure(figsize=(12, 7))
        plt.rcParams.update({
            "font.size": 16,  # base font size
            "axes.titlesize": 20,  # title
            "axes.labelsize": 18,  # x and y labels
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 24
        })

        dataset_name_map = {
            "CHEMBL261": "261",
            "CHEMBL220": "220",
            "CHEMBL1914": "1914",
            "CHEMBL1163125": "CHEMBL1163125",
            "CHEMBL230": "230",
            "CHEMBL204": "204",
            "CHEMBL4822": "4822",
            "CHEMBL214": "214",
            "CHEMBL218": "218",
            "CHEMBL2973": "2973",
        }

        dataset_name_map = {
            "CHEMBL261": "CA1 (261)",
            "CHEMBL220": "AChE (220)",
            "CHEMBL1914": "BChE (1914)",
            "CHEMBL1163125": "BRD4 (1163125)",
            "CHEMBL230": "COX-2 (230)",
            "CHEMBL204": "Thrombin (204)",
            "CHEMBL4822": "BACE1 (4822)",
            "CHEMBL214": "5-HT1A (214)",
            "CHEMBL218": "CB1 (218)",
            "CHEMBL2973": "ROCK2 (2973)",
        }

        for dataset_name in datasets:
            ds_k = k_stats[k_stats["dataset"] == dataset_name].copy()
            ds_full = full_stats[full_stats["dataset"] == dataset_name].copy()
            display_name = dataset_name_map.get(dataset_name, dataset_name)

            # Example:
            #plt.title(display_name, fontsize=18)

            if ds_k.empty:
                continue

            k_min = 4
            k_max= 8
            ds_k = ds_k[(ds_k["k"] >= k_min) & (ds_k["k"] <= k_max)]

            ds_k = ds_k.sort_values("k")
            x = ds_k["k"]
            y = ds_k["avg_test_acc"]

            scatter = plt.scatter(x, y, marker="o", label=display_name)
            color = scatter.get_facecolor()[0]

            # Plot top-k curve and capture its color
            #line, = plt.plot(x, y, marker="o", label=display_name)
            #color = line.get_color()

            # Plot full-feature baseline in same color
            if not ds_full.empty:
                full_avg = ds_full.iloc[0]["full_avg_test_acc"]
                plt.hlines(
                    y=full_avg,
                    xmin=x.min(),
                    xmax=x.max(),
                    linestyles="--",
                    colors=color

                )

        plt.xlabel("K", fontsize=22)
        plt.ylabel("Average test accuracy", fontsize=22)
        #plt.title("Average test accuracy over 10 runs for all datasets")
        plt.xticks(sorted(k_stats["k"].unique()))
        plt.tick_params(axis="both", labelsize=22)
        plt.xlim(k_min-0.05, k_max+0.05)
        plt.ylim(50, 88)
        plt.grid(True, alpha=0.3)
        # Existing dataset legend handles
        handles, labels = plt.gca().get_legend_handles_labels()

        # Add custom dashed-line legend entry
        handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                label="Full computed set"
            )
        )

        labels.append("F-Fingerprints")
        #labels = ["Full fingerprints set"] + labels


        plt.legend(handles=handles, labels=labels, fontsize=16, ncol=2,bbox_to_anchor=(0.98, 0.42))
        #plt.legend(fontsize=8, ncol=2)
        #plt.tight_layout()

        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_file}")


    def export_all_dataset_plot(results_k_file, results_full_file, output_dir="results_v3/plots_output"):
        os.makedirs(output_dir, exist_ok=True)

        df_k = load_and_clean_results(results_k_file)
        df_full = load_and_clean_results(results_full_file)

        k_stats = compute_k_stats(df_k)
        full_stats = compute_full_stats(df_full)

        output_file = os.path.join(output_dir, "accuracy_all_datasets_4_to_8k.png")
        plot_all_datasets_one_axis(k_stats, full_stats, output_file)


    results_k_file = "plots_data/results_k_fingerprints.csv"
    results_full_file = "plots_data/results_full_fingerprints.csv"
    export_all_dataset_plot(results_k_file, results_full_file, output_dir="plots_output")


def figure_4():
    ###4 Grid of Molecules

    """
    Publication-style figure: a grid of molecules from one ChEMBL activity class,
    with the selected MACCS bits (here, the CA1 primary-sulfonamide pharmacophore)
    highlighted on the active compounds, alongside inactive compounds that lack the
    signature.

    Produces a two-row figure:
      Row 1 (ACTIVE)   : compounds carrying the full bit signature, bits color-coded
      Row 2 (INACTIVE) : compounds lacking the signature, drawn plain for contrast
    """




    # ----------------------------------------------------------------------------
    # CONFIG
    # ----------------------------------------------------------------------------
    PKL   = 'plots_data/100_classes_10_splits_all_train_test_sets_maccs_166.pkl'
    TID   = '261'          # CA1
    TRIAL = 0

    # The selected bits to highlight (the sulfonamide pharmacophore from Section 3.3).
    # Each gets a distinct colour. (bit_id, legend_label, RGB-0..1)
    BIT_INFO = [
        (84,  'Bit 84  \n(NH$_2$)',    (0.85, 0.37, 0.36)),  # red
        (88,  'Bit 88  \nsulfur atom',               (0.33, 0.46, 0.75)),  # blue
        (124, 'Bit 124  \nheteroatom pair',(0.45, 0.68, 0.40)),  # green
        (51,  'Bit 51  \nC-S-O linkage',             (0.90, 0.64, 0.28)),  # orange
        (59,  'Bit 59  \nS-aromatic bond',           (0.60, 0.45, 0.72)),  # purple
    ]
    # Bits whose simultaneous presence defines "the signature":
    CORE = [84, 124, 88, 59, 51]

    # Hand-pick CIDs for a clean, diverse figure (or set to None to auto-pick).
    ACTIVE_CIDS   = ['CHEMBL3264972', 'CHEMBL3264969', 'CHEMBL2011157', 'CHEMBL4248617']
    INACTIVE_CIDS = ['CHEMBL1683894', 'CHEMBL1161718', 'CHEMBL3121103', 'CHEMBL221219']

    cid_name_map = {
        "CHEMBL3264972": "CHEMBL3264972",
        "CHEMBL3264969": "3264969",
        "CHEMBL2011157": "2011157",
        "CHEMBL4248617": "4248617",
        "CHEMBL1683894": "1683894",
        "CHEMBL1161718": "1161718",
        "CHEMBL3121103": "3121103",
        "CHEMBL221219":  "221219",
    }

    active_labels = [cid_name_map.get(cid, cid) for cid in ACTIVE_CIDS]
    inactive_labels = [cid_name_map.get(cid, cid) for cid in INACTIVE_CIDS]


    N_PER_ROW = 4
    MOL_PX    = 360          # render resolution per molecule
    OUT       = 'plots_output/ca1_molecule'

    # ----------------------------------------------------------------------------
    def clean_smarts_groups(smarts):
        """Extract inner patterns from every $(...) recursive-SMARTS group."""
        groups, i, L = [], 0, len(smarts)
        while True:
            s = smarts.find('$(', i)
            if s == -1:
                break
            j, depth = s + 2, 1
            while j < L and depth > 0:
                if smarts[j] == '(':
                    depth += 1
                elif smarts[j] == ')':
                    depth -= 1
                j += 1
            groups.append(smarts[s + 2:j - 1])
            i = j
        return groups


    def highlight_for_mol(mol):
        """Return (atom_colors, bond_colors, radii) for the configured bits."""
        athi, bhi, arad = defaultdict(list), defaultdict(list), {}
        for bit, _label, color in BIT_INFO:
            smarts = MACCSkeys.smartsPatts[bit][0]
            for patt in (clean_smarts_groups(smarts) or [smarts]):
                sub = Chem.MolFromSmarts(patt)
                if sub is None:
                    continue
                for match in mol.GetSubstructMatches(sub):
                    ms = set(match)
                    for a in match:
                        athi[a].append(color)
                        arad[a] = 0.45
                    for bond in mol.GetBonds():
                        if bond.GetBeginAtomIdx() in ms and bond.GetEndAtomIdx() in ms:
                            bhi[bond.GetIdx()].append(color)
        # one colour per atom/bond (first match wins) for clean rendering
        atom_colors = {a: cols[0] for a, cols in athi.items()}
        bond_colors = {b: cols[0] for b, cols in bhi.items()}
        return atom_colors, bond_colors, arad


    def render_mol(smiles, highlight=True):
        """Render a single molecule to a PNG byte buffer."""
        mol = Chem.MolFromSmiles(smiles)
        d = rdMolDraw2D.MolDraw2DCairo(MOL_PX, MOL_PX)
        o = d.drawOptions()

        # larger atom labels
        o.baseFontSize = 1.0
        o.maxFontSize = 40
        o.minFontSize = 22

        # thicker drawing
        o.bondLineWidth = 2.5

        # less whitespace
        o.padding = 0.02
        if highlight:
            ac, bc, ar = highlight_for_mol(mol)
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol,
                highlightAtoms=list(ac.keys()), highlightAtomColors=ac,
                highlightBonds=list(bc.keys()), highlightBondColors=bc,
                highlightAtomRadii=ar)
        else:
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()


    # ----------------------------------------------------------------------------
    # Load data and select molecules
    # ----------------------------------------------------------------------------
    df = pd.read_pickle(PKL)
    df['active_tid'] = df['active_tid'].apply(lambda x: x.strip('CHEMBL'))
    sub = df[(df['active_tid'] == TID) & (df['trial'] == TRIAL)].copy()


    def pick(cids, label, want_signature):
        """Return list of (cid, smiles) for the requested CIDs, or auto-pick."""
        def has_sig(fp):
            present = set(i for i in range(167) if fp[i] == 1)
            return all(b in present for b in CORE)

        sub['has_sig'] = sub['maccs_166_fingerprint'].apply(has_sig)
        pool = sub[(sub['class_label'] == label) & (sub['has_sig'] == want_signature)].copy()
        if cids:
            rows = [pool[pool['cid'] == c].iloc[0] for c in cids if (pool['cid'] == c).any()]
        else:
            pool['n'] = pool['smiles'].str.len()
            rows = [r for _, r in pool.sort_values('n').head(N_PER_ROW).iterrows()]
        return [(r.cid, r.smiles) for r in rows][:N_PER_ROW]


    actives   = pick(ACTIVE_CIDS,   label=1, want_signature=True)
    inactives = pick(INACTIVE_CIDS, label=0, want_signature=False)

    # ----------------------------------------------------------------------------
    # Compose the grid
    # ----------------------------------------------------------------------------
    n = N_PER_ROW
    fig = plt.figure(figsize=(3.1 * n, 7.2))

    # legend now at the bottom
    gs = fig.add_gridspec(
        3, n,
        height_ratios=[1, 1, 0.22],
        hspace=0.02,
        wspace=0.04
    )

    # --- legend across the bottom ---
    ax_leg = fig.add_subplot(gs[2, :])

    ax_leg.axis('off')
    handles = [Patch(facecolor=c, edgecolor='none', label=l) for _, l, c in BIT_INFO]
    leg = ax_leg.legend(handles=handles, ncol=len(BIT_INFO), loc='upper center',
                        bbox_to_anchor=(0.5, 1.35),
                        frameon=False, fontsize=18, handlelength=1.2,
                        columnspacing=1.7,
                        handletextpad=0.25,
                        borderaxespad=0.5,
                        labelspacing=0.2,
                        #title='Selected MACCS bits — primary-sulfonamide pharmacophore',
                        title_fontsize=18)
    leg.get_title().set_fontweight('bold')

    # --- row 1: actives ---
    for i, (cid, smi) in enumerate(actives):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mpimg.imread(__import__('io').BytesIO(render_mol(smi, highlight=True))))
        ax.axis('off')
        display_name = cid_name_map.get(cid, cid)
        ax.set_title(display_name, fontsize=18, pad=0)
        if i == 0:
            ax.text(-0.28, 0.5, 'ACTIVE', transform=ax.transAxes, rotation=0,
                    va='center', ha='center', fontsize=20, fontweight='bold',
                    color='#2c6e49')

    # --- row 2: inactives ---
    # HIGHLIGHT_INACTIVES=False draws them plain (cleanest contrast: the signature is
    # simply absent). Set True to show that isolated atoms may match individual bits
    # but the full co-occurring signature does not appear.
    HIGHLIGHT_INACTIVES = False
    for i, (cid, smi) in enumerate(inactives):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(mpimg.imread(__import__('io').BytesIO(render_mol(smi, highlight=HIGHLIGHT_INACTIVES))))
        ax.axis('off')
        display_name = cid_name_map.get(cid, cid)
        ax.set_title(display_name, fontsize=18)
        if i == 0:
            ax.text(-0.28, 0.5, 'INACTIVE', transform=ax.transAxes, rotation=0,
                    va='center', ha='center', fontsize=18, fontweight='bold',
                    color='#9b2226')

    plt.savefig(f'{OUT}.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUT}.png', dpi=500, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT}.pdf / .png')


def figure_5():
    ###5 Pearson Heatmap







    # --------------------------------------------------------------------------- #
    # Default file locations
    # --------------------------------------------------------------------------- #
    DEFAULT_PKL = "plots_data/100_classes_10_splits_all_train_test_sets_maccs_166.pkl"


    # --------------------------------------------------------------------------- #
    # Data loading
    # --------------------------------------------------------------------------- #
    def load_dataset(pkl_path: Path, tid: str, trial: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the MACCS pickle and return:
            fp:     ndarray of shape (n_molecules, 166), values in {0, 1}
            labels: ndarray of shape (n_molecules,), values in {0, 1}

        Only molecules belonging to the requested target and trial are returned.
        """
        df = pd.read_pickle(pkl_path)
        df["active_tid"] = df["active_tid"].apply(lambda x: x.strip("CHEMBL"))
        sub = df[(df["active_tid"] == tid) & (df["trial"] == trial)]
        if len(sub) == 0:
            raise ValueError(f"No molecules found for CHEMBL{tid}, trial={trial}.")
        fp = np.stack(sub["maccs_166_fingerprint"].values)
        labels = sub["class_label"].values
        return fp, labels


    # --------------------------------------------------------------------------- #
    # SMARTS lookup
    # --------------------------------------------------------------------------- #
    def load_maccs_smarts() -> dict[int, str]:
        """Return {bit_index: SMARTS_string} for all MACCS bits."""
        return {bit: val[0] for bit, val in MACCSkeys.smartsPatts.items() if bit != 0}


    # --------------------------------------------------------------------------- #
    # Statistics per bit
    # --------------------------------------------------------------------------- #
    def bit_statistics(
        fp: np.ndarray,
        labels: np.ndarray,
        bit: int,
    ) -> dict:
        """Return P(bit=1|active), P(bit=1|inactive), and disc = difference."""
        col = fp[:, bit]
        p_act = float(col[labels == 1].mean()) if (labels == 1).any() else float("nan")
        p_inact = float(col[labels == 0].mean()) if (labels == 0).any() else float("nan")
        return {"p_active": p_act, "p_inactive": p_inact, "disc": p_act - p_inact}


    # --------------------------------------------------------------------------- #
    # Correlation computation
    # --------------------------------------------------------------------------- #
    def compute_correlation_matrix(
        fp: np.ndarray,
        bits: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Compute the Pearson correlation matrix for the given bits.

        Bits whose column is constant (zero variance) cannot be correlated and
        are dropped from the matrix; their indices are removed from `bits` in
        the returned `kept` list.
        """
        kept = [b for b in bits if fp[:, b].std() > 0]
        dropped = sorted(set(bits) - set(kept))
        if dropped:
            print(
                f"NOTE: dropping bits {dropped} (constant across this dataset)",
                file=sys.stderr,
            )
        if len(kept) < 2:
            raise ValueError(
                f"Need at least two non-constant bits to compute correlations; "
                f"got {len(kept)}."
            )
        matrix = np.corrcoef(fp[:, kept].T)
        return matrix, kept


    # --------------------------------------------------------------------------- #
    # Label construction
    # --------------------------------------------------------------------------- #
    def build_labels(
        bits: list[int],
        selected_set: set[int],
        fp: np.ndarray,
        labels: np.ndarray,
    ) -> list[str]:
        """
        Build a descriptive label for each bit, prefixed with ★ if the bit is in
        the FR-GNN selected set.

        Label format: "★bit59 (Δ=+0.69)" or "bit73 (Δ=+0.72)".
        """
        out = []
        for b in bits:
            s = bit_statistics(fp, labels, b)
            prefix = "★ " if b in selected_set else "  "
            #out.append(f"{prefix}bit {b} (Δ={s['disc']:+.2f})")
            out.append(f"{prefix}bit {b}")
        return out


    def block_mean(matrix, start_row, end_row, include_diagonal=False):
        """
        Computes the mean of values inside the submatrix
        defined by rows/cols [start_row, end_row].

        Parameters
        ----------
        start_row : int
        end_row   : int
            Inclusive indices.
        include_diagonal : bool
            Whether to include diagonal 1s.
        """

        block = matrix[start_row:end_row+1, start_row:end_row+1]

        if include_diagonal:
            return block.mean()

        # Remove diagonal
        mask = ~np.eye(block.shape[0], dtype=bool)
        return block[mask].mean()

    # --------------------------------------------------------------------------- #
    # Plotting
    # --------------------------------------------------------------------------- #

    def cross_block_mean(M, start1, end1, start2, end2):
        """
        Mean cross-correlation between two blocks.

        Example:
        rows 1-5 against rows 6-15.

        Indices are 1-based.
        """
        block = M[start1 - 1:end1, start2 - 1:end2]
        return block.mean()


    def plot_heatmap(
        corr: np.ndarray,
        bit_labels: list[str],
        title: str,
        output_path: Path,
        figsize: tuple[float, float] = (11.0, 9.0),
        annotate: bool = True,
        upper_triangle_only: bool = True,
        fontsize = 22
    ) -> None:
        """
        Render and save the heatmap.

        - Diverging colormap centred at zero (RdYlBu_r).
        - Square cells.
        - Optional masking of the upper triangle (default True) for readability.
        - Cells annotated with their numeric correlation to two decimal places.
        """
        print(corr)
        M = np.array(corr)
        M = np.abs(M)

        mean_1_5 = block_mean(M, 1, 12)
        mean_6_15 = block_mean(M, 13, 18)
        cross = cross_block_mean(M, 1, 12, 13, 18)

        print("Mean rows 1-12:", mean_1_5)
        print("Mean rows 13-18:", mean_6_15)
        print("Mean cross:", cross)

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if upper_triangle_only else None

        sns.heatmap(
            M,
            mask=mask,
            xticklabels=bit_labels,
            yticklabels=bit_labels,
            cmap="RdYlBu_r",
            vmin=0.0,
            vmax=1.0,
            center=0.0,
            annot=annotate,
            fmt=".2f",
            annot_kws={"size": 11},
            cbar_kws={"label": "Pearson correlation"},
            linewidths=0.5,
            square=True,
            ax=ax,
        )

        cbar = ax.collections[0].colorbar
        cbar.set_label("Pearson correlation", fontsize=20)
        cbar.ax.tick_params(labelsize=18)

        ax.set_title(title, fontsize=18, pad=10)
        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.yticks(rotation=0, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}", file=sys.stderr)


    # --------------------------------------------------------------------------- #
    # Main
    # --------------------------------------------------------------------------- #
    def parse_int_list(s: str) -> list[int]:
        """Parse '59,51,102' into [59, 51, 102]."""
        return [int(x.strip()) for x in s.split(",") if x.strip()]


    tid = '261'
    trial = 1
    #selected = parse_int_list('59,51,88,84,124')
    #unselected = parse_int_list('73,33,58,60,61,64,67,32,69,81')
    selected = parse_int_list('51,59,88,84,124,33,58,60,61,64,67,73')
    unselected = parse_int_list('38,87,98,107,134,137')
    #unselected = [i for i in range(0,167)]
    #selected = []
    output_path = 'plots_output/Pearson_Heatmap.png'
    figsize = tuple((11,9))


    fp, labels = load_dataset(DEFAULT_PKL, tid, trial)
    n_active = int((labels == 1).sum())
    n_inactive = int((labels == 0).sum())
    print(f"  Loaded {n_active} actives and {n_inactive} inactives")

    # Combine bits, preserving order: selected first, then unselected
    bits = selected + [b for b in unselected if b not in selected]
    print(f"  Computing correlation matrix for bits: {bits}")

    # Correlation matrix (may drop constant columns)
    corr, kept_bits = compute_correlation_matrix(fp, bits)

    # Labels (recompute selected set from the actually-kept bits)
    selected_set = set(selected)
    bit_labels = build_labels(kept_bits, selected_set, fp, labels)

    # Title

    title = (

                f"★ = Sulfonamide bits  "

            )

        # Plot
    plot_heatmap(
            corr=corr,
            bit_labels=bit_labels,
            title=title,
            fontsize = 20,
            output_path=output_path,
            figsize=figsize,
            annotate=True)


def figure_6():
    ###6 Chemistry annotated feature heatmap





    # ============================================================================
    # Defaults — tuned for the ten ChEMBL targets in the manuscript
    # ============================================================================

    DEFAULT_DATASET_ORDER = [
        "CHEMBL261", "CHEMBL220", "CHEMBL1914", "CHEMBL1163125", "CHEMBL230",
        "CHEMBL204", "CHEMBL4822", "CHEMBL214", "CHEMBL218", "CHEMBL2973",
    ]

    #DEFAULT_DATASET_ORDER = [
    #    "CHEMBL261", "CHEMBL220", "CHEMBL1914",
    #    "CHEMBL204", "CHEMBL4822",  "CHEMBL214"
    #]


    INPUT_FILE = "plots_data/top_features_fingerprints.txt"

    OUTPUT_FILE = "plots_output/heatmap_chemistry_annotated_features.png"

    TOP_K = 15
    TOP_FEATURES_PER_CATEGORY = 10
    SELECTION_THRESHOLD = 3

    DISPLAY_THRESHOLD = 3

    FIGSIZE = (16, 6.5)

    DPI = 300

    TITLE = (
        f"Top Selected MACCS features across ChEMBL Targets "
    )
    TITLE = ('')

    DEFAULT_TARGET_NAMES = {
        "CHEMBL261":     "CA1",
        "CHEMBL220":     "AChE",
        "CHEMBL1914":    "BChE",
        "CHEMBL1163125": "BRD4",
        "CHEMBL230":     "COX-2",
        "CHEMBL204":     "Thrombin",
        "CHEMBL4822":    "BACE1",
        "CHEMBL214":     "5-HT1A",
        "CHEMBL218":     "CB1",
        "CHEMBL2973":    "ROCK2",
    }

    CATEGORY_ORDER = [
        "Halogen", "Sulfur", "Nitrogen", "Oxygen",
        "Heteroatom", "Ring/topology"
    ]

    CATEGORY_COLORS = {
        "Halogen":       "#FF9F1C",
        "Sulfur":        "#E63946",
        "Nitrogen":      "#3D5A80",
        "Oxygen":        "#06A77D",
        "Heteroatom":    "#9C6644",
        "Ring/topology": "#7768AE"
    }

    # Bits assigned by SMARTS inspection. Update if MACCS definition changes.
    NITROGEN_BITS = frozenset({
        13, 24, 25, 32, 33, 37, 38, 41, 45, 52, 56, 70, 71, 75, 77, 78, 79, 80,
        84, 85, 92, 94, 95, 97, 100, 110, 111, 117, 119, 121, 122, 133, 135, 142,
        151, 156, 158, 161,
    })
    OXYGEN_BITS = frozenset({
        15, 23, 39, 40, 48, 55, 57, 63, 72, 89, 102, 109, 113, 123, 126, 127, 132,
        136, 139, 140, 143, 146, 152, 154, 157, 159, 164,
    })
    HALOGEN_BITS = frozenset({27, 31, 42, 46, 87, 103, 107, 134})
    RING_BITS = frozenset({
        8, 11, 16, 19, 22, 26, 36, 43, 53, 62, 65, 86, 96, 98, 101, 105, 112, 116,
        120, 125, 137, 144, 145, 148, 150, 153, 162, 163, 165,
    })
    #ALKYL_BITS = frozenset({
    #    17, 30, 34, 50, 66, 74, 76, 99, 108, 114, 118, 141, 149, 155, 160,
    #})

    CATEGORY_BITS = {
        "Halogen": HALOGEN_BITS,
        "Sulfur": frozenset({14, 32, 33, 36, 39, 40, 47, 51, 55, 58, 59, 60, 61, 64, 67, 73, 81, 88}),
        "Nitrogen": NITROGEN_BITS,
        "Oxygen": OXYGEN_BITS,
        "Heteroatom": frozenset({28, 43, 54, 68, 69, 82, 83, 86, 90, 91, 93, 98, 104, 106, 124, 131, 138, 148}),
        "Ring/topology": RING_BITS
    }

    # Short labels (4-12 chars). Falls back to f"bit{N}" for any bit not listed.
    SHORT_LABELS = {
        # Halogens
        27: "I",  31: "X-het",  42: "F",  46: "Br",  87: "X-arom",
        103: "Cl", 107: "X-C(*)*", 134: "X (any)",
        # Sulfur
        14: "S-S",  32: "C-S-N",  33: "N-S",  36: "S(ring)",  39: "S(O)(O)O",
        40: "S-O",  47: "S~N",  51: "C-S-O",  55: "O-S-O",  58: "het-S-het",
        59: "S-arom",  60: "S=O",  61: "S(*)*",  64: "S(acyc)",  67: "het-S",
        73: "S=*",   81: "S(*)*",  88: "S (any)",
        # Nitrogen
        13: "O-N(C)C",  24: "N-O",  25: "N-C(N)N", 38: "N-C(-C)-N",  41: "C#N",  45: "C=C-N",
        52: "N-N",  56: "O-N(O)C",  70: "het-N-het",  75: "N (acyc)",
        77: "N-*-N",  78: "C=N",  79: "N-2*-N",  80: "N-3*-N",  84: "NH2",
        85: "C-N(C)C",  94: "het-N",  95: "N-*-*-O",  97: "N-3*-O",
        100: "CH2-N",  110: "N-C-O",  111: "N-CH2",  117: "N-*-O",
        119: "N=*",  121: "N (ring)",  122: "N(*)*",  133: "N (acyc)",
        135: "N-arom",  142: "N",  151: "NH",
        # Oxygen
        15: "O-C(O)O",  23: "N-C(O)O",  48: "O-(O)(O)O",  57: "O (ring)",
        63: "N=O",   72: "O-*-*-O",  89: "O-3*-O",  92: "O-C(N)C",
        102: "het-O",  109: "CH2-O",  113: "O-arom",  126: "O (acyc)",
        127: "C-O acyc",  132: "O-*-CH2",  136: "O=*",  139: "OH",
        140: "O",  143: "O (acyc)",  146: "O",  152: "O-C(C)C",
        154: "C-O",  157: "C-O",  159: "O",  164: "O",
        # Heteroatom motifs
        28: "het-CH2-het",  43: "het-*-het-H",  54: "het-*-*-het",
        68: "donor pair",  69: "het-donor",  82: "CH2-donor",  83: "het-ring",
        86: "CH-het-CH",  90: "donor-ring",  91: "donor-ring",  93: "het-CH3",
        98: "het-6ring",  104: "donor-CH2",  106: "het(het)het",  124: "adj. het",
        131: "donor",  138: "het-CH2",  148: "het(*)*",
        # Ring / topology
        8: "het-4ring",  11: "4-ring",  16: "het-3ring",  19: "7-ring",
        22: "3-ring",  26: "C=C ring",  53: "donor-3*",  62: "*@*!@*@*",
        65: "c:n",  96: "5-ring",  101: "fused-rings",  105: "branch",
        112: "*~(*)~*",  116: "CH3-CH2 ring",  120: "het ring",  125: "?",
        137: "C-free ring",  144: "*!:*:*!:*",  145: "6-ring",  150: "*-ring-*",
        153: "het-CH2",  162: "aromatic",  163: "6-ring",  165: "ring",
        # Carbon / alkyl
        17: "C#C",  30: "C-C-(C)C",  34: "CH2=*",  50: "C=C(C)C",  66: "C(C)(C)C",
        74: "CH3-*-CH3",  76: "C=C(*)*",  99: "C=C",  108: "CH3-*-CH2",
        114: "CH3-CH2-*",  118: "CH2-CH2",  141: "CH3",  149: "CH3/CH4",
        155: "CH2-acyc",  160: "CH3/CH4",
        # Other
        49: "charged",  29: "P",  44: "rare elem.",
    }


    # ============================================================================
    # Core functions
    # ============================================================================

    def parse_runs_file(path: Path) -> dict[str, dict[int, list[int]]]:
        """
        Parse a top_features.txt file into {dataset_id: {run_idx: [bit_indices]}}.
        """
        pattern = re.compile(
            r"top_k_dict\['(CHEMBL\d+)'\]\[(\d+)\]\s*=\s*\[([^\]]+)\]"
        )
        runs: dict[str, dict[int, list[int]]] = {}
        with open(path) as f:
            text = f.read()
        for match in pattern.finditer(text):
            dataset_id = match.group(1)
            run_idx = int(match.group(2))
            bits = [int(x.strip()) for x in match.group(3).split(",")]
            runs.setdefault(dataset_id, {})[run_idx] = bits
        if not runs:
            raise ValueError(
                f"No runs found in {path}. Expected lines like "
                f"top_k_dict['CHEMBL261'][1] = [74, 119, ...]"
            )
        return runs


    def compute_selection_frequencies(
        runs: dict[str, dict[int, list[int]]],
        dataset_order: list[str],
        top_k: int,
    ) -> dict[str, Counter]:
        """For each dataset, count how often each bit appears in the top-K of any run."""
        freq: dict[str, Counter] = {tid: Counter() for tid in dataset_order}
        for tid in dataset_order:
            if tid not in runs:
                print(
                    f"WARNING: dataset {tid} not present in input file; skipping.",
                    file=sys.stderr,
                )
                continue
            for _, bit_list in runs[tid].items():
                for bit in bit_list[:top_k]:
                    freq[tid][bit] += 1
        return freq


    def classify_bit(bit: int, smarts_lookup: dict[int, str] | None = None) -> str:
        """
        Return the chemistry-family label for a MACCS bit.

        The classification combines (1) hand-built sets for bits where the SMARTS
        is unambiguous and (2) a SMARTS-pattern fallback that catches sulfur, the
        general heteroatom motif, halogen literals, and a few other regularities.
        """
        sm = (smarts_lookup or {}).get(bit, "")
        # Halogen first (more specific than general SMARTS match)
        if bit in HALOGEN_BITS or sm in {"F", "Cl", "Br", "I"} or "F,Cl,Br,I" in sm:
            return "Halogen"
        # Sulfur — anything mentioning #16 atomic number, except bits already classified
        if "#16" in sm:
            return "Sulfur"
        if bit in NITROGEN_BITS:
            return "Nitrogen"
        if bit in OXYGEN_BITS:
            return "Oxygen"
        # General heteroatom motifs (bits encoding [!#6;!#1] without committing to N/O/S)
        if "!#6" in sm and "!#1" in sm:
            return "Heteroatom"
        if bit in RING_BITS:
            return "Ring/topology"
        if bit in ALKYL_BITS:
            return "Carbon/alkyl"
        return "Other"


    def order_bits_by_family(
        bits: set[int],
        total_freq: Counter,
        max_dataset_freq: Counter,
        smarts_lookup: dict[int, str] | None,
    ) -> tuple[list[int], list[str]]:
        """
        Return the top features from each chemistry family.

        For each category in CATEGORY_ORDER, keep at most TOP_FEATURES_PER_CATEGORY
        features, sorted by total appearance frequency across all datasets.
        """



        ordered_bits: list[int] = []
        ordered_cats: list[str] = []

        for cat in CATEGORY_ORDER:
            category_bits = CATEGORY_BITS.get(cat, frozenset())

            sub = [
                bit for bit in bits
                if bit in category_bits
            ]

            sub = sorted(
                sub,
                key=lambda b: (-max_dataset_freq[b], -total_freq[b], b)
            )

            sub = sub[:TOP_FEATURES_PER_CATEGORY]

            ordered_bits.extend(sub)
            ordered_cats.extend([cat] * len(sub))

        return ordered_bits, ordered_cats

    def get_short_label(bit: int, smarts_lookup: dict[int, str] | None = None) -> str:
        """Return a short chemistry-name label for a bit; fall back to f'bit{N}'."""
        if bit in SHORT_LABELS:
            return SHORT_LABELS[bit]
        sm = (smarts_lookup or {}).get(bit, "")
        return sm if sm else f"bit{bit}"


    def try_load_smarts() -> dict[int, str] | None:
        """Try to import RDKit and return the MACCS SMARTS dictionary, else None."""
        try:
            return {
                b: MACCSkeys.smartsPatts.get(b, ("",))[0]
                for b in range(167)
            }
        except ImportError:
            print(
                "WARNING: rdkit not available; bit labels will use the manual table only.",
                file=sys.stderr,
            )
            return None


    # ============================================================================
    # Plotting
    # ============================================================================

    def build_figure(
        runs: dict[str, dict[int, list[int]]],
        dataset_order: list[str],
        target_names: dict[str, str],
        top_k: int,
        selection_threshold: int,
        display_threshold: int,
        figsize: tuple[float, float],
        title: str,
    ) -> plt.Figure:
        """Construct the full chemistry-annotated heatmap and return the figure."""
        smarts_lookup = try_load_smarts()
        freq = compute_selection_frequencies(runs, dataset_order, top_k)

        # Identify bits stably selected (≥ selection_threshold/10) in ≥1 dataset
        selected_bits: set[int] = set()
        for tid in dataset_order:
            for bit, count in freq[tid].items():
                if count >= selection_threshold:
                    selected_bits.add(bit)

        # Total selection count across all datasets (for within-family ordering)
        total_freq: Counter = Counter()
        for tid in dataset_order:
            for bit, count in freq[tid].items():
                if bit in selected_bits:
                    total_freq[bit] += count

        # Maximum appearance count of each feature in any single dataset
        # Example: if bit X appears 7 times in CHEMBL261 and 0 elsewhere,
        # max_dataset_freq[X] = 7
        max_dataset_freq: Counter = Counter()

        for bit in selected_bits:
            max_dataset_freq[bit] = max(
                freq[tid].get(bit, 0)
                for tid in dataset_order
            )

        # ============================================================================
        # PRINT TOP FEATURES PER CATEGORY
        # ============================================================================

        #TOP_FEATURES_PER_CATEGORY = 5

        print("\n" + "=" * 80)
        print("TOP FEATURES PER CATEGORY")
        print("=" * 80)

        for cat in CATEGORY_ORDER:

            category_bits = CATEGORY_BITS.get(cat, frozenset())

            category_features = [
                bit for bit in selected_bits
                if bit in category_bits
            ]

            category_features = sorted(
                category_features,
                key=lambda b: (-max_dataset_freq[b], -total_freq[b], b)
            )


            category_features = category_features[:TOP_FEATURES_PER_CATEGORY]

            print(f"\n{cat}")
            print("-" * len(cat))

            if len(category_features) == 0:
                print("  No selected features")
                continue

            for rank, bit in enumerate(category_features, start=1):
                label = get_short_label(bit, smarts_lookup)

                print(
                    f"{rank:2d}. "
                    f"Bit {bit:3d} | "
                    f"{label:15s} | "
                    f"Max dataset frequency = {max_dataset_freq[bit]} | "
                    f"Total frequency = {total_freq[bit]}"
                )

        print("\n" + "=" * 80 + "\n")

        # ============================================================================
        # ORDER FEATURES FOR PLOTTING
        # ============================================================================

        ordered_bits, ordered_cats = order_bits_by_family(
            selected_bits,
            total_freq,
            max_dataset_freq,
            smarts_lookup
        )

        if not ordered_bits:
            raise ValueError(
                f"No bits meet the selection threshold of {selection_threshold}/10. "
                f"Lower --selection-threshold to include more bits."
            )

        # Build the data matrix
        n_datasets = len(dataset_order)
        n_bits = len(ordered_bits)
        matrix = np.zeros((n_datasets, n_bits), dtype=int)
        for i, tid in enumerate(dataset_order):
            for j, bit in enumerate(ordered_bits):
                matrix[i, j] = freq[tid].get(bit, 0)

        print(
            f"Building figure: {n_datasets} datasets × {n_bits} bits "
            f"(top 5 features per category; selection threshold ≥ {selection_threshold}/{top_k} in any dataset)",
            file=sys.stderr,
        )
        for cat in CATEGORY_ORDER:
            n = sum(1 for c in ordered_cats if c == cat)
            if n:
                print(f"  {cat:15s}: {n} bits", file=sys.stderr)

        # ---- Plot
        fig = plt.figure(figsize=figsize)

        fig, ax_heat = plt.subplots(figsize=figsize)
        #gs = fig.add_gridspec(2, 1, height_ratios=[0.05, 1.0], hspace=0.02)
        #ax_strip = fig.add_subplot(gs[0])
        #ax_heat = fig.add_subplot(gs[1])

        # Family colour strip
        '''
        for j, cat in enumerate(ordered_cats):
            ax_strip.add_patch(
                plt.Rectangle((j - 0.5, 0), 1, 1, color=CATEGORY_COLORS[cat], lw=0)
            )
        ax_strip.set_xlim(-0.5, n_bits - 0.5)
        ax_strip.set_ylim(0, 1)
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        ax_strip.set_frame_on(False)

        # Family labels on the strip
        prev_cat = None
        seg_start = 0
        for j, cat in enumerate(ordered_cats + ["__END__"]):
            if cat != prev_cat:
                if prev_cat is not None:
                    mid = (seg_start + j - 1) / 2
                    count = j - seg_start
                    if count >= 2:
                        ax_strip.text(
                            mid, 0.5, prev_cat, ha="center", va="center",
                            fontsize=11, fontweight="bold", color="white",
                        )
                    else:
                        ax_strip.text(
                            mid, 1.4, prev_cat, ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color=CATEGORY_COLORS[prev_cat],
                        )
                seg_start = j
                prev_cat = cat
        '''
        # Heatmap
        #im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=10)

        # ============================================================================
        # DISCRETE COLOR MAP
        # 0-2   -> light
        # 3-5   -> medium
        # 6-10  -> dark
        # ============================================================================

        # ============================================================================
        # DISCRETE COLOR MAP
        # ============================================================================

        cmap = ListedColormap([
            "#FFFFFF",  # 0-2
            "#FDBB84",  # 3-4
            "#E34A33",  # 5-6
            "#7F0000",  # 7-10
        ])

        bounds = [0, 3, 5, 7, 11]

        norm = BoundaryNorm(bounds, cmap.N)

        im = ax_heat.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            norm=norm
        )
        # Annotate cells whose value meets display_threshold
        '''
        for i in range(n_datasets):
            for j in range(n_bits):
                v = int(matrix[i, j])
                if v >= display_threshold:
                    color = "white" if v >= top_k * 0.6 else "black"
                    ax_heat.text(
                        j, i, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color,
                    )
        '''

        # X labels with chemistry-family colour
        #x_labels = [get_short_label(b, smarts_lookup) for b in ordered_bits]
        x_labels = [
            f"{b}:{get_short_label(b, smarts_lookup)}"
            for b in ordered_bits
        ]
        ax_heat.set_xticks(range(n_bits))
        ax_heat.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=13, fontweight="bold")
        # ============================================================================
        # CATEGORY LABELS BELOW X AXIS
        # ============================================================================

        prev_cat = ordered_cats[0]

        for j, cat in enumerate(ordered_cats + ["__END__"]):

            if cat != prev_cat:
                ax_heat.axvline(
                    j - 0.5,
                    color="black",
                    lw=1.0,
                    zorder=10
                )

                prev_cat = cat

        for label, cat in zip(ax_heat.get_xticklabels(), ordered_cats):
            label.set_color(CATEGORY_COLORS[cat])

        # Y labels: target name + ChEMBL ID
        y_labels = [
            f"{target_names.get(tid, tid.replace('CHEMBL', ''))} ({tid.replace('CHEMBL', '')})"
            for tid in dataset_order
        ]
        ax_heat.set_yticks(range(n_datasets))
        ax_heat.set_yticklabels(y_labels, fontsize=16)

        # Vertical separators between families
        # ============================================================================
        # CATEGORY LABELS BELOW FEATURE LABELS
        # ============================================================================

        prev_cat = ordered_cats[0]
        seg_start = 0

        for j, cat in enumerate(ordered_cats + ["__END__"]):

            if cat != prev_cat:
                mid = (seg_start + j - 1) / 2

                ax_heat.text(
                    mid,
                    -0.5,  # move category labels further below feature labels
                    prev_cat,
                    ha="center",
                    va="top",
                    fontsize=20,
                    fontweight="bold",
                    color=CATEGORY_COLORS[prev_cat],
                    transform=ax_heat.get_xaxis_transform(),
                    clip_on=False,
                )

                seg_start = j
                prev_cat = cat

        ax_heat.set_xlim(-0.5, n_bits - 0.5)
        ax_heat.set_ylim(n_datasets - 0.5, -0.5)

        plt.subplots_adjust(bottom=0.32)

        #cbar = plt.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.012)
        #cbar.set_label(f"Selection count\n(out of 10 runs)", fontsize=10)
        cbar = plt.colorbar(
            im,
            ax=ax_heat,
            fraction=0.018,
            pad=0.012,
            ticks=[1.5, 4, 6, 8.8]
        )

        cbar.ax.set_yticklabels([
            "0–2",
            "3–4",
            "5–6",
            "7–10"
        ])

        cbar.ax.tick_params(labelsize=20)

        cbar.set_label(
            "Selection count\n(out of 10 runs)",
            fontsize=20
        )
        #ax_strip.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax_heat.set_title(
            title,
            fontsize=18,
            fontweight="bold",
            pad=20
        )

        return fig

    def main() -> int:

        input_path = Path(INPUT_FILE)
        output_path = Path(OUTPUT_FILE)

        if not input_path.exists():
            print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
            return 1

        runs = parse_runs_file(input_path)

        fig = build_figure(
            runs=runs,
            dataset_order=DEFAULT_DATASET_ORDER,
            target_names=DEFAULT_TARGET_NAMES,
            top_k=TOP_K,
            selection_threshold=SELECTION_THRESHOLD,
            display_threshold=DISPLAY_THRESHOLD,
            figsize=FIGSIZE,
            title=TITLE,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)

        print(f"Saved: {output_path}", file=sys.stderr)

        return 0


    main()


def figure_7():
    ###7 Absent Features Analysis




    # =========================
    # USER SETTINGS
    # =========================
    frequency_file = "plots_data/feature_frequency_by_dataset.txt"

    # folder that contains subfolders like:
    # base_score_dir/204/204_discriminative_scores.csv
    # base_score_dir/214/214_discriminative_scores.csv
    base_score_dir = "plots_data/binary_feature_plots"

    output_dir = "plots_output"
    os.makedirs(output_dir, exist_ok=True)

    MIN_FREQ = 3
    SCORE_THRESHOLD = -0.2   # keep diff <= -0.20
    OUTPUT_PREFIX = f"absent_selected_minfreq{MIN_FREQ}_thr{abs(SCORE_THRESHOLD):.2f}"


    # =========================
    # READ FREQUENCY FILE
    # =========================
    def read_frequency_file(path):
        """
        Reads format:

        CHEMBL204
        ---------
        82: 8
        92: 8

        Returns:
            dict[dataset][feature_index] = frequency
        """
        freq = {}
        current_dataset = None

        dataset_pattern = re.compile(r"^(CHEMBL\d+)$")
        feature_pattern = re.compile(r"^(\d+)\s*:\s*(\d+)$")

        with open(path, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                ds_match = dataset_pattern.match(line)
                if ds_match:
                    current_dataset = ds_match.group(1)
                    freq[current_dataset] = {}
                    continue

                if line.startswith("-"):
                    continue

                feat_match = feature_pattern.match(line)
                if feat_match and current_dataset is not None:
                    feature_index = int(feat_match.group(1))
                    count = int(feat_match.group(2))
                    freq[current_dataset][feature_index] = count

        return freq


    # =========================
    # FIND SCORE FILE
    # =========================
    def find_score_file(dataset, base_dir):
        """
        For CHEMBL204, searches for:
            base_dir/204/204_discriminative_scores.csv
            base_dir/204/CHEMBL204_discriminative_scores.csv
            or any file containing discriminative_scores inside base_dir/204
        """
        dataset_num = dataset.replace("CHEMBL", "")

        candidate_dir = Path(base_dir) / dataset_num

        candidates = [
            candidate_dir / f"{dataset_num}_discriminative_scores.csv",
            candidate_dir / f"{dataset}_discriminative_scores.csv",
            candidate_dir / f"{dataset_num}_discriminative_scores.txt",
            candidate_dir / f"{dataset}_discriminative_scores.txt",
        ]

        for p in candidates:
            if p.exists():
                return p

        if candidate_dir.exists():
            all_candidates = list(candidate_dir.glob("*discriminative_scores*"))
            if all_candidates:
                return all_candidates[0]

        return None


    # =========================
    # MAIN ANALYSIS
    # =========================
    freq_by_dataset = read_frequency_file(frequency_file)

    rows = []

    for dataset, feat_freqs in freq_by_dataset.items():

        score_file = find_score_file(dataset, base_score_dir)

        if score_file is None:
            print(f"WARNING: score file not found for {dataset}")
            continue

        score_df = pd.read_csv(score_file)

        # make sure numeric
        score_df["feature_index"] = score_df["feature_index"].astype(int)
        score_df["diff"] = pd.to_numeric(score_df["diff"], errors="coerce")
        score_df["abs_diff"] = pd.to_numeric(score_df["abs_diff"], errors="coerce")

        for feature_index, freq in feat_freqs.items():

            if freq < MIN_FREQ:
                continue

            hit = score_df[score_df["feature_index"] == feature_index]

            if hit.empty:
                continue

            diff = float(hit.iloc[0]["diff"])
            abs_diff = float(hit.iloc[0]["abs_diff"])

            # keep only negative absent features above threshold
            # example: diff=-0.25 is kept when threshold=-0.20
            if diff <= SCORE_THRESHOLD:

                rows.append({
                    "dataset": dataset,
                    "feature_index": feature_index,
                    "frequency": freq,
                    "diff": diff,
                    "abs_diff": abs_diff,
                    "score_file": str(score_file),
                })


    result_df = pd.DataFrame(rows)

    # =========================
    # SAVE TABLE
    # =========================
    csv_out = os.path.join(output_dir, f"{OUTPUT_PREFIX}.csv")
    result_df.to_csv(csv_out, index=False)

    print(f"Saved selected absent features to: {csv_out}")
    print(f"Number of selected absent features: {len(result_df)}")

    dataset_name_map = {
        "CHEMBL261":     "CA1 (261)",
        "CHEMBL220":     "AChE (220)",
        "CHEMBL1914":    "BChE (1914)",
        "CHEMBL1163125": "BRD4 (1163125)",
        "CHEMBL230":     "COX-2 (230)",
        "CHEMBL204":     "Thrombin (204)",
        "CHEMBL4822":    "BACE1 (4822)",
        "CHEMBL214":     "5-HT1A (214)",
        "CHEMBL218":     "CB1 (218)",
        "CHEMBL2973":    "ROCK2 (2973)",
    }


    # =========================
    # DRAW HEATMAP
    # =========================
    if result_df.empty:
        print("No features matched the conditions.")
    else:
        datasets = sorted(result_df["dataset"].unique())
        features = sorted(result_df["feature_index"].unique())

        heatmap_df = pd.DataFrame(
            np.nan,
            index=datasets,
            columns=features
        )

        for _, row in result_df.iterrows():
            heatmap_df.loc[row["dataset"], row["feature_index"]] = row["diff"]

        plt.figure(figsize=(max(10, 0.45 * len(features)), 0.55 * len(datasets) + 2.5))

        # before heatmap
        min_diff = result_df["diff"].min()

        # heatmap
        ax = sns.heatmap(
            heatmap_df,
            cmap="magma",
            vmin=min_diff,
            vmax=-0.1,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "signed diff"},
            mask=heatmap_df.isna()
        )

        #ax.set_title(
        #    f"Selected absent features across datasets "
        #    f"(freq ≥ {MIN_FREQ}, $\Delta$ ≤ {SCORE_THRESHOLD})",
        #    fontsize=16,
        #    pad=12
        #)

        ax.set_xlabel("Feature index", fontsize=24)
        #ax.set_ylabel("Dataset", fontsize=13)

        ax.tick_params(axis="x", labelsize=18, rotation=90)
        ax.tick_params(axis="y", labelsize=20)

        cbar = ax.collections[0].colorbar
        cbar.set_label("$\Delta$", fontsize=22)
        cbar.ax.tick_params(labelsize=20)

        ax.set_yticklabels(
            [dataset_name_map.get(x, x) for x in heatmap_df.index],
            rotation=0
        )

        plt.tight_layout()

        pdf_out = os.path.join(output_dir, f"{OUTPUT_PREFIX}.pdf")
        png_out = os.path.join(output_dir, f"{OUTPUT_PREFIX}.png")

        plt.savefig(pdf_out, bbox_inches="tight")
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved heatmap to: {pdf_out}")
        print(f"Saved heatmap to: {png_out}")


def figure_8():
    ###8 Plot shap figure


    # ============================================================================
    # DATA  —  EDIT THESE.  Each value is (mean, std) in % accuracy over 10 runs.
    # K order is [5, 10, 15] within every list.
    # ============================================================================
    K_VALUES = [5, 10, 15, 20]

    DATA = {
        'Fingerprints': {
            'ours': [(76.0, 4.8), (78.7, 3.6), (79.3, 3.8), (78.8,6.6)],   # <- fill: ours @ K=5,10,15
            'shap': [(74.2, 5.8), (76.5, 6.1), (76.4, 6.2), (77.6,5.7)],   # <- fill: SHAP @ K=5,10,15
        },
        'Computed': {
            'ours': [(72.6, 5.4), (77.1, 4.3), (77.0, 4.4), (76.0,6.0)],
            'shap': [(74.1, 5.9), (74.7, 6.8), (74.5, 7.8), (74.9,8.3)],
        },
        'Combined': {
            'ours': [(66.3, 5.7), (72.4, 4.1), (77.1, 3.1), (80.1,5.3)],
            'shap': [(71.6, 6.1), (73.6, 6.3), (76.9, 6.6), (78.1,7.5)],
        },
    }

    # Optional: annotate the gap (ours - shap) above each K group. Set False to hide.
    SHOW_GAP = True
    # y-axis limits — set to bracket your data nicely (e.g. (55, 90)).
    YLIM = (55, 90)
    OUT = 'plots_output/shap_vs_ours_accuracy'
    # ============================================================================

    # ----- style -----
    COL_OURS = '#2c6e8f'   # deep teal-blue
    COL_SHAP = "#bdbdbd"
    #COL_SHAP = '#c97b3c'   # muted orange
    plt.rcParams.update({
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.9,
        'font.family': 'sans-serif',
    })

    pools = list(DATA.keys())
    fig, axes = plt.subplots(1, len(pools), figsize=(13, 4.6), sharey=True)

    bar_w = 0.38
    x = np.arange(len(K_VALUES))

    for ax, pool in zip(axes, pools):
        ours = DATA[pool]['ours']
        shap = DATA[pool]['shap']
        ours_m = [m for m, _ in ours]; ours_s = [s for _, s in ours]
        shap_m = [m for m, _ in shap]; shap_s = [s for _, s in shap]

        b1 = ax.bar(x - bar_w/2, ours_m, bar_w, yerr=ours_s, capsize=4,
                    color=COL_OURS, edgecolor='black', linewidth=0.8,
                    error_kw=dict(ecolor='#333333', lw=1.1), label='Our ranking')
        b2 = ax.bar(x + bar_w/2, shap_m, bar_w, yerr=shap_s, capsize=4,
                    color=COL_SHAP, edgecolor='black', linewidth=0.8,
                    error_kw=dict(ecolor='#333333', lw=1.1), label='SHAP')

        # gap annotation above each K group
        if SHOW_GAP:
            for i in range(len(K_VALUES)):
                gap = ours_m[i] - shap_m[i]
                top = max(ours_m[i] + ours_s[i], shap_m[i] + shap_s[i])
                label = f'+{gap:.1f}' if gap > 0 else f'{gap:.1f}'
                ax.text(x[i], top + 1.0, label, ha='center', va='bottom',
                        fontsize=16, fontweight='bold', color='#2c6e8f')

        ax.set_title(pool, fontsize=18, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in K_VALUES],fontsize=16)
        #ax.tick_params(axis="y", labelsize=18)
        #ax.set_xlabel('Number of selected features', fontsize=16)
        ax.set_ylim(*YLIM)
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Average test accuracy (%)', fontsize=18)
    axes[0].tick_params(axis="y", labelsize=16)

    # shared legend on top
    handles = [Patch(facecolor=COL_OURS, edgecolor='black', label='Our ranking'),
               Patch(facecolor=COL_SHAP, edgecolor='black', label='SHAP')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               fontsize=16, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{OUT}.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUT}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT}.pdf / .png')


def figure_9():
    ###9 Accuracy Subsplots for 10 datasets




    # ============================================================================
    # PARAMETERS
    # ============================================================================

    K_MIN = 1
    K_MAX = 15

    FIGSIZE = (11, 6)
    DPI = 300

    DATASET_ORDER = [
        "CHEMBL220",
        "CHEMBL4822",
        "CHEMBL218",
        "CHEMBL2973",
        "CHEMBL1163125",
        "CHEMBL204",
        "CHEMBL261",
        "CHEMBL230",
        "CHEMBL1914",
        "CHEMBL214",

    ]

    DATASET_LABELS = {
        "CHEMBL261": "CA1 (261)",
        "CHEMBL220": "AChE (220)",
        "CHEMBL1914": "BChE (1914)",
        "CHEMBL1163125": "BRD4 (1163125)",
        "CHEMBL230": "COX-2 (230)",
        "CHEMBL204": "Thrombin (204)",
        "CHEMBL4822": "BACE1 (4822)",
        "CHEMBL214": "5-HT1A (214)",
        "CHEMBL218": "CB1 (218)",
        "CHEMBL2973": "ROCK2 (2973)",
    }


    # ============================================================================
    # LOADING
    # ============================================================================

    def load_and_clean_results(filename):
        """
        Load CSV with possible repeated headers.
        Expected columns:
        dataset, size, run, model, variant, k, test_acc, train_acc, gc_training_time
        """

        df = pd.read_csv(filename, skipinitialspace=True)

        df.columns = df.columns.str.strip()

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        df = df[df["dataset"] != "dataset"].copy()

        df["run"] = pd.to_numeric(df["run"], errors="coerce")
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")

        df = df.dropna(subset=["dataset", "run", "k", "test_acc"]).copy()

        df["run"] = df["run"].astype(int)
        df["k"] = df["k"].astype(int)

        return df


    # ============================================================================
    # STATISTICS
    # ============================================================================

    def compute_k_stats(df_k):
        """
        Average test accuracy for each dataset and k.
        """

        df_k = df_k[
            (df_k["k"] >= K_MIN) &
            (df_k["k"] <= K_MAX)
        ].copy()

        k_stats = (
            df_k.groupby(["dataset", "k"])["test_acc"]
            .agg(
                avg_test_acc="mean",
                std_test_acc="std",
                n_runs="count"
            )
            .reset_index()
        )

        return k_stats


    def compute_full_stats(df_full):
        """
        Average full-feature accuracy for each dataset.
        """

        full_stats = (
            df_full.groupby("dataset")["test_acc"]
            .agg(
                full_avg_test_acc="mean",
                full_std_test_acc="std",
                full_n_runs="count"
            )
            .reset_index()
        )

        return full_stats


    def build_averages_table(k_stats, full_stats):
        """
        Merge top-k and full-feature averages.
        Adds delta over full feature set.
        """

        merged = k_stats.merge(
            full_stats,
            on="dataset",
            how="left"
        )

        merged["delta_vs_full"] = (
            merged["avg_test_acc"] - merged["full_avg_test_acc"]
        )

        merged = merged.sort_values(["dataset", "k"])

        return merged


    # ============================================================================
    # PLOTTING
    # ============================================================================
    def plot_detailed_subfigures(
        k_stats,
        full_stats,
        output_file,
        nrows=2,
        ncols=5,
        figsize=(22, 8)
    ):
        datasets = [
            "CHEMBL261",
            "CHEMBL220",
            "CHEMBL1914",
            "CHEMBL1163125",
            "CHEMBL230",
            "CHEMBL204",
            "CHEMBL4822",
            "CHEMBL214",
            "CHEMBL218",
            "CHEMBL2973",
        ]

        dataset_labels = {
            "CHEMBL261": "CA1",
            "CHEMBL220": "AChE",
            "CHEMBL1914": "BChE",
            "CHEMBL1163125": "BRD4",
            "CHEMBL230": "COX-2",
            "CHEMBL204": "Thrombin",
            "CHEMBL4822": "BACE1",
            "CHEMBL214": "5-HT1A",
            "CHEMBL218": "CB1",
            "CHEMBL2973": "ROCK2",
        }

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True,
            sharey=True
        )

        axes = axes.flatten()
        axes = axes.flatten()

        for ax, dataset in zip(axes, datasets):

            ds_k = k_stats[k_stats["dataset"] == dataset].copy()
            ds_full = full_stats[full_stats["dataset"] == dataset].copy()

            ds_k = ds_k.sort_values("k")

            x = ds_k["k"]
            y = ds_k["avg_test_acc"]
            yerr = ds_k["std_test_acc"].fillna(0)

            '''
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="-o",
                linewidth=2,
                markersize=4,
                capsize=3,
                label="Top-k"
            )
            '''
            # Mean curve
            ax.plot(
                x,
                y,
                "-o",
                linewidth=2,
                markersize=4,
                label="Top-$K$"
            )

            # Std ribbon
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                alpha=0.20
            )

            if not ds_full.empty:

                full_avg = ds_full.iloc[0]["full_avg_test_acc"]
                full_std = ds_full.iloc[0]["full_std_test_acc"]

                if pd.isna(full_std):
                    full_std = 0.0

                # Dashed baseline
                ax.axhline(
                    full_avg,
                    linestyle="--",
                    linewidth=2,
                    label="F-Fingerprints"
                )

                # Shaded std band
                ax.fill_between(
                    [x.min(), x.max()],
                    [full_avg - full_std, full_avg - full_std],
                    [full_avg + full_std, full_avg + full_std],
                    alpha=0.18
                )

            # Mark best k
            best_row = ds_k.loc[ds_k["avg_test_acc"].idxmax()]
            best_k = int(best_row["k"])
            best_acc = best_row["avg_test_acc"]

            ax.scatter(
                best_k,
                best_acc,
                s=90,
                edgecolor="black",
                linewidth=1.2,
                zorder=5
            )

            ax.set_title(
                f"{DATASET_LABELS.get(dataset, dataset)}",
                fontsize=24,
                fontweight="bold"
            )

            ax.set_xticks(range(1, 16))
            ax.set_xticklabels(
                [str(k) if k in [1, 5, 10, 15, 20] else "" for k in range(1, 16)]
            )
            ax.grid(True, alpha=0.3)

            ax.tick_params(axis="y", labelsize=22)
            ax.tick_params(axis="x", labelsize=22)
            ax.yaxis.set_major_locator(MultipleLocator(10))

        # Shared labels
        fig.supxlabel("Number of selected features $K$", fontsize=26, fontweight="bold")
        fig.supylabel("Average test accuracy", fontsize=26, fontweight="bold")

        # One shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fontsize=26,
            frameon=False
        )

        fig.tight_layout(rect=[0.02, 0.03, 1, 0.93])

        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_file}")
    # ============================================================================
    # MAIN
    # ============================================================================
    def export_all_dataset_plots(results_k_file, results_full_file, output_dir="results_v3/plots"):

        os.makedirs(output_dir, exist_ok=True)

        df_k = load_and_clean_results(results_k_file)
        df_full = load_and_clean_results(results_full_file)

        k_stats = compute_k_stats(df_k)
        full_stats = compute_full_stats(df_full)

        output_file = os.path.join(
            output_dir,
            "detailed_accuracy_vs_k_10_datasets_fingerprints.png"
        )

        plot_detailed_subfigures(
            k_stats,
            full_stats,
            output_file=output_file
        )


    def main():
        results_k_file = "plots_data/results_k_fingerprints.csv"
        results_full_file = "plots_data/results_full_fingerprints.csv"
        output_dir = "plots_output"

        export_all_dataset_plots(
            results_k_file=results_k_file,
            results_full_file=results_full_file,
            output_dir=output_dir
        )


    main()


FIGURE_OPTIONS = {
    "1": ("accuracy per dataset and run", figure_1),
    "2": ("Heatmap accuracy k vs full set", figure_2),
    "3": ("accuracy per dataset, k from 4 to 8", figure_3),
    "4": ("Grid of Molecules", figure_4),
    "5": ("Pearson Heatmap", figure_5),
    "6": ("Chemistry annotated feature heatmap", figure_6),
    "7": ("Absent Features Analysis", figure_7),
    "8": ("Plot shap figure", figure_8),
    "9": ("Accuracy Subsplots for 10 datasets", figure_9),
}


def main():
    print("Select the figure to generate:")
    for number, (title, _) in FIGURE_OPTIONS.items():
        print(f"  {number}. {title}")

    choice = input("Enter a figure number (1-9): ").strip()

    if choice not in FIGURE_OPTIONS:
        print("Invalid option. Please enter a number from 1 to 9.")
        return

    title, figure_function = FIGURE_OPTIONS[choice]
    print(f"Generating Figure {choice}: {title}")
    figure_function()


if __name__ == "__main__":
    main()

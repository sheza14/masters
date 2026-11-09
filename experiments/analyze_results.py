
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# 1. SETTINGS
# ==========================================================

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = {
    "PSO": RESULTS_DIR / "pso_results.csv",
    "Firefly": RESULTS_DIR / "firefly_results.csv",
    "ABC": RESULTS_DIR / "abc_results.csv"
}

FEATURE_FILE = RESULTS_DIR / "feature_values.csv"

ALGORITHMS = ["PSO", "Firefly", "ABC"]
BUDGETS = [50, 500, 5000, 20000]
DATASETS = ["Karate Club", "Netscience"]

COLORS = {
    "PSO": "#2878B5",
    "Firefly": "#E58B32",
    "ABC": "#399B68"
}

MARKERS = {
    "PSO": "o",
    "Firefly": "s",
    "ABC": "^"
}

# Best individual classical heuristic from your current results
BASELINES = {
    "Karate Club": {
        "name": "Adamic-Adar",
        "auc": 0.6720
    },
    "Netscience": {
        "name": "Resource Allocation",
        "auc": 0.8538
    }
}

WEIGHT_DATASET = "Netscience"
WEIGHT_BUDGET = 20000


# ==========================================================
# 2. LOAD EXPERIMENT RESULTS
# ==========================================================

frames = []

for algorithm, file_path in CSV_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Check the CSV filename in the settings section."
        )

    data = pd.read_csv(file_path)
    data["algorithm"] = algorithm
    frames.append(data)

results = pd.concat(frames, ignore_index=True)

required_columns = [
    "dataset",
    "budget",
    "seed",
    "best_validation_auc",
    "final_test_auc",
    "evaluations",
    "weight_cn",
    "weight_jaccard",
    "weight_aa",
    "weight_ra"
]

missing_columns = [
    column for column in required_columns
    if column not in results.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in your CSV files: {missing_columns}"
    )

# Check that budget and AUC columns are numeric
numeric_columns = [
    "budget",
    "seed",
    "best_validation_auc",
    "final_test_auc",
    "evaluations",
    "weight_cn",
    "weight_jaccard",
    "weight_aa",
    "weight_ra"
]

for column in numeric_columns:
    results[column] = pd.to_numeric(
        results[column],
        errors="raise"
    )

print("\nTotal experiment runs:", len(results))

print("\nRuns per algorithm:")
print(results.groupby("algorithm").size())


# ==========================================================
# 3. CALCULATE SUMMARY STATISTICS
# ==========================================================

summary = (
    results
    .groupby(["dataset", "algorithm", "budget"])
    .agg(
        mean_test_auc=("final_test_auc", "mean"),
        std_test_auc=("final_test_auc", "std"),
        mean_validation_auc=("best_validation_auc", "mean"),
        std_validation_auc=("best_validation_auc", "std"),
        mean_evaluations=("evaluations", "mean"),
        number_of_runs=("seed", "nunique")
    )
    .reset_index()
)

summary["std_test_auc"] = summary["std_test_auc"].fillna(0)
summary["std_validation_auc"] = (
    summary["std_validation_auc"].fillna(0)
)

summary.to_csv(
    RESULTS_DIR / "final_comparison_summary.csv",
    index=False
)

print("\nEXPERIMENT SUMMARY")
print(summary.to_string(index=False))


# ==========================================================
# 4. FIGURE 1: BUDGET CURVES
# ==========================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)

for ax, dataset in zip(axes, DATASETS):

    dataset_data = summary[
        summary["dataset"] == dataset
    ]

    for algorithm in ALGORITHMS:

        algorithm_data = (
            dataset_data[
                dataset_data["algorithm"] == algorithm
            ]
            .set_index("budget")
            .reindex(BUDGETS)
        )

        ax.errorbar(
            BUDGETS,
            algorithm_data["mean_test_auc"],
            yerr=algorithm_data["std_test_auc"],
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            markersize=8,
            linewidth=2.2,
            capsize=4,
            label=algorithm
        )

    baseline = BASELINES[dataset]

    ax.axhline(
        y=baseline["auc"],
        color="#555555",
        linestyle="--",
        linewidth=2,
        label=f'{baseline["name"]} baseline'
    )

    ax.set_xscale("log")

    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([
        "50", "500", "5,000", "20,000"
    ])

    ax.set_title(
        dataset,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Fitness Evaluation Budget",
        fontsize=11
    )

    ax.set_ylabel(
        "Mean Test AUC",
        fontsize=11
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Test Performance Across Evaluation Budgets",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "figure1_budget_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 1 saved: Budget curves")


# ==========================================================
# 5. FIGURE 2: VALIDATION VS TEST AUC
#    ALL THREE ALGORITHMS ON NETSCIENCE
# ==========================================================

dataset = "Netscience"

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 5),
    sharey=True
)

for ax, algorithm in zip(axes, ALGORITHMS):

    selected = summary[
        (summary["dataset"] == dataset) &
        (summary["algorithm"] == algorithm)
    ].sort_values("budget")

    if selected.empty:
        print(f"Warning: no data for {algorithm} on {dataset}")
        continue

    # Validation AUC
    ax.errorbar(
        selected["budget"],
        selected["mean_validation_auc"],
        yerr=selected["std_validation_auc"],
        marker="o",
        color=COLORS[algorithm],
        linewidth=2.3,
        markersize=7,
        capsize=3,
        label="Validation AUC"
    )

    # Test AUC
    ax.errorbar(
        selected["budget"],
        selected["mean_test_auc"],
        yerr=selected["std_test_auc"],
        marker="s",
        color="#555555",
        linestyle="--",
        linewidth=2.3,
        markersize=7,
        capsize=3,
        label="Test AUC"
    )

    ax.set_xscale("log")

    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([
        "50", "500", "5,000", "20,000"
    ])

    ax.set_title(
        algorithm,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Evaluation Budget",
        fontsize=10
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Mean AUC", fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04)
)

fig.suptitle(
    "Validation vs Test Performance on Netscience",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0.10, 1, 0.91]
)

plt.savefig(
    FIGURES_DIR / "figure2_validation_vs_test_all.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 2 saved: Validation vs test for all algorithms")


# ==========================================================
# 6. FIGURE 3: FEATURE CORRELATION HEATMAP
# ==========================================================

feature_columns = [
    "cn",
    "jaccard",
    "aa",
    "ra"
]

feature_labels = [
    "CN",
    "Jaccard",
    "AA",
    "RA"
]

if FEATURE_FILE.exists():

    feature_data = pd.read_csv(FEATURE_FILE)

    missing_features = [
        column for column in feature_columns
        if column not in feature_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    # If dataset column exists, create one heatmap per dataset.
    if "dataset" in feature_data.columns:

        for dataset_name in feature_data["dataset"].dropna().unique():

            dataset_features = feature_data[
                feature_data["dataset"] == dataset_name
            ]

            correlation = dataset_features[
                feature_columns
            ].corr(method="pearson")

            fig, ax = plt.subplots(figsize=(7, 6))

            image = ax.imshow(
                correlation,
                cmap="coolwarm",
                vmin=-1,
                vmax=1
            )

            ax.set_xticks(range(4))
            ax.set_yticks(range(4))

            ax.set_xticklabels(feature_labels)
            ax.set_yticklabels(feature_labels)

            for i in range(4):
                for j in range(4):

                    value = correlation.iloc[i, j]

                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="black"
                    )

            ax.set_title(
                f"Feature Correlation - {dataset_name}",
                fontsize=13,
                fontweight="bold"
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Pearson Correlation"
            )

            plt.tight_layout()

            safe_name = str(dataset_name).replace(" ", "_")

            plt.savefig(
                FIGURES_DIR /
                f"figure3_feature_correlation_{safe_name}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        print("\nFigure 3 saved: Dataset-wise feature heatmaps")

    else:

        # A single heatmap if no dataset column is supplied.
        correlation = feature_data[
            feature_columns
        ].corr(method="pearson")

        fig, ax = plt.subplots(figsize=(7, 6))

        image = ax.imshow(
            correlation,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))

        ax.set_xticklabels(feature_labels)
        ax.set_yticklabels(feature_labels)

        for i in range(4):
            for j in range(4):

                value = correlation.iloc[i, j]

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black"
                )

        ax.set_title(
            "Feature Correlation",
            fontsize=13,
            fontweight="bold"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Pearson Correlation"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "figure3_feature_correlation.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("\nFigure 3 saved: Feature correlation")

else:

    print("\nFigure 3 skipped.")
    print(
        "Feature file not found:",
        FEATURE_FILE
    )
    print(
        "Create feature_values.csv using actual feature "
        "values from your graph splits."
    )


# ==========================================================
# 7. FIGURE 4: MEAN OPTIMIZED FEATURE WEIGHTS
#    USING ALL 10 SEEDS
# ==========================================================

weight_columns = {
    "CN": "weight_cn",
    "Jaccard": "weight_jaccard",
    "AA": "weight_aa",
    "RA": "weight_ra"
}

weight_data = results[
    (results["dataset"] == WEIGHT_DATASET) &
    (results["budget"] == WEIGHT_BUDGET)
].copy()

if weight_data.empty:
    raise ValueError(
        f"No weight data for {WEIGHT_DATASET}, "
        f"budget {WEIGHT_BUDGET}"
    )

# Calculate mean and standard deviation across seeds
weight_summary = (
    weight_data
    .groupby("algorithm")
    .agg({
        column: ["mean", "std", "min", "max"]
        for column in weight_columns.values()
    })
)

weight_summary.to_csv(
    RESULTS_DIR / "weight_summary_all_seeds.csv"
)

print("\nWEIGHT SUMMARY ACROSS SEEDS")
print(weight_summary.to_string())


fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(weight_columns))
bar_width = 0.24

for i, algorithm in enumerate(ALGORITHMS):

    algorithm_rows = weight_data[
        weight_data["algorithm"] == algorithm
    ]

    if algorithm_rows.empty:
        continue

    means = [
        algorithm_rows[column].mean()
        for column in weight_columns.values()
    ]

    stds = [
        algorithm_rows[column].std()
        for column in weight_columns.values()
    ]

    stds = [
        0 if pd.isna(value) else value
        for value in stds
    ]

    positions = x + (i - 1) * bar_width

    bars = ax.bar(
        positions,
        means,
        width=bar_width,
        color=COLORS[algorithm],
        label=algorithm,
        yerr=stds,
        capsize=3,
        edgecolor="white",
        linewidth=0.8
    )

    # Add labels, including zero weights
    for bar, value in zip(bars, means):

        label_y = value + 0.025 if value > 0 else 0.025

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

ax.set_xticks(x)

ax.set_xticklabels(
    list(weight_columns.keys()),
    fontsize=11
)

ax.set_ylim(0, 1.15)

ax.set_xlabel(
    "Link Prediction Features",
    fontsize=11
)

ax.set_ylabel(
    "Mean Optimized Weight",
    fontsize=11
)

ax.set_title(
    f"Mean Optimized Feature Weights: "
    f"{WEIGHT_DATASET}, Budget {WEIGHT_BUDGET:,}",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.set_axisbelow(True)

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "figure4_mean_weight_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 4 saved: Mean feature weights across seeds")


# ==========================================================
# 8. JACCARD WEIGHT ANALYSIS ACROSS ALL 10 SEEDS
# ==========================================================

print("\n" + "=" * 65)
print("JACCARD WEIGHT ANALYSIS")
print("=" * 65)

jaccard_summary = (
    results[
        (results["dataset"] == WEIGHT_DATASET) &
        (results["budget"] == WEIGHT_BUDGET)
    ]
    .groupby("algorithm")["weight_jaccard"]
    .agg(
        mean="mean",
        std="std",
        minimum="min",
        maximum="max",
        number_of_runs="count"
    )
)

print(jaccard_summary.to_string())

jaccard_summary.to_csv(
    RESULTS_DIR / "jaccard_weight_analysis.csv"
)


# ==========================================================
# 9. CHECK EXPERIMENT COUNTS AND EVALUATION BUDGETS
# ==========================================================

print("\n" + "=" * 65)
print("EXPERIMENT VERIFICATION")
print("=" * 65)

print("\nRuns per dataset, algorithm and budget:")

run_counts = (
    results
    .groupby(["dataset", "algorithm", "budget"])
    .size()
    .rename("runs")
)

print(run_counts.to_string())

print("\nEvaluation count check:")

results["evaluation_check"] = (
    results["evaluations"] == results["budget"]
)

incorrect = results[
    ~results["evaluation_check"]
]

if incorrect.empty:
    print("All runs used exactly their specified budgets.")
else:
    print("WARNING: Some runs have incorrect evaluation counts.")
    print(
        incorrect[
            [
                "dataset",
                "algorithm",
                "budget",
                "seed",
                "evaluations"
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)

print("\nGenerated files are in:")
print(FIGURES_DIR.resolve())

print("\nSummary CSV files:")
print(RESULTS_DIR / "final_comparison_summary.csv")
print(RESULTS_DIR / "weight_summary_all_seeds.csv")
print(RESULTS_DIR / "jaccard_weight_analysis.csv")

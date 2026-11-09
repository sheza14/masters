
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Folder paths
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BUDGETS = [50, 500, 5000, 20000]

COLUMNS = [
    "dataset", "budget", "seed",
    "best_validation_auc", "final_test_auc",
    "evaluations", "weight_cn", "weight_jaccard",
    "weight_aa", "weight_ra"
]


def load_results(filename, fix_missing_header=False):
    path = ROOT / filename

    if not path.exists():
        print(f"Missing file: {filename}")
        return None

    # The uploaded Firefly CSV has no proper header row.
    if fix_missing_header:
        df = pd.read_csv(path, header=None, names=COLUMNS)
    else:
        df = pd.read_csv(path)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        print(f"{filename}: missing columns {sorted(missing)}")
        return None

    for col in ["budget", "seed", "evaluations",
                "final_test_auc", "best_validation_auc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "dataset", "budget", "seed", "final_test_auc"
    ])

    # Check whether each dataset-budget-seed combination is unique.
    keys = ["dataset", "budget", "seed"]
    duplicates = df.duplicated(keys, keep=False)

    print(f"\n{filename}: {len(df)} rows")
    print(f"Repeated run identities: {int(duplicates.sum())}")

    if duplicates.any():
        print("WARNING: resolve repeated run identities before "
              "using this file for final scientific comparisons.")
        print(df.loc[duplicates, keys].sort_values(keys).head(12))

    if not df["final_test_auc"].between(0, 1).all():
        raise ValueError(f"Invalid ROC-AUC values in {filename}")

    return df


pso = load_results("pso_results.csv")
firefly = load_results("firefly_results.csv",
                       fix_missing_header=True)
abc = load_results("abc_results.csv")

# Plot budget curves only from files with unique run identities.
frames = []
for method, df in [
    ("PSO", pso),
    ("Firefly", firefly),
    ("ABC", abc)
]:
    if df is None:
        continue

    keys = ["dataset", "budget", "seed"]
    if df.duplicated(keys).any():
        print(f"{method}: excluded from budget plots until "
              "duplicate records are resolved.")
        continue

    df = df.copy()
    df["method"] = method
    frames.append(df)

if frames:
    results = pd.concat(frames, ignore_index=True)

    for dataset in ["Karate Club", "Netscience"]:
        subset = results[results["dataset"] == dataset]

        if subset.empty:
            print(f"No valid runs available for {dataset}")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for method in subset["method"].unique():
            part = subset[subset["method"] == method]
            summary = (
                part.groupby("budget")["final_test_auc"]
                .agg(["mean", "std", "count"])
                .reindex(BUDGETS)
            )

            ax.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                marker="o",
                capsize=4,
                label=method
            )

        # Strongest classical baseline from the recorded results
        baseline = {
            "Karate Club": 0.6720,
            "Netscience": 0.8788
        }[dataset]

        ax.axhline(
            baseline, linestyle="--",
            label=f"Best classical baseline ({baseline:.4f})"
        )

        ax.set_xscale("log")
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels(["50", "500", "5,000", "20,000"])
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Held-out test ROC-AUC")
        ax.set_title(f"{dataset}: performance vs budget")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        name = dataset.lower().replace(" ", "_")
        fig.savefig(OUT / f"budget_curve_{name}.png", dpi=300)
        fig.savefig(OUT / f"budget_curve_{name}.pdf")
        plt.close(fig)

# Classical baseline comparison
baseline_data = pd.DataFrame([
    ["Karate Club", "CN", 0.6597],
    ["Karate Club", "Jaccard", 0.6134],
    ["Karate Club", "Adamic-Adar", 0.6720],
    ["Karate Club", "Resource Allocation", 0.6701],
    ["Netscience", "CN", 0.8780],
    ["Netscience", "Jaccard", 0.8757],
    ["Netscience", "Adamic-Adar", 0.8788],
    ["Netscience", "Resource Allocation", 0.8783],
], columns=["dataset", "method", "auc"])

fig, ax = plt.subplots(figsize=(9, 5))
baseline_data.pivot(
    index="method", columns="dataset", values="auc"
).plot(kind="bar", ax=ax)

ax.set_ylabel("ROC-AUC")
ax.set_title("Classical link prediction baselines")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "classical_baseline_comparison.png", dpi=300)
fig.savefig(OUT / "classical_baseline_comparison.pdf")
plt.close(fig)

print("\nFinished. Check the figures folder.")
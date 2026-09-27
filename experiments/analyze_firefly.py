import pandas as pd
from pathlib import Path

# Input and output files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results" / "firefly_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "firefly_summary.csv"

# Classical baseline AUCs
BASELINES = {
    "Karate Club": 0.6720,
    "Netscience": 0.8538
}

# Load Firefly results
df = pd.read_csv(INPUT_FILE)

# Calculate mean and standard deviation for each dataset and budget
summary = (
    df.groupby(["dataset", "budget"])["final_test_auc"]
    .agg(mean_test_auc="mean", std_test_auc="std")
    .reset_index()
)

# Add classical baseline and difference
summary["classical_baseline"] = summary["dataset"].map(BASELINES)
summary["difference"] = (
    summary["mean_test_auc"] - summary["classical_baseline"]
)

# Round values for display
summary[["mean_test_auc", "std_test_auc",
         "classical_baseline", "difference"]] = (
    summary[["mean_test_auc", "std_test_auc",
             "classical_baseline", "difference"]].round(4)
)

# Save summary
summary.to_csv(OUTPUT_FILE, index=False)

print("\nFIREFLY SUMMARY")
print("=" * 75)
print(summary.to_string(index=False))

print("\nSummary saved to:")
print(OUTPUT_FILE)
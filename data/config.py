from pathlib import Path

# Dataset root.
DATA_ROOT = Path("/scratch/cora_to_zoey/data")


# Result CSVs are stored under RESULTS_ROOT/<exp_name>/.
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

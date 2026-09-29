from pathlib import Path

#NOTE Set this to your dataset folder.
# DATA_ROOT = Path("../data")
DATA_ROOT = Path("/scratch/cora_to_zoey/data")


#NOTE Results CSVs go to RESULTS_ROOT/<exp_name>/; keep in sync with OUT_DIR in the
# submit_*.sh scripts and RESULT_ROOT in experiments_real/plot/make_plots.R.
RESULTS_ROOT = Path("/scratch/cora_to_zoey/chdc/exp_real_adaptive/results")

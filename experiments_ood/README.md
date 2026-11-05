# Additional OOD experiments

This directory contains the current four-benchmark OOD experiment pipeline. Python
generates the experiment results, Bash launches/submits jobs, and `make_plots.R`
generates the numerical report directly from completed raw-result files. The
retired ISOLET and Languages experiments are not part of this pipeline.

## Data and output locations

- Original datasets: `--data-root PATH` or `CHDC_DATA_ROOT`; otherwise the existing
  `DATA_ROOT` setting in `data/config.py` is used. This is a dataset cache, not the
  raw experiment-results directory.
- Raw experiment results: `CHDC_RESULTS_ROOT` or `--results-root PATH`, defaulting
  to `<repository>/results`.
- Generated tables: `<results-root>/plots/ood`.
- Manuscript updates: run this repository's report generator first, then copy its
  complete generated TeX files into the Overleaf checkout. Do not edit the
  generated tables or reconstruct inputs from manuscript files.

Each dataset has its own raw-result subdirectory:

| Dataset ID | Directory | Nominal alpha | Operating-point grid |
|---|---|---|---|
| `mnist` | `results/ood_mnist` | 0.05 | 0.025, 0.0375, 0.05, 0.0625, 0.075 |
| `fashion_mnist` | `results/ood_fashion_mnist` | 0.05 | 0.025, 0.0375, 0.05, 0.0625, 0.075 |
| `pamap2` | `results/ood_pamap2` | 0.1 | 0.05, 0.075, 0.1, 0.125, 0.15 |
| `har` | `results/ood_har` | 0.1 | 0.05, 0.075, 0.1, 0.125, 0.15 |

The grid is the nominal alpha multiplied by `0.5, 0.75, 1, 1.25, 1.5`.

## Experimental protocol

Use the current main experiments' four datasets, in-distribution/OOD labels,
hypervector dimension 10,000, and encoding families. The image data use an
80/15/5% ID training/calibration/test split; PAMAP2 and UCI HAR use 54/36/10%.
Feature-quantization ranges are fitted to ID training observations only, then
applied unchanged to calibration, test, and OOD observations. This corrects the
older experiments' pooled preprocessing. Use all intended OOD observations;
there is no ordered-prefix truncation to match the ID test count. AUROC does not
require balancing the numbers of ID and OOD examples.

All three methods use the same encoded observations and train-only prototypes.
Let `c_k(x)` be raw cosine similarity in `[-1, 1]`, and let
`delta_k(x) = (1 + c_k(x)) / 2`. All reported scores are oriented so that higher
values indicate ID:

- **Max similarity:** `max_k c_k(x)`.
- **Discount:** `max_k delta_k(x)^2 / sum_j delta_j(x)`.
- **Energy:** negative energy, `T * logsumexp(c_k(x) / T)`, with default `T = 1`.
  The input is raw cosine, not shifted similarity. The same temperature is used
  for every repetition and is recorded in the outputs; no test-OOD tuning is
  performed.

AUROC evaluates these raw OOD score rankings. A pooled conformal p-value is a
nondecreasing transformation of the corresponding raw score and is not treated
as a separate ranking baseline. These experiments compare scoring rules on HDC
representations, not all variants of neural energy-based OOD detection.

For operating points, calibrate each method's cutoff using that method's scores
on the held-out ID calibration observations. With `n_cal` calibration scores,
let `ell = floor(alpha * (n_cal + 1))`. The cutoff is the `ell`th smallest score,
or negative infinity when `ell = 0`. Reject an observation only when its score
is **strictly below** that cutoff. This handles ties conservatively. Report:

- ID false-rejection rate: rejected ID test observations / all ID test observations.
- OOD rejection rate: rejected OOD observations / all OOD observations.

Neither cutoff selection nor preprocessing uses OOD outcomes. Conformal
calibration controls ID false rejection under exchangeability; it does not
guarantee rejection of arbitrary OOD distributions.

## Run one job locally

From the repository root, choose the Python environment containing the main
experiment dependencies and the local dataset cache:

```bash
PYTHON=/path/to/python bash experiments_ood/submit_har.sh \
  --local --seed 1 --data-root /path/to/dataset-cache
```

One default seed-group job runs four repetitions. The wrappers also accept
`--results-root`, `--repetitions`, `--temperature`, and `--force`; use `--force`
only when intentionally replacing an already completed seed group. `PYTHON`
selects an existing interpreter; the launchers do not install dependencies.

For the approved local Windows environment, invoke the same Bash launcher through
WSL from PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 -- env 'PYTHON=C:/Users/liang/anaconda3/envs/hdc/python.exe' bash /mnt/c/Users/liang/Documents/GitHub/conformalHDC/experiments_ood/submit_har.sh --local --seed 1 --data-root C:/Users/liang/Documents/GitHub/conformalHDC/data
```

The launcher converts WSL paths for the selected Windows Python interpreter; the
Python experiment and data-processing functions are the same ones used remotely.

Direct Python execution is also available:

```bash
python experiments_ood/ood_har.py 1 --data-root /path/to/dataset-cache
```

Do not replace the local launcher test with a different analysis or generation
script. If loading, plotting, or execution fails, report the issue before
changing the experiment protocol or environment.

## Submit the complete remote runs

Configure `CHDC_DATA_ROOT` when the existing `data/config.py` dataset-cache path
does not describe the server. Outputs default to the remote checkout's
`results` directory; `CHDC_RESULTS_ROOT` can explicitly override it.

```bash
bash experiments_ood/submit_mnist.sh --dry-run
bash experiments_ood/submit_fashion_mnist.sh --dry-run
bash experiments_ood/submit_pamap2.sh --dry-run
bash experiments_ood/submit_har.sh --dry-run

bash experiments_ood/submit_mnist.sh
bash experiments_ood/submit_fashion_mnist.sh
bash experiments_ood/submit_pamap2.sh
bash experiments_ood/submit_har.sh
```

Defaults are seed groups 1--25, four repetitions per group, and thus 100
repetitions per dataset. The default Slurm partition is `biodatascience.p`;
`CHDC_PARTITION`, `CHDC_JOB_MEMORY`, `CHDC_JOB_TIME`, and `CHDC_JOB_CPUS` override
the submission resources. `--seed 1` or `--seeds 1 2 3` limits the selected groups.
An optional `CHDC_ENV_PATH` activates the configured Python environment.

## Raw-result files and completion

For seed group 1, a successful job writes:

```text
results/ood_har/seed1_ood.csv
results/ood_har/seed1_operating.csv
results/ood_har/seed1_complete.json
```

The AUROC file has one row per repetition and method. The operating-point file
has one row per repetition, method, and alpha. Split counts, score temperature,
nominal alpha, rejection counts, and provenance accompany the metrics. The JSON
completion marker is created only after both CSVs are successfully written and
records their SHA256 checksums, the exact repetitions, methods, alpha grid, and
generation metadata. A failed or partial seed group is not a completed job.

## Generate the report

After all four datasets finish:

```bash
Rscript experiments_ood/make_plots.R
```

The default full report requires all four datasets and repetition states 1--100
for each. It checks completion markers and checksums, complete method/alpha grids,
duplicate states, temperatures, numerical-library versions, split counts, and
metric/count consistency before
creating any report output. Required R packages are `jsonlite` and `digest`; the
generator does not auto-install packages or silently manufacture missing cells.

Full output files are:

```text
results/plots/ood/ood_summary_table.tex
results/plots/ood/ood_operating_mnist.tex
results/plots/ood/ood_operating_fashion_mnist.tex
results/plots/ood/ood_operating_pamap2.tex
results/plots/ood/ood_operating_har.tex
results/plots/ood/ood_auc_summary.csv
results/plots/ood/ood_operating_summary.csv
results/plots/ood/ood_report_metadata.json
```

TeX cells are mean (standard error) across repetitions, with equal weight for
each repetition. The operating-point rates are fractions in `[0, 1]`. AUROC is
summarized once per repetition and method, not repeatedly over the alpha grid.
The JSON report records exact input markers and checksums. An explicitly changed
full-run target can be specified with `--expected-repetitions N`.

After the one-job local test, a diagnostic report can be requested explicitly:

```bash
Rscript experiments_ood/make_plots.R --datasets har --allow-partial
```

These output names start with `partial_`, the tables visibly say **Partial run**,
and their repetition counts are displayed. They never overwrite the full report
filenames. A single local job is not a four-benchmark paper result and must not be
copied into the manuscript as a completed study.

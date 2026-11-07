# OOD ROC and score diagnostics

The diagnostic pipeline reuses the existing experiment's data preparation,
encoding, prototypes, score functions and ID-only cutoff function. It leaves the
original Python/Bash entrypoints and the four `results/ood_*` result folders
unchanged. One shared runner handles MNIST, Fashion-MNIST, PAMAP2 and UCI HAR.

The defaults reproduce the existing 100 states using **20 seed groups with five
repetitions per group**. For group `g`, the states are `5*(g-1)+1` through `5*g`.
Keep the same five-repetition grouping for these full runs.

## Locations and preservation

`--results-root` selects the base results directory, normally `<repo>/results`.
The new exports go only to:

```text
results/ood_diagnostics/ood_mnist/
results/ood_diagnostics/ood_fashion_mnist/
results/ood_diagnostics/ood_pamap2/
results/ood_diagnostics/ood_har/
results/logs/ood_diagnostics/
```

Each successful group writes score and ROC CSVs compressed with gzip, AUROC and
operating-point CSVs, and a checksummed completion marker. Score rows retain the
input sample index, original class label and partition. ROC curves treat OOD as
positive, use `-ID_score`, and retain all thresholds. Calibrated rejection remains
the original strict `ID_score < cutoff` rule. A compatible completed diagnostic
group is skipped; incompatible or partial existing outputs cause a clear error.
There is no force-overwrite option.

The baseline-versus-diagnostic comparisons, baseline checksum snapshots and their
reports are separate temporary verification artifacts outside both repositories.
They are not part of this directory or its submission workflow.

## One local job

Use an existing Python environment and cached datasets. For a one-repetition
smoke test, select a separate temporary base results root:

```bash
PYTHON=/path/to/hdc/python bash experiments_ood/submit_ood_diagnostics.sh \
  --local --dataset har --seed 1 --repetitions 1 \
  --data-root /path/to/data --results-root /path/to/temporary-smoke-results
```

The Windows HDC Python environment can be invoked through WSL, as with the
original OOD launcher:

```powershell
wsl.exe -d Ubuntu-22.04 -- env 'PYTHON=C:/Users/liang/anaconda3/envs/hdc/python.exe' bash /mnt/c/Users/liang/Documents/GitHub/conformalHDC/experiments_ood/submit_ood_diagnostics.sh --local --dataset har --seed 1 --repetitions 1 --data-root C:/Users/liang/Documents/GitHub/conformalHDC/data --results-root C:/path/to/temporary-smoke-results
```

## Slurm submission and monitoring

Activate the existing HDC environment and pull the repository first. Preview the
complete plan without creating files, running Python or submitting anything:

```bash
PYTHON="$(command -v python)" bash experiments_ood/submit_ood_diagnostics.sh \
  --dry-run --data-root /home/liangz25/conformalHDC/data
```

The complete run contains **80 tasks: 20 jobs per dataset, five repetitions per
job**, or 100 repetitions per dataset. It uses **one global array capped at 40
concurrent tasks**; four independently capped arrays are not used. The submitter
refuses to proceed while the user already has other queued or running jobs, so
the initial concurrency budget is explicit.

```bash
PYTHON="$(command -v python)" bash experiments_ood/submit_ood_diagnostics.sh \
  --monitor --data-root /home/liangz25/conformalHDC/data \
  --results-root /home/liangz25/conformalHDC/results
```

Use `--dataset mnist,fashion_mnist`, `--seed N` or `--seeds N ...` for an explicit
subset, and `--max-concurrent N` for a smaller cap (never above 40). Dataset order
and seed groups are recorded in the submission's `tasks.tsv`. Submission records
and per-task stdout/stderr remain under `results/logs/ood_diagnostics`.

The monitor checks this array only and returns an error on a failed task or when
accounting cannot confirm that every task finished with exit zero. It does not
change the protocol, install packages, download missing data, or silently retry
failed scientific jobs. Report missing inputs and non-trivial failures before
changing the environment or execution plan.

Resources default to the existing `biodatascience.p` partition, 8 GB, one CPU and
three hours per task. The existing `CHDC_PARTITION`, `CHDC_JOB_MEMORY`,
`CHDC_JOB_TIME` and `CHDC_JOB_CPUS` environment variables override those resources.
`PYTHON` selects the already installed interpreter; `CHDC_ENV_PATH` optionally
selects a Linux environment with `bin/activate`.

## Repository-generated plots

The plotting entrypoint is `experiments_ood/make_roc_plots.R`. It reads the new
diagnostic exports and writes figures under `results/plots/ood_diagnostics`.
Full output requires all four datasets and all 100 states. A deliberately small
local run must use its explicit `--allow-partial` mode and remain labelled as a
partial preview. No generated output should be hand-edited or reconstructed from
the manuscript.

# ConformalHDC

Code for [ConformalHDC: Uncertainty-Aware Hyperdimensional Computing with Application to Neural Decoding](https://arxiv.org/abs/2602.21446).

## Repository structure

| Directory | Purpose |
| --- | --- |
| `conformal_inference/` | Conformal scores and prediction methods. |
| `hdc_encoder/` | Hyperdimensional encoders. |
| `data/` | Dataset loading, preprocessing and configuration. |
| `experiments_real/` | Benchmark and neural-decoding experiments; R reporting in `plot/`. |
| `experiments_ood/` | OOD experiments and diagnostics. |
| `notebook/` | Paper illustration and neural preprocessing notebooks. |

## Workflow

Python (`.py`) scripts run experiments. Bash (`.sh`) scripts launch experiments and submit Slurm jobs. R (`.R`) scripts generate plots and tables from saved results.

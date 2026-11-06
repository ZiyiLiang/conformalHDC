#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm copies this launcher; submit_ood.sh sets the repository script directory.
if [[ ! -f "$SCRIPT_DIR/run_ood.sh" && -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="$PWD"
fi
exec bash "$SCRIPT_DIR/run_ood.sh" pamap2 "$@"
